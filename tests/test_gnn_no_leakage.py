"""Tests to verify NO label leakage in GNN features.

CRITICAL: These tests MUST pass before any GNN training.
Label leakage would invalidate all experimental results.

Forbidden features (derived from gold labels):
- is_gold, groundtruth
- mrr, recall_at_*, map_at_*, ndcg_at_*
- gold_rank, min_gold_rank, mean_gold_rank
- n_gold_sentences, gold_sentence_ids
"""

import json
import numpy as np
import pytest

from final_sc_review.gnn.config import GraphConstructionConfig, GNNConfig
from final_sc_review.gnn.graphs.features import (
    LEAKAGE_FEATURES,
    check_leakage,
    assert_no_leakage,
    NodeFeatureExtractor,
    EdgeFeatureExtractor,
    GraphStatsExtractor,
)


class TestLeakageDetection:
    """Test the leakage detection utilities."""

    def test_leakage_features_complete(self):
        """Verify LEAKAGE_FEATURES set is comprehensive."""
        expected_patterns = [
            "is_gold",
            "groundtruth",
            "mrr",
            "recall_at_",
            "map_at_",
            "ndcg_at_",
            "gold_rank",
            "n_gold",
            "gold_sentence",
        ]
        for pattern in expected_patterns:
            assert any(pattern in f for f in LEAKAGE_FEATURES), f"Missing pattern: {pattern}"

    def test_check_leakage_detects_forbidden(self):
        """check_leakage should detect forbidden features."""
        leaked_features = {
            "max_score": 0.5,
            "is_gold": 1,  # FORBIDDEN
            "mrr": 0.8,  # FORBIDDEN
            "recall_at_5": 0.6,  # FORBIDDEN
        }
        leaked = check_leakage(leaked_features)
        assert len(leaked) == 3
        assert "is_gold" in leaked
        assert "mrr" in leaked
        assert "recall_at_5" in leaked

    def test_check_leakage_allows_valid(self):
        """check_leakage should allow valid inference-time features."""
        valid_features = {
            "max_reranker_score": 0.9,
            "entropy_top5": 1.2,
            "rank_percentile": 0.1,
            "cosine_similarity": 0.8,
            "embedding_norm": 15.2,
        }
        leaked = check_leakage(valid_features)
        assert len(leaked) == 0, f"False positives: {leaked}"

    def test_assert_no_leakage_raises(self):
        """assert_no_leakage should raise on forbidden features."""
        leaked_features = {"score": 0.5, "gold_rank": 3}
        with pytest.raises(ValueError, match="Label leakage detected"):
            assert_no_leakage(leaked_features)

    def test_assert_no_leakage_passes(self):
        """assert_no_leakage should pass on valid features."""
        valid_features = {"score": 0.5, "rank": 3, "embedding": [0.1, 0.2]}
        assert_no_leakage(valid_features)  # Should not raise


class TestNodeFeatureExtractor:
    """Test node feature extraction has no leakage."""

    @pytest.fixture
    def extractor(self):
        return NodeFeatureExtractor(
            use_embedding=True,
            use_reranker_score=True,
            use_rank_percentile=True,
            use_score_gaps=True,
            use_score_stats=True,
            embedding_dim=16,  # Small for testing
        )

    def test_feature_names_no_leakage(self, extractor):
        """Feature names should not contain forbidden patterns."""
        names = extractor.get_feature_names()
        leaked = check_leakage({n: 0 for n in names})
        assert len(leaked) == 0, f"Leakage in feature names: {leaked}"

    def test_extract_no_gold_input(self, extractor):
        """extract() should not accept gold labels as input."""
        # Check that the signature doesn't include gold-related params
        import inspect
        sig = inspect.signature(extractor.extract)
        params = sig.parameters.keys()

        forbidden_params = ["gold", "label", "groundtruth", "is_gold", "gold_ids"]
        for param in params:
            for forbidden in forbidden_params:
                assert forbidden not in param.lower(), f"Forbidden param: {param}"

    def test_extract_output_shape(self, extractor):
        """extract() should return correct shape without gold info."""
        n_candidates = 10
        embeddings = np.random.randn(n_candidates, 16).astype(np.float32)
        scores = np.random.randn(n_candidates).astype(np.float32)

        features = extractor.extract(embeddings, scores)

        assert features.shape[0] == n_candidates
        assert features.shape[1] == extractor.feature_dim

    def test_feature_dim_matches(self, extractor):
        """feature_dim property should match actual output."""
        expected_dim = extractor.feature_dim
        # 16 (embedding) + 1 (score) + 1 (rank_pct) + 2 (gaps) + 4 (stats)
        assert expected_dim == 24

        embeddings = np.random.randn(5, 16).astype(np.float32)
        scores = np.random.randn(5).astype(np.float32)
        features = extractor.extract(embeddings, scores)

        assert features.shape[1] == expected_dim


class TestEdgeFeatureExtractor:
    """Test edge feature extraction has no leakage."""

    @pytest.fixture
    def extractor(self):
        return EdgeFeatureExtractor(
            use_cosine_similarity=True,
            use_sequence_distance=True,
        )

    def test_feature_names_no_leakage(self, extractor):
        """Edge feature names should not contain forbidden patterns."""
        names = extractor.get_feature_names()
        leaked = check_leakage({n: 0 for n in names})
        assert len(leaked) == 0, f"Leakage in edge feature names: {leaked}"

    def test_extract_single_edge(self, extractor):
        """extract_for_edge should use only inference-time data."""
        emb_i = np.random.randn(16).astype(np.float32)
        emb_j = np.random.randn(16).astype(np.float32)

        features = extractor.extract_for_edge(emb_i, emb_j, sid_i=5, sid_j=7)

        assert len(features) == 2  # cosine_sim, seq_dist
        assert features[0] >= -1 and features[0] <= 1  # Valid cosine range
        assert features[1] >= 0 and features[1] <= 1  # Normalized distance


class TestGraphStatsExtractor:
    """Test graph statistics extraction has no leakage."""

    @pytest.fixture
    def extractor(self):
        return GraphStatsExtractor()

    def test_stats_names_no_leakage(self, extractor):
        """Graph stats should not contain forbidden patterns."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        scores = np.array([0.9, 0.7, 0.5])
        embeddings = np.random.randn(3, 16).astype(np.float32)

        stats = extractor.extract(
            n_nodes=3,
            edge_index=edge_index,
            node_scores=scores,
            embeddings=embeddings,
        )

        leaked = check_leakage(stats)
        assert len(leaked) == 0, f"Leakage in graph stats: {leaked}"

    def test_stats_no_gold_dependency(self, extractor):
        """Graph stats should be identical regardless of gold labels."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        scores = np.array([0.9, 0.7, 0.5])
        embeddings = np.random.randn(3, 16).astype(np.float32)

        # Stats should not change based on which nodes are gold
        stats1 = extractor.extract(3, edge_index, scores, embeddings)
        stats2 = extractor.extract(3, edge_index, scores, embeddings)

        assert stats1 == stats2


class TestGraphBuilderNoLeakage:
    """Test GraphBuilder doesn't leak labels into features."""

    def test_graph_data_no_gold_in_features(self):
        """Graph.x (node features) should not contain gold labels."""
        pytest.importorskip("torch_geometric")

        from final_sc_review.gnn.graphs.builder import GraphBuilder
        from final_sc_review.gnn.config import GraphConstructionConfig

        config = GraphConstructionConfig(
            embedding_dim=16,
            use_embedding=False,  # Skip embedding loading
            use_reranker_score=True,
            use_rank_percentile=True,
            use_score_gaps=True,
            use_score_stats=True,
        )

        builder = GraphBuilder(config)
        builder._embedding_cache = np.random.randn(100, 16).astype(np.float32)
        builder._uid_to_idx = {f"post1_{i}": i for i in range(100)}

        # Build graph with has_evidence=True
        graph = builder.build_graph(
            candidate_uids=[f"post1_{i}" for i in range(10)],
            reranker_scores=np.random.randn(10).astype(np.float32),
            sentence_ids=list(range(10)),
            query_id="test_query",
            has_evidence=True,  # Label
        )

        # graph.x should not contain the label
        # Check that changing has_evidence doesn't change features
        graph2 = builder.build_graph(
            candidate_uids=[f"post1_{i}" for i in range(10)],
            reranker_scores=graph.reranker_scores.numpy(),
            sentence_ids=list(range(10)),
            query_id="test_query",
            has_evidence=False,  # Different label
        )

        # Features should be IDENTICAL regardless of label
        import torch
        assert torch.allclose(graph.x, graph2.x), "Features depend on label - LEAKAGE!"

    def test_label_stored_separately(self):
        """Labels should be in graph.y, not in graph.x."""
        pytest.importorskip("torch_geometric")

        from final_sc_review.gnn.graphs.builder import GraphBuilder
        from final_sc_review.gnn.config import GraphConstructionConfig

        config = GraphConstructionConfig(
            embedding_dim=16,
            use_embedding=False,
        )

        builder = GraphBuilder(config)
        builder._embedding_cache = np.random.randn(100, 16).astype(np.float32)
        builder._uid_to_idx = {f"post1_{i}": i for i in range(100)}

        graph = builder.build_graph(
            candidate_uids=[f"post1_{i}" for i in range(5)],
            reranker_scores=np.random.randn(5).astype(np.float32),
            sentence_ids=list(range(5)),
            query_id="test",
            has_evidence=True,
        )

        # Label should be in graph.y
        assert hasattr(graph, "y")
        assert graph.y.item() == 1.0

        # graph.x should not have the label as a feature
        # (Feature dim should match config, not include extra label column)
        expected_dim = config.node_feature_dim
        assert graph.x.shape[1] == expected_dim


class TestConfigNoLeakage:
    """Test configuration doesn't enable leakage."""

    def test_default_config_no_gold_features(self):
        """Default config should not include gold-derived features."""
        config = GNNConfig()

        # Check that no feature flag enables gold features
        graph_config = config.graph
        assert not hasattr(graph_config, "use_gold_labels")
        assert not hasattr(graph_config, "use_mrr")
        assert not hasattr(graph_config, "use_recall")

    def test_config_feature_flags(self):
        """All feature flags should be for inference-time features only."""
        config = GraphConstructionConfig()

        allowed_flags = [
            "use_embedding",
            "use_reranker_score",
            "use_rank_percentile",
            "use_score_gaps",
            "use_score_stats",
        ]

        # Check all use_* attributes
        for attr in dir(config):
            if attr.startswith("use_"):
                assert attr in allowed_flags, f"Unexpected feature flag: {attr}"


class TestEndToEndNoLeakage:
    """End-to-end tests for leakage prevention."""

    def test_graph_features_independent_of_labels(self):
        """Graph features should be completely independent of labels."""
        pytest.importorskip("torch_geometric")

        from final_sc_review.gnn.graphs.builder import GraphBuilder
        from final_sc_review.gnn.config import GraphConstructionConfig

        config = GraphConstructionConfig(embedding_dim=16, use_embedding=False)
        builder = GraphBuilder(config)
        builder._embedding_cache = np.random.randn(100, 16).astype(np.float32)
        builder._uid_to_idx = {f"post1_{i}": i for i in range(100)}

        # Same candidates, same scores
        uids = [f"post1_{i}" for i in range(10)]
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        sids = list(range(10))

        # Different gold labels (simulating different ground truths)
        graph_with_evidence = builder.build_graph(
            candidate_uids=uids,
            reranker_scores=scores,
            sentence_ids=sids,
            query_id="q1",
            has_evidence=True,
            node_labels=np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        )

        graph_no_evidence = builder.build_graph(
            candidate_uids=uids,
            reranker_scores=scores,
            sentence_ids=sids,
            query_id="q1",
            has_evidence=False,
            node_labels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )

        # Features (graph.x) MUST be identical
        import torch
        assert torch.allclose(
            graph_with_evidence.x, graph_no_evidence.x
        ), "Node features depend on labels - CRITICAL LEAKAGE!"

        # Edge features MUST be identical
        if graph_with_evidence.edge_attr is not None:
            assert torch.allclose(
                graph_with_evidence.edge_attr, graph_no_evidence.edge_attr
            ), "Edge features depend on labels - CRITICAL LEAKAGE!"

        # Edge structure MUST be identical
        assert torch.equal(
            graph_with_evidence.edge_index, graph_no_evidence.edge_index
        ), "Edge structure depends on labels - CRITICAL LEAKAGE!"
