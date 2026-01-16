"""Node and edge feature extraction for GNN graphs.

CRITICAL: NO GOLD LABELS OR DERIVED METRICS ALLOWED.
Features must be available at inference time only.

Forbidden features (cause label leakage):
- is_gold, groundtruth
- mrr, recall_at_*, map_at_*, ndcg_at_*
- gold_rank, min_gold_rank, mean_gold_rank
- n_gold_sentences, gold_sentence_ids

Allowed features (inference-time):
- Embeddings (dense, sparse, colbert)
- Reranker scores (raw, calibrated)
- Rank position (relative to other candidates)
- Score statistics (gaps, z-score, percentile)
- Text features (length, overlap with criterion)
- Graph structure features (degree, centrality)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)

# Features that are FORBIDDEN due to label leakage
LEAKAGE_FEATURES: Set[str] = {
    "is_gold",
    "groundtruth",
    "mrr",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "map_at_1",
    "map_at_3",
    "map_at_5",
    "map_at_10",
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "ndcg_at_10",
    "gold_rank",
    "min_gold_rank",
    "max_gold_rank",
    "mean_gold_rank",
    "n_gold_sentences",
    "gold_sentence_ids",
    "n_gold",
    "has_gold",
    "gold_positions",
    "gold_mask",
}


def check_leakage(features: Dict[str, Any]) -> List[str]:
    """Check if any features would cause label leakage.

    Args:
        features: Dictionary of feature names to values

    Returns:
        List of feature names that would cause leakage

    Raises:
        ValueError: If any leakage features are found
    """
    leaked = []
    for key in features.keys():
        key_lower = key.lower()
        for forbidden in LEAKAGE_FEATURES:
            if forbidden in key_lower:
                leaked.append(key)
                break
    return leaked


def assert_no_leakage(features: Dict[str, Any]) -> None:
    """Assert that no features cause label leakage.

    Args:
        features: Dictionary of feature names to values

    Raises:
        ValueError: If any leakage features are found
    """
    leaked = check_leakage(features)
    if leaked:
        raise ValueError(
            f"Label leakage detected! Forbidden features found: {leaked}. "
            f"These features are derived from gold labels and cannot be used at inference time."
        )


class NodeFeatureExtractor:
    """Extract node features from candidate data.

    Node features are extracted per-candidate within a query graph.
    All features must be available at inference time (NO gold labels).
    """

    def __init__(
        self,
        use_embedding: bool = True,
        use_reranker_score: bool = True,
        use_rank_percentile: bool = True,
        use_score_gaps: bool = True,
        use_score_stats: bool = True,
        embedding_dim: int = 1024,
    ):
        self.use_embedding = use_embedding
        self.use_reranker_score = use_reranker_score
        self.use_rank_percentile = use_rank_percentile
        self.use_score_gaps = use_score_gaps
        self.use_score_stats = use_score_stats
        self.embedding_dim = embedding_dim

    @property
    def feature_dim(self) -> int:
        """Compute total feature dimension."""
        dim = 0
        if self.use_embedding:
            dim += self.embedding_dim
        if self.use_reranker_score:
            dim += 1
        if self.use_rank_percentile:
            dim += 1
        if self.use_score_gaps:
            dim += 2  # gap_to_prev, gap_to_next
        if self.use_score_stats:
            dim += 4  # zscore, minmax, above_mean, below_median
        return dim

    def extract(
        self,
        embeddings: np.ndarray,  # [n_candidates, embedding_dim]
        reranker_scores: np.ndarray,  # [n_candidates]
        ranks: Optional[np.ndarray] = None,  # [n_candidates] 0-indexed
    ) -> np.ndarray:
        """Extract node features for all candidates in a query.

        Args:
            embeddings: Dense embeddings for each candidate
            reranker_scores: Reranker scores for each candidate (sorted descending)
            ranks: Rank positions (0-indexed). If None, inferred from score order.

        Returns:
            Node features array of shape [n_candidates, feature_dim]
        """
        n = len(reranker_scores)
        if n == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # Infer ranks if not provided
        if ranks is None:
            ranks = np.argsort(-reranker_scores)  # Descending order

        features_list = []

        # 1. Embeddings
        if self.use_embedding:
            if embeddings.shape[0] != n:
                raise ValueError(f"Embedding count {embeddings.shape[0]} != candidate count {n}")
            features_list.append(embeddings)

        # 2. Reranker scores (normalized)
        if self.use_reranker_score:
            # Normalize to [0, 1]
            score_min = reranker_scores.min()
            score_max = reranker_scores.max()
            if score_max > score_min:
                norm_scores = (reranker_scores - score_min) / (score_max - score_min)
            else:
                norm_scores = np.ones(n) * 0.5
            features_list.append(norm_scores.reshape(-1, 1))

        # 3. Rank percentile (0 = top, 1 = bottom)
        if self.use_rank_percentile:
            rank_pct = ranks / max(n - 1, 1)
            features_list.append(rank_pct.reshape(-1, 1))

        # 4. Score gaps (to neighbors in ranking)
        if self.use_score_gaps:
            sorted_idx = np.argsort(-reranker_scores)
            sorted_scores = reranker_scores[sorted_idx]

            # Compute gaps in sorted order, then map back
            gaps_prev = np.zeros(n)
            gaps_next = np.zeros(n)

            for i, idx in enumerate(sorted_idx):
                if i > 0:
                    gaps_prev[idx] = sorted_scores[i - 1] - sorted_scores[i]
                if i < n - 1:
                    gaps_next[idx] = sorted_scores[i] - sorted_scores[i + 1]

            features_list.append(gaps_prev.reshape(-1, 1))
            features_list.append(gaps_next.reshape(-1, 1))

        # 5. Score statistics
        if self.use_score_stats:
            mean_score = reranker_scores.mean()
            std_score = reranker_scores.std() if n > 1 else 1.0
            median_score = np.median(reranker_scores)

            # Z-score
            zscore = (reranker_scores - mean_score) / (std_score + 1e-8)

            # Min-max normalized
            if score_max > score_min:
                minmax = (reranker_scores - score_min) / (score_max - score_min)
            else:
                minmax = np.ones(n) * 0.5

            # Binary: above mean
            above_mean = (reranker_scores > mean_score).astype(np.float32)

            # Binary: below median
            below_median = (reranker_scores < median_score).astype(np.float32)

            features_list.append(zscore.reshape(-1, 1))
            features_list.append(minmax.reshape(-1, 1))
            features_list.append(above_mean.reshape(-1, 1))
            features_list.append(below_median.reshape(-1, 1))

        # Concatenate all features
        node_features = np.hstack(features_list).astype(np.float32)

        return node_features

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        names = []
        if self.use_embedding:
            names.extend([f"emb_{i}" for i in range(self.embedding_dim)])
        if self.use_reranker_score:
            names.append("reranker_score_norm")
        if self.use_rank_percentile:
            names.append("rank_percentile")
        if self.use_score_gaps:
            names.extend(["gap_to_prev", "gap_to_next"])
        if self.use_score_stats:
            names.extend(["score_zscore", "score_minmax", "above_mean", "below_median"])
        return names


class EdgeFeatureExtractor:
    """Extract edge features for graph construction.

    Edge features represent relationships between candidates:
    - Cosine similarity (semantic relatedness)
    - Sequence distance (positional proximity in post)
    """

    def __init__(
        self,
        use_cosine_similarity: bool = True,
        use_sequence_distance: bool = True,
    ):
        self.use_cosine_similarity = use_cosine_similarity
        self.use_sequence_distance = use_sequence_distance

    @property
    def feature_dim(self) -> int:
        """Compute edge feature dimension."""
        dim = 0
        if self.use_cosine_similarity:
            dim += 1
        if self.use_sequence_distance:
            dim += 1
        return dim

    def extract_for_edge(
        self,
        emb_i: np.ndarray,  # [embedding_dim]
        emb_j: np.ndarray,  # [embedding_dim]
        sid_i: int,
        sid_j: int,
        max_seq_dist: int = 10,
    ) -> np.ndarray:
        """Extract features for a single edge.

        Args:
            emb_i: Embedding of source node
            emb_j: Embedding of target node
            sid_i: Sentence index of source
            sid_j: Sentence index of target
            max_seq_dist: Maximum sequence distance for normalization

        Returns:
            Edge features array of shape [feature_dim]
        """
        features = []

        if self.use_cosine_similarity:
            # Cosine similarity
            norm_i = np.linalg.norm(emb_i)
            norm_j = np.linalg.norm(emb_j)
            if norm_i > 0 and norm_j > 0:
                cos_sim = np.dot(emb_i, emb_j) / (norm_i * norm_j)
            else:
                cos_sim = 0.0
            features.append(cos_sim)

        if self.use_sequence_distance:
            # Normalized sequence distance
            seq_dist = abs(sid_i - sid_j)
            norm_dist = min(seq_dist / max_seq_dist, 1.0)
            features.append(norm_dist)

        return np.array(features, dtype=np.float32)

    def extract_batch(
        self,
        embeddings: np.ndarray,  # [n_candidates, embedding_dim]
        edge_index: np.ndarray,  # [2, n_edges]
        sentence_ids: List[int],  # Sentence indices for each candidate
        max_seq_dist: int = 10,
    ) -> np.ndarray:
        """Extract edge features for all edges.

        Args:
            embeddings: Candidate embeddings
            edge_index: Edge index array [2, n_edges]
            sentence_ids: Sentence indices for sequence distance
            max_seq_dist: Max sequence distance for normalization

        Returns:
            Edge features array of shape [n_edges, feature_dim]
        """
        n_edges = edge_index.shape[1]
        if n_edges == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        edge_features = []
        for i in range(n_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            feat = self.extract_for_edge(
                embeddings[src],
                embeddings[dst],
                sentence_ids[src],
                sentence_ids[dst],
                max_seq_dist,
            )
            edge_features.append(feat)

        return np.stack(edge_features)

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        names = []
        if self.use_cosine_similarity:
            names.append("cosine_similarity")
        if self.use_sequence_distance:
            names.append("sequence_distance")
        return names


class GraphStatsExtractor:
    """Extract graph-level statistics for baseline models.

    These features capture graph structure without requiring GNN:
    - Edge density
    - Average/max degree
    - Clustering coefficient proxy
    - PageRank statistics
    - Score-connectivity correlation
    """

    def extract(
        self,
        n_nodes: int,
        edge_index: np.ndarray,  # [2, n_edges]
        node_scores: np.ndarray,  # [n_nodes]
        embeddings: Optional[np.ndarray] = None,  # [n_nodes, dim]
    ) -> Dict[str, float]:
        """Extract graph statistics.

        Args:
            n_nodes: Number of nodes in graph
            edge_index: Edge index array
            node_scores: Reranker scores for each node
            embeddings: Optional node embeddings for similarity stats

        Returns:
            Dictionary of graph statistics
        """
        stats = {}

        if n_nodes == 0:
            return {
                "edge_density": 0.0,
                "avg_degree": 0.0,
                "max_degree": 0.0,
                "clustering_proxy": 0.0,
                "score_degree_corr": 0.0,
                "avg_neighbor_sim": 0.0,
            }

        n_edges = edge_index.shape[1] if edge_index.size > 0 else 0

        # Edge density
        max_edges = n_nodes * (n_nodes - 1)  # Directed
        stats["edge_density"] = n_edges / max_edges if max_edges > 0 else 0.0

        # Degree statistics
        if n_edges > 0:
            degrees = np.bincount(edge_index[0], minlength=n_nodes)
            stats["avg_degree"] = float(degrees.mean())
            stats["max_degree"] = float(degrees.max())

            # Clustering proxy: avg triangles / possible triangles
            # Simplified: just use edge density as proxy
            stats["clustering_proxy"] = stats["edge_density"]

            # Score-degree correlation
            if degrees.std() > 0 and node_scores.std() > 0:
                corr = np.corrcoef(degrees, node_scores)[0, 1]
                stats["score_degree_corr"] = float(corr) if not np.isnan(corr) else 0.0
            else:
                stats["score_degree_corr"] = 0.0
        else:
            stats["avg_degree"] = 0.0
            stats["max_degree"] = 0.0
            stats["clustering_proxy"] = 0.0
            stats["score_degree_corr"] = 0.0

        # Average pairwise similarity among top-k
        if embeddings is not None and len(embeddings) > 1:
            # Take top 5 by score
            top_k = min(5, len(embeddings))
            top_idx = np.argsort(-node_scores)[:top_k]
            top_emb = embeddings[top_idx]

            # Compute pairwise cosine similarities
            norms = np.linalg.norm(top_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = top_emb / norms

            sim_matrix = normed @ normed.T
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
            if mask.sum() > 0:
                stats["avg_neighbor_sim"] = float(sim_matrix[mask].mean())
            else:
                stats["avg_neighbor_sim"] = 0.0
        else:
            stats["avg_neighbor_sim"] = 0.0

        return stats
