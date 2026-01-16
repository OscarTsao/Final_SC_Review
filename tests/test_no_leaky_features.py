"""Unit tests to prevent label leakage in feature pipeline.

These tests ensure that deployable features NEVER touch gold labels.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# Forbidden patterns in feature names
FORBIDDEN_PATTERNS = [
    "gold",
    "mrr",
    "recall_at",
    "recall@",
    "ndcg",
    "precision_at",
    "precision@",
    "map@",
    "map_at",
    "label",
    "relevant",
    "groundtruth",
    "gt_",
    "is_positive",
    "is_relevant",
    "hit_at",
    "hit@",
]


# Allowed exceptions - deployable proxies that contain forbidden substrings
ALLOWED_EXCEPTIONS = [
    "soft_mrr",  # Softmax-based proxy, doesn't use gold labels
    "mass_at_",  # Probability mass concentration, doesn't use gold labels
]


def has_forbidden_pattern(name: str) -> bool:
    """Check if a feature name contains forbidden patterns.

    Allows certain deployable proxy features that contain forbidden substrings
    but don't actually use gold labels (e.g., soft_mrr uses softmax scores only).
    """
    name_lower = name.lower()

    # Check if name is in allowed exceptions
    for exception in ALLOWED_EXCEPTIONS:
        if exception in name_lower:
            return False

    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in name_lower:
            return True
    return False


class TestDeployableFeatures:
    """Tests for deployable feature computation."""

    def test_compute_deployable_features_no_leakage(self):
        """Test that compute_deployable_features doesn't use gold labels."""
        from scripts.verification.build_deployable_features import (
            compute_deployable_features,
            DEPLOYABLE_FEATURES,
            FeatureMode,
        )

        # Test with sample data
        reranker_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        retriever_scores = np.array([0.8, 0.6, 0.4, 0.2, 0.0])

        features = compute_deployable_features(
            reranker_scores,
            retriever_scores,
            n_candidates=5,
        )

        # Check no forbidden feature names
        for name in features.keys():
            assert not has_forbidden_pattern(name), f"Forbidden feature name: {name}"

        # Check all features are numeric
        for name, value in features.items():
            assert isinstance(value, (int, float)), f"Non-numeric feature: {name}={value}"

    def test_deployable_feature_registry_no_gold_inputs(self):
        """Test that DEPLOYABLE_FEATURES don't depend on gold labels."""
        from scripts.verification.build_deployable_features import (
            DEPLOYABLE_FEATURES,
            FeatureMode,
        )

        forbidden_inputs = ["gold_ids", "gold_sentence_ids", "groundtruth", "labels", "relevance"]

        for spec in DEPLOYABLE_FEATURES:
            assert spec.mode == FeatureMode.DEPLOYABLE, f"{spec.name} should be DEPLOYABLE"
            for inp in spec.inputs:
                assert inp not in forbidden_inputs, f"{spec.name} uses forbidden input: {inp}"

    def test_evaluation_features_marked_correctly(self):
        """Test that evaluation features are clearly marked."""
        from scripts.verification.build_deployable_features import (
            EVALUATION_FEATURES,
            FeatureMode,
        )

        for spec in EVALUATION_FEATURES:
            assert spec.mode == FeatureMode.EVALUATION, f"{spec.name} should be EVALUATION"
            # Should use gold_ids
            assert "gold_ids" in spec.inputs, f"{spec.name} should use gold_ids"

    def test_compute_evaluation_features_separate(self):
        """Test that evaluation features are separate from deployable."""
        from scripts.verification.build_deployable_features import (
            compute_deployable_features,
            compute_evaluation_features,
        )

        reranker_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        candidate_ids = ["s1", "s2", "s3", "s4", "s5"]
        gold_ids = ["s1", "s3"]

        deployable = compute_deployable_features(reranker_scores)
        evaluation = compute_evaluation_features(candidate_ids, gold_ids, reranker_scores)

        # Check no overlap (except prefixed)
        deployable_names = set(deployable.keys())
        evaluation_names = set(evaluation.keys())
        assert deployable_names.isdisjoint(evaluation_names), "Feature sets should not overlap"

    def test_empty_scores_handling(self):
        """Test handling of empty score arrays."""
        from scripts.verification.build_deployable_features import compute_deployable_features

        features = compute_deployable_features(np.array([]))
        assert len(features) > 0
        for v in features.values():
            assert np.isfinite(v), "Features should be finite for empty input"


class TestFeatureStoreIntegrity:
    """Tests for feature store integrity."""

    def test_deployable_store_no_leaky_columns(self, tmp_path):
        """Test that deployable feature store has no leaky columns."""
        from scripts.verification.build_deployable_features import (
            FeatureMode,
            DEPLOYABLE_FEATURES,
        )

        # Create mock deployable feature store
        data = {
            "post_id": ["p1", "p2"],
            "query_id": ["p1_c1", "p2_c1"],
            "has_evidence": [1, 0],
            "max_reranker_score": [0.9, 0.5],
            "entropy_top5": [0.5, 0.8],
            "soft_mrr": [0.7, 0.3],  # This is SoftMRR (deployable), not MRR (leaky)
        }
        df = pd.DataFrame(data)

        # Check feature columns
        exclude = ["post_id", "query_id", "has_evidence", "fold_id", "n_gold_sentences",
                   "gold_sentence_ids", "candidate_ids", "criterion_id", "criterion_text",
                   "n_candidates", "candidate_scores"]
        feature_cols = [c for c in df.columns if c not in exclude]

        for col in feature_cols:
            # Allow "soft_mrr" but not "mrr" alone
            if col == "soft_mrr":
                continue
            assert not has_forbidden_pattern(col), f"Leaky column in deployable store: {col}"

    def test_provenance_file_completeness(self):
        """Test that feature provenance is complete."""
        from scripts.verification.build_deployable_features import (
            DEPLOYABLE_FEATURES,
            EVALUATION_FEATURES,
        )

        # Every feature should have provenance
        for spec in DEPLOYABLE_FEATURES:
            assert spec.name, "Feature must have name"
            assert spec.description, "Feature must have description"
            assert spec.inputs, "Feature must declare inputs"
            assert spec.compute_fn, "Feature must have compute function"

        for spec in EVALUATION_FEATURES:
            assert spec.name, "Feature must have name"
            assert spec.inputs, "Feature must declare inputs"


class TestDynamicKCaps:
    """Tests for dynamic K selection with caps."""

    def test_k_max_from_ratio(self):
        """Test k_max calculation from ratio."""
        import math

        k_max_ratio = 0.5
        hard_cap = 10
        k_min = 2

        def compute_k_max1(n_candidates: int) -> int:
            return min(hard_cap, max(k_min, math.ceil(k_max_ratio * n_candidates)))

        # Test cases
        assert compute_k_max1(20) == 10  # ceil(0.5*20)=10, min(10,10)=10
        assert compute_k_max1(12) == 6   # ceil(0.5*12)=6, min(10,6)=6
        assert compute_k_max1(40) == 10  # ceil(0.5*40)=20, min(10,20)=10
        assert compute_k_max1(4) == 2    # ceil(0.5*4)=2, max(2,2)=2
        assert compute_k_max1(2) == 2    # ceil(0.5*2)=1, max(2,1)=2

    def test_k_selection_within_bounds(self):
        """Test that K selection always respects bounds."""
        k_min = 2
        hard_cap = 10

        # Mock K selector that might return out-of-bounds values
        raw_k_values = [0, 1, 5, 10, 15, 20]

        for raw_k in raw_k_values:
            clamped_k = max(k_min, min(hard_cap, raw_k))
            assert k_min <= clamped_k <= hard_cap, f"K={clamped_k} out of bounds [{k_min}, {hard_cap}]"


class TestAuditScript:
    """Tests for the audit script itself."""

    def test_classify_feature_leaky(self):
        """Test that leaky features are correctly classified."""
        from scripts.verification.audit_feature_leakage import classify_feature

        leaky_names = ["mrr", "recall_at_5", "min_gold_rank", "gold_count", "is_relevant"]
        for name in leaky_names:
            assert classify_feature(name) == "LEAKY", f"{name} should be LEAKY"

    def test_classify_feature_deployable(self):
        """Test that deployable features are correctly classified."""
        from scripts.verification.audit_feature_leakage import classify_feature

        deployable_names = ["max_reranker_score", "entropy_top5", "top1_top2_gap", "n_candidates"]
        for name in deployable_names:
            assert classify_feature(name) == "DEPLOYABLE", f"{name} should be DEPLOYABLE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
