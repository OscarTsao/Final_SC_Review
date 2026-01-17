"""Comprehensive unit tests for metrics - Part of Gold Standard Audit.

This test suite verifies:
1. Ranking metrics against known reference values
2. Edge cases (empty lists, single elements)
3. Consistency with sklearn implementations
4. NE detection metrics
5. Calibration metrics

Audit Timestamp: 20260117_203822
"""

import math
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from final_sc_review.metrics.ranking import (
    recall_at_k,
    mrr_at_k,
    map_at_k,
    ndcg_at_k,
)
from final_sc_review.gnn.evaluation.metrics import (
    NEGateMetrics,
    compute_threshold_at_fpr,
)
from final_sc_review.postprocessing.calibration import compute_ece


class TestRecallAtK:
    """Test suite for Recall@K metric."""

    def test_perfect_recall(self):
        """All gold items in top-k."""
        gold = ["a", "b"]
        ranked = ["a", "b", "c", "d"]
        assert recall_at_k(gold, ranked, k=2) == 1.0

    def test_partial_recall(self):
        """Some gold items in top-k."""
        gold = ["a", "b"]
        ranked = ["a", "c", "b", "d"]
        assert recall_at_k(gold, ranked, k=2) == 0.5

    def test_zero_recall(self):
        """No gold items in top-k."""
        gold = ["x", "y"]
        ranked = ["a", "b", "c", "d"]
        assert recall_at_k(gold, ranked, k=3) == 0.0

    def test_empty_gold(self):
        """Edge case: empty gold set."""
        gold = []
        ranked = ["a", "b", "c"]
        assert recall_at_k(gold, ranked, k=2) == 0.0

    def test_empty_ranked(self):
        """Edge case: empty ranked list."""
        gold = ["a"]
        ranked = []
        assert recall_at_k(gold, ranked, k=3) == 0.0

    def test_k_larger_than_ranked(self):
        """K larger than ranked list length."""
        gold = ["a", "b"]
        ranked = ["a"]
        assert recall_at_k(gold, ranked, k=10) == 0.5

    def test_single_gold_at_k_boundary(self):
        """Gold item at exactly position k."""
        gold = ["c"]
        ranked = ["a", "b", "c", "d"]
        assert recall_at_k(gold, ranked, k=3) == 1.0
        assert recall_at_k(gold, ranked, k=2) == 0.0


class TestMRRAtK:
    """Test suite for MRR@K metric."""

    def test_first_position(self):
        """First relevant at position 1."""
        gold = ["a"]
        ranked = ["a", "b", "c"]
        assert mrr_at_k(gold, ranked, k=3) == 1.0

    def test_second_position(self):
        """First relevant at position 2."""
        gold = ["b"]
        ranked = ["a", "b", "c"]
        assert mrr_at_k(gold, ranked, k=3) == 0.5

    def test_third_position(self):
        """First relevant at position 3."""
        gold = ["c"]
        ranked = ["a", "b", "c"]
        assert mrr_at_k(gold, ranked, k=3) == pytest.approx(1/3)

    def test_no_relevant_in_top_k(self):
        """No relevant in top-k."""
        gold = ["d"]
        ranked = ["a", "b", "c", "d"]
        assert mrr_at_k(gold, ranked, k=3) == 0.0

    def test_multiple_gold_uses_first(self):
        """MRR uses rank of FIRST relevant item."""
        gold = ["b", "c"]
        ranked = ["a", "b", "c", "d"]
        assert mrr_at_k(gold, ranked, k=4) == 0.5

    def test_empty_gold(self):
        """Edge case: empty gold set."""
        gold = []
        ranked = ["a", "b"]
        assert mrr_at_k(gold, ranked, k=2) == 0.0


class TestMAPAtK:
    """Test suite for MAP@K metric."""

    def test_perfect_ranking(self):
        """Gold items ranked first."""
        gold = ["a", "b"]
        ranked = ["a", "b", "c", "d"]
        # P@1 = 1/1 = 1.0 (hit)
        # P@2 = 2/2 = 1.0 (hit)
        # MAP = (1.0 + 1.0) / 2 = 1.0
        assert map_at_k(gold, ranked, k=4) == 1.0

    def test_imperfect_ranking(self):
        """Gold items scattered."""
        gold = ["b", "d"]
        ranked = ["a", "b", "c", "d"]
        # P@2 = 1/2 = 0.5 (hit on b)
        # P@4 = 2/4 = 0.5 (hit on d)
        # MAP = (0.5 + 0.5) / 2 = 0.5
        assert map_at_k(gold, ranked, k=4) == 0.5

    def test_one_gold_not_at_top(self):
        """Single gold not at top."""
        gold = ["b"]
        ranked = ["a", "b", "c"]
        # P@2 = 1/2 = 0.5 (hit)
        # MAP = 0.5 / 1 = 0.5
        assert map_at_k(gold, ranked, k=3) == 0.5

    def test_empty_gold(self):
        """Edge case: empty gold set."""
        gold = []
        ranked = ["a", "b"]
        assert map_at_k(gold, ranked, k=2) == 0.0

    def test_k_smaller_than_gold(self):
        """K smaller than number of gold items."""
        gold = ["a", "b", "c"]
        ranked = ["a", "b", "c"]
        # Only look at top-2, both are hits
        # P@1 = 1, P@2 = 1
        # MAP = (1 + 1) / min(3, 2) = 2 / 2 = 1.0
        assert map_at_k(gold, ranked, k=2) == 1.0


class TestNDCGAtK:
    """Test suite for nDCG@K metric."""

    def test_perfect_ranking(self):
        """Gold items ranked optimally."""
        gold = ["a", "b"]
        ranked = ["a", "b", "c", "d"]
        # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
        # IDCG = 1/log2(2) + 1/log2(3) = 1.631
        # nDCG = 1.0
        assert ndcg_at_k(gold, ranked, k=4) == pytest.approx(1.0)

    def test_imperfect_ranking(self):
        """Gold items not at top."""
        gold = ["c", "d"]
        ranked = ["a", "b", "c", "d"]
        # DCG = 0/log2(2) + 0/log2(3) + 1/log2(4) + 1/log2(5)
        #     = 0 + 0 + 0.5 + 0.431 = 0.931
        # IDCG = 1/log2(2) + 1/log2(3) = 1.631
        # nDCG = 0.931 / 1.631 ≈ 0.571
        dcg = 1/math.log2(4) + 1/math.log2(5)
        idcg = 1/math.log2(2) + 1/math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(gold, ranked, k=4) == pytest.approx(expected, rel=1e-4)

    def test_single_gold_at_top(self):
        """Single gold at position 1."""
        gold = ["a"]
        ranked = ["a", "b", "c"]
        assert ndcg_at_k(gold, ranked, k=3) == pytest.approx(1.0)

    def test_single_gold_not_at_top(self):
        """Single gold not at position 1."""
        gold = ["b"]
        ranked = ["a", "b", "c"]
        # DCG = 1/log2(3) = 0.631
        # IDCG = 1/log2(2) = 1.0
        # nDCG = 0.631
        expected = 1 / math.log2(3)
        assert ndcg_at_k(gold, ranked, k=3) == pytest.approx(expected, rel=1e-4)

    def test_empty_gold(self):
        """Edge case: empty gold set."""
        gold = []
        ranked = ["a", "b"]
        assert ndcg_at_k(gold, ranked, k=2) == 0.0

    def test_k_limits_ideal_hits(self):
        """K smaller than gold count affects IDCG."""
        gold = ["a", "b", "c", "d", "e"]  # 5 gold items
        ranked = ["a", "b", "c", "d", "e"]
        # k=2: IDCG only counts 2 items
        # DCG = 1/log2(2) + 1/log2(3) = 1.631
        # IDCG = 1/log2(2) + 1/log2(3) = 1.631
        # nDCG = 1.0
        assert ndcg_at_k(gold, ranked, k=2) == pytest.approx(1.0)


class TestNEGateMetrics:
    """Test suite for NE Gate metrics."""

    def test_perfect_classifier(self):
        """Perfect binary classifier."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])

        metrics = NEGateMetrics.compute(y_prob, y_true)

        assert metrics.auroc == pytest.approx(1.0)
        assert metrics.n_samples == 6
        assert metrics.has_evidence_rate == pytest.approx(0.5)

    def test_random_classifier(self):
        """Random classifier should have AUROC near 0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=1000)
        y_prob = np.random.rand(1000)

        metrics = NEGateMetrics.compute(y_prob, y_true)

        # AUROC should be near 0.5 for random predictions
        assert 0.4 < metrics.auroc < 0.6

    def test_all_same_class(self):
        """Edge case: only one class in labels."""
        y_true = np.ones(10)
        y_prob = np.random.rand(10)

        metrics = NEGateMetrics.compute(y_prob, y_true)

        assert metrics.auroc == 0.5
        assert metrics.has_evidence_rate == 1.0

    def test_auroc_matches_sklearn(self):
        """Verify AUROC matches sklearn implementation."""
        np.random.seed(123)
        y_true = np.random.randint(0, 2, size=500)
        y_prob = np.random.rand(500)

        metrics = NEGateMetrics.compute(y_prob, y_true)
        sklearn_auroc = roc_auc_score(y_true, y_prob)

        assert metrics.auroc == pytest.approx(sklearn_auroc, rel=1e-6)

    def test_auprc_matches_sklearn(self):
        """Verify AUPRC matches sklearn implementation."""
        np.random.seed(456)
        y_true = np.random.randint(0, 2, size=500)
        y_prob = np.random.rand(500)

        metrics = NEGateMetrics.compute(y_prob, y_true)
        sklearn_auprc = average_precision_score(y_true, y_prob)

        assert metrics.auprc == pytest.approx(sklearn_auprc, rel=1e-6)


class TestTPRAtFPR:
    """Test suite for TPR at fixed FPR computation."""

    def test_tpr_at_5pct_fpr(self):
        """Test TPR at 5% FPR."""
        # Create labels with known ROC behavior
        np.random.seed(42)
        n = 1000
        y_true = np.concatenate([np.ones(100), np.zeros(900)])
        # Good classifier: positives have higher scores
        y_prob = np.concatenate([
            np.random.beta(5, 2, 100),  # Positives: higher scores
            np.random.beta(2, 5, 900),  # Negatives: lower scores
        ])

        threshold, tpr, fpr = compute_threshold_at_fpr(y_prob, y_true, target_fpr=0.05)

        assert fpr <= 0.05 + 0.01  # Allow small tolerance
        assert 0 <= tpr <= 1

    def test_tpr_at_0_fpr(self):
        """Test TPR at 0% FPR (most strict threshold)."""
        y_true = np.array([1, 1, 0, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.7])

        threshold, tpr, fpr = compute_threshold_at_fpr(y_prob, y_true, target_fpr=0.0)

        assert fpr == 0.0


class TestExpectedCalibrationError:
    """Test suite for ECE metric."""

    def test_perfectly_calibrated(self):
        """Perfectly calibrated predictions have ECE = 0."""
        # Create predictions that match actual rates
        n_per_bin = 100
        probs = []
        labels = []

        for bin_center in np.linspace(0.05, 0.95, 10):
            probs.extend([bin_center] * n_per_bin)
            # Generate labels with probability = bin_center
            labels.extend(np.random.binomial(1, bin_center, n_per_bin))

        ece = compute_ece(np.array(probs), np.array(labels), n_bins=10)

        # ECE should be small for well-calibrated predictions
        # Not exactly 0 due to randomness
        assert ece < 0.15

    def test_completely_miscalibrated(self):
        """Completely miscalibrated predictions."""
        # Predict 0.9 for everything, but actual rate is 0.1
        probs = np.ones(1000) * 0.9
        labels = np.zeros(1000)
        labels[:100] = 1  # 10% positive rate

        ece = compute_ece(probs, labels, n_bins=10)

        # ECE should be high (around 0.8)
        assert ece > 0.7

    def test_ece_bounds(self):
        """ECE should be bounded between 0 and 1."""
        np.random.seed(42)
        probs = np.random.rand(500)
        labels = np.random.randint(0, 2, size=500)

        ece = compute_ece(probs, labels, n_bins=10)

        assert 0 <= ece <= 1


class TestMetricConsistency:
    """Cross-metric consistency tests."""

    def test_recall_perfect_implies_ndcg_perfect(self):
        """If Recall@K = 1 and gold items at top, nDCG@K = 1."""
        gold = ["a", "b"]
        ranked = ["a", "b", "c", "d"]

        assert recall_at_k(gold, ranked, k=4) == 1.0
        assert ndcg_at_k(gold, ranked, k=4) == pytest.approx(1.0)

    def test_mrr_upper_bounds_map(self):
        """MRR@K >= MAP@K when single gold item."""
        gold = ["b"]
        ranked = ["a", "b", "c", "d"]

        mrr = mrr_at_k(gold, ranked, k=4)
        map_val = map_at_k(gold, ranked, k=4)

        # For single gold, MRR = MAP
        assert mrr == pytest.approx(map_val)

    def test_all_metrics_zero_when_no_hits(self):
        """All metrics should be 0 when no gold in top-k."""
        gold = ["x", "y", "z"]
        ranked = ["a", "b", "c", "d"]
        k = 4

        assert recall_at_k(gold, ranked, k) == 0.0
        assert mrr_at_k(gold, ranked, k) == 0.0
        assert map_at_k(gold, ranked, k) == 0.0
        assert ndcg_at_k(gold, ranked, k) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
