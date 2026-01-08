"""Tests for post-processing modules."""

import numpy as np
import pytest


def test_dynamic_k_score_gap():
    """Test dynamic-k selection by score gap."""
    from final_sc_review.postprocessing.dynamic_k import DynamicKSelector

    selector = DynamicKSelector(method="score_gap", score_gap_ratio=0.3, min_k=1)

    # Clear gap after position 3
    scores = [0.9, 0.85, 0.8, 0.4, 0.3, 0.2]
    result = selector.select_k(scores)

    assert result.selected_k >= 1
    assert result.method == "score_gap"


def test_dynamic_k_threshold():
    """Test dynamic-k selection by threshold."""
    from final_sc_review.postprocessing.dynamic_k import DynamicKSelector

    selector = DynamicKSelector(method="threshold", probability_threshold=0.5, min_k=1)

    probs = [0.9, 0.7, 0.6, 0.4, 0.3]
    result = selector.select_k(probs, calibrated_probs=probs)

    # Should select first 3 (0.9, 0.7, 0.6 are >= 0.5)
    assert result.selected_k == 3
    assert result.threshold_used == 0.5


def test_dynamic_k_elbow():
    """Test dynamic-k selection by elbow method."""
    from final_sc_review.postprocessing.dynamic_k import DynamicKSelector

    selector = DynamicKSelector(method="elbow", min_k=1, max_k=10)

    # Scores with clear elbow around position 4
    scores = [1.0, 0.95, 0.9, 0.85, 0.3, 0.25, 0.2]
    result = selector.select_k(scores)

    assert 1 <= result.selected_k <= 10
    assert result.method == "elbow"


def test_no_evidence_max_score():
    """Test no-evidence detection by max score."""
    from final_sc_review.postprocessing.no_evidence import NoEvidenceDetector

    detector = NoEvidenceDetector(method="max_score", max_score_threshold=0.5)

    # High scores = has evidence
    result = detector.detect([0.8, 0.6, 0.4])
    assert result.has_evidence is True

    # Low scores = no evidence
    result = detector.detect([0.3, 0.2, 0.1])
    assert result.has_evidence is False


def test_no_evidence_empty():
    """Test no-evidence detection with empty candidates."""
    from final_sc_review.postprocessing.no_evidence import NoEvidenceDetector

    detector = NoEvidenceDetector()
    result = detector.detect([])

    assert result.has_evidence is False
    assert result.confidence == 1.0
    assert result.reason == "no_candidates"


def test_score_distribution_analysis():
    """Test score distribution analysis."""
    from final_sc_review.postprocessing.dynamic_k import analyze_score_distribution

    scores = [0.9, 0.8, 0.7, 0.3, 0.2]
    stats = analyze_score_distribution(scores)

    assert stats["n_scores"] == 5
    assert stats["max"] == 0.9
    assert stats["min"] == 0.2
    assert len(stats["gaps"]) == 4
    assert stats["max_gap_idx"] == 2  # Gap between 0.7 and 0.3


def test_calibration_platt():
    """Test Platt scaling calibration."""
    from final_sc_review.postprocessing.calibration import ScoreCalibrator

    # Generate synthetic data
    np.random.seed(42)
    scores = np.concatenate([
        np.random.normal(0.7, 0.1, 50),  # Positives
        np.random.normal(0.3, 0.1, 50),  # Negatives
    ])
    labels = np.concatenate([np.ones(50), np.zeros(50)])

    calibrator = ScoreCalibrator(method="platt")
    calibrator.fit(scores, labels)

    calibrated = calibrator.calibrate(np.array([0.8, 0.5, 0.2]))

    # Higher score should give higher probability
    assert calibrated[0] > calibrated[1] > calibrated[2]
    # All probabilities should be in [0, 1]
    assert all(0 <= p <= 1 for p in calibrated)


def test_ece_computation():
    """Test Expected Calibration Error computation."""
    from final_sc_review.postprocessing.calibration import compute_ece

    # Perfectly calibrated predictions
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1, 1])

    ece = compute_ece(probs, labels, n_bins=5)
    assert 0 <= ece <= 1


def test_no_evidence_metrics():
    """Test no-evidence detection metrics."""
    from final_sc_review.postprocessing.no_evidence import compute_no_evidence_metrics

    predictions = [True, True, False, False, True]
    ground_truth = [True, False, False, True, True]

    metrics = compute_no_evidence_metrics(predictions, ground_truth)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
