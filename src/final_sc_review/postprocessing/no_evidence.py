"""No-evidence detection for queries without supporting evidence.

Detects when a post doesn't contain evidence for a given criterion by:
- Low max score: If best candidate score is below threshold
- Score distribution: If all candidates have similar (low) scores
- Combined: Weighted combination of signals
- RF Classifier: Random Forest trained on score features (dev-only fitting)

Note: RF classifier uses pickle for model serialization (standard for sklearn).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NoEvidenceResult:
    """Result of no-evidence detection."""

    has_evidence: bool
    confidence: float
    reason: str
    max_score: float
    score_std: float


def extract_score_features(scores: List[float]) -> Dict[str, float]:
    """Extract features from score distribution for classifier input.

    Args:
        scores: List of reranker scores for candidates

    Returns:
        Dictionary of extracted features
    """
    if not scores:
        return {
            "max_score": 0.0,
            "min_score": 0.0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "top3_mean": 0.0,
            "bottom3_mean": 0.0,
            "score_range": 0.0,
            "top1_minus_top2": 0.0,
            "top1_minus_mean": 0.0,
            "gini": 0.0,
            "entropy": 0.0,
            "n_candidates": 0,
        }

    scores_arr = np.array(sorted(scores, reverse=True))
    n = len(scores_arr)

    max_score = float(scores_arr[0])
    min_score = float(scores_arr[-1])
    mean_score = float(scores_arr.mean())
    std_score = float(scores_arr.std()) if n > 1 else 0.0

    top3_mean = float(scores_arr[:min(3, n)].mean())
    bottom3_mean = float(scores_arr[-min(3, n):].mean())
    score_range = max_score - min_score

    top1_minus_top2 = float(scores_arr[0] - scores_arr[1]) if n > 1 else 0.0
    top1_minus_mean = max_score - mean_score

    # Gini coefficient (normalized)
    scores_sorted = np.sort(scores_arr)
    cumsum = np.cumsum(scores_sorted)
    gini = (2.0 * np.sum((np.arange(1, n+1) * scores_sorted)) - (n+1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)
    gini = max(0.0, min(1.0, gini))

    # Entropy (normalized)
    scores_pos = scores_arr - scores_arr.min() + 1e-10
    probs = scores_pos / scores_pos.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-10))) / np.log(n + 1e-10) if n > 1 else 0.0

    return {
        "max_score": max_score,
        "min_score": min_score,
        "mean_score": mean_score,
        "std_score": std_score,
        "top3_mean": top3_mean,
        "bottom3_mean": bottom3_mean,
        "score_range": score_range,
        "top1_minus_top2": top1_minus_top2,
        "top1_minus_mean": top1_minus_mean,
        "gini": gini,
        "entropy": entropy,
        "n_candidates": n,
    }


class NoEvidenceDetector:
    """Detect queries with no evidence in the post.

    Supported methods:
    - max_score: Threshold on max reranker score
    - score_std: Threshold on score standard deviation
    - score_gap: Threshold on gap between top two scores (top1 - top2)
    - score_gap_mean: Threshold on gap between top score and mean (top1 - mean)
    - combined: Weighted combination of signals
    - rf_classifier: Random Forest classifier on score features
    """

    def __init__(
        self,
        method: str = "max_score",
        max_score_threshold: float = 0.3,
        score_std_threshold: float = 0.1,
        score_gap_threshold: float = 0.1,
        min_candidates: int = 5,
        model_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        """Initialize no-evidence detector.

        Args:
            method: Detection method ('max_score', 'score_std', 'score_gap', 'score_gap_mean', 'combined', 'rf_classifier')
            max_score_threshold: Below this max score, predict no evidence
            score_std_threshold: Below this std, scores are too uniform
            score_gap_threshold: Below this gap (top1-top2 or top1-mean), predict no evidence
            min_candidates: Minimum candidates needed for reliable detection
            model_path: Path to trained classifier model (for rf_classifier method)
            features: List of feature names to use (for rf_classifier)
            threshold: Classification threshold (for rf_classifier)
        """
        self.method = method
        self.max_score_threshold = max_score_threshold
        self.score_std_threshold = score_std_threshold
        self.score_gap_threshold = score_gap_threshold
        self.min_candidates = min_candidates
        self.model_path = model_path
        self.features = features or [
            "max_score", "std_score", "top3_mean", "score_range",
            "min_score", "mean_score", "bottom3_mean",
            "top1_minus_top2", "top1_minus_mean", "gini", "entropy", "n_candidates"
        ]
        self.threshold = threshold
        self._classifier = None
        self._scaler = None

        # Load classifier if needed
        if method == "rf_classifier" and model_path:
            self._load_classifier(model_path)

    def _load_classifier(self, model_path: str) -> None:
        """Load trained classifier from disk.

        Supports both pickle (.pkl) and joblib formats.
        For safety, also supports JSON-based model coefficients.
        """
        import pickle
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Classifier model not found at {model_path}, using combined fallback")
            self.method = "combined"
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                self._classifier = data.get("model")
                self._scaler = data.get("scaler")
            else:
                self._classifier = data
                self._scaler = None

            logger.info(f"Loaded classifier from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}, using combined fallback")
            self.method = "combined"

    def _detect_rf_classifier(
        self,
        scores: List[float],
        max_score: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect using trained RF classifier on score features."""
        if self._classifier is None:
            # Fallback to combined method
            return self._detect_combined(max_score, score_std, None)

        # Extract features
        features = extract_score_features(scores)
        X = np.array([[features.get(f, 0.0) for f in self.features]])

        # Scale if scaler available
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # Predict probability
        try:
            prob = self._classifier.predict_proba(X)[0, 1]  # P(has_evidence=1)
        except Exception:
            prob = float(self._classifier.predict(X)[0])

        has_evidence = prob >= self.threshold
        confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason="rf_classifier",
            max_score=max_score,
            score_std=score_std,
        )

    def detect(
        self,
        scores: List[float],
        calibrated_probs: Optional[List[float]] = None,
    ) -> NoEvidenceResult:
        """Detect if query has no evidence in the post.

        Args:
            scores: Reranker scores for candidates
            calibrated_probs: Optional calibrated probabilities

        Returns:
            NoEvidenceResult with prediction and confidence
        """
        if not scores:
            return NoEvidenceResult(
                has_evidence=False,
                confidence=1.0,
                reason="no_candidates",
                max_score=0.0,
                score_std=0.0,
            )

        scores_arr = np.array(scores)
        sorted_scores = np.sort(scores_arr)[::-1]  # Descending order
        max_score = float(sorted_scores[0])
        second_max = float(sorted_scores[1]) if len(sorted_scores) > 1 else max_score
        mean_score = float(scores_arr.mean())
        score_std = float(scores_arr.std()) if len(scores) > 1 else 0.0

        if self.method == "max_score":
            return self._detect_by_max_score(max_score, score_std)
        elif self.method == "score_std":
            return self._detect_by_score_std(max_score, score_std)
        elif self.method == "score_gap":
            return self._detect_by_score_gap(max_score, second_max, score_std)
        elif self.method == "score_gap_mean":
            return self._detect_by_score_gap_mean(max_score, mean_score, score_std)
        elif self.method == "score_gap_and_std":
            return self._detect_by_score_gap_and_std(max_score, second_max, score_std)
        elif self.method == "score_gap_or_std":
            return self._detect_by_score_gap_or_std(max_score, second_max, score_std)
        elif self.method == "score_gap_std_weighted":
            return self._detect_by_score_gap_std_weighted(max_score, second_max, score_std)
        elif self.method == "combined":
            return self._detect_combined(max_score, score_std, calibrated_probs)
        elif self.method == "rf_classifier":
            return self._detect_rf_classifier(scores, max_score, score_std)
        else:
            raise ValueError(f"Unknown method: {self.method}. Supported: max_score, score_std, score_gap, score_gap_mean, score_gap_and_std, score_gap_or_std, score_gap_std_weighted, combined, rf_classifier")

    def _detect_by_max_score(
        self,
        max_score: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect by max score threshold."""
        has_evidence = max_score >= self.max_score_threshold

        # Confidence based on distance from threshold
        if has_evidence:
            confidence = min(1.0, (max_score - self.max_score_threshold) / 0.3 + 0.5)
        else:
            confidence = min(1.0, (self.max_score_threshold - max_score) / 0.3 + 0.5)

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason="max_score_threshold",
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_by_score_std(
        self,
        max_score: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect by score standard deviation."""
        # Low std means all candidates have similar scores = no clear evidence
        has_evidence = score_std >= self.score_std_threshold

        if has_evidence:
            confidence = min(1.0, (score_std - self.score_std_threshold) / 0.2 + 0.5)
        else:
            confidence = min(1.0, (self.score_std_threshold - score_std) / 0.1 + 0.5)

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason="score_std_threshold",
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_by_score_gap(
        self,
        max_score: float,
        second_max: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect by gap between top two scores (top1 - top2).

        If the gap is large, the top result is clearly better = has evidence.
        If the gap is small, no clear winner = no evidence.
        """
        score_gap = max_score - second_max
        has_evidence = score_gap >= self.score_gap_threshold

        if has_evidence:
            confidence = min(1.0, (score_gap - self.score_gap_threshold) / 0.2 + 0.5)
        else:
            confidence = min(1.0, (self.score_gap_threshold - score_gap) / 0.1 + 0.5)

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason="score_gap_threshold",
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_by_score_gap_mean(
        self,
        max_score: float,
        mean_score: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect by gap between top score and mean (top1 - mean).

        If the gap is large, the top result stands out = has evidence.
        If the gap is small, top result not much better than average = no evidence.
        """
        score_gap = max_score - mean_score
        has_evidence = score_gap >= self.score_gap_threshold

        if has_evidence:
            confidence = min(1.0, (score_gap - self.score_gap_threshold) / 0.2 + 0.5)
        else:
            confidence = min(1.0, (self.score_gap_threshold - score_gap) / 0.1 + 0.5)

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason="score_gap_mean_threshold",
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_by_score_gap_and_std(
        self,
        max_score: float,
        second_max: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect using AND logic: both gap AND std must pass thresholds.

        Conservative approach - reduces false positives (fewer "has evidence" predictions).
        has_evidence = (gap >= gap_thresh) AND (std >= std_thresh)
        """
        score_gap = max_score - second_max
        gap_pass = score_gap >= self.score_gap_threshold
        std_pass = score_std >= self.score_std_threshold

        has_evidence = gap_pass and std_pass

        # Confidence based on how many criteria are met and by how much
        gap_margin = (score_gap - self.score_gap_threshold) / 0.2 if gap_pass else (self.score_gap_threshold - score_gap) / 0.1
        std_margin = (score_std - self.score_std_threshold) / 0.2 if std_pass else (self.score_std_threshold - score_std) / 0.1

        if has_evidence:
            confidence = min(1.0, 0.5 + 0.25 * min(gap_margin, std_margin))
        else:
            confidence = min(1.0, 0.5 + 0.25 * max(abs(gap_margin), abs(std_margin)))

        reason = f"gap_{'pass' if gap_pass else 'fail'}+std_{'pass' if std_pass else 'fail'}"

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason=reason,
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_by_score_gap_or_std(
        self,
        max_score: float,
        second_max: float,
        score_std: float,
    ) -> NoEvidenceResult:
        """Detect using OR logic: either gap OR std passing is enough.

        Lenient approach - reduces false negatives (fewer "no evidence" predictions).
        has_evidence = (gap >= gap_thresh) OR (std >= std_thresh)
        """
        score_gap = max_score - second_max
        gap_pass = score_gap >= self.score_gap_threshold
        std_pass = score_std >= self.score_std_threshold

        has_evidence = gap_pass or std_pass

        # Confidence based on strongest signal
        gap_margin = (score_gap - self.score_gap_threshold) / 0.2 if gap_pass else (self.score_gap_threshold - score_gap) / 0.1
        std_margin = (score_std - self.score_std_threshold) / 0.2 if std_pass else (self.score_std_threshold - score_std) / 0.1

        if has_evidence:
            confidence = min(1.0, 0.5 + 0.25 * max(gap_margin, std_margin))
        else:
            confidence = min(1.0, 0.5 + 0.25 * min(abs(gap_margin), abs(std_margin)))

        reason = f"gap_{'pass' if gap_pass else 'fail'}|std_{'pass' if std_pass else 'fail'}"

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason=reason,
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_by_score_gap_std_weighted(
        self,
        max_score: float,
        second_max: float,
        score_std: float,
        gap_weight: float = 0.6,
    ) -> NoEvidenceResult:
        """Detect using weighted combination of normalized gap and std signals.

        Balanced approach - weights both signals.
        score = gap_weight * norm_gap + (1 - gap_weight) * norm_std
        has_evidence = score >= 0.5
        """
        score_gap = max_score - second_max

        # Normalize signals to [0, 1] based on thresholds
        # Signal > threshold maps to > 0.5, signal < threshold maps to < 0.5
        norm_gap = 0.5 + (score_gap - self.score_gap_threshold) / 0.4  # Scale factor
        norm_std = 0.5 + (score_std - self.score_std_threshold) / 0.4
        norm_gap = max(0.0, min(1.0, norm_gap))
        norm_std = max(0.0, min(1.0, norm_std))

        # Weighted combination
        combined_score = gap_weight * norm_gap + (1 - gap_weight) * norm_std
        has_evidence = combined_score >= 0.5

        confidence = abs(combined_score - 0.5) * 2  # Scale to [0, 1]

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason=f"weighted(gap={norm_gap:.2f},std={norm_std:.2f})",
            max_score=max_score,
            score_std=score_std,
        )

    def _detect_combined(
        self,
        max_score: float,
        score_std: float,
        calibrated_probs: Optional[List[float]],
    ) -> NoEvidenceResult:
        """Combine multiple signals for detection."""
        signals = []

        # Signal 1: Max score
        if max_score < self.max_score_threshold:
            signals.append(("low_max_score", 0.4))
        else:
            signals.append(("high_max_score", -0.4))

        # Signal 2: Score std
        if score_std < self.score_std_threshold:
            signals.append(("low_std", 0.3))
        else:
            signals.append(("high_std", -0.3))

        # Signal 3: Calibrated probability if available
        if calibrated_probs:
            max_prob = max(calibrated_probs)
            if max_prob < 0.5:
                signals.append(("low_prob", 0.3))
            else:
                signals.append(("high_prob", -0.3))

        # Aggregate signals
        total_weight = sum(w for _, w in signals)
        has_evidence = total_weight < 0

        confidence = min(1.0, abs(total_weight) + 0.5)
        reasons = [name for name, w in signals if (w > 0) != has_evidence]

        return NoEvidenceResult(
            has_evidence=has_evidence,
            confidence=confidence,
            reason="+".join(reasons) if reasons else "combined",
            max_score=max_score,
            score_std=score_std,
        )


def compute_no_evidence_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
) -> dict:
    """Compute metrics for no-evidence detection.

    Args:
        predictions: Predicted has_evidence flags
        ground_truth: True has_evidence flags

    Returns:
        Dictionary with precision, recall, F1
    """
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }
