"""No-evidence detection for queries without supporting evidence.

Detects when a post doesn't contain evidence for a given criterion by:
- Low max score: If best candidate score is below threshold
- Score distribution: If all candidates have similar (low) scores
- Classifier: Binary classifier on score features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


class NoEvidenceDetector:
    """Detect queries with no evidence in the post."""

    def __init__(
        self,
        method: str = "max_score",
        max_score_threshold: float = 0.3,
        score_std_threshold: float = 0.1,
        min_candidates: int = 5,
    ):
        """Initialize no-evidence detector.

        Args:
            method: Detection method ('max_score', 'score_std', 'combined')
            max_score_threshold: Below this max score, predict no evidence
            score_std_threshold: Below this std, scores are too uniform
            min_candidates: Minimum candidates needed for reliable detection
        """
        self.method = method
        self.max_score_threshold = max_score_threshold
        self.score_std_threshold = score_std_threshold
        self.min_candidates = min_candidates

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
        max_score = float(scores_arr.max())
        score_std = float(scores_arr.std()) if len(scores) > 1 else 0.0

        if self.method == "max_score":
            return self._detect_by_max_score(max_score, score_std)
        elif self.method == "score_std":
            return self._detect_by_score_std(max_score, score_std)
        elif self.method == "combined":
            return self._detect_combined(max_score, score_std, calibrated_probs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

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
