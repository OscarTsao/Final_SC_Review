"""Dynamic-k selection for adaptive top-k cutoff.

Instead of using a fixed k, dynamically select k based on:
- Score gap: Large drop in score indicates end of relevant results
- Threshold: Only return results above a calibrated probability threshold
- Elbow method: Find the "elbow" in the score curve
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DynamicKResult:
    """Result of dynamic-k selection."""

    selected_k: int
    scores: List[float]
    method: str
    threshold_used: Optional[float] = None


class DynamicKSelector:
    """Select adaptive k based on score distribution."""

    def __init__(
        self,
        method: str = "score_gap",
        min_k: int = 1,
        max_k: int = 20,
        score_gap_ratio: float = 0.3,
        probability_threshold: float = 0.5,
    ):
        """Initialize dynamic-k selector.

        Args:
            method: Selection method ('score_gap', 'threshold', 'elbow')
            min_k: Minimum k to return
            max_k: Maximum k to return
            score_gap_ratio: For score_gap method, cutoff if score drops by this ratio
            probability_threshold: For threshold method, only return above this
        """
        self.method = method
        self.min_k = min_k
        self.max_k = max_k
        self.score_gap_ratio = score_gap_ratio
        self.probability_threshold = probability_threshold

    def select_k(
        self,
        scores: List[float],
        calibrated_probs: Optional[List[float]] = None,
    ) -> DynamicKResult:
        """Select k based on score distribution.

        Args:
            scores: Ranked scores (highest first)
            calibrated_probs: Optional calibrated probabilities

        Returns:
            DynamicKResult with selected k
        """
        if not scores:
            return DynamicKResult(selected_k=0, scores=[], method=self.method)

        if self.method == "score_gap":
            k = self._select_by_score_gap(scores)
        elif self.method == "threshold":
            if calibrated_probs is None:
                raise ValueError("threshold method requires calibrated_probs")
            k = self._select_by_threshold(calibrated_probs)
        elif self.method == "elbow":
            k = self._select_by_elbow(scores)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply min/max bounds
        k = max(self.min_k, min(k, self.max_k, len(scores)))

        return DynamicKResult(
            selected_k=k,
            scores=scores[:k],
            method=self.method,
            threshold_used=self.probability_threshold if self.method == "threshold" else None,
        )

    def _select_by_score_gap(self, scores: List[float]) -> int:
        """Select k by finding large score gaps."""
        if len(scores) <= 1:
            return len(scores)

        scores_arr = np.array(scores)
        max_score = scores_arr[0]

        if max_score <= 0:
            return self.min_k

        # Find where score drops below threshold of max
        threshold = max_score * (1 - self.score_gap_ratio)

        for i, score in enumerate(scores_arr):
            if score < threshold:
                return max(i, self.min_k)

        return len(scores)

    def _select_by_threshold(self, probs: List[float]) -> int:
        """Select k by probability threshold."""
        for i, prob in enumerate(probs):
            if prob < self.probability_threshold:
                return max(i, self.min_k)
        return len(probs)

    def _select_by_elbow(self, scores: List[float]) -> int:
        """Select k using elbow detection on score curve."""
        if len(scores) <= 2:
            return len(scores)

        scores_arr = np.array(scores)

        # Normalize scores to [0, 1]
        min_s, max_s = scores_arr.min(), scores_arr.max()
        if max_s > min_s:
            normalized = (scores_arr - min_s) / (max_s - min_s)
        else:
            return self.min_k

        # Find elbow using distance to line from first to last point
        n = len(normalized)
        x = np.arange(n) / (n - 1)  # Normalize x to [0, 1]

        # Line from (0, normalized[0]) to (1, normalized[-1])
        line_vec = np.array([1, normalized[-1] - normalized[0]])
        line_vec = line_vec / np.linalg.norm(line_vec)

        # Distance from each point to line
        distances = []
        for i in range(n):
            point_vec = np.array([x[i], normalized[i] - normalized[0]])
            # Distance = |cross product| / |line_vec|
            dist = abs(point_vec[0] * line_vec[1] - point_vec[1] * line_vec[0])
            distances.append(dist)

        # Elbow is point with max distance
        elbow_idx = np.argmax(distances)

        return max(elbow_idx + 1, self.min_k)


def analyze_score_distribution(scores: List[float]) -> dict:
    """Analyze score distribution for debugging.

    Args:
        scores: List of scores (highest first)

    Returns:
        Dictionary with distribution statistics
    """
    if not scores:
        return {"empty": True}

    scores_arr = np.array(scores)

    return {
        "n_scores": len(scores),
        "max": float(scores_arr.max()),
        "min": float(scores_arr.min()),
        "mean": float(scores_arr.mean()),
        "std": float(scores_arr.std()),
        "range": float(scores_arr.max() - scores_arr.min()),
        "gaps": list(np.diff(scores_arr)),
        "max_gap": float(np.abs(np.diff(scores_arr)).max()) if len(scores) > 1 else 0,
        "max_gap_idx": int(np.abs(np.diff(scores_arr)).argmax()) if len(scores) > 1 else 0,
    }
