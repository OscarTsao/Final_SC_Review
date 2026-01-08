"""Score calibration for reranker outputs.

Converts raw reranker scores to calibrated probabilities using:
- Platt scaling (logistic regression on validation set)
- Isotonic regression for non-parametric calibration
- Temperature scaling for neural network outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of score calibration."""

    raw_scores: np.ndarray
    calibrated_probs: np.ndarray
    threshold: float = 0.5


class ScoreCalibrator:
    """Calibrate reranker scores to probabilities."""

    def __init__(
        self,
        method: str = "platt",
        temperature: float = 1.0,
    ):
        """Initialize calibrator.

        Args:
            method: Calibration method ('platt', 'isotonic', 'temperature')
            temperature: Temperature for temperature scaling
        """
        self.method = method
        self.temperature = temperature
        self._fitted = False
        self._platt_params: Optional[Tuple[float, float]] = None
        self._isotonic_model = None

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> "ScoreCalibrator":
        """Fit calibrator on validation data.

        Args:
            scores: Raw reranker scores
            labels: Binary labels (1 = positive, 0 = negative)

        Returns:
            Self for method chaining
        """
        if self.method == "platt":
            self._fit_platt(scores, labels)
        elif self.method == "isotonic":
            self._fit_isotonic(scores, labels)
        elif self.method == "temperature":
            self._fit_temperature(scores, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self._fitted = True
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate scores to probabilities.

        Args:
            scores: Raw reranker scores

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        if self.method == "platt":
            return self._calibrate_platt(scores)
        elif self.method == "isotonic":
            return self._calibrate_isotonic(scores)
        elif self.method == "temperature":
            return self._calibrate_temperature(scores)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_platt(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression)."""
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            a, b = params
            logits = a * scores + b
            probs = 1 / (1 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            ll = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            return -np.mean(ll)

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="L-BFGS-B")
        self._platt_params = tuple(result.x)
        logger.info("Platt scaling params: a=%.4f, b=%.4f", *self._platt_params)

    def _calibrate_platt(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        a, b = self._platt_params
        logits = a * scores + b
        return 1 / (1 + np.exp(-logits))

    def _fit_isotonic(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression."""
        from sklearn.isotonic import IsotonicRegression

        self._isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self._isotonic_model.fit(scores, labels)

    def _calibrate_isotonic(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic regression."""
        return self._isotonic_model.predict(scores)

    def _fit_temperature(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature scaling."""
        from scipy.optimize import minimize_scalar

        def nll(temp):
            scaled = scores / temp
            probs = 1 / (1 + np.exp(-scaled))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            ll = labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            return -np.mean(ll)

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        logger.info("Temperature scaling: T=%.4f", self.temperature)

    def _calibrate_temperature(self, scores: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        scaled = scores / self.temperature
        return 1 / (1 + np.exp(-scaled))


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_weight = mask.sum() / len(probs)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)
