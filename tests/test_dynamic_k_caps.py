"""Tests for dynamic K selection with deployment caps.

Requirements:
- k_min = 2 (default minimum)
- hard_cap = 10 (deployment constraint)
- k_max_ratio = 0.5 (adaptive cap)
- k_max1 = min(hard_cap, ceil(k_max_ratio * N_candidates))
"""

import math
import pytest
import numpy as np


class DynamicKConfig:
    """Configuration for dynamic K selection."""

    def __init__(
        self,
        k_min: int = 2,
        hard_cap: int = 10,
        k_max_ratio: float = 0.5,
    ):
        self.k_min = k_min
        self.hard_cap = hard_cap
        self.k_max_ratio = k_max_ratio

    def compute_k_max1(self, n_candidates: int) -> int:
        """Compute adaptive k_max1 for given number of candidates."""
        adaptive = math.ceil(self.k_max_ratio * n_candidates)
        return min(self.hard_cap, max(self.k_min, adaptive))

    def clamp_k(self, k: int, n_candidates: int) -> int:
        """Clamp K to valid range [k_min, k_max1]."""
        k_max1 = self.compute_k_max1(n_candidates)
        return max(self.k_min, min(k_max1, k))


class MassThresholdKSelector:
    """Dynamic K by probability mass threshold (DK1).

    Choose smallest K such that cumulative sum of calibrated probs >= gamma.
    K = min{k: sum_{i<=k} p_i >= gamma}
    """

    def __init__(self, gamma: float = 0.9, config: DynamicKConfig = None):
        self.gamma = gamma
        self.config = config or DynamicKConfig()

    def select_k(self, probs: np.ndarray, n_candidates: int) -> int:
        """Select K based on probability mass threshold."""
        if len(probs) == 0:
            return self.config.k_min

        # Sort probs descending
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)

        # Find smallest K where cumsum >= gamma
        k = 1
        for i, cum in enumerate(cumsum):
            if cum >= self.gamma:
                k = i + 1
                break
        else:
            k = len(sorted_probs)

        # Clamp to valid range
        return self.config.clamp_k(k, n_candidates)


class ScoreGapKneeSelector:
    """Dynamic K by score gap/knee detection (DK2).

    Find position with largest score gap.
    """

    def __init__(self, gap_threshold: float = 0.1, config: DynamicKConfig = None):
        self.gap_threshold = gap_threshold
        self.config = config or DynamicKConfig()

    def select_k(self, scores: np.ndarray, n_candidates: int) -> int:
        """Select K based on score gap detection."""
        if len(scores) <= 1:
            return self.config.k_min

        # Sort scores descending
        sorted_scores = np.sort(scores)[::-1]
        gaps = np.abs(np.diff(sorted_scores))

        # Find first position with large gap
        k = len(sorted_scores)  # Default: return all
        for i, gap in enumerate(gaps):
            if gap > self.gap_threshold:
                k = i + 1
                break

        return self.config.clamp_k(k, n_candidates)


class TestDynamicKConfig:
    """Tests for DynamicKConfig."""

    def test_k_max1_n20(self):
        """N=20 => k_max1=10 (hard cap)."""
        config = DynamicKConfig(k_min=2, hard_cap=10, k_max_ratio=0.5)
        assert config.compute_k_max1(20) == 10

    def test_k_max1_n12(self):
        """N=12 => k_max1=6."""
        config = DynamicKConfig(k_min=2, hard_cap=10, k_max_ratio=0.5)
        assert config.compute_k_max1(12) == 6

    def test_k_max1_n40(self):
        """N=40 => k_max1=10 (capped)."""
        config = DynamicKConfig(k_min=2, hard_cap=10, k_max_ratio=0.5)
        assert config.compute_k_max1(40) == 10

    def test_k_max1_n4(self):
        """N=4 => k_max1=2 (k_min floor)."""
        config = DynamicKConfig(k_min=2, hard_cap=10, k_max_ratio=0.5)
        assert config.compute_k_max1(4) == 2

    def test_k_max1_n2(self):
        """N=2 => k_max1=2."""
        config = DynamicKConfig(k_min=2, hard_cap=10, k_max_ratio=0.5)
        assert config.compute_k_max1(2) == 2

    def test_clamp_within_bounds(self):
        """Test clamping is always within [k_min, k_max1]."""
        config = DynamicKConfig(k_min=2, hard_cap=10, k_max_ratio=0.5)

        test_cases = [
            (0, 20, 2),   # 0 -> k_min=2
            (1, 20, 2),   # 1 -> k_min=2
            (5, 20, 5),   # 5 within bounds
            (10, 20, 10), # 10 = k_max1
            (15, 20, 10), # 15 -> hard_cap=10
            (5, 8, 4),    # k_max1=4 for N=8, so 5->4
        ]

        for raw_k, n_candidates, expected in test_cases:
            result = config.clamp_k(raw_k, n_candidates)
            assert result == expected, f"clamp({raw_k}, n={n_candidates}) = {result}, expected {expected}"


class TestMassThresholdKSelector:
    """Tests for mass threshold K selector."""

    def test_high_confidence_low_k(self):
        """High confidence in top-1 => low K."""
        selector = MassThresholdKSelector(gamma=0.9)
        probs = np.array([0.95, 0.02, 0.01, 0.01, 0.01])
        k = selector.select_k(probs, n_candidates=20)
        assert k == 2  # k_min is 2

    def test_spread_confidence_high_k(self):
        """Spread confidence => higher K."""
        selector = MassThresholdKSelector(gamma=0.9)
        probs = np.array([0.2, 0.18, 0.17, 0.16, 0.15, 0.14])
        k = selector.select_k(probs, n_candidates=20)
        assert k >= 4  # Need multiple to reach 90%

    def test_respects_hard_cap(self):
        """Should never exceed hard cap."""
        config = DynamicKConfig(hard_cap=10)
        selector = MassThresholdKSelector(gamma=0.99, config=config)
        probs = np.array([0.05] * 30)  # Very spread out
        k = selector.select_k(probs, n_candidates=30)
        assert k <= 10


class TestScoreGapKneeSelector:
    """Tests for score gap knee selector."""

    def test_clear_gap_detected(self):
        """Clear score gap => cut at gap."""
        selector = ScoreGapKneeSelector(gap_threshold=0.1)
        scores = np.array([0.9, 0.85, 0.8, 0.3, 0.2, 0.1])  # Gap after position 3
        k = selector.select_k(scores, n_candidates=20)
        assert k == 3

    def test_no_gap_returns_all(self):
        """No clear gap => return more candidates."""
        config = DynamicKConfig(hard_cap=10)
        selector = ScoreGapKneeSelector(gap_threshold=0.3, config=config)
        scores = np.array([0.5, 0.48, 0.46, 0.44, 0.42])  # Small gaps
        k = selector.select_k(scores, n_candidates=5)
        assert k >= 2  # At least k_min

    def test_respects_k_min(self):
        """Should never go below k_min."""
        config = DynamicKConfig(k_min=2)
        selector = ScoreGapKneeSelector(gap_threshold=0.01, config=config)
        scores = np.array([0.9, 0.1, 0.05, 0.02])  # Gap immediately after top-1
        k = selector.select_k(scores, n_candidates=20)
        assert k >= 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_probs(self):
        """Empty probs array."""
        selector = MassThresholdKSelector()
        k = selector.select_k(np.array([]), n_candidates=0)
        assert k == 2  # k_min

    def test_single_candidate(self):
        """Single candidate."""
        config = DynamicKConfig(k_min=1)  # Allow k_min=1 for this test
        selector = MassThresholdKSelector(config=config)
        k = selector.select_k(np.array([1.0]), n_candidates=1)
        assert k == 1

    def test_all_equal_probs(self):
        """All equal probabilities."""
        selector = MassThresholdKSelector(gamma=0.5)
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        k = selector.select_k(probs, n_candidates=10)
        assert k >= 2  # Need at least 2-3 to reach 50%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
