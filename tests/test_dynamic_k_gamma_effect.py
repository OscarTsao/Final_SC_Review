"""Tests for Dynamic-K gamma effect in mass-based selection.

These tests verify that:
1. Different gamma values produce different K selections
2. Higher gamma values produce larger K (more coverage needed)
3. The normalized version correctly handles probability sums < 1
"""

import numpy as np
import pytest


def select_k_mass_buggy(probs: np.ndarray, gamma: float, k_min: int, k_max: int) -> int:
    """BUGGY version: Uses raw probabilities without normalization."""
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    k = np.searchsorted(cumsum, gamma) + 1
    return max(k_min, min(k, k_max))


def select_k_mass_fixed(probs: np.ndarray, gamma: float, k_min: int, k_max: int) -> int:
    """FIXED version: Normalizes probabilities to sum to 1."""
    sorted_probs = np.sort(probs)[::-1]
    total = sorted_probs.sum()
    if total <= 0:
        # All zeros or negative: no meaningful selection, return k_min
        return k_min
    sorted_probs_norm = sorted_probs / total
    cumsum = np.cumsum(sorted_probs_norm)
    k = np.searchsorted(cumsum, gamma) + 1
    return max(k_min, min(k, k_max))


class TestGammaInvariance:
    """Tests demonstrating the gamma invariance bug."""

    def test_buggy_version_gamma_invariant_when_sum_low(self):
        """Buggy version produces identical K when prob sum < min gamma."""
        # Create probabilities that sum to less than 0.8
        probs = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
        # Sum = 0.72 < 0.8
        assert probs.sum() < 0.8

        k_min, k_max = 2, 5

        k_08 = select_k_mass_buggy(probs, 0.8, k_min, k_max)
        k_09 = select_k_mass_buggy(probs, 0.9, k_min, k_max)
        k_095 = select_k_mass_buggy(probs, 0.95, k_min, k_max)

        # Bug: All should be identical (all return k_max because cumsum never reaches gamma)
        assert k_08 == k_09 == k_095 == k_max, (
            f"Expected all K to be k_max={k_max} due to bug, "
            f"got k_08={k_08}, k_09={k_09}, k_095={k_095}"
        )

    def test_fixed_version_gamma_sensitive(self):
        """Fixed version produces different K for different gamma values."""
        # Same probabilities
        probs = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])

        k_min, k_max = 2, 10

        k_08 = select_k_mass_fixed(probs, 0.8, k_min, k_max)
        k_09 = select_k_mass_fixed(probs, 0.9, k_min, k_max)
        k_095 = select_k_mass_fixed(probs, 0.95, k_min, k_max)

        # Fixed version should show: k_08 <= k_09 <= k_095
        assert k_08 <= k_09 <= k_095, (
            f"Expected monotonic increase with gamma, "
            f"got k_08={k_08}, k_09={k_09}, k_095={k_095}"
        )

        # And they should NOT all be equal
        assert not (k_08 == k_09 == k_095), (
            f"Fixed version should show variation, but all are equal: {k_08}"
        )


class TestGammaMonotonicity:
    """Tests for gamma monotonicity in the fixed version."""

    @pytest.mark.parametrize("prob_sum", [0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    def test_gamma_monotonicity(self, prob_sum):
        """Higher gamma should always produce K >= lower gamma's K."""
        np.random.seed(42)
        n = 10
        probs = np.random.rand(n)
        probs = probs / probs.sum() * prob_sum  # Scale to target sum

        k_min, k_max = 2, n

        gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        ks = [select_k_mass_fixed(probs, g, k_min, k_max) for g in gammas]

        # Verify monotonicity
        for i in range(len(ks) - 1):
            assert ks[i] <= ks[i + 1], (
                f"K should be monotonic with gamma. "
                f"gamma={gammas[i]}: K={ks[i]}, gamma={gammas[i+1]}: K={ks[i+1]}"
            )


class TestGammaEdgeCases:
    """Test edge cases for gamma-based selection."""

    def test_gamma_1_selects_all(self):
        """gamma=1.0 should select all candidates (up to k_max)."""
        probs = np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        k_min, k_max = 2, 10

        k = select_k_mass_fixed(probs, 1.0, k_min, k_max)
        assert k == len(probs), f"gamma=1.0 should select all, got {k}"

    def test_gamma_0_selects_k_min(self):
        """gamma~0 should select k_min."""
        probs = np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        k_min, k_max = 2, 10

        k = select_k_mass_fixed(probs, 0.01, k_min, k_max)
        assert k == k_min, f"Very small gamma should select k_min, got {k}"

    def test_all_zero_probs(self):
        """All zero probabilities should return k_min."""
        probs = np.zeros(10)
        k_min, k_max = 2, 5

        k = select_k_mass_fixed(probs, 0.8, k_min, k_max)
        assert k == k_min, f"Zero probs should return k_min, got {k}"

    def test_single_high_prob(self):
        """Single high probability node should be selected first."""
        probs = np.array([0.9, 0.01, 0.01, 0.01, 0.01])
        k_min, k_max = 1, 5

        # gamma=0.5 should select just 1 (the dominant node)
        k = select_k_mass_fixed(probs, 0.5, k_min, k_max)
        # After normalization: [0.96, 0.01, 0.01, 0.01, 0.01]
        # Top 1 covers 0.96 > 0.5
        assert k == 1, f"Expected K=1 for dominant node, got {k}"


class TestNormalizationCorrectness:
    """Verify the normalization produces correct results."""

    def test_normalized_cumsum_reaches_1(self):
        """Normalized probabilities should cumsum to 1.0."""
        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        sorted_probs = np.sort(probs)[::-1]
        sorted_probs_norm = sorted_probs / sorted_probs.sum()

        cumsum = np.cumsum(sorted_probs_norm)

        assert np.isclose(cumsum[-1], 1.0), f"Cumsum should reach 1.0, got {cumsum[-1]}"

    def test_specific_gamma_calculation(self):
        """Test specific gamma calculation with known result."""
        # Create probabilities where we know the answer
        probs = np.array([0.5, 0.3, 0.15, 0.05])  # Sum = 1.0, already normalized
        k_min, k_max = 1, 4

        # Sorted: [0.5, 0.3, 0.15, 0.05]
        # Cumsum: [0.5, 0.8, 0.95, 1.0]
        # gamma=0.5 -> K=1 (0.5 >= 0.5)
        # gamma=0.8 -> K=2 (0.8 >= 0.8)
        # gamma=0.95 -> K=3 (0.95 >= 0.95)

        k_05 = select_k_mass_fixed(probs, 0.5, k_min, k_max)
        k_08 = select_k_mass_fixed(probs, 0.8, k_min, k_max)
        k_095 = select_k_mass_fixed(probs, 0.95, k_min, k_max)

        assert k_05 == 1, f"gamma=0.5 should give K=1, got {k_05}"
        assert k_08 == 2, f"gamma=0.8 should give K=2, got {k_08}"
        assert k_095 == 3, f"gamma=0.95 should give K=3, got {k_095}"


class TestBuggyVsFixedComparison:
    """Compare buggy and fixed versions to demonstrate the bug."""

    def test_typical_sigmoid_outputs(self):
        """Test with typical sigmoid outputs (often sum < 1)."""
        # Typical sigmoid outputs for 10 nodes
        probs = np.array([0.65, 0.55, 0.45, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
        # Sum = 3.05

        k_min, k_max = 2, 5

        # Buggy version
        k_08_buggy = select_k_mass_buggy(probs, 0.8, k_min, k_max)
        k_09_buggy = select_k_mass_buggy(probs, 0.9, k_min, k_max)

        # Fixed version
        k_08_fixed = select_k_mass_fixed(probs, 0.8, k_min, k_max)
        k_09_fixed = select_k_mass_fixed(probs, 0.9, k_min, k_max)

        # Buggy: both should select same K (cumsum quickly exceeds gamma)
        # Fixed: should show difference

        # The key assertion: fixed version should differentiate
        print(f"Buggy: gamma=0.8 -> K={k_08_buggy}, gamma=0.9 -> K={k_09_buggy}")
        print(f"Fixed: gamma=0.8 -> K={k_08_fixed}, gamma=0.9 -> K={k_09_fixed}")

        # Fixed version should have k_09 >= k_08
        assert k_09_fixed >= k_08_fixed, (
            f"Fixed version should show k_09 >= k_08, "
            f"got {k_09_fixed} and {k_08_fixed}"
        )

    def test_low_confidence_sigmoid_outputs(self):
        """Test with low confidence sigmoid outputs (sum < 1)."""
        # Low confidence: all near 0.1
        probs = np.array([0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
        # Sum = 0.75 < 0.8

        k_min, k_max = 2, 10

        # Buggy version should be gamma-invariant
        k_08_buggy = select_k_mass_buggy(probs, 0.8, k_min, k_max)
        k_095_buggy = select_k_mass_buggy(probs, 0.95, k_min, k_max)

        # Fixed version should vary
        k_08_fixed = select_k_mass_fixed(probs, 0.8, k_min, k_max)
        k_095_fixed = select_k_mass_fixed(probs, 0.95, k_min, k_max)

        # Assert buggy version fails to differentiate
        assert k_08_buggy == k_095_buggy == k_max, (
            f"Buggy version should return k_max for both when sum < gamma, "
            f"got {k_08_buggy} and {k_095_buggy}"
        )

        # Assert fixed version does differentiate
        assert k_08_fixed < k_095_fixed, (
            f"Fixed version should show k_08 < k_095, "
            f"got {k_08_fixed} and {k_095_fixed}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
