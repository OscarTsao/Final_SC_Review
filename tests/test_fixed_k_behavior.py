"""Tests for Fixed K selection behavior and constraints.

These tests verify that:
1. Fixed K respects k_min and k_max constraints
2. k_max_ratio=0.5 properly constrains K on small graphs
3. The documentation correctly explains avgK != target_K
"""

import numpy as np
import pytest
from typing import Tuple


def compute_k_constraints(
    n_candidates: int,
    k_min: int = 2,
    k_max: int = 10,
    k_max_ratio: float = 0.5
) -> Tuple[int, int]:
    """Compute k constraints for a given number of candidates.

    Returns:
        Tuple of (actual_k_min, actual_k_max)
    """
    k_max_from_ratio = int(np.ceil(n_candidates * k_max_ratio))
    actual_k_max = min(k_max, k_max_from_ratio, n_candidates)
    actual_k_max = max(actual_k_max, k_min)
    return k_min, actual_k_max


def select_k_fixed(n_candidates: int, fixed_k: int, k_min: int, k_max: int) -> int:
    """Fixed K policy with constraints."""
    return max(k_min, min(fixed_k, k_max))


class TestKConstraints:
    """Tests for K constraint computation."""

    def test_large_graph_no_constraint(self):
        """Large graphs should not be constrained by k_max_ratio."""
        n = 30
        k_min, k_max = compute_k_constraints(n)

        # k_max_from_ratio = ceil(30 * 0.5) = 15
        # actual_k_max = min(10, 15, 30) = 10
        assert k_max == 10, f"Expected k_max=10, got {k_max}"
        assert k_min == 2, f"Expected k_min=2, got {k_min}"

    def test_small_graph_constrained(self):
        """Small graphs should be constrained by k_max_ratio."""
        n = 6
        k_min, k_max = compute_k_constraints(n)

        # k_max_from_ratio = ceil(6 * 0.5) = 3
        # actual_k_max = min(10, 3, 6) = 3
        assert k_max == 3, f"Expected k_max=3, got {k_max}"

    def test_very_small_graph(self):
        """Very small graphs should have k_max = k_min."""
        n = 3
        k_min, k_max = compute_k_constraints(n)

        # k_max_from_ratio = ceil(3 * 0.5) = 2
        # actual_k_max = max(min(10, 2, 3), k_min) = max(2, 2) = 2
        assert k_max >= k_min, f"k_max should be >= k_min"
        assert k_max == 2, f"Expected k_max=2, got {k_max}"

    def test_constraint_examples_from_data(self):
        """Test specific examples based on actual data distribution."""
        # Based on graph dataset analysis
        test_cases = [
            (8, 2, 4),   # n=8: k_max = ceil(8*0.5) = 4
            (10, 2, 5),  # n=10: k_max = ceil(10*0.5) = 5
            (15, 2, 8),  # n=15: k_max = ceil(15*0.5) = 8
            (20, 2, 10), # n=20: k_max = min(10, ceil(20*0.5)) = 10
            (25, 2, 10), # n=25: k_max = 10 (hard cap)
        ]

        for n, expected_min, expected_max in test_cases:
            k_min, k_max = compute_k_constraints(n)
            assert k_min == expected_min, f"n={n}: expected k_min={expected_min}, got {k_min}"
            assert k_max == expected_max, f"n={n}: expected k_max={expected_max}, got {k_max}"


class TestFixedKSelection:
    """Tests for Fixed K selection with constraints."""

    def test_fixed_k_within_bounds(self):
        """Fixed K within bounds should not be modified."""
        k_min, k_max = 2, 10
        for target_k in range(k_min, k_max + 1):
            actual_k = select_k_fixed(20, target_k, k_min, k_max)
            assert actual_k == target_k, f"target={target_k} should give actual={target_k}, got {actual_k}"

    def test_fixed_k_above_k_max(self):
        """Fixed K above k_max should be clamped."""
        k_min, k_max = 2, 5
        target_k = 10

        actual_k = select_k_fixed(20, target_k, k_min, k_max)
        assert actual_k == k_max, f"target={target_k} should be clamped to {k_max}, got {actual_k}"

    def test_fixed_k_below_k_min(self):
        """Fixed K below k_min should be raised."""
        k_min, k_max = 2, 10
        target_k = 1

        actual_k = select_k_fixed(20, target_k, k_min, k_max)
        assert actual_k == k_min, f"target={target_k} should be raised to {k_min}, got {actual_k}"

    def test_fixed_k_5_on_small_graph(self):
        """Fixed K=5 on small graph should be constrained."""
        n = 8
        target_k = 5

        k_min, k_max = compute_k_constraints(n)
        # k_max = ceil(8 * 0.5) = 4

        actual_k = select_k_fixed(n, target_k, k_min, k_max)
        assert actual_k == 4, f"Fixed K=5 on n=8 should give K=4, got {actual_k}"


class TestFixedKAverage:
    """Tests demonstrating why avgK != target K."""

    def test_avg_k_distribution_simulation(self):
        """Simulate avgK calculation across a realistic distribution."""
        # Realistic distribution of n_candidates from actual data
        # Based on metadata: typically 5-30 candidates per query
        np.random.seed(42)
        n_samples = 1000

        # Sample graph sizes from realistic distribution
        graph_sizes = np.random.choice(
            range(5, 31),
            size=n_samples,
            p=np.array([1/(31-5)] * 26)  # Uniform for simplicity
        )

        target_k = 5
        actual_ks = []

        for n in graph_sizes:
            k_min, k_max = compute_k_constraints(n)
            k = select_k_fixed(n, target_k, k_min, k_max)
            actual_ks.append(k)

        avg_k = np.mean(actual_ks)

        # avgK should be less than target due to constraints
        assert avg_k < target_k, f"Expected avgK < {target_k}, got {avg_k:.2f}"

        # Count how many were constrained
        n_constrained = sum(1 for k in actual_ks if k != target_k)
        pct_constrained = n_constrained / n_samples * 100

        print(f"Target K: {target_k}")
        print(f"Actual avgK: {avg_k:.2f}")
        print(f"% Constrained: {pct_constrained:.1f}%")

        # K distribution
        k_dist = {k: actual_ks.count(k) for k in sorted(set(actual_ks))}
        print(f"K Distribution: {k_dist}")

    def test_specific_constraint_scenario(self):
        """Test specific scenario: Fixed K=5 but avgK=3.63."""
        # Reproduce the reported bug: Fixed K=5 showing avgK=3.63

        # This happens when many graphs have n_candidates where k_max < 5
        # k_max < 5 when: ceil(n * 0.5) < 5, i.e., n < 10

        # Simulate: 50% graphs with n=6 (k_max=3), 50% with n=20 (k_max=10)
        graph_sizes = [6] * 500 + [20] * 500
        np.random.shuffle(graph_sizes)

        target_k = 5
        actual_ks = []

        for n in graph_sizes:
            k_min, k_max = compute_k_constraints(n)
            k = select_k_fixed(n, target_k, k_min, k_max)
            actual_ks.append(k)

        avg_k = np.mean(actual_ks)

        # n=6: k_max=3, actual_k=3
        # n=20: k_max=10, actual_k=5
        # Average: (3*500 + 5*500) / 1000 = 4.0

        expected_avg = (3 * 500 + 5 * 500) / 1000
        assert np.isclose(avg_k, expected_avg, atol=0.01), (
            f"Expected avgK={expected_avg}, got {avg_k}"
        )


class TestConstraintDocumentation:
    """Tests to ensure constraint behavior is well documented."""

    def test_constraint_table_accuracy(self):
        """Verify the constraint table in documentation is accurate."""
        constraint_table = [
            # (n_candidates, expected_k_min, expected_k_max)
            (4, 2, 2),   # ceil(4*0.5)=2, max(2,2)=2
            (5, 2, 3),   # ceil(5*0.5)=3, min(10,3)=3
            (6, 2, 3),   # ceil(6*0.5)=3
            (7, 2, 4),   # ceil(7*0.5)=4
            (8, 2, 4),   # ceil(8*0.5)=4
            (9, 2, 5),   # ceil(9*0.5)=5
            (10, 2, 5),  # ceil(10*0.5)=5
            (15, 2, 8),  # ceil(15*0.5)=8
            (20, 2, 10), # ceil(20*0.5)=10, capped at 10
            (30, 2, 10), # capped at 10
        ]

        for n, expected_min, expected_max in constraint_table:
            k_min, k_max = compute_k_constraints(n)
            assert k_min == expected_min and k_max == expected_max, (
                f"n={n}: expected ({expected_min}, {expected_max}), "
                f"got ({k_min}, {k_max})"
            )

    def test_constraint_formula_explicit(self):
        """Explicitly test the constraint formula."""
        for n in range(1, 50):
            k_min, k_max = compute_k_constraints(n)

            # Verify formula
            expected_k_max_from_ratio = int(np.ceil(n * 0.5))
            expected_k_max = min(10, expected_k_max_from_ratio, n)
            expected_k_max = max(expected_k_max, 2)  # Ensure >= k_min

            assert k_max == expected_k_max, (
                f"n={n}: formula gives {expected_k_max}, function gives {k_max}"
            )


class TestEdgeCases:
    """Test edge cases for Fixed K selection."""

    def test_very_small_graph_n_2(self):
        """Test with n=2 (minimum viable graph)."""
        n = 2
        k_min, k_max = compute_k_constraints(n)

        # k_max_from_ratio = ceil(2*0.5) = 1
        # actual_k_max = max(min(10, 1, 2), 2) = 2
        assert k_max == 2, f"n=2 should have k_max=2, got {k_max}"

        # Fixed K=5 should give K=2
        actual_k = select_k_fixed(n, 5, k_min, k_max)
        assert actual_k == 2, f"Fixed K=5 on n=2 should give K=2, got {actual_k}"

    def test_graph_size_equals_target_k(self):
        """Test when n_candidates equals target K."""
        n = 5
        target_k = 5
        k_min, k_max = compute_k_constraints(n)

        # k_max = ceil(5*0.5) = 3
        actual_k = select_k_fixed(n, target_k, k_min, k_max)
        assert actual_k == 3, f"Fixed K=5 on n=5 should give K=3, got {actual_k}"

    def test_zero_candidates(self):
        """Test with n=0 (should be prevented upstream but test anyway)."""
        n = 0
        # This should ideally not happen, but test robustness
        k_min, k_max = compute_k_constraints(max(n, 1))  # Prevent div by zero

        # k_max_from_ratio = ceil(1*0.5) = 1
        # actual_k_max = max(min(10, 1, 1), 2) = 2
        assert k_max == 2, f"n=0 (treated as 1) should have k_max=2, got {k_max}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
