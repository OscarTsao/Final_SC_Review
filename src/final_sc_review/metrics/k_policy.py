"""K policy definition for evaluation metrics.

This module defines the standard K values for evaluation metrics that are:
1. Deployment-relevant: Reflect real-world usage patterns
2. Fair: Use K_eff = min(K, n_candidates) to handle short posts
3. Interpretable: Oracle@200 is meaningless when posts have ~20 sentences

Paper-standard K policy:
- Primary K: [1, 3, 5, 10] - covers typical deployment scenarios
- Extended K: [20] - for longer posts (only if p95 >= 20)
- Ceiling K: [ALL] - oracle sanity check (K = n_candidates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# Standard K values for paper reporting (per R3)
# K_primary = [3, 5, 10] are mandatory deployment metrics
# K_optional = [1] is optional for very short outputs
K_PRIMARY = [3, 5, 10]  # Mandatory per R3
K_OPTIONAL = [1]  # Optional per R3
K_EXTENDED = []  # Removed - primary K is sufficient
K_CEILING = None  # Represents ALL candidates (sanity-only per R3)

# For backwards compatibility
K_DEFAULT = [3, 5, 10]


@dataclass
class KPolicy:
    """K policy configuration for evaluation.

    Attributes:
        primary_k: Primary K values for deployment metrics
        extended_k: Extended K values for long-tail analysis
        use_ceiling: Whether to compute ceiling metrics (oracle@ALL)
        k_eff_enabled: Whether to use K_eff = min(K, n_candidates)
    """

    primary_k: List[int]
    extended_k: List[int]
    use_ceiling: bool = True
    k_eff_enabled: bool = True

    @classmethod
    def paper_standard(cls) -> "KPolicy":
        """Return paper-standard K policy."""
        return cls(
            primary_k=K_PRIMARY.copy(),
            extended_k=K_EXTENDED.copy(),
            use_ceiling=True,
            k_eff_enabled=True,
        )

    @classmethod
    def from_post_stats(cls, p95_sentences: float) -> "KPolicy":
        """Create K policy based on post length statistics.

        Args:
            p95_sentences: 95th percentile of sentences per post

        Returns:
            KPolicy with appropriate K values
        """
        primary = K_PRIMARY.copy()
        extended = []

        # Add extended K only if p95 is large enough
        if p95_sentences >= 20:
            extended = [20]
        if p95_sentences >= 50:
            extended.extend([50])
        if p95_sentences >= 100:
            extended.extend([100])

        return cls(
            primary_k=primary,
            extended_k=extended,
            use_ceiling=True,
            k_eff_enabled=True,
        )

    @property
    def all_k(self) -> List[int]:
        """Return all K values (primary + extended)."""
        return sorted(set(self.primary_k + self.extended_k))

    def get_k_eff(self, k: int, n_candidates: int) -> int:
        """Compute effective K: min(K, n_candidates).

        This ensures fair comparison when candidate pool is smaller than K.

        Args:
            k: Requested K value
            n_candidates: Number of available candidates

        Returns:
            Effective K value
        """
        if self.k_eff_enabled:
            return min(k, n_candidates)
        return k


def compute_k_eff(k: int, n_candidates: int) -> int:
    """Compute effective K = min(K, n_candidates).

    This is the standalone function for use outside of KPolicy class.

    Args:
        k: Requested K value
        n_candidates: Number of available candidates

    Returns:
        Effective K value
    """
    return min(k, n_candidates)


def get_paper_k_values() -> List[int]:
    """Return paper-standard K values for metrics reporting."""
    return K_PRIMARY + K_EXTENDED


def validate_k_for_metrics(
    k_values: List[int],
    n_candidates_percentiles: Tuple[float, float, float],
) -> Tuple[List[int], List[str]]:
    """Validate K values against candidate count distribution.

    Args:
        k_values: Requested K values
        n_candidates_percentiles: (p50, p90, p99) of n_candidates per query

    Returns:
        Tuple of (valid_k_values, warnings)
    """
    p50, p90, p99 = n_candidates_percentiles
    valid_k = []
    warnings = []

    for k in k_values:
        if k > p99:
            warnings.append(
                f"K={k} exceeds p99={p99:.0f} of candidate counts; "
                f"this is effectively a ceiling metric"
            )
        elif k > p90:
            warnings.append(
                f"K={k} exceeds p90={p90:.0f}; "
                f"K_eff will be used for many queries"
            )
        valid_k.append(k)

    return valid_k, warnings
