"""Supersearch: Post-processing optimization framework.

This module implements a comprehensive search over post-processing approaches
for the S-C evidence retrieval pipeline, targeting deployment constraints:
- FPR (no-evidence) <= 5%
- TPR (has-evidence) >= 90%
- Evidence recall >= 93%
- Avg evidence K <= 5

Architecture:
- Stage-1: Fast NE gate (threshold, linear, tree models)
- UncertainSelector: Routes uncertain queries to Stage-2
- Stage-2: Heavy NE models (NO_EVIDENCE margin, GNN, verifier, ensembles)
- KSelector: Evidence count selection (fixed/dynamic, capped at K=5)
- Calibrator: Probability calibration (Platt, isotonic, temperature)
"""

from final_sc_review.supersearch.registry import (
    PluginRegistry,
    NEStage1Model,
    UncertainSelector,
    NEStage2Model,
    KSelector,
    Calibrator,
)

__all__ = [
    "PluginRegistry",
    "NEStage1Model",
    "UncertainSelector",
    "NEStage2Model",
    "KSelector",
    "Calibrator",
]
