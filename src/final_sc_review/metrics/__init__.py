"""Metrics package for S-C retrieval evaluation.

Exports:
- Ranking metrics: recall_at_k, mrr_at_k, map_at_k, ndcg_at_k
- K policy: compute_k_eff, get_paper_k_values, K_PRIMARY, K_EXTENDED
- Evaluation: evaluate_rankings, dual_evaluate, paper_evaluate
"""

from final_sc_review.metrics.ranking import (
    recall_at_k,
    mrr_at_k,
    map_at_k,
    ndcg_at_k,
)
from final_sc_review.metrics.k_policy import (
    compute_k_eff,
    get_paper_k_values,
    K_PRIMARY,
    K_EXTENDED,
    K_CEILING,
)
from final_sc_review.metrics.retrieval_eval import (
    evaluate_rankings,
    dual_evaluate,
    paper_evaluate,
    evaluate_with_k_eff,
    format_dual_metrics,
)

__all__ = [
    # Ranking metrics
    "recall_at_k",
    "mrr_at_k",
    "map_at_k",
    "ndcg_at_k",
    # K policy
    "compute_k_eff",
    "get_paper_k_values",
    "K_PRIMARY",
    "K_EXTENDED",
    "K_CEILING",
    # Evaluation
    "evaluate_rankings",
    "dual_evaluate",
    "paper_evaluate",
    "evaluate_with_k_eff",
    "format_dual_metrics",
]
