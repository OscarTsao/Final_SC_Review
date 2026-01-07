"""Strict retrieval evaluation requiring groundtruth labels."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from final_sc_review.metrics.ranking import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def evaluate_rankings(
    results: Sequence[Dict],
    ks: Sequence[int],
    skip_no_positives: bool = True,
) -> Dict[str, float]:
    """Evaluate retrieval rankings.

    Args:
        results: List of dicts with keys: post_id, criterion_id, gold_ids, ranked_ids.
        ks: List of K values for metrics.
        skip_no_positives: If True, queries with no positives are excluded.

    Returns:
        Dict with aggregated metrics and counts.
    """
    metrics: Dict[str, float] = {}
    counts = {
        "queries_total": len(results),
        "queries_with_positives": 0,
    }

    # Accumulators
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"mrr@{k}": 0.0 for k in ks})
    sums.update({f"map@{k}": 0.0 for k in ks})
    sums.update({f"ndcg@{k}": 0.0 for k in ks})

    effective = 0
    for res in results:
        gold_ids = res.get("gold_ids", [])
        ranked_ids = res.get("ranked_ids", [])
        has_pos = len(gold_ids) > 0
        if not has_pos and skip_no_positives:
            continue
        if has_pos:
            counts["queries_with_positives"] += 1
        effective += 1
        for k in ks:
            sums[f"recall@{k}"] += recall_at_k(gold_ids, ranked_ids, k)
            sums[f"mrr@{k}"] += mrr_at_k(gold_ids, ranked_ids, k)
            sums[f"map@{k}"] += map_at_k(gold_ids, ranked_ids, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(gold_ids, ranked_ids, k)

    denom = max(effective, 1)
    for key, value in sums.items():
        metrics[key] = value / denom

    metrics.update(counts)
    metrics["queries_evaluated"] = effective
    metrics["skip_no_positives"] = bool(skip_no_positives)
    return metrics
