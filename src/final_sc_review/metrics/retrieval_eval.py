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


def dual_evaluate(
    results: Sequence[Dict],
    ks: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """Run evaluation in both modes: positives-only and all-queries.

    Args:
        results: List of dicts with keys: post_id, criterion_id, gold_ids, ranked_ids.
        ks: List of K values for metrics.

    Returns:
        Dict with two keys:
        - "positives_only": metrics computed only on queries with gold positives
        - "all_queries": metrics computed on all queries (includes negatives)

    Rationale:
        skip_no_positives=True inflates metrics (only measures queries where success is possible).
        skip_no_positives=False gives realistic system-level "hit rate".
    """
    return {
        "positives_only": evaluate_rankings(results, ks, skip_no_positives=True),
        "all_queries": evaluate_rankings(results, ks, skip_no_positives=False),
    }


def format_dual_metrics(dual_results: Dict[str, Dict[str, float]], k: int = 10) -> str:
    """Format dual evaluation results for display.

    Args:
        dual_results: Output from dual_evaluate().
        k: The K value to highlight.

    Returns:
        Formatted string with key metrics.
    """
    pos = dual_results["positives_only"]
    all_q = dual_results["all_queries"]

    lines = [
        f"Evaluation at k={k}:",
        f"  Positives-only (n={pos['queries_evaluated']}):",
        f"    nDCG@{k}:   {pos.get(f'ndcg@{k}', 0):.4f}",
        f"    Recall@{k}: {pos.get(f'recall@{k}', 0):.4f}",
        f"    MRR@{k}:    {pos.get(f'mrr@{k}', 0):.4f}",
        f"  All-queries (n={all_q['queries_evaluated']}):",
        f"    nDCG@{k}:   {all_q.get(f'ndcg@{k}', 0):.4f}",
        f"    Recall@{k}: {all_q.get(f'recall@{k}', 0):.4f}",
        f"    MRR@{k}:    {all_q.get(f'mrr@{k}', 0):.4f}",
        f"  Coverage: {pos['queries_with_positives']}/{all_q['queries_total']} queries have positives "
        f"({100*pos['queries_with_positives']/max(all_q['queries_total'],1):.1f}%)",
    ]
    return "\n".join(lines)
