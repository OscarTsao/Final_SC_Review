"""Strict retrieval evaluation requiring groundtruth labels.

Key features:
- K_eff support: min(K, n_candidates) for fair short-post evaluation
- Dual evaluation: positives-only vs all-queries metrics
- Paper-standard K policy: [1, 3, 5, 10, 20] with ceiling
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from final_sc_review.metrics.ranking import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from final_sc_review.metrics.k_policy import compute_k_eff, get_paper_k_values, K_PRIMARY


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


def evaluate_with_k_eff(
    results: Sequence[Dict],
    ks: Optional[Sequence[int]] = None,
    skip_no_positives: bool = True,
    include_ceiling: bool = True,
) -> Dict[str, float]:
    """Evaluate rankings using K_eff = min(K, n_candidates) for fair comparison.

    This is the paper-standard evaluation that:
    1. Uses K_eff to handle short posts fairly
    2. Computes oracle@ALL as ceiling metric
    3. Reports candidate count distribution

    Args:
        results: List of dicts with keys: post_id, criterion_id, gold_ids, ranked_ids.
        ks: K values to compute. Defaults to paper-standard [1,3,5,10,20].
        skip_no_positives: If True, exclude queries with no positives.
        include_ceiling: If True, include oracle@ALL metric.

    Returns:
        Dict with metrics using K_eff and candidate distribution stats.
    """
    if ks is None:
        ks = get_paper_k_values()

    metrics: Dict[str, float] = {}

    # Accumulators
    sums = {f"recall@{k}": 0.0 for k in ks}
    sums.update({f"mrr@{k}": 0.0 for k in ks})
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    sums.update({f"oracle@{k}": 0.0 for k in ks})  # Oracle = recall with K_eff
    if include_ceiling:
        sums["oracle@ALL"] = 0.0

    # Track candidate counts for distribution
    candidate_counts = []
    effective = 0
    queries_with_positives = 0

    for res in results:
        gold_ids = res.get("gold_ids", [])
        ranked_ids = res.get("ranked_ids", [])
        n_candidates = len(ranked_ids)
        has_pos = len(gold_ids) > 0

        if not has_pos and skip_no_positives:
            continue

        if has_pos:
            queries_with_positives += 1

        effective += 1
        candidate_counts.append(n_candidates)

        for k in ks:
            k_eff = compute_k_eff(k, n_candidates)
            sums[f"recall@{k}"] += recall_at_k(gold_ids, ranked_ids, k_eff)
            sums[f"mrr@{k}"] += mrr_at_k(gold_ids, ranked_ids, k_eff)
            sums[f"ndcg@{k}"] += ndcg_at_k(gold_ids, ranked_ids, k_eff)
            # Oracle: did we find any positive in top-K_eff?
            sums[f"oracle@{k}"] += 1.0 if any(uid in gold_ids for uid in ranked_ids[:k_eff]) else 0.0

        if include_ceiling and n_candidates > 0:
            # Oracle@ALL: recall at full candidate set
            sums["oracle@ALL"] += recall_at_k(gold_ids, ranked_ids, n_candidates)

    denom = max(effective, 1)
    for key, value in sums.items():
        metrics[key] = value / denom

    # Add counts
    metrics["queries_total"] = len(results)
    metrics["queries_with_positives"] = queries_with_positives
    metrics["queries_evaluated"] = effective
    metrics["skip_no_positives"] = bool(skip_no_positives)
    metrics["k_eff_enabled"] = True

    # Candidate count distribution
    if candidate_counts:
        import numpy as np
        arr = np.array(candidate_counts)
        metrics["n_candidates_mean"] = float(arr.mean())
        metrics["n_candidates_median"] = float(np.median(arr))
        metrics["n_candidates_p50"] = float(np.percentile(arr, 50))
        metrics["n_candidates_p90"] = float(np.percentile(arr, 90))
        metrics["n_candidates_p99"] = float(np.percentile(arr, 99))

    return metrics


def paper_evaluate(
    results: Sequence[Dict],
    ks: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Paper-standard evaluation with K_eff, dual mode, and ceiling.

    This is the recommended evaluation function for paper reporting.

    Args:
        results: List of dicts with keys: post_id, criterion_id, gold_ids, ranked_ids.
        ks: K values to compute. Defaults to [1,3,5,10,20].

    Returns:
        Dict with:
        - "positives_only": Metrics on queries with gold positives (K_eff applied)
        - "all_queries": Metrics on all queries (K_eff applied)
        - "candidate_distribution": Stats on n_candidates per query
    """
    if ks is None:
        ks = get_paper_k_values()

    pos_metrics = evaluate_with_k_eff(results, ks, skip_no_positives=True, include_ceiling=True)
    all_metrics = evaluate_with_k_eff(results, ks, skip_no_positives=False, include_ceiling=True)

    return {
        "positives_only": pos_metrics,
        "all_queries": all_metrics,
        "candidate_distribution": {
            "mean": pos_metrics.get("n_candidates_mean", 0),
            "median": pos_metrics.get("n_candidates_median", 0),
            "p50": pos_metrics.get("n_candidates_p50", 0),
            "p90": pos_metrics.get("n_candidates_p90", 0),
            "p99": pos_metrics.get("n_candidates_p99", 0),
        },
    }
