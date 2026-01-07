#!/usr/bin/env python3
"""Diagnostic analysis script for S-C retrieval pipeline.

Computes:
A) Candidate ceiling analysis: % queries with gold in top-K
B) Failure taxonomy: per-criterion breakdown
C) Data quality: sentence corpus statistics
D) Runtime profiling estimates
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnostics")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ceiling_ks", type=str, default="10,20,50,100")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ceiling_ks = [int(k) for k in args.ceiling_ks.split(",")]

    # Load data
    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}

    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    sent_uid_to_sent = {s.sent_uid: s for s in sentences}

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    eval_posts = set(splits[args.split])

    # Group groundtruth by (post_id, criterion_id)
    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in eval_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    cache_dir = Path(cfg["paths"]["cache_dir"])
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        model_name=cfg["models"]["bge_m3"],
        device=cfg.get("device"),
        query_max_length=cfg["models"].get("bge_query_max_length", 128),
        passage_max_length=cfg["models"].get("bge_passage_max_length", 256),
        use_fp16=cfg["models"].get("bge_use_fp16", True),
        batch_size=cfg["models"].get("bge_batch_size", 64),
        rebuild_cache=False,
    )

    report = {
        "split": args.split,
        "total_queries": len(grouped),
        "ceiling_analysis": {},
        "failure_taxonomy": {},
        "data_quality": {},
        "per_criterion": {},
    }

    # A) Candidate ceiling analysis
    logger.info("Running ceiling analysis...")
    ceiling_stats = _run_ceiling_analysis(
        grouped, criteria_map, retriever, ceiling_ks, cfg["retriever"]
    )
    report["ceiling_analysis"] = ceiling_stats

    # B) Failure taxonomy
    logger.info("Computing failure taxonomy...")
    taxonomy = _compute_failure_taxonomy(grouped, criteria_map)
    report["failure_taxonomy"] = taxonomy

    # C) Data quality
    logger.info("Analyzing data quality...")
    data_quality = _analyze_data_quality(sentences, groundtruth, eval_posts)
    report["data_quality"] = data_quality

    # D) Per-criterion breakdown
    logger.info("Computing per-criterion metrics...")
    per_criterion = _per_criterion_analysis(grouped, criteria_map, retriever, cfg["retriever"])
    report["per_criterion"] = per_criterion

    # Save JSON report
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    _write_markdown_report(report, output_dir / "report.md")

    logger.info("Saved diagnostics to %s", output_dir)


def _run_ceiling_analysis(
    grouped: Dict[tuple, List],
    criteria_map: Dict[str, str],
    retriever: BgeM3Retriever,
    ceiling_ks: List[int],
    retriever_cfg: Dict,
) -> Dict:
    """Compute % of queries where gold appears in top-K retriever results."""
    max_k = max(ceiling_ks)
    gold_in_topk = {k: 0 for k in ceiling_ks}
    gold_best_ranks = []
    queries_with_positives = 0

    for (post_id, criterion_id), rows in sorted(grouped.items()):
        query = criteria_map.get(criterion_id)
        if query is None:
            continue

        gold_ids = set(r.sent_uid for r in rows if r.groundtruth == 1)
        if not gold_ids:
            continue

        queries_with_positives += 1

        # Get retriever results at max_k
        results = retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=max_k,
            top_k_colbert=max_k,
            dense_weight=retriever_cfg.get("dense_weight", 0.6),
            sparse_weight=retriever_cfg.get("sparse_weight", 0.2),
            colbert_weight=retriever_cfg.get("colbert_weight", 0.2),
            use_sparse=retriever_cfg.get("use_sparse", True),
            use_colbert=retriever_cfg.get("use_colbert", True),
            fusion_method=retriever_cfg.get("fusion_method", "weighted_sum"),
            score_normalization=retriever_cfg.get("score_normalization", "minmax_per_query"),
            rrf_k=retriever_cfg.get("rrf_k", 60),
        )

        retrieved_ids = [sid for sid, _, _ in results]

        # Check at which K gold first appears
        best_rank = None
        for rank, sid in enumerate(retrieved_ids, start=1):
            if sid in gold_ids:
                if best_rank is None or rank < best_rank:
                    best_rank = rank

        if best_rank is not None:
            gold_best_ranks.append(best_rank)
            for k in ceiling_ks:
                if best_rank <= k:
                    gold_in_topk[k] += 1

    # Compute percentages
    result = {
        "queries_with_positives": queries_with_positives,
        "ceiling_by_k": {},
        "best_rank_distribution": {},
    }

    for k in ceiling_ks:
        pct = gold_in_topk[k] / queries_with_positives if queries_with_positives > 0 else 0
        result["ceiling_by_k"][f"top_{k}"] = {
            "count": gold_in_topk[k],
            "percentage": round(pct * 100, 2),
        }

    # Rank distribution
    if gold_best_ranks:
        result["best_rank_distribution"] = {
            "min": min(gold_best_ranks),
            "max": max(gold_best_ranks),
            "mean": round(sum(gold_best_ranks) / len(gold_best_ranks), 2),
            "median": sorted(gold_best_ranks)[len(gold_best_ranks) // 2],
        }

    # Queries with gold never in top-K (hard ceiling)
    never_retrieved = queries_with_positives - gold_in_topk[max_k]
    result["never_in_top_k"] = {
        "count": never_retrieved,
        "percentage": round(never_retrieved / queries_with_positives * 100, 2) if queries_with_positives > 0 else 0,
        "k": max_k,
    }

    return result


def _compute_failure_taxonomy(grouped: Dict[tuple, List], criteria_map: Dict[str, str]) -> Dict:
    """Analyze query types: with positives vs no positives."""
    with_positives = 0
    no_positives = 0
    criterion_counts = defaultdict(lambda: {"with_pos": 0, "no_pos": 0})

    for (post_id, criterion_id), rows in grouped.items():
        gold_ids = [r.sent_uid for r in rows if r.groundtruth == 1]
        if gold_ids:
            with_positives += 1
            criterion_counts[criterion_id]["with_pos"] += 1
        else:
            no_positives += 1
            criterion_counts[criterion_id]["no_pos"] += 1

    return {
        "total_queries": with_positives + no_positives,
        "queries_with_positives": with_positives,
        "queries_no_positives": no_positives,
        "by_criterion": dict(criterion_counts),
    }


def _analyze_data_quality(
    sentences: List,
    groundtruth: List,
    eval_posts: Set[str],
) -> Dict:
    """Analyze sentence corpus quality."""
    total_sentences = len(sentences)
    empty_text = sum(1 for s in sentences if not s.text.strip())

    # Check for potential backfilled sentences (sid that seem high)
    post_sentence_counts = defaultdict(int)
    for s in sentences:
        post_sentence_counts[s.post_id] += 1

    # Sentence length distribution
    lengths = [len(s.text) for s in sentences]

    # Eval split sentence coverage
    eval_sentences = [s for s in sentences if s.post_id in eval_posts]

    # Groundtruth coverage
    gt_sent_uids = set(r.sent_uid for r in groundtruth if r.post_id in eval_posts)
    corpus_sent_uids = set(s.sent_uid for s in eval_sentences)
    missing_in_corpus = gt_sent_uids - corpus_sent_uids

    return {
        "total_sentences": total_sentences,
        "empty_text_count": empty_text,
        "empty_text_pct": round(empty_text / total_sentences * 100, 4) if total_sentences > 0 else 0,
        "sentence_length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "unique_posts": len(post_sentence_counts),
        "avg_sentences_per_post": round(total_sentences / len(post_sentence_counts), 2) if post_sentence_counts else 0,
        "eval_split_sentences": len(eval_sentences),
        "groundtruth_sent_uids_in_eval": len(gt_sent_uids),
        "missing_in_corpus": len(missing_in_corpus),
    }


def _per_criterion_analysis(
    grouped: Dict[tuple, List],
    criteria_map: Dict[str, str],
    retriever: BgeM3Retriever,
    retriever_cfg: Dict,
) -> Dict:
    """Compute per-criterion retrieval metrics."""
    from final_sc_review.metrics.retrieval_eval import evaluate_rankings

    criterion_results = defaultdict(list)

    for (post_id, criterion_id), rows in sorted(grouped.items()):
        query = criteria_map.get(criterion_id)
        if query is None:
            continue

        gold_ids = [r.sent_uid for r in rows if r.groundtruth == 1]
        if not gold_ids:
            continue

        results = retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=50,
            top_k_colbert=50,
            dense_weight=retriever_cfg.get("dense_weight", 0.6),
            sparse_weight=retriever_cfg.get("sparse_weight", 0.2),
            colbert_weight=retriever_cfg.get("colbert_weight", 0.2),
            use_sparse=retriever_cfg.get("use_sparse", True),
            use_colbert=retriever_cfg.get("use_colbert", True),
            fusion_method=retriever_cfg.get("fusion_method", "weighted_sum"),
            score_normalization=retriever_cfg.get("score_normalization", "minmax_per_query"),
            rrf_k=retriever_cfg.get("rrf_k", 60),
        )

        ranked_ids = [sid for sid, _, _ in results]
        criterion_results[criterion_id].append({
            "post_id": post_id,
            "gold_ids": gold_ids,
            "ranked_ids": ranked_ids,
        })

    # Compute metrics per criterion
    per_criterion = {}
    for criterion_id, results_list in criterion_results.items():
        metrics = evaluate_rankings(
            results_list,
            ks=[1, 5, 10, 20],
            skip_no_positives=True,
        )
        criterion_name = criteria_map.get(criterion_id, criterion_id)[:50]  # Truncate for display
        per_criterion[criterion_id] = {
            "name": criterion_name,
            "num_queries": len(results_list),
            "ndcg@10": round(metrics.get("ndcg@10", 0), 4),
            "recall@10": round(metrics.get("recall@10", 0), 4),
            "mrr@10": round(metrics.get("mrr@10", 0), 4),
        }

    # Sort by nDCG@10 to find best/worst
    sorted_criteria = sorted(per_criterion.items(), key=lambda x: x[1]["ndcg@10"], reverse=True)

    return {
        "by_criterion": per_criterion,
        "best_criterion": sorted_criteria[0] if sorted_criteria else None,
        "worst_criterion": sorted_criteria[-1] if sorted_criteria else None,
        "macro_avg_ndcg@10": round(
            sum(c["ndcg@10"] for c in per_criterion.values()) / len(per_criterion), 4
        ) if per_criterion else 0,
    }


def _write_markdown_report(report: Dict, output_path: Path) -> None:
    """Generate human-readable markdown report."""
    lines = [
        "# S-C Retrieval Diagnostic Report",
        "",
        f"**Split:** {report['split']}",
        f"**Total Queries:** {report['total_queries']}",
        "",
        "## A) Candidate Ceiling Analysis",
        "",
        "What percentage of queries have gold sentences in the retriever's top-K?",
        "",
        "| Top-K | Count | Percentage |",
        "|-------|-------|------------|",
    ]

    ceiling = report["ceiling_analysis"]
    for k, stats in sorted(ceiling.get("ceiling_by_k", {}).items()):
        lines.append(f"| {k} | {stats['count']} | {stats['percentage']}% |")

    if "never_in_top_k" in ceiling:
        never = ceiling["never_in_top_k"]
        lines.extend([
            "",
            f"**Hard ceiling:** {never['count']} queries ({never['percentage']}%) have no gold in top-{never['k']}",
            "",
        ])

    if "best_rank_distribution" in ceiling:
        dist = ceiling["best_rank_distribution"]
        lines.extend([
            "**Best gold rank distribution:**",
            f"- Min: {dist['min']}, Max: {dist['max']}, Mean: {dist['mean']}, Median: {dist['median']}",
            "",
        ])

    lines.extend([
        "## B) Failure Taxonomy",
        "",
    ])

    taxonomy = report["failure_taxonomy"]
    lines.extend([
        f"- Total queries: {taxonomy['total_queries']}",
        f"- Queries with positives: {taxonomy['queries_with_positives']}",
        f"- Queries with no positives: {taxonomy['queries_no_positives']}",
        "",
    ])

    lines.extend([
        "## C) Data Quality",
        "",
    ])

    quality = report["data_quality"]
    lines.extend([
        f"- Total sentences in corpus: {quality['total_sentences']}",
        f"- Empty text: {quality['empty_text_count']} ({quality['empty_text_pct']}%)",
        f"- Average sentences per post: {quality['avg_sentences_per_post']}",
        f"- Sentence length: min={quality['sentence_length_stats']['min']}, "
        f"max={quality['sentence_length_stats']['max']}, mean={quality['sentence_length_stats']['mean']}",
        f"- Missing groundtruth sents in corpus: {quality['missing_in_corpus']}",
        "",
    ])

    lines.extend([
        "## D) Per-Criterion Performance",
        "",
        "| Criterion | #Queries | nDCG@10 | Recall@10 | MRR@10 |",
        "|-----------|----------|---------|-----------|--------|",
    ])

    per_crit = report["per_criterion"]
    for crit_id, stats in sorted(per_crit.get("by_criterion", {}).items(), key=lambda x: x[1]["ndcg@10"], reverse=True):
        lines.append(
            f"| {crit_id} | {stats['num_queries']} | {stats['ndcg@10']} | {stats['recall@10']} | {stats['mrr@10']} |"
        )

    if per_crit.get("macro_avg_ndcg@10"):
        lines.extend([
            "",
            f"**Macro-average nDCG@10:** {per_crit['macro_avg_ndcg@10']}",
        ])

    if per_crit.get("best_criterion"):
        best = per_crit["best_criterion"]
        lines.append(f"**Best criterion:** {best[0]} (nDCG@10={best[1]['ndcg@10']})")

    if per_crit.get("worst_criterion"):
        worst = per_crit["worst_criterion"]
        lines.append(f"**Worst criterion:** {worst[0]} (nDCG@10={worst[1]['ndcg@10']})")

    lines.extend([
        "",
        "## E) Key Takeaways",
        "",
        "Based on this analysis:",
        "",
        "1. **Ceiling Analysis:** Check if gold sentences are being missed at retrieval stage",
        "2. **Per-Criterion Variance:** High variance suggests criterion-specific tuning may help",
        "3. **Data Quality:** Missing/empty sentences can cause silent failures",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
