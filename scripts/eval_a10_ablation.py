#!/usr/bin/env python3
"""
Evaluate pipeline performance with and without A.10 criterion.

Compares metrics when A.10 is included vs excluded to understand
its impact on overall performance.

Usage:
    python scripts/eval_a10_ablation.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def compute_metrics(ranked_uids, gold_uids, k):
    """Compute ranking metrics at K."""
    gold_set = set(gold_uids)
    ranked_k = ranked_uids[:k]

    # Recall@K
    hits = sum(1 for uid in ranked_k if uid in gold_set)
    recall = hits / len(gold_uids) if gold_uids else 0.0

    # MRR@K
    mrr = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@K
    dcg = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_uids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def evaluate_subset(retriever, queries, criterion_text, k_values, subset_name):
    """Evaluate retriever on a subset of queries."""
    results_by_k = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    n_queries = 0
    n_positive = 0

    for post_id, criterion_id, gold_uids, query_text in queries:
        n_queries += 1
        if not gold_uids:
            continue

        n_positive += 1

        results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k_retriever=50,
        )

        ranked_uids = [r[0] for r in results]

        for k in k_values:
            metrics = compute_metrics(ranked_uids, gold_uids, k)
            for metric_name, value in metrics.items():
                results_by_k[k][metric_name].append(value)

    # Aggregate
    summary = {
        "subset": subset_name,
        "n_queries_total": n_queries,
        "n_queries_positive": n_positive,
    }

    for k in k_values:
        if results_by_k[k]["recall"]:
            summary[f"recall@{k}"] = float(np.mean(results_by_k[k]["recall"]))
            summary[f"mrr@{k}"] = float(np.mean(results_by_k[k]["mrr"]))
            summary[f"ndcg@{k}"] = float(np.mean(results_by_k[k]["ndcg"]))
        else:
            summary[f"recall@{k}"] = 0.0
            summary[f"mrr@{k}"] = 0.0
            summary[f"ndcg@{k}"] = 0.0

    return summary


def main():
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth, load_criteria
    from final_sc_review.data.splits import split_post_ids
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("="*70)
    print("A.10 ABLATION STUDY")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    # Load data
    print("\n[1/5] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    print(f"  Corpus: {len(sentences)} sentences")
    print(f"  Groundtruth: {len(gt_rows)} rows")
    print(f"  Criteria: {list(criterion_text.keys())}")

    # Build queries grouped by criterion
    print("\n[2/5] Building queries...")
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}

    # Count by criterion
    criterion_counts = defaultdict(lambda: {"total": 0, "positive": 0})
    for row in gt_rows:
        criterion_counts[row.criterion_id]["total"] += 1
        if row.groundtruth == 1:
            criterion_counts[row.criterion_id]["positive"] += 1

    print("\n  Criterion distribution:")
    for cid in sorted(criterion_counts.keys()):
        stats = criterion_counts[cid]
        print(f"    {cid}: {stats['total']} total, {stats['positive']} positive")

    # Split posts
    print("\n[3/5] Splitting posts...")
    splits = split_post_ids(list(all_posts), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    # Use val (dev_select) for this evaluation
    eval_posts = set(splits["val"])
    print(f"  Evaluation posts (dev_select): {len(eval_posts)}")

    # Build query lists
    queries_all = []
    queries_no_a10 = []
    queries_a10_only = []

    for (post_id, criterion_id), gold_uids in query_gold.items():
        if post_id not in eval_posts:
            continue

        query_text = criterion_text.get(criterion_id, criterion_id)
        query_tuple = (post_id, criterion_id, list(gold_uids), query_text)

        queries_all.append(query_tuple)

        if criterion_id == "A.10":
            queries_a10_only.append(query_tuple)
        else:
            queries_no_a10.append(query_tuple)

    print(f"\n  Query counts:")
    print(f"    All criteria: {len(queries_all)} queries with gold")
    print(f"    Excluding A.10: {len(queries_no_a10)} queries with gold")
    print(f"    A.10 only: {len(queries_a10_only)} queries with gold")

    # Initialize retriever
    print("\n[4/5] Initializing retriever...")
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        rebuild_cache=False,
    )

    # Evaluate
    print("\n[5/5] Evaluating...")
    k_values = [1, 5, 10, 20]

    print("\n  Evaluating ALL criteria...")
    results_all = evaluate_subset(retriever, queries_all, criterion_text, k_values, "all_criteria")

    print("  Evaluating WITHOUT A.10...")
    results_no_a10 = evaluate_subset(retriever, queries_no_a10, criterion_text, k_values, "excluding_A10")

    print("  Evaluating A.10 ONLY...")
    results_a10_only = evaluate_subset(retriever, queries_a10_only, criterion_text, k_values, "A10_only")

    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    print("\n### Retriever Performance (dev_select split)")
    print("\n| Metric | All Criteria | Excluding A.10 | A.10 Only | Delta (excl vs all) |")
    print("|--------|--------------|----------------|-----------|---------------------|")

    for k in k_values:
        for metric in ["recall", "mrr", "ndcg"]:
            key = f"{metric}@{k}"
            val_all = results_all.get(key, 0)
            val_no_a10 = results_no_a10.get(key, 0)
            val_a10 = results_a10_only.get(key, 0)
            delta = val_no_a10 - val_all
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            print(f"| {key.upper()} | {val_all:.3f} | {val_no_a10:.3f} | {val_a10:.3f} | {delta_str} |")

    print(f"\n### Query Counts")
    print(f"| Subset | Queries with Gold |")
    print(f"|--------|-------------------|")
    print(f"| All Criteria | {results_all['n_queries_positive']} |")
    print(f"| Excluding A.10 | {results_no_a10['n_queries_positive']} |")
    print(f"| A.10 Only | {results_a10_only['n_queries_positive']} |")

    # Save results
    output_dir = repo_root / "outputs" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "split": "dev_select",
        "k_values": k_values,
        "all_criteria": results_all,
        "excluding_A10": results_no_a10,
        "A10_only": results_a10_only,
        "criterion_distribution": dict(criterion_counts),
    }

    results_path = output_dir / "a10_ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {results_path}")

    # Also save markdown report
    report_path = output_dir / "a10_ablation_report.md"
    with open(report_path, "w") as f:
        f.write(f"# A.10 Ablation Study\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Split:** dev_select ({len(eval_posts)} posts)\n\n")

        f.write("## Summary\n\n")
        f.write("This study compares retriever performance with and without the A.10 criterion.\n\n")

        f.write("## Criterion Distribution\n\n")
        f.write("| Criterion | Total Rows | Positive Labels |\n")
        f.write("|-----------|------------|----------------|\n")
        for cid in sorted(criterion_counts.keys()):
            stats = criterion_counts[cid]
            f.write(f"| {cid} | {stats['total']} | {stats['positive']} |\n")

        f.write("\n## Performance Comparison\n\n")
        f.write("| Metric | All Criteria | Excluding A.10 | A.10 Only | Delta |\n")
        f.write("|--------|--------------|----------------|-----------|-------|\n")

        for k in k_values:
            for metric in ["recall", "mrr", "ndcg"]:
                key = f"{metric}@{k}"
                val_all = results_all.get(key, 0)
                val_no_a10 = results_no_a10.get(key, 0)
                val_a10 = results_a10_only.get(key, 0)
                delta = val_no_a10 - val_all
                delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
                f.write(f"| {key.upper()} | {val_all:.3f} | {val_no_a10:.3f} | {val_a10:.3f} | {delta_str} |\n")

        f.write("\n## Key Findings\n\n")

        # Calculate average impact
        ndcg10_all = results_all.get("ndcg@10", 0)
        ndcg10_no_a10 = results_no_a10.get("ndcg@10", 0)
        ndcg10_a10 = results_a10_only.get("ndcg@10", 0)

        if ndcg10_a10 < ndcg10_no_a10:
            f.write(f"- A.10 has **lower** performance (nDCG@10={ndcg10_a10:.3f}) compared to other criteria ({ndcg10_no_a10:.3f})\n")
            f.write(f"- Excluding A.10 **improves** overall metrics by {ndcg10_no_a10 - ndcg10_all:.3f} nDCG@10\n")
        else:
            f.write(f"- A.10 has **comparable or higher** performance (nDCG@10={ndcg10_a10:.3f}) vs other criteria ({ndcg10_no_a10:.3f})\n")

        f.write(f"\n## Query Counts\n\n")
        f.write(f"- All criteria: {results_all['n_queries_positive']} positive queries\n")
        f.write(f"- Excluding A.10: {results_no_a10['n_queries_positive']} positive queries\n")
        f.write(f"- A.10 only: {results_a10_only['n_queries_positive']} positive queries\n")

    print(f"Report saved to: {report_path}")

    print("\n" + "="*70)
    print("[SUCCESS] A.10 ablation study complete")
    print("="*70)


if __name__ == "__main__":
    main()
