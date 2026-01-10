#!/usr/bin/env python3
"""Comprehensive retriever comparison across all models in the zoo.

Evaluates all available retrievers on dev_select split and generates:
- Per-retriever metrics (nDCG@K, Recall@K, MRR@K)
- Comparison tables
- A.10 ablation for each retriever

Usage:
    python scripts/retriever/compare_all_retrievers.py [--retrievers bge-m3,splade-cocondenser]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

# Add src to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.zoo import RetrieverZoo, RetrieverConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(ranked_uids: List[str], gold_uids: Set[str], k: int) -> Dict[str, float]:
    """Compute ranking metrics at K."""
    ranked_k = ranked_uids[:k]
    gold_set = set(gold_uids)

    hits = sum(1 for uid in ranked_k if uid in gold_set)
    recall = hits / len(gold_uids) if gold_uids else 0.0

    mrr = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            mrr = 1.0 / (i + 1)
            break

    dcg = sum(1.0 / np.log2(i + 2) for i, uid in enumerate(ranked_k) if uid in gold_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_uids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def evaluate_retriever(
    retriever,
    queries: List[tuple],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Evaluate a retriever on given queries."""
    metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for post_id, criterion_id, gold_uids, query_text in queries:
        if not gold_uids:
            continue

        try:
            results = retriever.retrieve_within_post(
                query=query_text,
                post_id=post_id,
                top_k=max(k_values),
            )
            ranked_uids = [r.sent_uid for r in results]
        except Exception as e:
            logger.warning(f"Error retrieving for post {post_id}: {e}")
            ranked_uids = []

        for k in k_values:
            m = compute_metrics(ranked_uids, gold_uids, k)
            metrics[k]["recall"].append(m["recall"])
            metrics[k]["mrr"].append(m["mrr"])
            metrics[k]["ndcg"].append(m["ndcg"])

    summary = {"n_queries": len([q for q in queries if q[2]])}
    for k in k_values:
        for metric in ["recall", "mrr", "ndcg"]:
            summary[f"{metric}@{k}"] = float(np.mean(metrics[k][metric])) if metrics[k][metric] else 0.0

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare all retrievers")
    parser.add_argument(
        "--retrievers",
        type=str,
        default=None,
        help="Comma-separated list of retriever names to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev_select",
        choices=["dev_select", "test"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_root / "outputs" / "retriever_comparison"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE RETRIEVER COMPARISON")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Split: {args.split}")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache"

    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    print(f"  Loaded {len(sentences)} sentences")

    # Build queries
    print("\n[2/4] Building queries...")
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}
    splits = split_post_ids(list(all_posts), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    split_posts = set(splits["val"] if args.split == "dev_select" else splits["test"])

    queries_all = []
    queries_no_a10 = []

    for (post_id, crit_id), gold_uids in query_gold.items():
        if post_id not in split_posts:
            continue
        query_text = criterion_text.get(crit_id, crit_id)
        query_tuple = (post_id, crit_id, gold_uids, query_text)
        queries_all.append(query_tuple)
        if crit_id != "A.10":
            queries_no_a10.append(query_tuple)

    print(f"  All criteria: {len(queries_all)} queries")
    print(f"  Excluding A.10: {len(queries_no_a10)} queries")

    # Initialize retriever zoo
    print("\n[3/4] Initializing retriever zoo...")
    zoo = RetrieverZoo(sentences=sentences, cache_dir=cache_dir, device="cuda")

    retriever_names = args.retrievers.split(",") if args.retrievers else zoo.list_retrievers()
    print(f"  Retrievers to evaluate: {retriever_names}")

    # Evaluate each retriever
    print("\n[4/4] Evaluating retrievers...")
    all_results = {}
    k_values = [1, 5, 10]

    for name in retriever_names:
        print(f"\n  === {name.upper()} ===")
        try:
            retriever = zoo.get_retriever(name)
            print(f"    Encoding corpus...")
            retriever.encode_corpus(rebuild=False)
        except Exception as e:
            print(f"    [ERROR] Failed to initialize {name}: {e}")
            continue

        # Evaluate on all criteria
        print(f"    Evaluating (all criteria)...")
        results_all = evaluate_retriever(retriever, queries_all, k_values)

        # Evaluate excluding A.10
        print(f"    Evaluating (excluding A.10)...")
        results_no_a10 = evaluate_retriever(retriever, queries_no_a10, k_values)

        all_results[name] = {
            "all_criteria": results_all,
            "excluding_a10": results_no_a10,
            "delta_ndcg10": results_no_a10["ndcg@10"] - results_all["ndcg@10"],
        }

        # Print results
        print(f"    nDCG@10: {results_all['ndcg@10']:.4f} (all) / {results_no_a10['ndcg@10']:.4f} (excl A.10)")
        print(f"    Recall@10: {results_all['recall@10']:.4f} (all) / {results_no_a10['recall@10']:.4f} (excl A.10)")

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n### {args.split.upper()} Split - All Criteria")
    print("\n| Retriever | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | MRR@10 |")
    print("|-----------|--------|--------|---------|-----------|--------|")
    for name in sorted(all_results.keys(), key=lambda x: -all_results[x]["all_criteria"]["ndcg@10"]):
        r = all_results[name]["all_criteria"]
        print(f"| {name} | {r['ndcg@1']:.3f} | {r['ndcg@5']:.3f} | {r['ndcg@10']:.3f} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} |")

    print(f"\n### {args.split.upper()} Split - Excluding A.10")
    print("\n| Retriever | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | MRR@10 | Delta |")
    print("|-----------|--------|--------|---------|-----------|--------|-------|")
    for name in sorted(all_results.keys(), key=lambda x: -all_results[x]["excluding_a10"]["ndcg@10"]):
        r = all_results[name]["excluding_a10"]
        delta = all_results[name]["delta_ndcg10"]
        print(f"| {name} | {r['ndcg@1']:.3f} | {r['ndcg@5']:.3f} | {r['ndcg@10']:.3f} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} | {delta:+.3f} |")

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "k_values": k_values,
        "retrievers_evaluated": list(all_results.keys()),
        "results": all_results,
    }

    results_path = output_dir / f"comparison_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\n\nResults saved to: {results_path}")

    # Also save a latest version
    latest_path = output_dir / f"comparison_{args.split}_latest.json"
    latest_path.write_text(json.dumps(results_data, indent=2))

    print("\n" + "=" * 80)
    print("[SUCCESS] Retriever comparison complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
