#!/usr/bin/env python3
"""Evaluate retriever zoo - compute oracle recalls and ranking metrics for all retrievers.

Usage:
    python scripts/eval_retriever_zoo.py --config configs/retriever_zoo.yaml --output outputs/maxout/retriever_zoo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus, load_criteria
from final_sc_review.data.splits import split_post_ids
from final_sc_review.data.schemas import Sentence
from final_sc_review.retriever.zoo import RetrieverZoo, RetrieverConfig
from final_sc_review.metrics.k_policy import get_paper_k_values, compute_k_eff, K_PRIMARY


def build_queries_from_groundtruth(
    groundtruth_path: Path,
    criteria_path: Path,
    split_post_ids: Set[str],
) -> List[Dict]:
    """Build query list with gold labels for oracle recall computation."""
    # Load groundtruth
    gt_rows = load_groundtruth(groundtruth_path)

    # Load criteria
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Group by (post_id, criterion_id)
    query_groups = defaultdict(lambda: {"gold_uids": set(), "post_id": None, "criterion_id": None})

    for row in gt_rows:
        if row.post_id not in split_post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        query_groups[key]["post_id"] = row.post_id
        query_groups[key]["criterion_id"] = row.criterion_id
        if row.groundtruth == 1:
            query_groups[key]["gold_uids"].add(row.sent_uid)

    # Build query list
    queries = []
    for key, data in query_groups.items():
        crit_id = data["criterion_id"]
        query_text = criterion_text.get(crit_id, crit_id)
        queries.append({
            "query": query_text,
            "post_id": data["post_id"],
            "criterion_id": crit_id,
            "gold_uids": data["gold_uids"],
        })

    return queries


def compute_ranking_metrics(
    queries: List[Dict],
    retriever,
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Compute ranking metrics (nDCG, MRR, Recall) for positives-only queries."""
    metrics = {f"ndcg@{k}": [] for k in k_values}
    metrics.update({f"mrr@{k}": [] for k in k_values})
    metrics.update({f"recall@{k}": [] for k in k_values})

    for q in queries:
        if not q.get("gold_uids"):
            continue  # Skip empty queries for positives-only metrics

        candidates = retriever.retrieve_within_post(
            query=q["query"],
            post_id=q["post_id"],
            top_k=max(k_values),
        )

        retrieved_uids = [r.sent_uid for r in candidates]
        gold_uids = set(q["gold_uids"])

        for k in k_values:
            top_k = retrieved_uids[:k]

            # Recall@k
            hits = len(gold_uids & set(top_k))
            recall = hits / len(gold_uids) if gold_uids else 0.0
            metrics[f"recall@{k}"].append(recall)

            # MRR@k
            mrr = 0.0
            for i, uid in enumerate(top_k):
                if uid in gold_uids:
                    mrr = 1.0 / (i + 1)
                    break
            metrics[f"mrr@{k}"].append(mrr)

            # nDCG@k
            dcg = 0.0
            for i, uid in enumerate(top_k):
                if uid in gold_uids:
                    dcg += 1.0 / np.log2(i + 2)

            # Ideal DCG
            ideal_hits = min(len(gold_uids), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"ndcg@{k}"].append(ndcg)

    # Aggregate
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever zoo")
    parser.add_argument("--config", default="configs/retriever_zoo.yaml", help="Zoo config path")
    parser.add_argument("--output", default="outputs/maxout/retriever_zoo", help="Output directory")
    parser.add_argument("--split", default="val", help="Evaluation split")
    parser.add_argument("--retrievers", nargs="*", help="Specific retrievers to evaluate (default: all)")
    parser.add_argument("--skip_slow", action="store_true", help="Skip slow/large models")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Paths
    data_dir = Path("data")
    groundtruth_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"
    cache_dir = data_dir / "cache"

    print("Loading data...")
    sentences = load_sentence_corpus(corpus_path)

    # Get split
    all_post_ids = list({s.post_id for s in sentences})
    splits = split_post_ids(all_post_ids, seed=42)
    eval_post_ids = set(splits[args.split])

    print(f"Split: {args.split} ({len(eval_post_ids)} posts)")

    # Build queries
    queries = build_queries_from_groundtruth(groundtruth_path, criteria_path, eval_post_ids)
    queries_with_positives = [q for q in queries if q.get("gold_uids")]

    print(f"Total queries: {len(queries)}")
    print(f"Queries with positives: {len(queries_with_positives)}")

    # Initialize zoo
    print("\nInitializing retriever zoo...")
    zoo = RetrieverZoo(sentences=sentences, cache_dir=cache_dir)

    # Determine which retrievers to evaluate
    if args.retrievers:
        retriever_names = args.retrievers
    else:
        retriever_names = zoo.list_retrievers()

    if args.skip_slow:
        # Skip large models
        retriever_names = [n for n in retriever_names if n not in ["stella-en-1.5B-v5", "gte-large-en-v1.5"]]

    print(f"Evaluating retrievers: {retriever_names}")

    # Evaluate each retriever
    results = {}
    # Use paper-standard K values for oracle recall (not misleading high K like 200)
    oracle_recall_ks = config.get("evaluation", {}).get("oracle_recall_ks", get_paper_k_values() + [50, 100])
    # Use paper-standard K values for ranking metrics
    ranking_ks = config.get("evaluation", {}).get("ranking_ks", get_paper_k_values())

    for name in retriever_names:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        try:
            retriever = zoo.get_retriever(name)
            print("Encoding corpus...")
            retriever.encode_corpus()
        except Exception as e:
            print(f"ERROR: Failed to initialize {name}: {e}")
            results[name] = {"error": str(e)}
            continue

        # Compute oracle recall
        print("Computing oracle recall...")
        oracle_recalls = {}
        for k in oracle_recall_ks:
            recalls = []
            for q in queries_with_positives:
                candidates = retriever.retrieve_within_post(
                    query=q["query"],
                    post_id=q["post_id"],
                    top_k=k,
                )
                retrieved_uids = set(r.sent_uid for r in candidates)
                gold_uids = q["gold_uids"]
                hits = len(gold_uids & retrieved_uids)
                recall = hits / len(gold_uids) if gold_uids else 0.0
                recalls.append(recall)
            oracle_recalls[f"oracle_recall@{k}"] = float(np.mean(recalls))

        # Compute ranking metrics
        print("Computing ranking metrics...")
        ranking_metrics = compute_ranking_metrics(queries_with_positives, retriever, ranking_ks)

        results[name] = {
            "oracle_recalls": oracle_recalls,
            "ranking_metrics": ranking_metrics,
            "queries_evaluated": len(queries_with_positives),
        }

        # Print summary using paper-standard K=20 (not misleading K=200)
        print(f"Oracle Recall@20: {oracle_recalls.get('oracle_recall@20', 0):.4f}")
        print(f"nDCG@10: {ranking_metrics.get('ndcg@10', 0):.4f}")
        print(f"MRR@10: {ranking_metrics.get('mrr@10', 0):.4f}")

    # Save results
    results_path = output_dir / "retriever_zoo_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "split": args.split,
            "n_queries": len(queries),
            "n_queries_with_positives": len(queries_with_positives),
            "retrievers": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    # Print leaderboard
    # Use paper-standard K=20 for decision gate (not misleading K=200)
    print(f"\n{'Retriever':<25} {'Oracle@20':<12} {'nDCG@10':<10} {'MRR@10':<10}")
    print("-" * 60)

    leaderboard = []
    for name, res in results.items():
        if "error" in res:
            print(f"{name:<25} ERROR: {res['error'][:30]}")
        else:
            oracle = res["oracle_recalls"].get("oracle_recall@20", 0)
            ndcg = res["ranking_metrics"].get("ndcg@10", 0)
            mrr = res["ranking_metrics"].get("mrr@10", 0)
            print(f"{name:<25} {oracle:<12.4f} {ndcg:<10.4f} {mrr:<10.4f}")
            leaderboard.append((name, oracle, ndcg, mrr))

    # Sort by oracle recall
    leaderboard.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 3 by Oracle Recall@20:")
    for i, (name, oracle, ndcg, mrr) in enumerate(leaderboard[:3], 1):
        print(f"  {i}. {name}: {oracle:.4f}")

    # Generate report
    report_path = output_dir / "retrieval_zoo_report.md"
    with open(report_path, "w") as f:
        f.write("# Retriever Zoo Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n")
        f.write(f"**Split**: {args.split}\n")
        f.write(f"**Queries with positives**: {len(queries_with_positives)}\n\n")

        f.write("## Leaderboard (Paper-Standard K Values)\n\n")
        f.write("| Retriever | Oracle@20 | nDCG@10 | MRR@10 |\n")
        f.write("|-----------|-----------|---------|--------|\n")
        for name, oracle, ndcg, mrr in leaderboard:
            f.write(f"| {name} | {oracle:.4f} | {ndcg:.4f} | {mrr:.4f} |\n")

        f.write("\n## Decision Gate D2 (Using Paper-Standard K=20)\n\n")
        best_oracle = leaderboard[0][1] if leaderboard else 0
        # Threshold for Oracle@20 is lower than Oracle@200 since K is smaller
        # With median=7 sentences per post, Oracle@20 threshold should be ~0.85
        oracle_threshold = 0.85
        if best_oracle >= oracle_threshold:
            f.write(f"Best Oracle@20 = {best_oracle:.4f} >= {oracle_threshold}\n")
            f.write("**Decision**: Retriever finetuning is OPTIONAL\n")
        else:
            f.write(f"Best Oracle@20 = {best_oracle:.4f} < {oracle_threshold}\n")
            f.write("**Decision**: Retriever finetuning is REQUIRED\n")

        f.write("\n## Top 3 Retrievers (Promoted)\n\n")
        for i, (name, oracle, ndcg, mrr) in enumerate(leaderboard[:3], 1):
            f.write(f"{i}. **{name}**: Oracle@20={oracle:.4f}, nDCG@10={ndcg:.4f}\n")

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
