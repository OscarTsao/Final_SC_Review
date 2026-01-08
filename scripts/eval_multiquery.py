#!/usr/bin/env python3
"""Evaluate multi-query retrieval using criterion paraphrases.

Tests whether using multiple paraphrases per criterion improves retrieval metrics.

Usage:
    python scripts/eval_multiquery.py --paraphrases outputs/maxout/multiquery/criteria_paraphrases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus, load_criteria
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.bge_m3 import BgeM3Retriever


def rrf_fusion(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    top_n: int = 100,
) -> List[Tuple[str, float]]:
    """Fuse multiple rankings using Reciprocal Rank Fusion (RRF)."""
    scores: Dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, (uid, _) in enumerate(ranking):
            scores[uid] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_items[:top_n]


def max_score_fusion(
    rankings: List[List[Tuple[str, float]]],
    top_n: int = 100,
) -> List[Tuple[str, float]]:
    """Fuse rankings by taking max score per candidate."""
    scores: Dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for uid, score in ranking:
            scores[uid] = max(scores[uid], score)

    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_items[:top_n]


def compute_metrics(
    retrieved_uids: List[str],
    gold_uids: Set[str],
    ks: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Compute retrieval metrics."""
    metrics = {}

    for k in ks:
        top_k = set(retrieved_uids[:k])

        # Recall
        hits = len(gold_uids & top_k)
        recall = hits / len(gold_uids) if gold_uids else 0.0
        metrics[f"recall@{k}"] = recall

        # MRR
        mrr = 0.0
        for i, uid in enumerate(retrieved_uids[:k]):
            if uid in gold_uids:
                mrr = 1.0 / (i + 1)
                break
        metrics[f"mrr@{k}"] = mrr

        # nDCG
        dcg = 0.0
        for i, uid in enumerate(retrieved_uids[:k]):
            if uid in gold_uids:
                dcg += 1.0 / np.log2(i + 2)
        ideal_hits = min(len(gold_uids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-query retrieval")
    parser.add_argument("--paraphrases", default="outputs/maxout/multiquery/criteria_paraphrases.json")
    parser.add_argument("--output", default="outputs/maxout/multiquery/eval_results.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n_paraphrases", type=int, default=8, help="Number of paraphrases to use")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k to retrieve per query")
    parser.add_argument("--fusion_method", choices=["rrf", "max_score", "both"], default="both")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load paraphrases
    with open(args.paraphrases) as f:
        paraphrase_data = json.load(f)
    criterion_paraphrases = paraphrase_data["criteria_paraphrases"]

    # Load data
    data_dir = Path("data")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")
    gt_rows = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Get split
    all_post_ids = list({s.post_id for s in sentences})
    splits = split_post_ids(all_post_ids, seed=42)
    eval_post_ids = set(splits[args.split])

    print(f"Split: {args.split} ({len(eval_post_ids)} posts)")

    # Build query groups
    query_groups = defaultdict(lambda: {"gold_uids": set(), "post_id": None, "criterion_id": None})
    for row in gt_rows:
        if row.post_id not in eval_post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        query_groups[key]["post_id"] = row.post_id
        query_groups[key]["criterion_id"] = row.criterion_id
        if row.groundtruth == 1:
            query_groups[key]["gold_uids"].add(row.sent_uid)

    queries = [
        {"post_id": v["post_id"], "criterion_id": v["criterion_id"], "gold_uids": v["gold_uids"]}
        for v in query_groups.values()
        if v["gold_uids"]  # Only queries with positives
    ]
    print(f"Queries with positives: {len(queries)}")

    # Initialize retriever
    cache_dir = data_dir / "cache" / "bge_m3"
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        model_name="BAAI/bge-m3",
        rebuild_cache=False,
    )

    # Evaluate different settings
    results = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "n_queries": len(queries),
        "settings": {},
    }

    settings_to_test = [
        {"name": "single_query", "n_paraphrases": 1},
        {"name": f"multi_query_{args.n_paraphrases}", "n_paraphrases": args.n_paraphrases},
    ]

    fusion_methods = ["rrf", "max_score"] if args.fusion_method == "both" else [args.fusion_method]

    for setting in settings_to_test:
        for fusion in fusion_methods:
            if setting["n_paraphrases"] == 1 and fusion != "rrf":
                continue  # Single query doesn't need fusion variants

            setting_name = f"{setting['name']}_{fusion}" if setting["n_paraphrases"] > 1 else setting["name"]
            print(f"\n{'='*60}")
            print(f"Evaluating: {setting_name}")
            print(f"{'='*60}")

            all_metrics = defaultdict(list)

            for i, q in enumerate(queries):
                crit_id = q["criterion_id"]
                post_id = q["post_id"]
                gold_uids = q["gold_uids"]

                # Get paraphrases
                if crit_id in criterion_paraphrases:
                    paras = criterion_paraphrases[crit_id]["paraphrases"][:setting["n_paraphrases"]]
                else:
                    paras = [criterion_text.get(crit_id, crit_id)]

                # Retrieve for each paraphrase
                rankings = []
                for para in paras:
                    candidates = retriever.retrieve_within_post(
                        query=para,
                        post_id=post_id,
                        top_k_retriever=args.top_k,
                        top_k_colbert=args.top_k,
                        use_sparse=True,
                        use_colbert=True,
                        fusion_method="rrf",
                    )
                    # Convert to (uid, score) tuples
                    ranking = [(uid, score) for uid, _, score in candidates]
                    rankings.append(ranking)

                # Fuse rankings
                if len(rankings) == 1:
                    fused = rankings[0]
                elif fusion == "rrf":
                    fused = rrf_fusion(rankings, top_n=args.top_k)
                else:
                    fused = max_score_fusion(rankings, top_n=args.top_k)

                # Compute metrics
                retrieved_uids = [uid for uid, _ in fused]
                metrics = compute_metrics(retrieved_uids, gold_uids)

                for k, v in metrics.items():
                    all_metrics[k].append(v)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(queries)} queries")

            # Aggregate metrics
            agg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
            results["settings"][setting_name] = {
                "n_paraphrases": setting["n_paraphrases"],
                "fusion_method": fusion if setting["n_paraphrases"] > 1 else "none",
                "metrics": agg_metrics,
            }

            print(f"  nDCG@10: {agg_metrics['ndcg@10']:.4f}")
            print(f"  Recall@20: {agg_metrics['recall@20']:.4f}")
            print(f"  MRR@10: {agg_metrics['mrr@10']:.4f}")

    # Compare single vs multi-query
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    baseline = results["settings"].get("single_query", {}).get("metrics", {})
    print(f"\n{'Setting':<30} {'nDCG@10':<10} {'Δ':<8} {'Recall@20':<12} {'Δ':<8}")
    print("-" * 70)

    for name, data in results["settings"].items():
        m = data["metrics"]
        ndcg_delta = m["ndcg@10"] - baseline.get("ndcg@10", m["ndcg@10"])
        recall_delta = m["recall@20"] - baseline.get("recall@20", m["recall@20"])
        print(f"{name:<30} {m['ndcg@10']:<10.4f} {ndcg_delta:+.4f}  {m['recall@20']:<12.4f} {recall_delta:+.4f}")

    # Decision gate D3
    best_multi = None
    for name, data in results["settings"].items():
        if "multi_query" in name:
            if best_multi is None or data["metrics"]["ndcg@10"] > best_multi["metrics"]["ndcg@10"]:
                best_multi = data

    if best_multi and baseline:
        improvement = best_multi["metrics"]["ndcg@10"] - baseline["ndcg@10"]
        results["decision_gate_D3"] = {
            "improvement": improvement,
            "threshold": 0.005,
            "recommendation": "KEEP" if improvement > 0.005 else "SKIP",
        }
        print(f"\nDecision Gate D3: Multi-query improvement = {improvement:+.4f}")
        print(f"  Threshold: 0.005")
        print(f"  Recommendation: {results['decision_gate_D3']['recommendation']}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
