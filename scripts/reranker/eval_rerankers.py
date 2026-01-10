#!/usr/bin/env python3
"""Evaluate rerankers on candidate sets (R0: off-the-shelf inference).

Implements Section 8.R0 from the research plan:
- Sweep inference configs: top_k_rerank, max_length, instruction templates
- Compute nDCG@10, MRR@10, Recall@K, false-evidence rate

Note: Uses pickle for loading cached candidate sets (internal use only).

Usage:
    python scripts/reranker/eval_rerankers.py --candidates candidates_dev_select_k100.pkl
"""

import argparse
import json
import pickle  # Internal cache loading only
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.data.io import load_sentence_corpus
from final_sc_review.reranker.zoo import RerankerZoo, RerankerResult
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ndcg(relevances: List[int], k: int) -> float:
    """Compute nDCG@k."""
    relevances = relevances[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Compute Recall@k."""
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & gold)
    return hits / len(gold) if gold else 0.0


def compute_mrr(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Compute MRR@k."""
    for i, uid in enumerate(retrieved[:k]):
        if uid in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_reranker(
    reranker_name: str,
    zoo: RerankerZoo,
    candidates_data: Dict,
    generator: str,
    top_k_rerank: int,
    sent_uid_to_text: Dict[str, str],
    k_values: List[int] = [1, 5, 10],
    threshold: float = 0.0,
) -> Dict:
    """Evaluate a single reranker on candidate sets.

    Returns:
        Dict with metrics for has-evidence and no-evidence queries
    """
    reranker = zoo.get_reranker(reranker_name)
    reranker.load_model()

    all_candidates = candidates_data["candidates"]

    # Metrics storage
    metrics_has_evidence = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
    metrics_no_evidence = {"false_evidence_rate": []}

    for query_key, data in tqdm(all_candidates.items(), desc=f"Evaluating {reranker_name}"):
        query = data["query"]
        cands = data["candidates"].get(generator, [])

        if not cands:
            continue

        # Limit candidates to rerank
        cands_to_rerank = cands[:top_k_rerank]

        # Build candidate tuples with actual text
        candidate_tuples = []
        for uid, _ in cands_to_rerank:
            text = sent_uid_to_text.get(uid, "")
            candidate_tuples.append((uid, text))

        # Rerank
        reranked = reranker.rerank(
            query=query["criterion_text"],
            candidates=candidate_tuples,
            top_k=max(k_values),
        )

        retrieved_uids = [r.sent_uid for r in reranked]
        gold_uids = query["gold_uids"]

        if query["has_evidence"]:
            # Compute ranking metrics
            for k in k_values:
                relevances = [1 if uid in gold_uids else 0 for uid in retrieved_uids[:k]]
                metrics_has_evidence[k]["recall"].append(compute_recall(retrieved_uids, gold_uids, k))
                metrics_has_evidence[k]["mrr"].append(compute_mrr(retrieved_uids, gold_uids, k))
                metrics_has_evidence[k]["ndcg"].append(compute_ndcg(relevances, k))
        else:
            # No-evidence query: check if any returned above threshold
            top_scores = [r.score for r in reranked[:10]]
            false_positive = any(s > threshold for s in top_scores)
            metrics_no_evidence["false_evidence_rate"].append(1.0 if false_positive else 0.0)

    # Aggregate results
    results = {
        "reranker": reranker_name,
        "generator": generator,
        "top_k_rerank": top_k_rerank,
        "has_evidence": {
            "n_queries": len(metrics_has_evidence[k_values[0]]["recall"]) if k_values else 0,
        },
        "no_evidence": {
            "n_queries": len(metrics_no_evidence["false_evidence_rate"]),
            "false_evidence_rate": float(np.mean(metrics_no_evidence["false_evidence_rate"]))
            if metrics_no_evidence["false_evidence_rate"] else 0.0,
        },
    }

    for k in k_values:
        if metrics_has_evidence[k]["recall"]:
            results["has_evidence"][f"recall@{k}"] = float(np.mean(metrics_has_evidence[k]["recall"]))
            results["has_evidence"][f"mrr@{k}"] = float(np.mean(metrics_has_evidence[k]["mrr"]))
            results["has_evidence"][f"ndcg@{k}"] = float(np.mean(metrics_has_evidence[k]["ndcg"]))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate rerankers on candidate sets")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Path to candidates pickle file")
    parser.add_argument("--rerankers", type=str, nargs="+",
                        default=None, help="Rerankers to evaluate (None = all)")
    parser.add_argument("--generator", type=str, default="fusion-rrf",
                        help="Candidate generator to use")
    parser.add_argument("--top_k_rerank", type=int, nargs="+", default=[20, 50, 100],
                        help="Top-k candidates to rerank")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/reranker_research/leaderboards"))
    args = parser.parse_args()

    print("=" * 80)
    print("RERANKER EVALUATION (R0: Off-the-shelf)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Candidates: {args.candidates}")
    print(f"Generator: {args.generator}")
    print(f"Top-K rerank values: {args.top_k_rerank}")
    print("=" * 80)

    # Load sentence corpus for text lookup
    print("\n[1/4] Loading sentence corpus...")
    sentences = load_sentence_corpus(Path("data/groundtruth/sentence_corpus.jsonl"))
    sent_uid_to_text = {s.sent_uid: s.text for s in sentences}
    print(f"  Loaded {len(sent_uid_to_text)} sentences")

    # Load candidates (internal pickle cache)
    print("\n[2/4] Loading candidates...")
    with open(args.candidates, "rb") as f:
        candidates_data = pickle.load(f)
    print(f"  Loaded {candidates_data['n_queries']} queries")
    print(f"  Available generators: {candidates_data['generators']}")

    # Initialize reranker zoo
    print("\n[3/4] Initializing reranker zoo...")
    zoo = RerankerZoo(device="cuda")
    available_rerankers = zoo.list_rerankers()
    print(f"  Available rerankers: {available_rerankers}")

    rerankers_to_eval = args.rerankers or available_rerankers

    # Evaluate
    print("\n[4/4] Evaluating rerankers...")
    all_results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for reranker_name in rerankers_to_eval:
        print(f"\n  === {reranker_name.upper()} ===")

        for top_k in args.top_k_rerank:
            try:
                results = evaluate_reranker(
                    reranker_name=reranker_name,
                    zoo=zoo,
                    candidates_data=candidates_data,
                    generator=args.generator,
                    top_k_rerank=top_k,
                    sent_uid_to_text=sent_uid_to_text,
                )
                all_results.append(results)

                # Print summary
                he = results["has_evidence"]
                ne = results["no_evidence"]
                print(f"    K={top_k}: nDCG@10={he.get('ndcg@10', 0):.4f}, "
                      f"MRR@10={he.get('mrr@10', 0):.4f}, "
                      f"FalseEvidence={ne['false_evidence_rate']:.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {reranker_name} with K={top_k}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    output_path = args.output_dir / f"reranker_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "candidates_file": str(args.candidates),
            "generator": args.generator,
            "results": all_results,
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Print leaderboard
    print("\n" + "=" * 80)
    print("LEADERBOARD (sorted by nDCG@10)")
    print("=" * 80)
    print(f"{'Reranker':<30} {'K':>5} {'nDCG@10':>10} {'MRR@10':>10} {'FalseEvid':>10}")
    print("-" * 80)

    sorted_results = sorted(all_results, key=lambda x: -x["has_evidence"].get("ndcg@10", 0))
    for r in sorted_results:
        he = r["has_evidence"]
        ne = r["no_evidence"]
        print(f"{r['reranker']:<30} {r['top_k_rerank']:>5} "
              f"{he.get('ndcg@10', 0):>10.4f} {he.get('mrr@10', 0):>10.4f} "
              f"{ne['false_evidence_rate']:>10.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
