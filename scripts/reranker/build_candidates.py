#!/usr/bin/env python3
"""Build and cache candidate sets for reranker research.

Implements candidate generators from the research plan:
- G1: Dense-best (qwen3-embed-0.6b)
- G2: Sparse-best (splade-cocondenser)
- G4: Fusion-RRF (dense + sparse)

Note: Uses pickle for internal caching (same pattern as bge_m3.py).
These caches are only written/read by this system.

Usage:
    python scripts/reranker/build_candidates.py --top_k 100 200
"""

import argparse
import json
import pickle  # Internal caching only, same pattern as existing codebase
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.schemas import Sentence
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.zoo import RetrieverZoo, RetrievalResult
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def rrf_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k0: int = 60,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple ranked lists.

    RRF(d) = sum_i 1/(k0 + rank_i(d))

    Args:
        ranked_lists: List of [(sent_uid, score), ...] per retriever
        k0: RRF constant (default 60 per Cormack et al.)

    Returns:
        Fused ranked list [(sent_uid, rrf_score), ...]
    """
    scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (sent_uid, _) in enumerate(ranked_list):
            scores[sent_uid] += 1.0 / (k0 + rank + 1)

    # Sort by RRF score descending
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused


def build_candidates_for_query(
    zoo: RetrieverZoo,
    query: str,
    post_id: str,
    top_k: int,
    generators: List[str],
    rrf_k0: int = 60,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build candidate sets for a single query using multiple generators.

    Returns:
        Dict mapping generator name to [(sent_uid, score), ...]
    """
    candidates = {}
    ranked_lists_for_fusion = []

    for gen_name in generators:
        if gen_name.startswith("fusion"):
            continue  # Handle fusion after individual retrievers

        try:
            retriever = zoo.get_retriever(gen_name)
            results = retriever.retrieve_within_post(query, post_id, top_k=top_k)
            ranked_list = [(r.sent_uid, r.score) for r in results]
            candidates[gen_name] = ranked_list
            ranked_lists_for_fusion.append(ranked_list)
        except Exception as e:
            logger.warning(f"Failed to get candidates from {gen_name}: {e}")

    # Build fusion candidates if we have multiple retrievers
    if len(ranked_lists_for_fusion) >= 2 and "fusion-rrf" in generators:
        fused = rrf_fusion(ranked_lists_for_fusion, k0=rrf_k0)[:top_k]
        candidates["fusion-rrf"] = fused

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Build candidate sets for reranker research")
    parser.add_argument("--top_k", type=int, nargs="+", default=[50, 100, 200],
                        help="Candidate pool sizes to generate")
    parser.add_argument("--generators", type=str, nargs="+",
                        default=["qwen3-embed-0.6b", "splade-cocondenser", "fusion-rrf"],
                        help="Candidate generators to use")
    parser.add_argument("--split", type=str, default="dev_select",
                        choices=["dev_select", "test"])
    parser.add_argument("--rrf_k0", type=int, default=60, help="RRF constant")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/reranker_research/candidates"))
    args = parser.parse_args()

    print("=" * 80)
    print("CANDIDATE SET GENERATION FOR RERANKER RESEARCH")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Generators: {args.generators}")
    print(f"Top-K values: {args.top_k}")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    data_dir = Path("data")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")

    print(f"  Loaded {len(sentences)} sentences")
    print(f"  Loaded {len(groundtruth)} groundtruth rows")
    print(f"  Loaded {len(criteria)} criteria")

    # Build split
    all_post_ids = list(set(s.post_id for s in sentences))
    splits = split_post_ids(all_post_ids, seed=42, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    split_posts = set(splits["val"] if args.split == "dev_select" else splits["test"])
    print(f"  Split {args.split}: {len(split_posts)} posts")

    # Build queries with gold labels
    print("\n[2/4] Building queries...")
    gt_lookup = defaultdict(set)
    for row in groundtruth:
        if row.groundtruth == 1:
            gt_lookup[(row.post_id, row.criterion_id)].add(row.sent_uid)

    queries = []
    for post_id in split_posts:
        for criterion in criteria:
            gold_uids = gt_lookup.get((post_id, criterion.criterion_id), set())
            queries.append({
                "post_id": post_id,
                "criterion_id": criterion.criterion_id,
                "criterion_text": criterion.text,
                "gold_uids": gold_uids,
                "has_evidence": len(gold_uids) > 0,
            })

    has_evidence = sum(1 for q in queries if q["has_evidence"])
    no_evidence = len(queries) - has_evidence
    print(f"  Total queries: {len(queries)}")
    print(f"  Has evidence: {has_evidence}, No evidence: {no_evidence}")

    # Initialize retriever zoo
    print("\n[3/4] Initializing retriever zoo...")
    cache_dir = data_dir / "cache"
    zoo = RetrieverZoo(sentences, cache_dir, device="cuda")

    # Pre-encode corpora for needed retrievers
    for gen_name in args.generators:
        if gen_name.startswith("fusion"):
            continue
        try:
            print(f"  Loading {gen_name}...")
            retriever = zoo.get_retriever(gen_name)
            retriever.encode_corpus()
        except Exception as e:
            logger.error(f"Failed to initialize {gen_name}: {e}")

    # Build candidates for each top_k
    print("\n[4/4] Building candidate sets...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for top_k in args.top_k:
        print(f"\n  === Top-K = {top_k} ===")

        all_candidates = {}

        for q in tqdm(queries, desc=f"Generating K={top_k}"):
            query_key = f"{q['post_id']}_{q['criterion_id']}"
            candidates = build_candidates_for_query(
                zoo=zoo,
                query=q["criterion_text"],
                post_id=q["post_id"],
                top_k=top_k,
                generators=args.generators,
                rrf_k0=args.rrf_k0,
            )
            all_candidates[query_key] = {
                "query": q,
                "candidates": candidates,
            }

        # Compute recall statistics per generator
        print(f"\n  Recall@{top_k} per generator:")
        for gen_name in args.generators:
            recalls = []
            for qkey, data in all_candidates.items():
                if not data["query"]["has_evidence"]:
                    continue
                gold = data["query"]["gold_uids"]
                if gen_name in data["candidates"]:
                    retrieved = set(uid for uid, _ in data["candidates"][gen_name])
                    recall = len(gold & retrieved) / len(gold)
                    recalls.append(recall)
            if recalls:
                print(f"    {gen_name}: {np.mean(recalls):.4f}")

        # Save candidates (using pickle for internal caching, same as bge_m3.py)
        output_path = args.output_dir / f"candidates_{args.split}_k{top_k}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().isoformat(),
                "split": args.split,
                "top_k": top_k,
                "generators": args.generators,
                "rrf_k0": args.rrf_k0,
                "n_queries": len(queries),
                "candidates": all_candidates,
            }, f)
        print(f"  Saved to {output_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Candidate generation complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
