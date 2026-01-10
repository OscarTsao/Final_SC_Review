#!/usr/bin/env python3
"""
Build candidate pools for all queries.

Pre-computes retrieval results for HPO and evaluation.
Note: Uses pickle for efficient serialization of large numerical arrays.

Usage:
    python scripts/retriever/build_candidates.py
    python scripts/retriever/build_candidates.py --config configs/default.yaml
    python scripts/retriever/build_candidates.py --top_k 200
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Build candidate pools")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--top_k", type=int, default=200, help="Top K candidates to keep")
    parser.add_argument("--split", type=str, default="dev_tune", help="Split to process")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth, load_criteria
    from final_sc_review.data.splits import split_post_ids
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever

    # Load config
    if args.config:
        config_path = Path(args.config)
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    paths = config.get("paths", {})
    corpus_path = Path(paths.get(
        "sentence_corpus",
        repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    ))
    gt_path = Path(paths.get(
        "evidence_groundtruth",
        repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    ))
    criteria_path = Path(paths.get(
        "criteria_path",
        repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    ))
    cache_dir = Path(paths.get(
        "cache_dir",
        repo_root / "data" / "cache" / "bge_m3"
    ))

    print("="*60)
    print("BUILD CANDIDATE POOLS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Top K: {args.top_k}")
    print(f"Split: {args.split}")
    print("="*60)

    # Load data
    print("\n[1/5] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)

    criterion_text = {c.criterion_id: c.text for c in criteria}
    print(f"  Corpus: {len(sentences)} sentences")
    print(f"  Groundtruth: {len(gt_rows)} rows")
    print(f"  Criteria: {len(criterion_text)}")

    # Build queries
    print("\n[2/5] Building queries...")
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}
    print(f"  Total posts: {len(all_posts)}")
    print(f"  Queries with gold: {len(query_gold)}")

    # Split posts
    print("\n[3/5] Splitting posts...")
    split_cfg = config.get("split", {})
    splits = split_post_ids(
        list(all_posts),
        train_ratio=split_cfg.get("train_ratio", 0.6),
        val_ratio=split_cfg.get("val_ratio", 0.2),
        test_ratio=split_cfg.get("test_ratio", 0.2),
        seed=split_cfg.get("seed", 42),
    )

    split_map = {"dev_tune": splits["train"], "dev_select": splits["val"], "test": splits["test"]}
    target_posts = set(split_map.get(args.split, splits["train"]))
    print(f"  {args.split} posts: {len(target_posts)}")

    # Filter queries to target split
    split_queries = [(k, v) for k, v in query_gold.items() if k[0] in target_posts]
    print(f"  {args.split} queries with gold: {len(split_queries)}")

    # Initialize retriever
    print("\n[4/5] Initializing retriever...")
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        rebuild_cache=False,
    )

    # Build candidates
    print(f"\n[5/5] Building candidates for {len(split_queries)} queries...")
    candidates = {}

    for (post_id, criterion_id), gold_uids in tqdm(split_queries, desc="Queries"):
        query_text = criterion_text.get(criterion_id, criterion_id)

        results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k_retriever=args.top_k,
            return_component_scores=True,
        )

        key = f"{post_id}_{criterion_id}"
        candidates[key] = {
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gold_uids": list(gold_uids),
            "candidates": [
                {
                    "sent_uid": r[0],
                    "text": r[1],
                    "score": r[2],
                    "component_scores": r[3] if len(r) > 3 else {},
                }
                for r in results
            ]
        }

    # Save candidates using pickle for efficiency with large arrays
    output_dir = repo_root / "outputs" / "retriever" / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"candidates_{args.split}_k{args.top_k}.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(candidates, f)

    # Also save summary JSON
    summary_path = output_dir / f"candidates_{args.split}_k{args.top_k}_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "top_k": args.top_k,
        "n_queries": len(candidates),
        "n_with_gold": sum(1 for v in candidates.values() if v["gold_uids"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  Saved candidates to: {output_path}")
    print(f"  Saved summary to: {summary_path}")

    print("\n" + "="*60)
    print("[SUCCESS] Candidate build complete")
    print("="*60)


if __name__ == "__main__":
    main()
