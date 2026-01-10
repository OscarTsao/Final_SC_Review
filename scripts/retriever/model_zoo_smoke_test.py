#!/usr/bin/env python3
"""
Quick smoke test for retriever models from the zoo.

Tests basic functionality on a small sample to verify models work.

Usage:
    python scripts/retriever/model_zoo_smoke_test.py
    python scripts/retriever/model_zoo_smoke_test.py --n_samples 50
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Retriever model zoo smoke test")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--models", nargs="+", default=None, help="Models to test")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("="*60)
    print("RETRIEVER MODEL ZOO SMOKE TEST")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"N samples: {args.n_samples}")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    print(f"  Corpus: {len(sentences)} sentences")
    print(f"  Groundtruth: {len(gt_rows)} rows")

    # Sample positive queries
    print("\n[2/4] Sampling positive queries...")
    positive_queries = []
    seen_pairs = set()
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            if key not in seen_pairs:
                seen_pairs.add(key)
                positive_queries.append({
                    "post_id": row.post_id,
                    "criterion_id": row.criterion_id,
                    "gold_uid": row.sent_uid,
                })
                if len(positive_queries) >= args.n_samples:
                    break

    print(f"  Sampled {len(positive_queries)} positive queries")

    # Test BGE-M3 retriever
    print("\n[3/4] Testing BGE-M3 retriever...")
    try:
        retriever = BgeM3Retriever(
            sentences=sentences,
            cache_dir=cache_dir,
            rebuild_cache=False,
        )

        hits = 0
        k_values = [5, 10, 20, 50]
        recall_at_k = {k: 0 for k in k_values}

        for i, q in enumerate(positive_queries[:args.n_samples]):
            # Use post_id as query for smoke test
            results = retriever.retrieve_within_post(
                query=q["criterion_id"],  # Use criterion as query
                post_id=q["post_id"],
                top_k_retriever=50,
            )

            retrieved_uids = [r[0] for r in results]

            for k in k_values:
                if q["gold_uid"] in retrieved_uids[:k]:
                    recall_at_k[k] += 1

            if q["gold_uid"] in retrieved_uids[:10]:
                hits += 1

            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(positive_queries)} queries")

        print(f"\n  BGE-M3 Results (n={len(positive_queries[:args.n_samples])}):")
        for k in k_values:
            recall = recall_at_k[k] / len(positive_queries[:args.n_samples])
            print(f"    Recall@{k}: {recall:.3f}")

    except Exception as e:
        print(f"  [ERROR] BGE-M3 test failed: {e}")
        sys.exit(1)

    # Save results
    print("\n[4/4] Saving results...")
    output_dir = repo_root / "outputs" / "retriever"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "smoke_test_results.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": args.n_samples,
        "n_tested": len(positive_queries[:args.n_samples]),
        "bge_m3": {
            "recall_at_k": {str(k): recall_at_k[k] / len(positive_queries[:args.n_samples])
                           for k in k_values}
        }
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved to: {results_path}")

    print("\n" + "="*60)
    print("[PASS] Smoke test complete")
    print("="*60)


if __name__ == "__main__":
    main()
