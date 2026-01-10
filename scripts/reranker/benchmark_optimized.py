#!/usr/bin/env python3
"""Benchmark optimized vs baseline reranker inference.

Compares:
1. Baseline (sequential query processing)
2. Optimized batch inference with bucketing + prefetching

Note: Uses pickle to load existing internal candidate cache files.

Usage:
    python scripts/reranker/benchmark_optimized.py \
        --candidates outputs/reranker_research/candidates/candidates_dev_select_k100.pkl \
        --reranker jina-reranker-v2 \
        --n_queries 500
"""

import argparse
import json
import pickle  # Internal cache loading only
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.data.io import load_sentence_corpus
from final_sc_review.reranker.zoo import RerankerZoo
from final_sc_review.reranker.optimized_inference import (
    BatchReranker,
    QueryCandidates,
    benchmark_batch_reranker,
)
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_candidates_as_queries(
    candidates_path: Path,
    generator: str,
    sent_uid_to_text: Dict[str, str],
    top_k: int = 20,
    max_queries: int = None,
) -> List[QueryCandidates]:
    """Load candidate pickle and convert to QueryCandidates format."""
    with open(candidates_path, "rb") as f:
        data = pickle.load(f)

    queries = []
    for query_key, qdata in data["candidates"].items():
        if max_queries and len(queries) >= max_queries:
            break

        cands = qdata["candidates"].get(generator, [])
        if not cands:
            continue

        # Build candidate tuples
        candidate_tuples = []
        for uid, _ in cands[:top_k]:
            text = sent_uid_to_text.get(uid, "")
            candidate_tuples.append((uid, text))

        queries.append(QueryCandidates(
            query_key=query_key,
            query_text=qdata["query"]["criterion_text"],
            candidates=candidate_tuples,
            gold_uids=qdata["query"]["gold_uids"],
            has_evidence=qdata["query"]["has_evidence"],
        ))

    return queries


def benchmark_baseline(
    zoo: RerankerZoo,
    reranker_name: str,
    queries: List[QueryCandidates],
    top_k: int = 10,
) -> Dict:
    """Benchmark baseline sequential processing."""
    reranker = zoo.get_reranker(reranker_name)
    reranker.load_model()

    start = time.time()
    total_pairs = 0

    for q in tqdm(queries, desc="Baseline"):
        results = reranker.rerank(
            query=q.query_text,
            candidates=q.candidates,
            top_k=top_k,
        )
        total_pairs += len(q.candidates)

    elapsed = time.time() - start
    throughput = total_pairs / elapsed

    return {
        "method": "baseline",
        "elapsed_seconds": elapsed,
        "total_pairs": total_pairs,
        "n_queries": len(queries),
        "throughput_pairs_per_sec": throughput,
        "throughput_queries_per_sec": len(queries) / elapsed,
    }


def benchmark_optimized(
    zoo: RerankerZoo,
    reranker_name: str,
    queries: List[QueryCandidates],
    top_k: int = 10,
    batch_size: int = None,
    use_bucketing: bool = True,
    use_cache: bool = False,
) -> Dict:
    """Benchmark optimized batch processing."""
    cache_dir = Path("outputs/reranker_research/cache") if use_cache else None

    batch_reranker = BatchReranker(
        zoo=zoo,
        reranker_name=reranker_name,
        cache_dir=cache_dir,
    )

    start = time.time()

    results = batch_reranker.rerank_batch(
        queries=queries,
        top_k=top_k,
        batch_size=batch_size,
        use_bucketing=use_bucketing,
        show_progress=True,
    )

    elapsed = time.time() - start
    total_pairs = sum(len(q.candidates) for q in queries)
    throughput = total_pairs / elapsed

    return {
        "method": "optimized",
        "batch_size": batch_size or batch_reranker.config.batch_size,
        "use_bucketing": use_bucketing,
        "use_cache": use_cache,
        "elapsed_seconds": elapsed,
        "total_pairs": total_pairs,
        "n_queries": len(queries),
        "throughput_pairs_per_sec": throughput,
        "throughput_queries_per_sec": len(queries) / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimized reranker inference")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Path to candidates pickle file")
    parser.add_argument("--reranker", type=str, default="jina-reranker-v2",
                        help="Reranker to benchmark")
    parser.add_argument("--generator", type=str, default="fusion-rrf",
                        help="Candidate generator")
    parser.add_argument("--n_queries", type=int, default=500,
                        help="Number of queries to benchmark")
    parser.add_argument("--top_k_rerank", type=int, default=20,
                        help="Top-k candidates to rerank")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64, 128],
                        help="Batch sizes to test")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/reranker_research/benchmarks"))
    args = parser.parse_args()

    print("=" * 80)
    print("RERANKER INFERENCE BENCHMARK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Reranker: {args.reranker}")
    print(f"N queries: {args.n_queries}")
    print(f"Top-K rerank: {args.top_k_rerank}")
    print("=" * 80)

    # Load sentence corpus
    print("\n[1/4] Loading sentence corpus...")
    sentences = load_sentence_corpus(Path("data/groundtruth/sentence_corpus.jsonl"))
    sent_uid_to_text = {s.sent_uid: s.text for s in sentences}
    print(f"  Loaded {len(sent_uid_to_text)} sentences")

    # Load candidates
    print("\n[2/4] Loading candidates...")
    queries = load_candidates_as_queries(
        args.candidates,
        args.generator,
        sent_uid_to_text,
        top_k=args.top_k_rerank,
        max_queries=args.n_queries,
    )
    print(f"  Loaded {len(queries)} queries")
    total_pairs = sum(len(q.candidates) for q in queries)
    print(f"  Total pairs: {total_pairs}")

    # Initialize zoo
    print("\n[3/4] Initializing reranker zoo...")
    zoo = RerankerZoo(device="cuda")

    # Run benchmarks
    print("\n[4/4] Running benchmarks...")
    all_results = []

    # Baseline
    print("\n--- BASELINE (Sequential) ---")
    baseline_result = benchmark_baseline(zoo, args.reranker, queries, top_k=10)
    all_results.append(baseline_result)
    print(f"  Throughput: {baseline_result['throughput_pairs_per_sec']:.1f} pairs/sec")
    print(f"  Time: {baseline_result['elapsed_seconds']:.2f}s")

    # Optimized with different batch sizes
    for bs in args.batch_sizes:
        print(f"\n--- OPTIMIZED (batch_size={bs}, bucketing=True) ---")
        opt_result = benchmark_optimized(
            zoo, args.reranker, queries,
            top_k=10,
            batch_size=bs,
            use_bucketing=True,
        )
        all_results.append(opt_result)
        speedup = opt_result['throughput_pairs_per_sec'] / baseline_result['throughput_pairs_per_sec']
        print(f"  Throughput: {opt_result['throughput_pairs_per_sec']:.1f} pairs/sec")
        print(f"  Time: {opt_result['elapsed_seconds']:.2f}s")
        print(f"  Speedup vs baseline: {speedup:.2f}x")

    # Optimized without bucketing (to measure bucketing impact)
    print(f"\n--- OPTIMIZED (batch_size={args.batch_sizes[0]}, bucketing=False) ---")
    no_bucket_result = benchmark_optimized(
        zoo, args.reranker, queries,
        top_k=10,
        batch_size=args.batch_sizes[0],
        use_bucketing=False,
    )
    all_results.append(no_bucket_result)
    speedup = no_bucket_result['throughput_pairs_per_sec'] / baseline_result['throughput_pairs_per_sec']
    print(f"  Throughput: {no_bucket_result['throughput_pairs_per_sec']:.1f} pairs/sec")
    print(f"  Time: {no_bucket_result['elapsed_seconds']:.2f}s")
    print(f"  Speedup vs baseline: {speedup:.2f}x")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"benchmark_{args.reranker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "reranker": args.reranker,
            "n_queries": len(queries),
            "total_pairs": total_pairs,
            "top_k_rerank": args.top_k_rerank,
            "results": all_results,
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<40} {'Throughput':>15} {'Speedup':>10}")
    print("-" * 80)
    for r in all_results:
        method = r.get('method', 'unknown')
        if method == 'optimized':
            method = f"optimized (bs={r.get('batch_size')}, bucket={r.get('use_bucketing')})"
        speedup = r['throughput_pairs_per_sec'] / baseline_result['throughput_pairs_per_sec']
        print(f"{method:<40} {r['throughput_pairs_per_sec']:>12.1f}/s {speedup:>9.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
