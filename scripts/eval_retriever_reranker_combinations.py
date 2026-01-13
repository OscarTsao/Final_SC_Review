#!/usr/bin/env python
"""Comprehensive Retriever + Reranker Combination Evaluation.

This script implements Section 12 of the reranker_research_plan.md:
- Tests all retriever (G) + reranker (M) combinations
- Evaluates with different candidate pool sizes (K)
- Finds the best overall system configuration

Usage:
    # Run all combinations (requires cached embeddings)
    python scripts/eval_retriever_reranker_combinations.py \
        --output_dir outputs/retriever_reranker_sweep \
        --retrievers bge-m3 bge-large-en-v1.5 e5-large-v2 bm25 \
        --rerankers bge-reranker-v2-m3 jina-reranker-v2 qwen3-reranker-0.6b \
        --top_k_values 50 100 200

    # Quick test with subset
    python scripts/eval_retriever_reranker_combinations.py \
        --output_dir outputs/quick_sweep \
        --retrievers bge-m3 \
        --rerankers bge-reranker-v2-m3 \
        --max_queries 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.schemas import Sentence
from final_sc_review.data.splits import create_splits
from final_sc_review.metrics.ranking import (
    mean_average_precision_at_k,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
    recall_at_k,
)
from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.reranker.zoo import RerankerZoo
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CombinationResult:
    """Result for a single retriever+reranker combination."""
    retriever: str
    reranker: str
    top_k_retriever: int
    top_k_rerank: int
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    map_at_10: float
    false_evidence_rate: float
    avg_latency_ms: float
    num_queries: int
    timestamp: str


@dataclass
class SweepConfig:
    """Configuration for the combination sweep."""
    retrievers: List[str]
    rerankers: List[str]
    top_k_retriever_values: List[int]
    top_k_rerank_values: List[int]
    max_queries: Optional[int]
    split: str
    output_dir: Path


def build_queries(
    groundtruth: pd.DataFrame,
    criteria: Dict[str, str],
    split_post_ids: List[str],
    max_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build query list from groundtruth."""
    queries = []

    # Get unique (post_id, criterion) pairs from split
    split_gt = groundtruth[groundtruth["post_id"].isin(split_post_ids)]

    for (post_id, criterion), group in split_gt.groupby(["post_id", "criterion"]):
        if criterion not in criteria:
            continue

        gold_uids = set(group[group["groundtruth"] == 1]["sent_uid"].tolist())

        queries.append({
            "post_id": post_id,
            "criterion": criterion,
            "query": criteria[criterion],
            "gold_uids": gold_uids,
            "has_evidence": len(gold_uids) > 0,
        })

    if max_queries:
        queries = queries[:max_queries]

    return queries


def evaluate_combination(
    retriever_zoo: RetrieverZoo,
    reranker_zoo: RerankerZoo,
    retriever_name: str,
    reranker_name: str,
    queries: List[Dict[str, Any]],
    top_k_retriever: int,
    top_k_rerank: int,
) -> CombinationResult:
    """Evaluate a single retriever+reranker combination."""
    try:
        retriever = retriever_zoo.get_retriever(retriever_name)
        retriever.encode_corpus()
    except Exception as e:
        logger.warning(f"Failed to load retriever {retriever_name}: {e}")
        return None

    try:
        reranker = reranker_zoo.get_reranker(reranker_name)
    except Exception as e:
        logger.warning(f"Failed to load reranker {reranker_name}: {e}")
        return None

    # Metrics accumulators
    all_retrieved = []
    all_gold = []
    latencies = []
    false_evidence_count = 0
    no_evidence_queries = 0

    for q in tqdm(queries, desc=f"{retriever_name} → {reranker_name}", leave=False):
        start_time = time.time()

        # Stage 1: Retrieve candidates
        candidates = retriever.retrieve_within_post(
            query=q["query"],
            post_id=q["post_id"],
            top_k=top_k_retriever,
        )

        if not candidates:
            all_retrieved.append([])
            all_gold.append(list(q["gold_uids"]))
            continue

        # Stage 2: Rerank
        rerank_input = [(c.text, c.sent_uid) for c in candidates[:top_k_rerank]]
        texts = [t for t, _ in rerank_input]
        uids = [u for _, u in rerank_input]

        scores = reranker.rerank(q["query"], texts)

        # Sort by reranker score
        ranked = sorted(zip(uids, scores), key=lambda x: -x[1])
        ranked_uids = [uid for uid, _ in ranked]

        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)

        all_retrieved.append(ranked_uids)
        all_gold.append(list(q["gold_uids"]))

        # Track false evidence on no-evidence queries
        if not q["has_evidence"]:
            no_evidence_queries += 1
            # Check if top result has positive score (model thinks there's evidence)
            if ranked and ranked[0][1] > 0:
                false_evidence_count += 1

    # Compute metrics
    if not all_retrieved:
        return None

    ndcg = ndcg_at_k(all_retrieved, all_gold, k=10)
    mrr = mean_reciprocal_rank_at_k(all_retrieved, all_gold, k=10)
    recall = recall_at_k(all_retrieved, all_gold, k=10)
    map_score = mean_average_precision_at_k(all_retrieved, all_gold, k=10)

    false_evidence_rate = (
        false_evidence_count / no_evidence_queries
        if no_evidence_queries > 0 else 0.0
    )

    return CombinationResult(
        retriever=retriever_name,
        reranker=reranker_name,
        top_k_retriever=top_k_retriever,
        top_k_rerank=top_k_rerank,
        ndcg_at_10=ndcg,
        mrr_at_10=mrr,
        recall_at_10=recall,
        map_at_10=map_score,
        false_evidence_rate=false_evidence_rate,
        avg_latency_ms=np.mean(latencies) if latencies else 0.0,
        num_queries=len(queries),
        timestamp=datetime.now().isoformat(),
    )


def run_sweep(config: SweepConfig) -> List[CombinationResult]:
    """Run the full retriever+reranker sweep."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_dir = Path("data")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "MDD_Criteria.json")

    # Load sentence corpus
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            data = json.loads(line)
            sentences.append(Sentence(
                post_id=data["post_id"],
                sid=data["sid"],
                sent_uid=data["sent_uid"],
                text=data["sentence"],
            ))

    logger.info(f"Loaded {len(sentences)} sentences, {len(criteria)} criteria")

    # Create splits
    splits = create_splits(groundtruth, train_ratio=0.7, val_ratio=0.15, seed=42)
    split_post_ids = splits[config.split]
    logger.info(f"Using {config.split} split with {len(split_post_ids)} posts")

    # Build queries
    queries = build_queries(groundtruth, criteria, split_post_ids, config.max_queries)
    logger.info(f"Built {len(queries)} queries")

    # Initialize zoos
    cache_dir = data_dir / "cache"
    retriever_zoo = RetrieverZoo(sentences, cache_dir)
    reranker_zoo = RerankerZoo()

    # Run sweep
    results = []
    total_combinations = (
        len(config.retrievers) *
        len(config.rerankers) *
        len(config.top_k_retriever_values) *
        len(config.top_k_rerank_values)
    )

    logger.info(f"Running {total_combinations} combinations...")

    with tqdm(total=total_combinations, desc="Sweep progress") as pbar:
        for retriever_name in config.retrievers:
            for reranker_name in config.rerankers:
                for top_k_retriever in config.top_k_retriever_values:
                    for top_k_rerank in config.top_k_rerank_values:
                        # Skip if top_k_rerank > top_k_retriever
                        if top_k_rerank > top_k_retriever:
                            pbar.update(1)
                            continue

                        result = evaluate_combination(
                            retriever_zoo=retriever_zoo,
                            reranker_zoo=reranker_zoo,
                            retriever_name=retriever_name,
                            reranker_name=reranker_name,
                            queries=queries,
                            top_k_retriever=top_k_retriever,
                            top_k_rerank=top_k_rerank,
                        )

                        if result:
                            results.append(result)

                            # Save incremental results
                            save_results(results, config.output_dir)

                        pbar.update(1)

                        # Clear GPU memory between combinations
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    return results


def save_results(results: List[CombinationResult], output_dir: Path):
    """Save results to CSV and JSON."""
    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])

    # Sort by nDCG@10
    df = df.sort_values("ndcg_at_10", ascending=False)

    # Save CSV
    csv_path = output_dir / "combination_results.csv"
    df.to_csv(csv_path, index=False)

    # Save JSON with best configs
    best_overall = df.iloc[0].to_dict()
    best_per_reranker = df.groupby("reranker").apply(
        lambda x: x.nlargest(1, "ndcg_at_10").iloc[0].to_dict()
    ).to_dict()
    best_per_retriever = df.groupby("retriever").apply(
        lambda x: x.nlargest(1, "ndcg_at_10").iloc[0].to_dict()
    ).to_dict()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_combinations": len(results),
        "best_overall": best_overall,
        "best_per_reranker": best_per_reranker,
        "best_per_retriever": best_per_retriever,
    }

    json_path = output_dir / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print leaderboard
    print("\n" + "=" * 80)
    print("RETRIEVER + RERANKER COMBINATION LEADERBOARD")
    print("=" * 80)
    print(f"\nTop 10 combinations by nDCG@10:\n")
    print(df[["retriever", "reranker", "top_k_retriever", "top_k_rerank",
              "ndcg_at_10", "mrr_at_10", "recall_at_10", "false_evidence_rate"]].head(10).to_string())
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever+reranker combinations")
    parser.add_argument("--output_dir", type=str, default="outputs/retriever_reranker_sweep")
    parser.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        default=["bge-m3", "bge-large-en-v1.5", "e5-large-v2", "bm25"],
        help="Retriever names to evaluate"
    )
    parser.add_argument(
        "--rerankers",
        type=str,
        nargs="+",
        default=["bge-reranker-v2-m3", "jina-reranker-v2", "ms-marco-minilm"],
        help="Reranker names to evaluate"
    )
    parser.add_argument(
        "--top_k_retriever",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Top-K values for retriever"
    )
    parser.add_argument(
        "--top_k_rerank",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="Top-K values for reranking"
    )
    parser.add_argument("--max_queries", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])

    args = parser.parse_args()

    config = SweepConfig(
        retrievers=args.retrievers,
        rerankers=args.rerankers,
        top_k_retriever_values=args.top_k_retriever,
        top_k_rerank_values=args.top_k_rerank,
        max_queries=args.max_queries,
        split=args.split,
        output_dir=Path(args.output_dir),
    )

    print("=" * 80)
    print("RETRIEVER + RERANKER COMBINATION SWEEP")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Retrievers: {config.retrievers}")
    print(f"Rerankers: {config.rerankers}")
    print(f"Top-K retriever: {config.top_k_retriever_values}")
    print(f"Top-K rerank: {config.top_k_rerank_values}")
    print(f"Split: {config.split}")
    print("=" * 80)

    results = run_sweep(config)

    logger.info(f"Sweep complete! {len(results)} combinations evaluated")
    logger.info(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
