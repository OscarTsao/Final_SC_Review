#!/usr/bin/env python3
"""Build OOF (Out-of-Fold) cache with real retriever and reranker scores.

This script scores all query-candidate pairs using the retriever and reranker zoos,
producing a cache file that the feature store can use for real metrics.

Output: data/cache/oof_cache/<retriever>_<reranker>_cache.parquet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.data.io import load_criteria
from final_sc_review.data.schemas import Sentence

logger = get_logger(__name__)


def load_sentence_corpus(corpus_path: Path) -> List[Sentence]:
    """Load sentence corpus from JSONL file."""
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            data = json.loads(line)
            sentences.append(Sentence(
                post_id=data["post_id"],
                sid=data["sid"],
                sent_uid=data["sent_uid"],
                text=data["text"],
            ))
    return sentences


def build_oof_cache(
    groundtruth_path: Path,
    corpus_path: Path,
    criteria_path: Path,
    cache_dir: Path,
    retriever_name: str = "bge-m3",
    reranker_name: str = "jina-reranker-v3",
    top_k_retrieval: int = 50,
    top_k_rerank: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
) -> Path:
    """Build OOF cache with real scores."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{retriever_name}_{reranker_name}_cache.parquet"

    logger.info(f"Building OOF cache: {retriever_name} + {reranker_name}")
    logger.info(f"Output: {output_path}")

    # Load data
    logger.info("Loading corpus and groundtruth...")
    sentences = load_sentence_corpus(corpus_path)
    logger.info(f"  Loaded {len(sentences)} sentences")

    gt_df = pd.read_csv(groundtruth_path)
    logger.info(f"  Loaded {len(gt_df)} groundtruth rows")

    criteria = load_criteria(criteria_path)
    criteria_map = {c.criterion_id: c.text for c in criteria}
    logger.info(f"  Loaded {len(criteria)} criteria")

    # Initialize retriever
    logger.info(f"Initializing retriever: {retriever_name}")
    from final_sc_review.retriever.zoo import RetrieverZoo

    retriever_zoo = RetrieverZoo(
        sentences=sentences,
        cache_dir=Path("data/cache"),
        device=device,
    )
    retriever = retriever_zoo.get_retriever(retriever_name)
    retriever.encode_corpus()

    # Initialize reranker
    logger.info(f"Initializing reranker: {reranker_name}")
    from final_sc_review.reranker.zoo import RerankerZoo

    reranker_zoo = RerankerZoo(device=device)
    reranker = reranker_zoo.get_reranker(reranker_name)
    reranker.load_model()

    # Build cache records
    records = []

    # Group by query (post_id, criterion)
    query_groups = gt_df.groupby(["post_id", "criterion"])
    total_queries = len(query_groups)
    logger.info(f"Processing {total_queries} queries...")

    # Process in batches for efficiency
    queries_batch = []
    candidates_batch = []
    metadata_batch = []

    for (post_id, criterion_id), group in tqdm(query_groups, desc="Building cache"):
        query_text = criteria_map.get(criterion_id, "")
        if not query_text:
            continue

        # Get gold sentence IDs
        gold_rows = group[group["groundtruth"] == 1]
        gold_ids = set(gold_rows["sent_uid"].tolist())
        has_evidence = len(gold_ids) > 0

        # Retrieve candidates
        retrieval_results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k=top_k_retrieval,
        )

        if not retrieval_results:
            continue

        # Prepare for reranking
        candidates = [(r.sent_uid, r.text) for r in retrieval_results]
        retriever_scores = {r.sent_uid: r.score for r in retrieval_results}

        queries_batch.append(query_text)
        candidates_batch.append(candidates)
        metadata_batch.append({
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gold_ids": gold_ids,
            "has_evidence": has_evidence,
            "retriever_scores": retriever_scores,
        })

        # Process batch when full
        if len(queries_batch) >= batch_size:
            _process_batch(
                queries_batch, candidates_batch, metadata_batch,
                reranker, top_k_rerank, records
            )
            queries_batch = []
            candidates_batch = []
            metadata_batch = []

    # Process remaining
    if queries_batch:
        _process_batch(
            queries_batch, candidates_batch, metadata_batch,
            reranker, top_k_rerank, records
        )

    # Create DataFrame and save
    cache_df = pd.DataFrame(records)
    cache_df.to_parquet(output_path, index=False)

    logger.info(f"\nOOF cache built successfully!")
    logger.info(f"  Total records: {len(cache_df)}")
    logger.info(f"  Unique queries: {cache_df[['post_id', 'criterion_id']].drop_duplicates().shape[0]}")
    logger.info(f"  Saved to: {output_path}")

    # Print summary stats
    if len(cache_df) > 0:
        logger.info(f"\nScore statistics:")
        logger.info(f"  Retriever score: mean={cache_df['retriever_score'].mean():.4f}, std={cache_df['retriever_score'].std():.4f}")
        logger.info(f"  Reranker score: mean={cache_df['reranker_score'].mean():.4f}, std={cache_df['reranker_score'].std():.4f}")

    return output_path


def _process_batch(
    queries_batch: List[str],
    candidates_batch: List[List[Tuple[str, str]]],
    metadata_batch: List[Dict],
    reranker,
    top_k_rerank: int,
    records: List[Dict],
):
    """Process a batch of queries through the reranker."""
    # Batch rerank
    queries_and_candidates = list(zip(queries_batch, candidates_batch))

    try:
        all_results = reranker.rerank_batch(queries_and_candidates, top_k=top_k_rerank)
    except Exception as e:
        logger.warning(f"Batch rerank failed, falling back to sequential: {e}")
        all_results = [
            reranker.rerank(q, c, top_k=top_k_rerank)
            for q, c in queries_and_candidates
        ]

    # Build records
    for i, results in enumerate(all_results):
        meta = metadata_batch[i]
        retriever_scores = meta["retriever_scores"]

        for result in results:
            records.append({
                "post_id": meta["post_id"],
                "criterion_id": meta["criterion_id"],
                "sent_uid": result.sent_uid,
                "sentence": result.text,
                "retriever_score": retriever_scores.get(result.sent_uid, 0.0),
                "reranker_score": result.score,
                "reranker_rank": result.rank,
                "is_gold": result.sent_uid in meta["gold_ids"],
                "has_evidence": meta["has_evidence"],
            })


def main():
    parser = argparse.ArgumentParser(description="Build OOF cache with real scores")
    parser.add_argument("--groundtruth", type=str,
                        default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--corpus", type=str,
                        default="data/groundtruth/sentence_corpus.jsonl")
    parser.add_argument("--criteria", type=str,
                        default="data/DSM5/MDD_Criteira.json")
    parser.add_argument("--cache_dir", type=str,
                        default="data/cache/oof_cache")
    parser.add_argument("--retriever", type=str, default="bge-m3",
                        help="Retriever name from zoo")
    parser.add_argument("--reranker", type=str, default="jina-reranker-v3",
                        help="Reranker name from zoo")
    parser.add_argument("--top_k_retrieval", type=int, default=50,
                        help="Top-k candidates from retriever")
    parser.add_argument("--top_k_rerank", type=int, default=20,
                        help="Top-k candidates after reranking")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for reranking")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_path = build_oof_cache(
        groundtruth_path=Path(args.groundtruth),
        corpus_path=Path(args.corpus),
        criteria_path=Path(args.criteria),
        cache_dir=Path(args.cache_dir),
        retriever_name=args.retriever,
        reranker_name=args.reranker,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\nOOF cache built at: {output_path}")


if __name__ == "__main__":
    main()
