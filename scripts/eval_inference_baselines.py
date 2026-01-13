#!/usr/bin/env python3
"""Evaluate inference baselines for all retriever x reranker combinations.

This computes nDCG@10 for each combo using default parameters,
establishing baselines for fine-tuning comparison.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.reranker.zoo import RerankerZoo
from final_sc_review.metrics.ranking import ndcg_at_k, recall_at_k
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)

# Available combinations
RETRIEVERS = [
    "bge-m3",
    "nv-embed-v2",
    "qwen3-embed-4b",
    "qwen3-embed-0.6b",
    # "llama-embed-8b",  # Skip for now - very large
]

RERANKERS = [
    "jina-reranker-v3",
    "bge-reranker-v2-m3",
    "jina-reranker-v2",
]


def load_data(data_dir: Path, groundtruth_path: Path, corpus_path: Path, criteria_path: Path):
    """Load all required data."""
    groundtruth = load_groundtruth(groundtruth_path)
    sentences = load_sentence_corpus(corpus_path)
    criteria = load_criteria(criteria_path)
    return groundtruth, sentences, criteria


def get_split_posts(groundtruth, seed=42, split="val"):
    """Get post IDs for the specified split."""
    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(post_ids, seed=seed, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    return set(splits[split])


def evaluate_combo(
    retriever_name: str,
    reranker_name: str,
    groundtruth,
    sentences,
    criteria,
    split_posts: set,
    top_k_retriever: int = 50,
    top_k_rerank: int = 20,
    cache_dir: Path = None,
) -> Dict:
    """Evaluate a single retriever+reranker combination."""

    logger.info(f"Evaluating: {retriever_name} + {reranker_name}")

    # Initialize retriever
    retriever_zoo = RetrieverZoo(sentences=sentences, cache_dir=cache_dir)
    retriever = retriever_zoo.get_retriever(retriever_name)

    # Encode corpus (uses cache if available)
    logger.info(f"  Encoding corpus for {retriever_name}...")
    retriever.encode_corpus(rebuild=False)

    # Initialize reranker
    reranker_zoo = RerankerZoo()
    reranker = reranker_zoo.get_reranker(reranker_name)

    # Build sentence index by post
    sentences_by_post = {}
    for sent in sentences:
        if sent.post_id not in sentences_by_post:
            sentences_by_post[sent.post_id] = []
        sentences_by_post[sent.post_id].append(sent)

    # Build criteria map
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Filter groundtruth to split
    split_gt = [row for row in groundtruth if row.post_id in split_posts]

    # Group by (post_id, criterion)
    queries = {}
    for row in split_gt:
        key = (row.post_id, row.criterion_id)
        if key not in queries:
            queries[key] = {"positives": set(), "post_id": row.post_id, "criterion": row.criterion_id}
        if row.groundtruth == 1:
            queries[key]["positives"].add(row.sent_uid)

    # Encode corpus per post (use cache if available)
    all_ndcg = []
    all_ndcg_retriever = []

    n_queries = len(queries)
    for i, ((post_id, criterion), query_data) in enumerate(queries.items()):
        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{n_queries}")

        if not query_data["positives"]:
            continue  # Skip queries with no positives

        # Get sentences for this post
        post_sentences = sentences_by_post.get(post_id, [])
        if not post_sentences:
            continue

        # Create query
        criterion_text = criteria_map.get(criterion, criterion)
        query_text = criterion_text

        # Retrieve
        doc_texts = [s.text for s in post_sentences]
        doc_ids = [s.sent_uid for s in post_sentences]

        try:
            retrieval_results = retriever.retrieve_within_post(
                query=query_text,
                post_id=post_id,
                top_k=min(top_k_retriever, len(post_sentences)),
            )

            # Get retriever-only nDCG
            retriever_ranking = [r.sent_uid for r in retrieval_results]
            if query_data["positives"]:
                ndcg_ret = ndcg_at_k(query_data["positives"], retriever_ranking, 10)
                all_ndcg_retriever.append(ndcg_ret)

            # Rerank top-k
            rerank_candidates = retrieval_results[:top_k_rerank]
            if len(rerank_candidates) > 0:
                rerank_texts = [r.text for r in rerank_candidates]
                rerank_ids = [r.sent_uid for r in rerank_candidates]

                rerank_scores = reranker.score_pairs(query_text, rerank_texts)

                # Sort by rerank score
                sorted_indices = np.argsort(rerank_scores)[::-1]
                final_ranking = [rerank_ids[i] for i in sorted_indices]

                # Compute nDCG@10
                if query_data["positives"]:
                    ndcg = ndcg_at_k(query_data["positives"], final_ranking, 10)
                    all_ndcg.append(ndcg)

        except Exception as e:
            logger.warning(f"Error processing query {post_id}/{criterion}: {e}")
            continue

    # Cleanup
    del retriever
    del reranker
    torch.cuda.empty_cache()

    mean_ndcg = np.mean(all_ndcg) if all_ndcg else 0.0
    mean_ndcg_retriever = np.mean(all_ndcg_retriever) if all_ndcg_retriever else 0.0

    return {
        "retriever": retriever_name,
        "reranker": reranker_name,
        "ndcg_at_10_reranked": mean_ndcg,
        "ndcg_at_10_retriever": mean_ndcg_retriever,
        "n_queries": len(all_ndcg),
        "improvement": mean_ndcg - mean_ndcg_retriever,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="data/cache/retriever_zoo")
    parser.add_argument("--output", type=str, default="outputs/inference_baselines.csv")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--retrievers", type=str, nargs="+", default=None)
    parser.add_argument("--rerankers", type=str, nargs="+", default=None)
    parser.add_argument("--top_k_retriever", type=int, default=50)
    parser.add_argument("--top_k_rerank", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    groundtruth, sentences, criteria = load_data(
        data_dir,
        data_dir / "groundtruth/evidence_sentence_groundtruth.csv",
        data_dir / "groundtruth/sentence_corpus.jsonl",
        data_dir / "DSM5/MDD_Criteira.json",
    )

    split_posts = get_split_posts(groundtruth, split=args.split)
    logger.info(f"Split '{args.split}' has {len(split_posts)} posts")

    # Filter combos
    retrievers = args.retrievers if args.retrievers else RETRIEVERS
    rerankers = args.rerankers if args.rerankers else RERANKERS

    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for retriever_name in retrievers:
        for reranker_name in rerankers:
            try:
                result = evaluate_combo(
                    retriever_name=retriever_name,
                    reranker_name=reranker_name,
                    groundtruth=groundtruth,
                    sentences=sentences,
                    criteria=criteria,
                    split_posts=split_posts,
                    top_k_retriever=args.top_k_retriever,
                    top_k_rerank=args.top_k_rerank,
                    cache_dir=cache_dir,
                )
                results.append(result)
                logger.info(f"  nDCG@10: {result['ndcg_at_10_reranked']:.4f} "
                           f"(retriever: {result['ndcg_at_10_retriever']:.4f}, "
                           f"improvement: {result['improvement']:.4f})")

                # Save incrementally
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)

            except Exception as e:
                logger.error(f"Failed {retriever_name} + {reranker_name}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print("\n" + "="*70)
    print("INFERENCE BASELINE RESULTS")
    print("="*70)
    print(f"{'Retriever':<20} {'Reranker':<25} {'nDCG@10':>10} {'Improvement':>12}")
    print("-"*70)
    for r in sorted(results, key=lambda x: x["ndcg_at_10_reranked"], reverse=True):
        print(f"{r['retriever']:<20} {r['reranker']:<25} {r['ndcg_at_10_reranked']:>10.4f} {r['improvement']:>+12.4f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
