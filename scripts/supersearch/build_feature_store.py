#!/usr/bin/env python3
"""Build Feature Store / OOF cache (single source of truth).

This script loads existing candidates + scores and outputs a parquet per fold with:
- identifiers: post_id, query_id, criterion_id, fold_id
- labels: has_evidence (binary), gold_sentence_ids list
- candidate list (top 20): sentence ids + text optional
- retriever scores, reranker scores
- calibrated probabilities if available (Platt / isotonic / temperature)
- derived features:
  max, second, mean, std, entropy of topK probs, top1-top2 gap, top1-top5 gap, topk sum probs,
  rank features, lexical overlap features (BM25/TFIDF optional),
  coherence features (embedding similarity stats among top candidates)

Output: outputs/feature_store/<timestamp>/fold_*.parquet
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.data.io import load_criteria

logger = get_logger(__name__)


def compute_derived_features(
    candidate_scores: np.ndarray,  # [n_candidates] reranker scores
    retriever_scores: Optional[np.ndarray] = None,
    top_k: int = 5,
) -> Dict[str, float]:
    """Compute derived features from candidate scores."""
    features = {}

    if len(candidate_scores) == 0:
        # Return zeros for empty candidates
        return {
            "max_reranker_score": 0.0,
            "second_reranker_score": 0.0,
            "mean_reranker_score": 0.0,
            "std_reranker_score": 0.0,
            "top1_top2_gap": 0.0,
            "top1_top5_gap": 0.0,
            "topk_sum_score": 0.0,
            "entropy_top5": 0.0,
            "entropy_top10": 0.0,
            "score_range": 0.0,
            "score_skewness": 0.0,
        }

    # Sort scores descending
    sorted_scores = np.sort(candidate_scores)[::-1]

    # Basic statistics
    features["max_reranker_score"] = float(sorted_scores[0])
    features["second_reranker_score"] = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    features["mean_reranker_score"] = float(np.mean(sorted_scores[:top_k]))
    features["std_reranker_score"] = float(np.std(sorted_scores[:top_k]))

    # Gap features
    features["top1_top2_gap"] = features["max_reranker_score"] - features["second_reranker_score"]
    features["top1_top5_gap"] = features["max_reranker_score"] - (
        float(sorted_scores[4]) if len(sorted_scores) > 4 else 0.0
    )

    # Sum features
    features["topk_sum_score"] = float(np.sum(sorted_scores[:top_k]))

    # Convert to probabilities for entropy (softmax)
    top5_scores = sorted_scores[:min(5, len(sorted_scores))]
    top10_scores = sorted_scores[:min(10, len(sorted_scores))]

    # Softmax normalization
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    if len(top5_scores) > 0:
        probs_5 = softmax(top5_scores)
        features["entropy_top5"] = float(entropy(probs_5))
    else:
        features["entropy_top5"] = 0.0

    if len(top10_scores) > 0:
        probs_10 = softmax(top10_scores)
        features["entropy_top10"] = float(entropy(probs_10))
    else:
        features["entropy_top10"] = 0.0

    # Range and distribution
    features["score_range"] = float(sorted_scores[0] - sorted_scores[-1]) if len(sorted_scores) > 1 else 0.0

    # Skewness (simplified)
    if len(sorted_scores) > 2 and features["std_reranker_score"] > 0:
        features["score_skewness"] = float(
            (features["mean_reranker_score"] - sorted_scores[len(sorted_scores) // 2]) / features["std_reranker_score"]
        )
    else:
        features["score_skewness"] = 0.0

    # Add retriever features if available
    if retriever_scores is not None and len(retriever_scores) > 0:
        sorted_ret = np.sort(retriever_scores)[::-1]
        features["max_retriever_score"] = float(sorted_ret[0])
        features["mean_retriever_score"] = float(np.mean(sorted_ret[:top_k]))
        features["retriever_reranker_corr"] = float(
            np.corrcoef(candidate_scores, retriever_scores)[0, 1]
            if len(candidate_scores) > 1 else 0.0
        )

    return features


def compute_rank_features(
    candidate_ids: List[str],
    gold_ids: List[str],
    candidate_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute rank-based features."""
    features = {}

    gold_set = set(gold_ids)

    # Find ranks of gold sentences
    sorted_indices = np.argsort(-candidate_scores)
    gold_ranks = []

    for rank, idx in enumerate(sorted_indices):
        if candidate_ids[idx] in gold_set:
            gold_ranks.append(rank + 1)  # 1-indexed

    if gold_ranks:
        features["min_gold_rank"] = float(min(gold_ranks))
        features["max_gold_rank"] = float(max(gold_ranks))
        features["mean_gold_rank"] = float(np.mean(gold_ranks))
        features["mrr"] = float(1.0 / min(gold_ranks))
    else:
        features["min_gold_rank"] = float(len(candidate_ids) + 1)
        features["max_gold_rank"] = float(len(candidate_ids) + 1)
        features["mean_gold_rank"] = float(len(candidate_ids) + 1)
        features["mrr"] = 0.0

    # Recall at various K
    for k in [1, 3, 5, 10]:
        top_k_ids = [candidate_ids[sorted_indices[i]] for i in range(min(k, len(sorted_indices)))]
        hits = len(gold_set & set(top_k_ids))
        features[f"recall_at_{k}"] = float(hits / len(gold_set)) if gold_set else 0.0

    return features


def build_feature_store(
    groundtruth_path: Path,
    criteria_path: Path,
    cache_dir: Path,
    output_dir: Path,
    n_folds: int = 5,
    top_k_candidates: int = 20,
    seed: int = 42,
    retriever_name: str = "bge-m3",
    reranker_name: str = "jina-reranker-v3",
) -> Path:
    """Build feature store from existing cache."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    store_dir = output_dir / timestamp
    store_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building feature store in {store_dir}")

    # Load data
    logger.info("Loading groundtruth and criteria...")
    # Load groundtruth as DataFrame directly (load_groundtruth returns List[GroundTruthRow])
    gt_df = pd.read_csv(groundtruth_path)
    logger.info(f"Loaded {len(gt_df)} groundtruth rows")
    criteria = load_criteria(criteria_path)
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Get unique post_ids
    post_ids = gt_df["post_id"].unique().tolist()
    logger.info(f"Total posts: {len(post_ids)}")

    # Create folds
    np.random.seed(seed)
    np.random.shuffle(post_ids)
    fold_size = len(post_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(post_ids)
        folds.append(set(post_ids[start:end]))

    logger.info(f"Created {n_folds} folds with sizes: {[len(f) for f in folds]}")

    # Check for existing cache
    cache_file = cache_dir / f"{retriever_name}_{reranker_name}_cache.parquet"
    if cache_file.exists():
        logger.info(f"Loading existing cache from {cache_file}")
        cache_df = pd.read_parquet(cache_file)
    else:
        logger.warning(f"No cache found at {cache_file}, will compute features from groundtruth only")
        cache_df = None

    # Build feature store per fold
    all_records = []

    # Group by query (post_id, criterion)
    query_groups = gt_df.groupby(["post_id", "criterion"])

    for (post_id, criterion_id), group in tqdm(query_groups, desc="Building features"):
        # Determine fold
        fold_id = -1
        for i, fold_posts in enumerate(folds):
            if post_id in fold_posts:
                fold_id = i
                break

        # Get gold sentence IDs
        gold_rows = group[group["groundtruth"] == 1]
        gold_sent_ids = gold_rows["sent_uid"].tolist()
        has_evidence = len(gold_sent_ids) > 0

        # Get all candidates for this query
        all_candidates = group["sent_uid"].tolist()
        all_texts = group["sentence"].tolist() if "sentence" in group.columns else [""] * len(all_candidates)

        # Get scores from cache if available
        if cache_df is not None and len(cache_df) > 0:
            query_cache = cache_df[
                (cache_df["post_id"] == post_id) &
                (cache_df["criterion_id"] == criterion_id)
            ]
            if len(query_cache) > 0:
                candidate_scores = query_cache["reranker_score"].values[:top_k_candidates]
                retriever_scores = query_cache["retriever_score"].values[:top_k_candidates] if "retriever_score" in query_cache.columns else None
                candidate_ids = query_cache["sent_uid"].tolist()[:top_k_candidates]
            else:
                # No cache, use placeholder scores
                candidate_scores = np.zeros(min(top_k_candidates, len(all_candidates)))
                retriever_scores = None
                candidate_ids = all_candidates[:top_k_candidates]
        else:
            # No cache available, use placeholder scores
            candidate_scores = np.zeros(min(top_k_candidates, len(all_candidates)))
            retriever_scores = None
            candidate_ids = all_candidates[:top_k_candidates]

        # Compute derived features
        derived = compute_derived_features(
            candidate_scores,
            retriever_scores,
            top_k=5,
        )

        # Compute rank features
        rank_features = compute_rank_features(
            candidate_ids,
            gold_sent_ids,
            candidate_scores,
        )

        # Build record
        record = {
            # Identifiers
            "post_id": post_id,
            "query_id": f"{post_id}_{criterion_id}",
            "criterion_id": criterion_id,
            "criterion_text": criteria_map.get(criterion_id, ""),
            "fold_id": fold_id,

            # Labels
            "has_evidence": int(has_evidence),
            "n_gold_sentences": len(gold_sent_ids),
            "gold_sentence_ids": json.dumps(gold_sent_ids),

            # Candidates
            "n_candidates": len(candidate_ids),
            "candidate_ids": json.dumps(candidate_ids),
            "candidate_scores": json.dumps(candidate_scores.tolist() if isinstance(candidate_scores, np.ndarray) else candidate_scores),

            # Derived features
            **derived,

            # Rank features
            **rank_features,
        }

        all_records.append(record)

    # Create DataFrame
    feature_df = pd.DataFrame(all_records)
    logger.info(f"Built feature store with {len(feature_df)} queries")

    # Save per-fold parquets
    for fold_id in range(n_folds):
        fold_df = feature_df[feature_df["fold_id"] == fold_id]
        fold_path = store_dir / f"fold_{fold_id}.parquet"
        fold_df.to_parquet(fold_path, index=False)
        logger.info(f"  Saved fold {fold_id}: {len(fold_df)} queries to {fold_path}")

    # Save full dataset
    full_path = store_dir / "full_features.parquet"
    feature_df.to_parquet(full_path, index=False)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "n_folds": n_folds,
        "n_queries": len(feature_df),
        "n_posts": len(post_ids),
        "top_k_candidates": top_k_candidates,
        "seed": seed,
        "retriever": retriever_name,
        "reranker": reranker_name,
        "has_evidence_rate": float(feature_df["has_evidence"].mean()),
        "feature_columns": list(feature_df.columns),
    }
    with open(store_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary statistics
    logger.info("\n=== Feature Store Summary ===")
    logger.info(f"Total queries: {len(feature_df)}")
    logger.info(f"Has evidence rate: {feature_df['has_evidence'].mean():.2%}")
    logger.info(f"Mean gold sentences: {feature_df['n_gold_sentences'].mean():.2f}")
    logger.info(f"Feature columns: {len(feature_df.columns)}")

    return store_dir


def main():
    parser = argparse.ArgumentParser(description="Build feature store for supersearch")
    parser.add_argument("--groundtruth", type=str, default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--criteria", type=str, default="data/DSM5/MDD_Criteira.json")
    parser.add_argument("--cache_dir", type=str, default="data/cache/oof_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_store")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever", type=str, default="bge-m3")
    parser.add_argument("--reranker", type=str, default="jina-reranker-v3")
    args = parser.parse_args()

    store_path = build_feature_store(
        groundtruth_path=Path(args.groundtruth),
        criteria_path=Path(args.criteria),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        n_folds=args.n_folds,
        top_k_candidates=args.top_k,
        seed=args.seed,
        retriever_name=args.retriever,
        reranker_name=args.reranker,
    )

    print(f"\nFeature store created at: {store_path}")


if __name__ == "__main__":
    main()
