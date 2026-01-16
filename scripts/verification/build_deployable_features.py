#!/usr/bin/env python3
"""Build deployable feature store (NO label leakage).

This script creates feature stores with ONLY deployment-safe features.
Features are explicitly declared with provenance information.

Two modes:
- mode="deployable": ONLY features computable at inference time (for modeling)
- mode="evaluation": Includes ground-truth metrics (ONLY for reporting)

Feature Provenance Rules:
------------------------
ALLOWED inputs (inference-time):
- retriever_scores: Dense/sparse/ColBERT scores from retriever
- reranker_scores: Logits/probabilities from reranker
- candidate_count: Number of candidates per query
- candidate_ranks: Positions in ranked list (1-indexed)
- text_stats: Sentence lengths, overlap stats (if available)

FORBIDDEN inputs (ground-truth, NOT available at inference):
- gold_sentence_ids
- relevance_labels
- any evaluation metric (MRR, Recall@K, nDCG, etc.)
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.data.io import load_criteria

logger = get_logger(__name__)


class FeatureMode(Enum):
    """Feature computation modes."""
    DEPLOYABLE = "deployable"  # Only inference-time features
    EVALUATION = "evaluation"  # Includes ground-truth metrics (for reporting only)


@dataclass
class FeatureSpec:
    """Specification for a feature with provenance."""
    name: str
    description: str
    inputs: List[str]  # Raw inputs this feature depends on
    mode: FeatureMode  # Whether deployable or evaluation-only
    compute_fn: str    # Name of computation function


# ============================================================================
# Feature Registry - Explicit provenance for every feature
# ============================================================================

DEPLOYABLE_FEATURES = [
    # Score statistics (from reranker)
    FeatureSpec("max_reranker_score", "Maximum reranker score", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_max"),
    FeatureSpec("second_reranker_score", "Second highest reranker score", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_second"),
    FeatureSpec("mean_reranker_score", "Mean of top-K reranker scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_mean_topk"),
    FeatureSpec("std_reranker_score", "Std of top-K reranker scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_std_topk"),
    FeatureSpec("median_reranker_score", "Median of top-K reranker scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_median"),

    # Gap features
    FeatureSpec("top1_top2_gap", "Score gap between rank 1 and 2", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_top1_top2_gap"),
    FeatureSpec("top1_top5_gap", "Score gap between rank 1 and 5", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_top1_top5_gap"),
    FeatureSpec("top1_mean_gap", "Gap between top1 and mean of rest", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_top1_mean_gap"),

    # Sum/concentration features
    FeatureSpec("topk_sum_score", "Sum of top-K scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_sum_topk"),
    FeatureSpec("top3_concentration", "Fraction of score mass in top 3", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_top3_concentration"),
    FeatureSpec("top5_concentration", "Fraction of score mass in top 5", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_top5_concentration"),

    # Distribution features
    FeatureSpec("score_range", "Max - min score", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_range"),
    FeatureSpec("score_skewness", "Skewness of score distribution", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_skewness"),
    FeatureSpec("score_kurtosis", "Kurtosis of score distribution", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_kurtosis"),

    # Entropy features (uncertainty proxies)
    FeatureSpec("entropy_top5", "Entropy of softmax over top-5 scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_entropy_top5"),
    FeatureSpec("entropy_top10", "Entropy of softmax over top-10 scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_entropy_top10"),
    FeatureSpec("entropy_full", "Entropy of softmax over all scores", ["reranker_scores"], FeatureMode.DEPLOYABLE, "compute_entropy_full"),

    # Retriever features (if available)
    FeatureSpec("max_retriever_score", "Maximum retriever score", ["retriever_scores"], FeatureMode.DEPLOYABLE, "compute_max"),
    FeatureSpec("mean_retriever_score", "Mean of top-K retriever scores", ["retriever_scores"], FeatureMode.DEPLOYABLE, "compute_mean_topk"),
    FeatureSpec("retriever_reranker_corr", "Correlation between retriever and reranker", ["retriever_scores", "reranker_scores"], FeatureMode.DEPLOYABLE, "compute_correlation"),

    # Count features
    FeatureSpec("n_candidates", "Number of candidates", ["candidate_count"], FeatureMode.DEPLOYABLE, "identity"),

    # Proxy rank features (SoftMRR - no gold labels!)
    FeatureSpec("soft_mrr", "SoftMRR: sum_i(p_i / i)", ["reranker_probs"], FeatureMode.DEPLOYABLE, "compute_soft_mrr"),
    FeatureSpec("mass_at_1", "Cumulative prob mass at k=1", ["reranker_probs"], FeatureMode.DEPLOYABLE, "compute_mass_at_1"),
    FeatureSpec("mass_at_3", "Cumulative prob mass at k=3", ["reranker_probs"], FeatureMode.DEPLOYABLE, "compute_mass_at_3"),
    FeatureSpec("mass_at_5", "Cumulative prob mass at k=5", ["reranker_probs"], FeatureMode.DEPLOYABLE, "compute_mass_at_5"),
    FeatureSpec("mass_at_10", "Cumulative prob mass at k=10", ["reranker_probs"], FeatureMode.DEPLOYABLE, "compute_mass_at_10"),
]

EVALUATION_FEATURES = [
    # These require gold labels - ONLY for reporting, NEVER for modeling
    FeatureSpec("mrr", "Mean Reciprocal Rank (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_mrr"),
    FeatureSpec("recall_at_1", "Recall@1 (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_recall_at_k"),
    FeatureSpec("recall_at_3", "Recall@3 (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_recall_at_k"),
    FeatureSpec("recall_at_5", "Recall@5 (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_recall_at_k"),
    FeatureSpec("recall_at_10", "Recall@10 (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_recall_at_k"),
    FeatureSpec("min_gold_rank", "Rank of first gold (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_gold_ranks"),
    FeatureSpec("mean_gold_rank", "Mean rank of gold (EVAL ONLY)", ["gold_ids", "reranker_scores"], FeatureMode.EVALUATION, "compute_gold_ranks"),
]


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    if len(x) == 0:
        return np.array([])
    exp_x = np.exp(x - np.max(x))
    return exp_x / (exp_x.sum() + 1e-10)


def compute_deployable_features(
    reranker_scores: np.ndarray,
    retriever_scores: Optional[np.ndarray] = None,
    n_candidates: int = 0,
    top_k: int = 5,
) -> Dict[str, float]:
    """Compute ONLY deployable features (no gold labels)."""
    features = {}

    if len(reranker_scores) == 0:
        # Return zeros for empty
        return {spec.name: 0.0 for spec in DEPLOYABLE_FEATURES}

    # Sort scores descending
    sorted_scores = np.sort(reranker_scores)[::-1]
    n = len(sorted_scores)

    # Basic statistics
    features["max_reranker_score"] = float(sorted_scores[0])
    features["second_reranker_score"] = float(sorted_scores[1]) if n > 1 else 0.0
    features["mean_reranker_score"] = float(np.mean(sorted_scores[:min(top_k, n)]))
    features["std_reranker_score"] = float(np.std(sorted_scores[:min(top_k, n)])) if n > 1 else 0.0
    features["median_reranker_score"] = float(np.median(sorted_scores[:min(top_k, n)]))

    # Gap features
    features["top1_top2_gap"] = features["max_reranker_score"] - features["second_reranker_score"]
    features["top1_top5_gap"] = features["max_reranker_score"] - (float(sorted_scores[4]) if n > 4 else 0.0)
    rest_mean = float(np.mean(sorted_scores[1:])) if n > 1 else 0.0
    features["top1_mean_gap"] = features["max_reranker_score"] - rest_mean

    # Sum/concentration
    features["topk_sum_score"] = float(np.sum(sorted_scores[:min(top_k, n)]))
    total_mass = float(np.sum(np.abs(sorted_scores))) + 1e-10
    features["top3_concentration"] = float(np.sum(np.abs(sorted_scores[:min(3, n)]))) / total_mass
    features["top5_concentration"] = float(np.sum(np.abs(sorted_scores[:min(5, n)]))) / total_mass

    # Distribution
    features["score_range"] = float(sorted_scores[0] - sorted_scores[-1]) if n > 1 else 0.0
    if n > 2 and features["std_reranker_score"] > 0:
        mean_val = np.mean(sorted_scores)
        features["score_skewness"] = float(np.mean(((sorted_scores - mean_val) / features["std_reranker_score"]) ** 3))
        features["score_kurtosis"] = float(np.mean(((sorted_scores - mean_val) / features["std_reranker_score"]) ** 4) - 3)
    else:
        features["score_skewness"] = 0.0
        features["score_kurtosis"] = 0.0

    # Entropy (uncertainty)
    top5_probs = softmax(sorted_scores[:min(5, n)])
    top10_probs = softmax(sorted_scores[:min(10, n)])
    full_probs = softmax(sorted_scores)
    features["entropy_top5"] = float(scipy_entropy(top5_probs)) if len(top5_probs) > 0 else 0.0
    features["entropy_top10"] = float(scipy_entropy(top10_probs)) if len(top10_probs) > 0 else 0.0
    features["entropy_full"] = float(scipy_entropy(full_probs)) if len(full_probs) > 0 else 0.0

    # Retriever features
    if retriever_scores is not None and len(retriever_scores) > 0:
        sorted_ret = np.sort(retriever_scores)[::-1]
        features["max_retriever_score"] = float(sorted_ret[0])
        features["mean_retriever_score"] = float(np.mean(sorted_ret[:min(top_k, len(sorted_ret))]))
        if len(reranker_scores) > 1 and len(retriever_scores) == len(reranker_scores):
            corr = np.corrcoef(reranker_scores, retriever_scores)[0, 1]
            features["retriever_reranker_corr"] = float(corr) if not np.isnan(corr) else 0.0
        else:
            features["retriever_reranker_corr"] = 0.0
    else:
        features["max_retriever_score"] = 0.0
        features["mean_retriever_score"] = 0.0
        features["retriever_reranker_corr"] = 0.0

    # Count
    features["n_candidates"] = float(n_candidates if n_candidates > 0 else n)

    # Proxy rank features (SoftMRR - using predicted probs, NOT gold labels)
    probs = softmax(sorted_scores)
    features["soft_mrr"] = float(np.sum(probs / (np.arange(len(probs)) + 1)))
    features["mass_at_1"] = float(probs[0]) if len(probs) > 0 else 0.0
    features["mass_at_3"] = float(np.sum(probs[:min(3, len(probs))])) if len(probs) > 0 else 0.0
    features["mass_at_5"] = float(np.sum(probs[:min(5, len(probs))])) if len(probs) > 0 else 0.0
    features["mass_at_10"] = float(np.sum(probs[:min(10, len(probs))])) if len(probs) > 0 else 0.0

    return features


def compute_evaluation_features(
    candidate_ids: List[str],
    gold_ids: List[str],
    reranker_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation-only features (requires gold labels).

    WARNING: These features must NEVER be used for model training.
    They are for evaluation reporting only.
    """
    features = {}
    gold_set = set(gold_ids)

    if len(reranker_scores) == 0 or len(candidate_ids) == 0:
        return {
            "mrr": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "min_gold_rank": float(len(candidate_ids) + 1),
            "mean_gold_rank": float(len(candidate_ids) + 1),
        }

    # Sort by scores
    sorted_indices = np.argsort(-reranker_scores)
    gold_ranks = []

    for rank, idx in enumerate(sorted_indices):
        if idx < len(candidate_ids) and candidate_ids[idx] in gold_set:
            gold_ranks.append(rank + 1)

    if gold_ranks:
        features["mrr"] = float(1.0 / min(gold_ranks))
        features["min_gold_rank"] = float(min(gold_ranks))
        features["mean_gold_rank"] = float(np.mean(gold_ranks))
    else:
        features["mrr"] = 0.0
        features["min_gold_rank"] = float(len(candidate_ids) + 1)
        features["mean_gold_rank"] = float(len(candidate_ids) + 1)

    # Recall at K
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
    mode: FeatureMode = FeatureMode.DEPLOYABLE,
    n_folds: int = 5,
    top_k_candidates: int = 20,
    seed: int = 42,
    retriever_name: str = "bge-m3",
    reranker_name: str = "jina-reranker-v3",
) -> Path:
    """Build feature store with explicit mode control."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    store_dir = output_dir / f"{timestamp}_{mode.value}"
    store_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building {mode.value.upper()} feature store in {store_dir}")

    # Load data
    gt_df = pd.read_csv(groundtruth_path)
    logger.info(f"Loaded {len(gt_df)} groundtruth rows")
    criteria = load_criteria(criteria_path)
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Get unique post_ids and create folds
    post_ids = gt_df["post_id"].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(post_ids)
    fold_size = len(post_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(post_ids)
        folds.append(set(post_ids[start:end]))

    # Check for cache
    cache_file = cache_dir / f"{retriever_name}_{reranker_name}_cache.parquet"
    cache_df = pd.read_parquet(cache_file) if cache_file.exists() else None

    # Build features
    all_records = []
    query_groups = gt_df.groupby(["post_id", "criterion"])

    for (post_id, criterion_id), group in tqdm(query_groups, desc=f"Building {mode.value} features"):
        # Determine fold
        fold_id = -1
        for i, fold_posts in enumerate(folds):
            if post_id in fold_posts:
                fold_id = i
                break

        # Get labels
        gold_rows = group[group["groundtruth"] == 1]
        gold_sent_ids = gold_rows["sent_uid"].tolist()
        has_evidence = len(gold_sent_ids) > 0

        # Get candidates
        all_candidates = group["sent_uid"].tolist()

        # Get scores from cache
        if cache_df is not None:
            query_cache = cache_df[
                (cache_df["post_id"] == post_id) &
                (cache_df["criterion_id"] == criterion_id)
            ]
            if len(query_cache) > 0:
                candidate_scores = query_cache["reranker_score"].values[:top_k_candidates]
                retriever_scores = query_cache["retriever_score"].values[:top_k_candidates] if "retriever_score" in query_cache.columns else None
                candidate_ids = query_cache["sent_uid"].tolist()[:top_k_candidates]
            else:
                candidate_scores = np.zeros(min(top_k_candidates, len(all_candidates)))
                retriever_scores = None
                candidate_ids = all_candidates[:top_k_candidates]
        else:
            candidate_scores = np.zeros(min(top_k_candidates, len(all_candidates)))
            retriever_scores = None
            candidate_ids = all_candidates[:top_k_candidates]

        # Compute DEPLOYABLE features (always)
        deployable = compute_deployable_features(
            candidate_scores,
            retriever_scores,
            n_candidates=len(candidate_ids),
        )

        # Build record
        record = {
            "post_id": post_id,
            "query_id": f"{post_id}_{criterion_id}",
            "criterion_id": criterion_id,
            "criterion_text": criteria_map.get(criterion_id, ""),
            "fold_id": fold_id,
            "has_evidence": int(has_evidence),
            "n_gold_sentences": len(gold_sent_ids),
            "gold_sentence_ids": json.dumps(gold_sent_ids),
            "n_candidates": len(candidate_ids),
            "candidate_ids": json.dumps(candidate_ids),
            **deployable,
        }

        # Add EVALUATION features only if requested (for reporting, NOT modeling)
        if mode == FeatureMode.EVALUATION:
            eval_features = compute_evaluation_features(
                candidate_ids, gold_sent_ids, candidate_scores
            )
            # Prefix with "eval_" to make it clear these are not for modeling
            record.update({f"eval_{k}": v for k, v in eval_features.items()})

        all_records.append(record)

    # Create DataFrame
    feature_df = pd.DataFrame(all_records)
    logger.info(f"Built feature store with {len(feature_df)} queries")

    # Save
    full_path = store_dir / "full_features.parquet"
    feature_df.to_parquet(full_path, index=False)

    # Save per-fold
    for fold_id in range(n_folds):
        fold_df = feature_df[feature_df["fold_id"] == fold_id]
        fold_df.to_parquet(store_dir / f"fold_{fold_id}.parquet", index=False)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "mode": mode.value,
        "n_folds": n_folds,
        "n_queries": len(feature_df),
        "n_posts": len(post_ids),
        "seed": seed,
        "retriever": retriever_name,
        "reranker": reranker_name,
        "deployable_features": [f.name for f in DEPLOYABLE_FEATURES],
        "evaluation_features": [f.name for f in EVALUATION_FEATURES] if mode == FeatureMode.EVALUATION else [],
        "feature_columns": list(feature_df.columns),
    }
    with open(store_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save feature provenance
    provenance = {
        "deployable": [{"name": f.name, "description": f.description, "inputs": f.inputs} for f in DEPLOYABLE_FEATURES],
        "evaluation": [{"name": f.name, "description": f.description, "inputs": f.inputs} for f in EVALUATION_FEATURES],
    }
    with open(store_dir / "feature_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    logger.info(f"\n=== {mode.value.upper()} Feature Store Summary ===")
    logger.info(f"Total queries: {len(feature_df)}")
    logger.info(f"Has evidence rate: {feature_df['has_evidence'].mean():.2%}")
    logger.info(f"Deployable features: {len([f for f in DEPLOYABLE_FEATURES])}")
    if mode == FeatureMode.EVALUATION:
        logger.info(f"Evaluation features: {len([f for f in EVALUATION_FEATURES])}")

    return store_dir


def main():
    parser = argparse.ArgumentParser(description="Build deployable feature store (no leakage)")
    parser.add_argument("--groundtruth", type=str, default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--criteria", type=str, default="data/DSM5/MDD_Criteira.json")
    parser.add_argument("--cache_dir", type=str, default="data/cache/oof_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_store_clean")
    parser.add_argument("--mode", type=str, choices=["deployable", "evaluation"], default="deployable")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever", type=str, default="bge-m3")
    parser.add_argument("--reranker", type=str, default="jina-reranker-v3")
    args = parser.parse_args()

    mode = FeatureMode(args.mode)

    store_path = build_feature_store(
        groundtruth_path=Path(args.groundtruth),
        criteria_path=Path(args.criteria),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        mode=mode,
        n_folds=args.n_folds,
        seed=args.seed,
        retriever_name=args.retriever,
        reranker_name=args.reranker,
    )

    print(f"\n{mode.value.upper()} feature store created at: {store_path}")


if __name__ == "__main__":
    main()
