#!/usr/bin/env python3
"""Stage E: Exhaustive postprocessing HPO for no-evidence detection.

This script:
1. Runs the pipeline to extract actual reranker scores
2. Computes score distribution features for each query
3. Trains abstention classifiers using train split
4. Tunes thresholds on val split
5. Reports deployment metrics

Critical goal: Reduce false evidence rate from 100% to <30%
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.pipeline.three_stage import PipelineConfig, ThreeStagePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class QueryScoreFeatures:
    """Features extracted from score distribution for a query."""

    post_id: str
    criterion_id: str
    has_evidence: bool  # Ground truth
    n_candidates: int
    max_score: float
    min_score: float
    mean_score: float
    std_score: float
    score_range: float
    top1_minus_top2: float  # Gap between top scores
    top1_minus_mean: float
    entropy: float  # Score distribution entropy
    gini: float  # Score inequality
    top3_mean: float
    bottom3_mean: float


def compute_score_entropy(scores: np.ndarray) -> float:
    """Compute entropy of score distribution."""
    if len(scores) < 2:
        return 0.0
    # Normalize to probabilities
    probs = np.exp(scores - np.max(scores))  # Softmax-like
    probs = probs / (probs.sum() + 1e-10)
    probs = probs + 1e-10  # Avoid log(0)
    return float(-np.sum(probs * np.log(probs)))


def compute_gini(scores: np.ndarray) -> float:
    """Compute Gini coefficient of score distribution."""
    if len(scores) < 2:
        return 0.0
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_scores) - (n + 1) * np.sum(sorted_scores)) / (n * np.sum(sorted_scores) + 1e-10))


def extract_query_features(
    post_id: str,
    criterion_id: str,
    scores: List[float],
    has_evidence: bool,
) -> QueryScoreFeatures:
    """Extract features from score distribution for abstention classification."""
    scores_arr = np.array(scores) if scores else np.array([0.0])

    if len(scores_arr) == 0:
        scores_arr = np.array([0.0])

    sorted_scores = np.sort(scores_arr)[::-1]  # Descending

    features = QueryScoreFeatures(
        post_id=post_id,
        criterion_id=criterion_id,
        has_evidence=has_evidence,
        n_candidates=len(scores_arr),
        max_score=float(sorted_scores[0]),
        min_score=float(sorted_scores[-1]),
        mean_score=float(np.mean(scores_arr)),
        std_score=float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0,
        score_range=float(sorted_scores[0] - sorted_scores[-1]),
        top1_minus_top2=float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0,
        top1_minus_mean=float(sorted_scores[0] - np.mean(scores_arr)),
        entropy=compute_score_entropy(scores_arr),
        gini=compute_gini(scores_arr),
        top3_mean=float(np.mean(sorted_scores[:3])) if len(sorted_scores) >= 3 else float(sorted_scores[0]),
        bottom3_mean=float(np.mean(sorted_scores[-3:])) if len(sorted_scores) >= 3 else float(sorted_scores[-1]),
    )

    return features


def features_to_array(features: QueryScoreFeatures) -> np.ndarray:
    """Convert features to array for classification."""
    return np.array([
        features.n_candidates,
        features.max_score,
        features.min_score,
        features.mean_score,
        features.std_score,
        features.score_range,
        features.top1_minus_top2,
        features.top1_minus_mean,
        features.entropy,
        features.gini,
        features.top3_mean,
        features.bottom3_mean,
    ])


FEATURE_NAMES = [
    "n_candidates",
    "max_score",
    "min_score",
    "mean_score",
    "std_score",
    "score_range",
    "top1_minus_top2",
    "top1_minus_mean",
    "entropy",
    "gini",
    "top3_mean",
    "bottom3_mean",
]


def run_pipeline_and_extract_features(
    cfg: dict,
    split_name: str = "val",
) -> List[QueryScoreFeatures]:
    """Run the pipeline and extract score features for each query."""

    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    eval_posts = set(splits[split_name])

    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    cache_dir = Path(cfg["paths"]["cache_dir"])

    # Support both old (top_k_colbert) and new (top_k_rerank) config options
    top_k_rerank = cfg["retriever"].get("top_k_rerank")
    if top_k_rerank is None:
        top_k_rerank = cfg["retriever"].get("top_k_colbert", cfg["retriever"]["top_k_retriever"])

    pipeline_cfg = PipelineConfig(
        bge_model=cfg["models"]["bge_m3"],
        jina_model=cfg["models"]["jina_v3"],
        bge_query_max_length=cfg["models"].get("bge_query_max_length", 128),
        bge_passage_max_length=cfg["models"].get("bge_passage_max_length", 256),
        bge_use_fp16=cfg["models"].get("bge_use_fp16", True),
        bge_batch_size=cfg["models"].get("bge_batch_size", 64),
        dense_weight=cfg["retriever"].get("dense_weight", 0.7),
        sparse_weight=cfg["retriever"].get("sparse_weight", 0.3),
        colbert_weight=cfg["retriever"].get("colbert_weight", 0.0),
        fusion_method=cfg["retriever"].get("fusion_method", "weighted_sum"),
        score_normalization=cfg["retriever"].get("score_normalization", "none"),
        rrf_k=cfg["retriever"].get("rrf_k", 60),
        use_sparse=cfg["retriever"].get("use_sparse", True),
        use_colbert=cfg["retriever"].get("use_colbert", True),
        top_k_retriever=cfg["retriever"]["top_k_retriever"],
        top_k_colbert=top_k_rerank,
        top_k_rerank=top_k_rerank,
        top_k_final=cfg["retriever"]["top_k_final"],
        reranker_max_length=cfg["models"].get("reranker_max_length", 512),
        reranker_chunk_size=cfg["models"].get("reranker_chunk_size", 64),
        reranker_dtype=cfg["models"].get("reranker_dtype", "auto"),
        reranker_use_listwise=cfg["models"].get("reranker_use_listwise", True),
        device=cfg.get("device"),
    )
    pipeline = ThreeStagePipeline(sentences, cache_dir, pipeline_cfg, rebuild_cache=False)

    # Group groundtruth by (post_id, criterion)
    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in eval_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    all_features = []
    logger.info(f"Processing {len(grouped)} queries from {split_name} split...")

    for (post_id, criterion_id), rows in tqdm(sorted(grouped.items()), desc=f"Extracting {split_name} features"):
        query = criteria_map.get(criterion_id)
        if query is None:
            continue

        has_evidence = any(r.groundtruth == 1 for r in rows)

        # Get reranked results with scores
        reranked = pipeline.retrieve(query=query, post_id=post_id)
        scores = [score for _, _, score in reranked]

        features = extract_query_features(post_id, criterion_id, scores, has_evidence)
        all_features.append(features)

    return all_features


def train_abstention_classifier(
    train_features: List[QueryScoreFeatures],
    classifier_type: str = "logistic",
) -> Tuple[object, StandardScaler]:
    """Train an abstention classifier."""

    X_train = np.array([features_to_array(f) for f in train_features])
    y_train = np.array([f.has_evidence for f in train_features])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if classifier_type == "logistic":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    elif classifier_type == "gbm":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    clf.fit(X_train_scaled, y_train)

    return clf, scaler


def evaluate_abstention(
    features: List[QueryScoreFeatures],
    clf: object,
    scaler: StandardScaler,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate abstention classifier."""

    X = np.array([features_to_array(f) for f in features])
    y_true = np.array([f.has_evidence for f in features])

    X_scaled = scaler.transform(X)
    y_proba = clf.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    # Note: has_evidence=True means should return evidence
    # has_evidence=False means should abstain

    # TP = correctly predict has_evidence
    # FP = predict has_evidence when should abstain (FALSE EVIDENCE!)
    # FN = abstain when there is evidence
    # TN = correctly abstain

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # False evidence
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_with_evidence = int(y_true.sum())
    n_empty = int((~y_true.astype(bool)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # False evidence rate = FP / n_empty
    false_evidence_rate = fp / n_empty if n_empty > 0 else 0.0

    # Empty detection metrics (inverse perspective)
    empty_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    empty_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    empty_f1 = 2 * empty_precision * empty_recall / (empty_precision + empty_recall) if (empty_precision + empty_recall) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.5

    return {
        "threshold": threshold,
        "n_queries": len(features),
        "n_with_evidence": n_with_evidence,
        "n_empty": n_empty,
        "empty_prevalence": n_empty / len(features),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_evidence_rate": false_evidence_rate,
        "false_evidence_count": fp,
        "empty_precision": empty_precision,
        "empty_recall": empty_recall,
        "empty_f1": empty_f1,
        "auc": auc,
        "accuracy": (tp + tn) / len(features),
    }


def find_optimal_threshold(
    features: List[QueryScoreFeatures],
    clf: object,
    scaler: StandardScaler,
    target_metric: str = "empty_f1",
    thresholds: List[float] = None,
) -> Tuple[float, Dict]:
    """Find optimal threshold for abstention."""

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17).tolist()

    best_threshold = 0.5
    best_score = 0.0
    best_metrics = None

    for t in thresholds:
        metrics = evaluate_abstention(features, clf, scaler, threshold=t)
        score = metrics.get(target_metric, 0.0)

        if score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Stage E: Postprocessing HPO for abstention")
    parser.add_argument("--config", default="configs/locked_best_stageE_deploy.yaml")
    parser.add_argument("--output_dir", default="outputs/stageE/hpo")
    parser.add_argument("--target_metric", default="empty_f1", choices=["empty_f1", "false_evidence_rate", "f1"])
    parser.add_argument("--classifiers", nargs="+", default=["logistic", "rf", "gbm"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE E: POSTPROCESSING HPO")
    logger.info("=" * 70)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract features from train split (for training classifier)
    logger.info("\nExtracting features from TRAIN split...")
    train_features = run_pipeline_and_extract_features(cfg, split_name="train")
    logger.info(f"Train: {len(train_features)} queries, {sum(1 for f in train_features if f.has_evidence)} with evidence")

    # Extract features from val split (for threshold tuning)
    logger.info("\nExtracting features from VAL split...")
    cfg_val = cfg.copy()
    cfg_val["evaluation"] = cfg_val.get("evaluation", {})
    cfg_val["evaluation"]["split"] = "val"
    val_features = run_pipeline_and_extract_features(cfg, split_name="val")
    logger.info(f"Val: {len(val_features)} queries, {sum(1 for f in val_features if f.has_evidence)} with evidence")

    # Save features for analysis
    train_df = pd.DataFrame([
        {**{"post_id": f.post_id, "criterion_id": f.criterion_id, "has_evidence": f.has_evidence},
         **dict(zip(FEATURE_NAMES, features_to_array(f)))}
        for f in train_features
    ])
    val_df = pd.DataFrame([
        {**{"post_id": f.post_id, "criterion_id": f.criterion_id, "has_evidence": f.has_evidence},
         **dict(zip(FEATURE_NAMES, features_to_array(f)))}
        for f in val_features
    ])
    train_df.to_csv(output_dir / "train_features.csv", index=False)
    val_df.to_csv(output_dir / "val_features.csv", index=False)

    # Train and evaluate classifiers
    results = {}
    best_classifier = None
    best_overall_score = 0.0
    best_overall_threshold = 0.5
    best_overall_metrics = None
    best_clf = None
    best_scaler = None

    for clf_type in args.classifiers:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training {clf_type} classifier...")

        clf, scaler = train_abstention_classifier(train_features, classifier_type=clf_type)

        # Find optimal threshold on val
        opt_threshold, opt_metrics = find_optimal_threshold(
            val_features, clf, scaler, target_metric=args.target_metric
        )

        results[clf_type] = {
            "optimal_threshold": opt_threshold,
            "val_metrics": opt_metrics,
        }

        logger.info(f"  Optimal threshold: {opt_threshold:.2f}")
        logger.info(f"  False evidence rate: {opt_metrics['false_evidence_rate']:.2%}")
        logger.info(f"  Empty F1: {opt_metrics['empty_f1']:.3f}")
        logger.info(f"  Evidence recall: {opt_metrics['recall']:.3f}")

        if opt_metrics[args.target_metric] > best_overall_score:
            best_overall_score = opt_metrics[args.target_metric]
            best_classifier = clf_type
            best_overall_threshold = opt_threshold
            best_overall_metrics = opt_metrics
            best_clf = clf
            best_scaler = scaler

    # Report best results
    logger.info("\n" + "=" * 70)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Classifier: {best_classifier}")
    logger.info(f"Threshold: {best_overall_threshold:.2f}")
    logger.info(f"Target metric ({args.target_metric}): {best_overall_score:.4f}")
    logger.info(f"\nDeployment metrics:")
    logger.info(f"  Empty prevalence: {best_overall_metrics['empty_prevalence']:.2%}")
    logger.info(f"  False evidence rate: {best_overall_metrics['false_evidence_rate']:.2%} (target: <30%)")
    logger.info(f"  False evidence count: {best_overall_metrics['false_evidence_count']}")
    logger.info(f"  Empty detection F1: {best_overall_metrics['empty_f1']:.3f}")
    logger.info(f"  Evidence precision: {best_overall_metrics['precision']:.3f}")
    logger.info(f"  Evidence recall: {best_overall_metrics['recall']:.3f}")

    # Save results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "target_metric": args.target_metric,
        "best_classifier": best_classifier,
        "best_threshold": best_overall_threshold,
        "best_val_metrics": best_overall_metrics,
        "all_classifiers": results,
    }

    with open(output_dir / "hpo_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Feature importance (for logistic/rf)
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        importance_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(output_dir / "feature_importances.csv", index=False)
        logger.info("\nTop features:")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    elif hasattr(best_clf, "coef_"):
        coefs = np.abs(best_clf.coef_[0])
        importance_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": coefs,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(output_dir / "feature_importances.csv", index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
