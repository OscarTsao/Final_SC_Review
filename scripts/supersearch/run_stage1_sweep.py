#!/usr/bin/env python3
"""Run Stage-1 NE model sweep.

Evaluates multiple Stage-1 models on feature store:
- Threshold rules: max_score, gap, margin, calibrated_prob
- Linear models: LogisticRegression (with class weights), RidgeClassifier
- Tree/GBM: RandomForest, XGBoost or LightGBM if available (else sklearn HistGBDT)
- Small MLP (PyTorch) as optional

All models are tuned for low-FPR regime:
- Maximize TPR subject to FPR<=alpha (alpha in {0.03, 0.05})
- Report TPR@3%FPR and TPR@5%FPR, plus AUROC/AUPRC

Output: Pareto frontier table and best Stage-1 configs.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.supersearch.registry import (
    PluginRegistry, NEStage1Model, EvaluationMetrics
)

logger = get_logger(__name__)


# Feature column groups for different models
SCORE_FEATURES = [
    "max_reranker_score", "second_reranker_score", "mean_reranker_score",
    "std_reranker_score", "top1_top2_gap", "top1_top5_gap", "topk_sum_score",
]

ENTROPY_FEATURES = ["entropy_top5", "entropy_top10"]

RANK_FEATURES = [
    "min_gold_rank", "max_gold_rank", "mean_gold_rank", "mrr",
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
]

DISTRIBUTION_FEATURES = ["score_range", "score_skewness"]

ALL_NUMERIC_FEATURES = SCORE_FEATURES + ENTROPY_FEATURES + DISTRIBUTION_FEATURES


def get_model_configs() -> List[Dict[str, Any]]:
    """Get all Stage-1 model configurations to sweep."""
    configs = []

    # Threshold models (fast baselines)
    configs.append({
        "name": "threshold_max_score",
        "model_type": "threshold_max_score",
        "config": {},
    })

    configs.append({
        "name": "threshold_gap",
        "model_type": "threshold_gap",
        "config": {},
    })

    # Logistic regression variants
    for C in [0.01, 0.1, 1.0, 10.0]:
        configs.append({
            "name": f"logreg_C{C}",
            "model_type": "logistic_regression",
            "config": {"C": C, "class_weight": "balanced"},
        })

    # Logistic regression with specific feature subsets
    configs.append({
        "name": "logreg_scores_only",
        "model_type": "logistic_regression",
        "config": {"feature_cols": SCORE_FEATURES, "C": 1.0},
    })

    configs.append({
        "name": "logreg_scores_entropy",
        "model_type": "logistic_regression",
        "config": {"feature_cols": SCORE_FEATURES + ENTROPY_FEATURES, "C": 1.0},
    })

    # Random forest variants
    for n_est in [50, 100, 200]:
        for max_depth in [5, 10, 20]:
            configs.append({
                "name": f"rf_n{n_est}_d{max_depth}",
                "model_type": "random_forest",
                "config": {
                    "n_estimators": n_est,
                    "max_depth": max_depth,
                    "class_weight": "balanced",
                },
            })

    # XGBoost variants
    for n_est in [50, 100]:
        for max_depth in [3, 6]:
            for lr in [0.05, 0.1]:
                configs.append({
                    "name": f"xgb_n{n_est}_d{max_depth}_lr{lr}",
                    "model_type": "xgboost",
                    "config": {
                        "n_estimators": n_est,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                    },
                })

    return configs


def run_cross_validation(
    model_config: Dict[str, Any],
    feature_store_dir: Path,
    n_folds: int = 5,
) -> Tuple[Dict[str, float], List[EvaluationMetrics]]:
    """Run cross-validation for a single model config."""

    all_metrics = []
    all_probs = []
    all_labels = []

    for fold_id in range(n_folds):
        # Load fold data
        fold_path = feature_store_dir / f"fold_{fold_id}.parquet"
        if not fold_path.exists():
            logger.warning(f"Fold {fold_id} not found at {fold_path}")
            continue

        fold_df = pd.read_parquet(fold_path)

        # Train on other folds, test on this fold
        train_dfs = []
        for other_fold in range(n_folds):
            if other_fold != fold_id:
                other_path = feature_store_dir / f"fold_{other_fold}.parquet"
                if other_path.exists():
                    train_dfs.append(pd.read_parquet(other_path))

        if not train_dfs:
            continue

        train_df = pd.concat(train_dfs, ignore_index=True)

        # Filter to numeric features only
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove non-feature columns
        exclude_cols = ["fold_id", "has_evidence", "n_gold_sentences", "n_candidates"]
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        train_features = train_df[feature_cols].fillna(0)
        train_labels = train_df["has_evidence"].values

        test_features = fold_df[feature_cols].fillna(0)
        test_labels = fold_df["has_evidence"].values

        # Create and train model
        model_cls = PluginRegistry.get_stage1(model_config["model_type"])
        model = model_cls(model_config.get("config", {}))

        try:
            model.fit(train_features, train_labels)
            metrics = model.evaluate(test_features, test_labels)
            probs = model.predict_proba(test_features)

            all_metrics.append(metrics)
            all_probs.extend(probs.tolist())
            all_labels.extend(test_labels.tolist())
        except Exception as e:
            logger.warning(f"Model {model_config['name']} failed on fold {fold_id}: {e}")
            continue

    if not all_metrics:
        return {}, []

    # Aggregate metrics
    aggregated = {
        "tpr": np.mean([m.tpr for m in all_metrics]),
        "fpr": np.mean([m.fpr for m in all_metrics]),
        "precision": np.mean([m.precision for m in all_metrics]),
        "f1": np.mean([m.f1 for m in all_metrics]),
        "auroc": np.mean([m.auroc for m in all_metrics]),
        "auprc": np.mean([m.auprc for m in all_metrics]),
        "tpr_at_fpr_3pct": np.mean([m.tpr_at_fpr_3pct for m in all_metrics]),
        "tpr_at_fpr_5pct": np.mean([m.tpr_at_fpr_5pct for m in all_metrics]),
        "threshold_at_fpr_5pct": np.mean([m.threshold_at_fpr_5pct for m in all_metrics]),
    }

    # Also compute overall metrics from aggregated predictions
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    from sklearn.metrics import roc_curve, roc_auc_score

    if len(np.unique(all_labels)) > 1:
        fpr_curve, tpr_curve, thresholds = roc_curve(all_labels, all_probs)

        # Find TPR at specific FPR levels (overall)
        idx_3pct = np.where(fpr_curve <= 0.03)[0]
        idx_5pct = np.where(fpr_curve <= 0.05)[0]

        aggregated["overall_tpr_at_fpr_3pct"] = float(tpr_curve[idx_3pct[-1]]) if len(idx_3pct) > 0 else 0.0
        aggregated["overall_tpr_at_fpr_5pct"] = float(tpr_curve[idx_5pct[-1]]) if len(idx_5pct) > 0 else 0.0
        aggregated["overall_auroc"] = float(roc_auc_score(all_labels, all_probs))

    return aggregated, all_metrics


def run_stage1_sweep(
    feature_store_dir: Path,
    output_dir: Path,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Run full Stage-1 model sweep."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir / f"stage1_sweep_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running Stage-1 sweep, output to {results_dir}")

    # Get all model configs
    model_configs = get_model_configs()
    logger.info(f"Testing {len(model_configs)} model configurations")

    # Run sweep
    results = []
    for config in tqdm(model_configs, desc="Stage-1 sweep"):
        logger.info(f"Evaluating {config['name']}...")

        metrics, fold_metrics = run_cross_validation(
            config, feature_store_dir, n_folds
        )

        if metrics:
            result = {
                "model_name": config["name"],
                "model_type": config["model_type"],
                **metrics,
            }
            results.append(result)

            # Save individual model config
            config_path = results_dir / "configs" / f"{config['name']}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by TPR@5%FPR (primary) then AUROC (secondary)
    if "overall_tpr_at_fpr_5pct" in results_df.columns:
        results_df = results_df.sort_values(
            ["overall_tpr_at_fpr_5pct", "overall_auroc"],
            ascending=[False, False]
        )

    # Save results
    results_df.to_csv(results_dir / "sweep_results.csv", index=False)

    # Identify Pareto frontier (TPR@5%FPR vs AUROC)
    pareto_df = compute_pareto_frontier(results_df)
    pareto_df.to_csv(results_dir / "pareto_frontier.csv", index=False)

    # Save best config
    if len(results_df) > 0:
        best_row = results_df.iloc[0]
        best_config = model_configs[[c["name"] for c in model_configs].index(best_row["model_name"])]
        with open(results_dir / "best_config.json", "w") as f:
            json.dump(best_config, f, indent=2)

    # Print summary
    logger.info("\n=== Stage-1 Sweep Summary ===")
    logger.info(f"Total models evaluated: {len(results_df)}")

    if len(results_df) > 0:
        logger.info(f"\nTop 5 by TPR@5%FPR:")
        top5 = results_df.head(5)[["model_name", "overall_tpr_at_fpr_5pct", "overall_auroc"]]
        logger.info(f"\n{top5.to_string()}")

        logger.info(f"\nBest model: {results_df.iloc[0]['model_name']}")
        logger.info(f"  TPR@5%FPR: {results_df.iloc[0].get('overall_tpr_at_fpr_5pct', 'N/A'):.4f}")
        logger.info(f"  AUROC: {results_df.iloc[0].get('overall_auroc', 'N/A'):.4f}")

    return results_df


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto frontier for TPR@5%FPR vs AUROC."""
    if len(df) == 0:
        return df

    x_col = "overall_tpr_at_fpr_5pct"
    y_col = "overall_auroc"

    if x_col not in df.columns or y_col not in df.columns:
        return df

    pareto_mask = np.ones(len(df), dtype=bool)

    for i, row in df.iterrows():
        # Check if any other point dominates this one
        for j, other in df.iterrows():
            if i != j:
                if (other[x_col] >= row[x_col] and other[y_col] >= row[y_col] and
                    (other[x_col] > row[x_col] or other[y_col] > row[y_col])):
                    pareto_mask[df.index.get_loc(i)] = False
                    break

    return df[pareto_mask].copy()


def main():
    parser = argparse.ArgumentParser(description="Run Stage-1 NE model sweep")
    parser.add_argument("--feature_store", type=str, required=True,
                        help="Path to feature store directory")
    parser.add_argument("--output_dir", type=str, default="outputs/supersearch")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    results = run_stage1_sweep(
        feature_store_dir=Path(args.feature_store),
        output_dir=Path(args.output_dir),
        n_folds=args.n_folds,
    )

    print(f"\nResults saved to outputs/supersearch/stage1_sweep_*/")


if __name__ == "__main__":
    main()
