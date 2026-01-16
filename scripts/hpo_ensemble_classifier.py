#!/usr/bin/env python3
"""
PHASE 3B: Ensemble Classifier for NE Detection

This script trains a lightweight classifier on score features to make
NE detection decisions, potentially learning complex decision boundaries
that threshold-based methods cannot capture.

Features used:
- max_score, min_score, mean_score, std_score
- top3_mean, score_range
- top1_minus_top2, top1_minus_mean
- ne_score, ne_calibrated_prob, best_minus_ne
- n_candidates
- criterion_id (one-hot encoded)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")


def load_targets(targets_path: str, profile: str) -> dict:
    """Load deployment targets from config."""
    with open(targets_path) as f:
        config = yaml.safe_load(f)
    return config["profiles"][profile]["targets"]


def prepare_features(df: pd.DataFrame, include_criterion: bool = True) -> tuple:
    """
    Prepare feature matrix from OOF cache.

    Returns:
        X: Feature matrix
        feature_names: List of feature names
    """
    # Numeric features
    numeric_features = [
        "max_score", "min_score", "mean_score", "std_score",
        "top3_mean", "score_range",
        "top1_minus_top2", "top1_minus_mean",
        "n_candidates"
    ]

    # NE-related features (may have NaN)
    ne_features = ["ne_score", "ne_calibrated_prob", "best_minus_ne"]

    X_numeric = df[numeric_features].copy()

    # Handle NE features with NaN
    for feat in ne_features:
        if feat in df.columns:
            # Fill NaN with a sentinel value
            X_numeric[feat] = df[feat].fillna(-999)

    feature_names = numeric_features + ne_features

    # One-hot encode criterion if requested
    if include_criterion:
        criterion_dummies = pd.get_dummies(df["criterion_id"], prefix="crit")
        X_numeric = pd.concat([X_numeric, criterion_dummies], axis=1)
        feature_names.extend(criterion_dummies.columns.tolist())

    return X_numeric.values, feature_names


def train_evaluate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    clf_class,
    clf_params: dict,
    targets: dict,
    include_criterion: bool = True
) -> dict:
    """
    Train classifier on train fold and evaluate on val fold.
    """
    # Prepare features
    X_train, feature_names = prepare_features(train_df, include_criterion)
    X_val, _ = prepare_features(val_df, include_criterion)

    y_train = train_df["gt_has_evidence"].values
    y_val = val_df["gt_has_evidence"].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train classifier
    clf = clf_class(**clf_params)
    clf.fit(X_train, y_train)

    # Get predictions
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
    mcc = matthews_corrcoef(y_val, y_pred)

    try:
        auroc = roc_auc_score(y_val, y_prob)
    except:
        auroc = 0.5

    # Find optimal threshold for target FPR
    fpr_curve, tpr_curve, thresholds = roc_curve(y_val, y_prob)
    optimal_threshold = 0.5
    optimal_tpr_at_target_fpr = 0

    # Find threshold that achieves target FPR
    for i, (fpr_i, tpr_i) in enumerate(zip(fpr_curve, tpr_curve)):
        if fpr_i <= targets["max_fpr"]:
            if tpr_i > optimal_tpr_at_target_fpr:
                optimal_tpr_at_target_fpr = tpr_i
                if i < len(thresholds):
                    optimal_threshold = thresholds[i]

    return {
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "auroc": auroc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "optimal_threshold": optimal_threshold,
        "tpr_at_target_fpr": optimal_tpr_at_target_fpr,
        "feature_names": feature_names,
        "clf": clf,
        "scaler": scaler
    }


def cross_validate_classifier(
    df: pd.DataFrame,
    clf_class,
    clf_params: dict,
    targets: dict,
    include_criterion: bool = True
) -> dict:
    """
    Perform 5-fold cross-validation for a classifier.
    """
    folds = sorted(df["fold_id"].unique())
    fold_results = []

    for val_fold in folds:
        train_df = df[df["fold_id"] != val_fold]
        val_df = df[df["fold_id"] == val_fold]

        result = train_evaluate_fold(
            train_df, val_df, clf_class, clf_params, targets, include_criterion
        )
        fold_results.append(result)

    # Aggregate metrics
    metrics = ["tpr", "fpr", "tnr", "precision", "f1", "mcc", "auroc", "tpr_at_target_fpr"]
    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in fold_results]
        aggregated[metric] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)

    # Check if meets targets
    meets_fpr = aggregated["fpr"] <= targets["max_fpr"]
    meets_tpr = aggregated["tpr"] >= targets["min_tpr"]

    return {
        "mean_metrics": aggregated,
        "fold_results": fold_results,
        "meets_fpr": meets_fpr,
        "meets_tpr": meets_tpr,
        "meets_both": meets_fpr and meets_tpr
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble classifier for NE detection")
    parser.add_argument("--oof_cache", type=str, required=True, help="Path to OOF cache parquet")
    parser.add_argument("--targets", type=str, required=True, help="Path to deployment targets YAML")
    parser.add_argument("--profile", type=str, default="high_recall_low_hallucination")
    parser.add_argument("--outdir", type=str, default="outputs/hpo_ensemble")
    parser.add_argument("--include_criterion", action="store_true", default=True,
                        help="Include criterion as feature")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading OOF cache from {args.oof_cache}")
    df = pd.read_parquet(args.oof_cache)
    print(f"Loaded {len(df)} records")
    print(f"Positive rate: {df['gt_has_evidence'].mean():.2%}")

    targets = load_targets(args.targets, args.profile)
    print(f"Targets: max_fpr={targets['max_fpr']}, min_tpr={targets['min_tpr']}")

    # Define classifiers to try
    classifiers = {
        "LogisticRegression": (
            LogisticRegression,
            {"class_weight": "balanced", "max_iter": 1000, "C": 0.1}
        ),
        "LogisticRegression_L1": (
            LogisticRegression,
            {"class_weight": "balanced", "max_iter": 1000, "C": 0.1, "penalty": "l1", "solver": "saga"}
        ),
        "RandomForest": (
            RandomForestClassifier,
            {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced", "random_state": 42}
        ),
        "RandomForest_Deep": (
            RandomForestClassifier,
            {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced", "random_state": 42}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier,
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier,
            {"max_depth": 5, "learning_rate": 0.1, "random_state": 42}
        ),
    }

    # Train and evaluate each classifier
    results = {}

    print("\n" + "="*70)
    print("CLASSIFIER COMPARISON")
    print("="*70)

    for clf_name, (clf_class, clf_params) in classifiers.items():
        print(f"\nTraining {clf_name}...")

        try:
            cv_result = cross_validate_classifier(
                df, clf_class, clf_params, targets, args.include_criterion
            )

            metrics = cv_result["mean_metrics"]
            status = "✅" if cv_result["meets_both"] else "❌"

            print(f"  TPR: {metrics['tpr']:.2%} ± {metrics['tpr_std']:.2%}")
            print(f"  FPR: {metrics['fpr']:.2%} ± {metrics['fpr_std']:.2%}")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  AUROC: {metrics['auroc']:.3f}")
            print(f"  TPR@{targets['max_fpr']:.0%}FPR: {metrics['tpr_at_target_fpr']:.2%}")
            print(f"  Meets targets: {status}")

            results[clf_name] = {
                "clf_class": clf_class.__name__,
                "clf_params": clf_params,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "meets_fpr": cv_result["meets_fpr"],
                "meets_tpr": cv_result["meets_tpr"],
                "meets_both": cv_result["meets_both"]
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[clf_name] = {"error": str(e)}

    # Find best classifier
    print("\n" + "="*70)
    print("BEST CLASSIFIER")
    print("="*70)

    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if valid_results:
        # Prioritize: meets_both > maximize tpr_at_target_fpr > maximize auroc
        def score_classifier(item):
            name, result = item
            return (
                result["meets_both"],
                result["metrics"]["tpr_at_target_fpr"],
                result["metrics"]["auroc"]
            )

        best_name, best_result = max(valid_results.items(), key=score_classifier)

        print(f"\nBest classifier: {best_name}")
        print(f"  TPR: {best_result['metrics']['tpr']:.2%}")
        print(f"  FPR: {best_result['metrics']['fpr']:.2%}")
        print(f"  TPR@{targets['max_fpr']:.0%}FPR: {best_result['metrics']['tpr_at_target_fpr']:.2%}")
        print(f"  AUROC: {best_result['metrics']['auroc']:.3f}")

        # Compare with threshold baseline
        print("\n" + "="*70)
        print("COMPARISON WITH THRESHOLD-BASED METHOD")
        print("="*70)

        global_results_path = Path("outputs/hpo_ne_gate/results.json")
        if global_results_path.exists():
            with open(global_results_path) as f:
                global_results = json.load(f)

            # Handle different result formats
            if "best" in global_results:
                baseline_tpr = global_results["best"]["metrics"]["tpr"]
                baseline_fpr = global_results["best"]["metrics"]["fpr"]
            elif "best_method_name" in global_results:
                best_method = global_results["best_method_name"]
                baseline_tpr = global_results["method_results"][best_method]["avg_metrics"]["tpr"]
                baseline_fpr = global_results["method_results"][best_method]["avg_metrics"]["fpr"]
            else:
                baseline_tpr = baseline_fpr = None

            if baseline_tpr is not None:
                print(f"\nThreshold-based (margin):")
                print(f"  TPR: {baseline_tpr:.2%}, FPR: {baseline_fpr:.2%}")
                print(f"\nEnsemble classifier ({best_name}):")
                print(f"  TPR: {best_result['metrics']['tpr']:.2%}, FPR: {best_result['metrics']['fpr']:.2%}")
                print(f"\nImprovement:")
                print(f"  TPR: {best_result['metrics']['tpr'] - baseline_tpr:+.2%}")
                print(f"  FPR: {best_result['metrics']['fpr'] - baseline_fpr:+.2%}")

        results["best"] = best_name

    # Save results
    output = {
        "profile": args.profile,
        "targets": targets,
        "classifiers": results,
        "timestamp": datetime.now().isoformat()
    }

    with open(outdir / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Generate report
    report_lines = [
        "# Ensemble Classifier for NE Detection Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Configuration",
        f"\n- Profile: `{args.profile}`",
        f"- Target FPR: ≤ {targets['max_fpr']:.0%}",
        f"- Target TPR: ≥ {targets['min_tpr']:.0%}",
        f"- Include criterion feature: {args.include_criterion}",
        f"\n---",
        f"\n## Classifier Comparison",
        f"\n| Classifier | TPR | FPR | Precision | AUROC | TPR@5%FPR | Meets Targets |",
        f"|------------|-----|-----|-----------|-------|-----------|---------------|"
    ]

    for clf_name, result in results.items():
        if clf_name == "best" or "error" in result:
            continue
        m = result["metrics"]
        status = "✅" if result["meets_both"] else "❌"
        report_lines.append(
            f"| {clf_name} | {m['tpr']:.1%} | {m['fpr']:.1%} | {m['precision']:.1%} | "
            f"{m['auroc']:.3f} | {m['tpr_at_target_fpr']:.1%} | {status} |"
        )

    if "best" in results:
        best = results[results["best"]]
        report_lines.extend([
            f"\n---",
            f"\n## Best Classifier: {results['best']}",
            f"\n- **TPR:** {best['metrics']['tpr']:.2%} ± {best['metrics']['tpr_std']:.2%}",
            f"- **FPR:** {best['metrics']['fpr']:.2%} ± {best['metrics']['fpr_std']:.2%}",
            f"- **Precision:** {best['metrics']['precision']:.2%}",
            f"- **AUROC:** {best['metrics']['auroc']:.3f}",
            f"- **TPR@5%FPR:** {best['metrics']['tpr_at_target_fpr']:.2%}",
        ])

    with open(outdir / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nResults saved to {outdir}")


if __name__ == "__main__":
    main()
