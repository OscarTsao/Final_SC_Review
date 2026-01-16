#!/usr/bin/env python3
"""
PHASE 3A: Per-Criterion Threshold Optimization

This script optimizes NE gate thresholds independently for each criterion,
potentially achieving better overall performance than a global threshold.

Key insight: Different criteria have vastly different positive rates (2.6% to 21.8%),
suggesting that optimal thresholds may vary by criterion.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)


def load_targets(targets_path: str, profile: str) -> dict:
    """Load deployment targets from config."""
    with open(targets_path) as f:
        config = yaml.safe_load(f)
    return config["profiles"][profile]["targets"]


def compute_ne_score(row: pd.Series, method: str) -> float:
    """Compute NE detection score for a single row."""
    if method == "max_score":
        return row["max_score"]
    elif method == "margin":
        ne_score = row.get("ne_score")
        if ne_score is None or pd.isna(ne_score):
            return row["max_score"]
        return row["max_score"] - ne_score
    elif method == "calibrated_prob":
        probs = json.loads(row["calibrated_probs"]) if isinstance(row["calibrated_probs"], str) else row["calibrated_probs"]
        if not probs:
            return 0.0
        return max(probs)
    else:
        raise ValueError(f"Unknown method: {method}")


def optimize_threshold_for_criterion(
    df: pd.DataFrame,
    criterion_id: str,
    method: str,
    targets: dict,
    n_steps: int = 100
) -> dict:
    """
    Optimize threshold for a single criterion using cross-validation.

    Returns the best threshold and its CV performance.
    """
    criterion_df = df[df["criterion_id"] == criterion_id].copy()

    # Compute scores
    criterion_df["ne_decision_score"] = criterion_df.apply(
        lambda row: compute_ne_score(row, method), axis=1
    )

    # Get score range for threshold sweep
    scores = criterion_df["ne_decision_score"].values
    score_min, score_max = np.percentile(scores, [1, 99])
    thresholds = np.linspace(score_min, score_max, n_steps)

    folds = criterion_df["fold_id"].unique()

    # Results per threshold
    threshold_results = []

    for threshold in thresholds:
        fold_metrics = []

        for fold in folds:
            val_mask = criterion_df["fold_id"] == fold
            val_df = criterion_df[val_mask]

            y_true = val_df["gt_has_evidence"].values
            y_pred = (val_df["ne_decision_score"].values >= threshold).astype(int)

            # Skip if no positive or negative samples
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            fold_metrics.append({
                "tpr": tpr,
                "fpr": fpr,
                "precision": precision,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            })

        if not fold_metrics:
            continue

        # Average across folds
        avg_tpr = np.mean([m["tpr"] for m in fold_metrics])
        avg_fpr = np.mean([m["fpr"] for m in fold_metrics])
        avg_precision = np.mean([m["precision"] for m in fold_metrics])

        # Check if meets targets
        meets_fpr = avg_fpr <= targets["max_fpr"]
        meets_tpr = avg_tpr >= targets["min_tpr"]

        threshold_results.append({
            "threshold": threshold,
            "tpr": avg_tpr,
            "fpr": avg_fpr,
            "precision": avg_precision,
            "meets_fpr": meets_fpr,
            "meets_tpr": meets_tpr,
            "meets_both": meets_fpr and meets_tpr
        })

    # Find best threshold using lexicographic ordering
    # Priority: meets_both > meets_fpr > maximize precision
    best = None

    # First, try to find one that meets both targets
    candidates = [r for r in threshold_results if r["meets_both"]]
    if candidates:
        # Among those meeting both, maximize precision
        best = max(candidates, key=lambda x: x["precision"])
    else:
        # Try to meet FPR constraint while maximizing TPR
        candidates = [r for r in threshold_results if r["meets_fpr"]]
        if candidates:
            best = max(candidates, key=lambda x: x["tpr"])
        else:
            # Fallback: minimize FPR while keeping reasonable TPR
            candidates = [r for r in threshold_results if r["tpr"] >= 0.5]
            if candidates:
                best = min(candidates, key=lambda x: x["fpr"])
            else:
                # Last resort: balance FPR and TPR
                best = max(threshold_results, key=lambda x: x["tpr"] - x["fpr"])

    return {
        "criterion_id": criterion_id,
        "method": method,
        "threshold": best["threshold"],
        "tpr": best["tpr"],
        "fpr": best["fpr"],
        "precision": best["precision"],
        "meets_targets": best["meets_both"],
        "n_positive": int(criterion_df["gt_has_evidence"].sum()),
        "n_total": len(criterion_df),
        "positive_rate": criterion_df["gt_has_evidence"].mean()
    }


def evaluate_per_criterion_thresholds(
    df: pd.DataFrame,
    thresholds: dict,  # criterion_id -> threshold
    method: str
) -> dict:
    """
    Evaluate performance using per-criterion thresholds.

    Returns overall and per-criterion metrics.
    """
    df = df.copy()

    # Compute scores
    df["ne_decision_score"] = df.apply(
        lambda row: compute_ne_score(row, method), axis=1
    )

    # Apply per-criterion thresholds
    def apply_threshold(row):
        threshold = thresholds.get(row["criterion_id"], 0)
        return 1 if row["ne_decision_score"] >= threshold else 0

    df["y_pred"] = df.apply(apply_threshold, axis=1)

    folds = df["fold_id"].unique()
    fold_metrics = []

    for fold in folds:
        val_df = df[df["fold_id"] == fold]

        y_true = val_df["gt_has_evidence"].values
        y_pred = val_df["y_pred"].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

        fold_metrics.append({
            "tpr": tpr,
            "fpr": fpr,
            "tnr": tnr,
            "precision": precision,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

    # Average across folds
    return {
        "tpr": np.mean([m["tpr"] for m in fold_metrics]),
        "fpr": np.mean([m["fpr"] for m in fold_metrics]),
        "tnr": np.mean([m["tnr"] for m in fold_metrics]),
        "precision": np.mean([m["precision"] for m in fold_metrics]),
        "f1": np.mean([m["f1"] for m in fold_metrics]),
        "tpr_std": np.std([m["tpr"] for m in fold_metrics]),
        "fpr_std": np.std([m["fpr"] for m in fold_metrics]),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-criterion threshold optimization")
    parser.add_argument("--oof_cache", type=str, required=True, help="Path to OOF cache parquet")
    parser.add_argument("--targets", type=str, required=True, help="Path to deployment targets YAML")
    parser.add_argument("--profile", type=str, default="high_recall_low_hallucination")
    parser.add_argument("--outdir", type=str, default="outputs/hpo_per_criterion")
    parser.add_argument("--method", type=str, default="margin", choices=["max_score", "margin", "calibrated_prob"])
    parser.add_argument("--n_steps", type=int, default=100, help="Number of threshold steps per criterion")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading OOF cache from {args.oof_cache}")
    df = pd.read_parquet(args.oof_cache)
    print(f"Loaded {len(df)} records")

    targets = load_targets(args.targets, args.profile)
    print(f"Targets: max_fpr={targets['max_fpr']}, min_tpr={targets['min_tpr']}")

    criteria = sorted(df["criterion_id"].unique())
    print(f"Optimizing thresholds for {len(criteria)} criteria")

    # Optimize per-criterion thresholds
    per_criterion_results = []
    thresholds = {}

    for criterion_id in criteria:
        print(f"\nOptimizing {criterion_id}...")
        result = optimize_threshold_for_criterion(
            df, criterion_id, args.method, targets, args.n_steps
        )
        per_criterion_results.append(result)
        thresholds[criterion_id] = result["threshold"]

        status = "✅" if result["meets_targets"] else "❌"
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  TPR: {result['tpr']:.2%}, FPR: {result['fpr']:.2%} {status}")
        print(f"  Positive rate: {result['positive_rate']:.2%} ({result['n_positive']}/{result['n_total']})")

    # Evaluate overall performance with per-criterion thresholds
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE WITH PER-CRITERION THRESHOLDS")
    print("="*60)

    overall = evaluate_per_criterion_thresholds(df, thresholds, args.method)

    meets_fpr = overall["fpr"] <= targets["max_fpr"]
    meets_tpr = overall["tpr"] >= targets["min_tpr"]

    print(f"\nOverall TPR: {overall['tpr']:.2%} ± {overall['tpr_std']:.2%} (target: ≥{targets['min_tpr']:.0%}) {'✅' if meets_tpr else '❌'}")
    print(f"Overall FPR: {overall['fpr']:.2%} ± {overall['fpr_std']:.2%} (target: ≤{targets['max_fpr']:.0%}) {'✅' if meets_fpr else '❌'}")
    print(f"Overall Precision: {overall['precision']:.2%}")
    print(f"Overall F1: {overall['f1']:.3f}")

    # Compare with global threshold baseline
    print("\n" + "="*60)
    print("COMPARISON WITH GLOBAL THRESHOLD")
    print("="*60)

    # Load global threshold results if available
    global_results_path = Path("outputs/hpo_ne_gate/results.json")
    if global_results_path.exists():
        with open(global_results_path) as f:
            global_results = json.load(f)

        # Handle different result formats
        if "best" in global_results:
            global_tpr = global_results["best"]["metrics"]["tpr"]
            global_fpr = global_results["best"]["metrics"]["fpr"]
        elif "best_method_name" in global_results:
            best_method = global_results["best_method_name"]
            global_tpr = global_results["method_results"][best_method]["avg_metrics"]["tpr"]
            global_fpr = global_results["method_results"][best_method]["avg_metrics"]["fpr"]
        else:
            global_tpr = global_fpr = None

        if global_tpr is not None:
            print(f"\nGlobal threshold ({args.method}):")
            print(f"  TPR: {global_tpr:.2%}, FPR: {global_fpr:.2%}")
            print(f"\nPer-criterion thresholds:")
            print(f"  TPR: {overall['tpr']:.2%}, FPR: {overall['fpr']:.2%}")
            print(f"\nImprovement:")
            print(f"  TPR: {overall['tpr'] - global_tpr:+.2%}")
            print(f"  FPR: {overall['fpr'] - global_fpr:+.2%}")

    # Save results
    # Convert numpy types to native Python for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = {
        "method": args.method,
        "profile": args.profile,
        "targets": targets,
        "per_criterion": convert_numpy(per_criterion_results),
        "thresholds": convert_numpy(thresholds),
        "overall": convert_numpy(overall),
        "timestamp": datetime.now().isoformat()
    }

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    report_lines = [
        "# Per-Criterion Threshold Optimization Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Configuration",
        f"\n- Method: `{args.method}`",
        f"- Profile: `{args.profile}`",
        f"- Target FPR: ≤ {targets['max_fpr']:.0%}",
        f"- Target TPR: ≥ {targets['min_tpr']:.0%}",
        f"\n---",
        f"\n## Per-Criterion Thresholds",
        f"\n| Criterion | Threshold | TPR | FPR | Pos Rate | Meets Targets |",
        f"|-----------|-----------|-----|-----|----------|---------------|"
    ]

    for r in per_criterion_results:
        status = "✅" if r["meets_targets"] else "❌"
        report_lines.append(
            f"| {r['criterion_id']} | {r['threshold']:.4f} | {r['tpr']:.1%} | {r['fpr']:.1%} | {r['positive_rate']:.1%} | {status} |"
        )

    report_lines.extend([
        f"\n---",
        f"\n## Overall Performance",
        f"\n| Metric | Value | Target | Status |",
        f"|--------|-------|--------|--------|",
        f"| TPR | {overall['tpr']:.2%} | ≥ {targets['min_tpr']:.0%} | {'✅' if meets_tpr else '❌'} |",
        f"| FPR | {overall['fpr']:.2%} | ≤ {targets['max_fpr']:.0%} | {'✅' if meets_fpr else '❌'} |",
        f"| Precision | {overall['precision']:.2%} | - | - |",
        f"| F1 | {overall['f1']:.3f} | - | - |",
    ])

    with open(outdir / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nResults saved to {outdir}")


if __name__ == "__main__":
    main()
