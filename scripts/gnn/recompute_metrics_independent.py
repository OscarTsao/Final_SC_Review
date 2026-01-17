#!/usr/bin/env python3
"""Independent Metric Recomputation and Verification Script.

This script provides an independent recomputation of all metrics from raw
predictions files. It compares the recomputed values with reported values
in cv_results.json and flags any mismatches.

Purpose:
1. Verify reported metrics are correct
2. Provide audit trail for gold-standard evaluation
3. Detect any accidental metric computation errors

Usage:
    python scripts/gnn/recompute_metrics_independent.py --experiment_dir outputs/gnn_research/20260117_004627/p1_ne_gate
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
)


@dataclass
class RecomputedMetrics:
    """Container for recomputed metrics."""
    auroc: float
    auprc: float
    tpr_at_5pct_fpr: float
    tpr_at_10pct_fpr: float
    precision_at_05: float
    recall_at_05: float
    f1_at_05: float
    mcc_at_05: float
    n_samples: int
    n_positive: int
    n_negative: int


@dataclass
class MetricComparison:
    """Comparison between reported and recomputed metrics."""
    metric_name: str
    reported: float
    recomputed: float
    difference: float
    match: bool
    tolerance: float


def compute_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """Compute TPR at a specific FPR threshold.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_fpr: Target FPR (e.g., 0.05 for 5%)

    Returns:
        Tuple of (TPR achieved, actual FPR at that threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find the largest threshold that achieves FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0

    best_idx = valid_idx[-1]  # Largest index where FPR <= target
    return float(tpr[best_idx]), float(fpr[best_idx])


def recompute_metrics_from_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> RecomputedMetrics:
    """Recompute all metrics from raw predictions.

    Args:
        preds: Predicted probabilities (sigmoid outputs)
        labels: Ground truth binary labels
        threshold: Decision threshold for binary metrics

    Returns:
        RecomputedMetrics dataclass
    """
    y_true = labels.astype(int)
    y_prob = preds.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_samples - n_positive

    # Handle edge cases
    if n_positive == 0 or n_positive == n_samples:
        # All same class - metrics undefined
        return RecomputedMetrics(
            auroc=0.5,
            auprc=n_positive / n_samples if n_positive > 0 else 0.0,
            tpr_at_5pct_fpr=0.0,
            tpr_at_10pct_fpr=0.0,
            precision_at_05=0.0,
            recall_at_05=0.0,
            f1_at_05=0.0,
            mcc_at_05=0.0,
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
        )

    # Threshold-independent metrics
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # TPR at fixed FPR
    tpr_at_5pct, _ = compute_tpr_at_fpr(y_true, y_prob, 0.05)
    tpr_at_10pct, _ = compute_tpr_at_fpr(y_true, y_prob, 0.10)

    # Threshold-dependent metrics at threshold=0.5
    precision_at_05 = precision_score(y_true, y_pred, zero_division=0.0)
    recall_at_05 = recall_score(y_true, y_pred, zero_division=0.0)
    f1_at_05 = f1_score(y_true, y_pred, zero_division=0.0)
    mcc_at_05 = matthews_corrcoef(y_true, y_pred)

    return RecomputedMetrics(
        auroc=float(auroc),
        auprc=float(auprc),
        tpr_at_5pct_fpr=float(tpr_at_5pct),
        tpr_at_10pct_fpr=float(tpr_at_10pct),
        precision_at_05=float(precision_at_05),
        recall_at_05=float(recall_at_05),
        f1_at_05=float(f1_at_05),
        mcc_at_05=float(mcc_at_05),
        n_samples=n_samples,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def extract_reported_metrics(cv_results: Dict, fold_id: int) -> Dict[str, float]:
    """Extract reported metrics from cv_results.json for a specific fold.

    Args:
        cv_results: Loaded cv_results.json data
        fold_id: Fold ID to extract

    Returns:
        Dict mapping metric names to reported values
    """
    reported = {}

    # Try different structures
    if "fold_results" in cv_results:
        # Structure: {"fold_results": [{"fold_id": 0, "metrics": {...}}, ...]}
        for fold_data in cv_results["fold_results"]:
            if fold_data.get("fold_id") == fold_id:
                metrics = fold_data.get("metrics", fold_data)
                reported["auroc"] = metrics.get("auroc", metrics.get("auc_roc"))
                reported["auprc"] = metrics.get("auprc", metrics.get("auc_pr"))
                reported["tpr_at_5pct_fpr"] = metrics.get("tpr_at_5pct_fpr", metrics.get("tpr@5%fpr"))
                reported["tpr_at_10pct_fpr"] = metrics.get("tpr_at_10pct_fpr", metrics.get("tpr@10%fpr"))
                break

    elif f"fold_{fold_id}" in cv_results:
        # Structure: {"fold_0": {"auroc": ..., ...}, ...}
        metrics = cv_results[f"fold_{fold_id}"]
        reported["auroc"] = metrics.get("auroc", metrics.get("auc_roc"))
        reported["auprc"] = metrics.get("auprc", metrics.get("auc_pr"))

    return {k: v for k, v in reported.items() if v is not None}


def compare_metrics(
    reported: Dict[str, float],
    recomputed: RecomputedMetrics,
    tolerance: float = 1e-5,
) -> List[MetricComparison]:
    """Compare reported and recomputed metrics.

    Args:
        reported: Dict of reported metric values
        recomputed: Recomputed metrics dataclass
        tolerance: Tolerance for floating point comparison

    Returns:
        List of MetricComparison objects
    """
    recomputed_dict = asdict(recomputed)
    comparisons = []

    for metric_name, reported_value in reported.items():
        if metric_name not in recomputed_dict:
            continue

        recomputed_value = recomputed_dict[metric_name]
        if recomputed_value is None:
            continue

        diff = abs(reported_value - recomputed_value)
        match = diff <= tolerance

        comparisons.append(MetricComparison(
            metric_name=metric_name,
            reported=reported_value,
            recomputed=recomputed_value,
            difference=diff,
            match=match,
            tolerance=tolerance,
        ))

    return comparisons


def load_predictions(fold_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions and labels from a fold directory.

    Args:
        fold_dir: Path to fold directory containing predictions.npz

    Returns:
        Tuple of (preds, labels)
    """
    npz_path = fold_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"predictions.npz not found in {fold_dir}")

    data = np.load(npz_path)
    return data["preds"], data["labels"]


def verify_experiment(
    experiment_dir: Path,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify all folds in an experiment directory.

    Args:
        experiment_dir: Path to experiment directory
        tolerance: Tolerance for metric comparison

    Returns:
        Verification results dict
    """
    results = {
        "experiment_dir": str(experiment_dir),
        "folds": [],
        "summary": {
            "total_comparisons": 0,
            "matches": 0,
            "mismatches": 0,
            "overall_pass": True,
        },
    }

    # Load cv_results.json
    cv_results_path = experiment_dir / "cv_results.json"
    if cv_results_path.exists():
        with open(cv_results_path) as f:
            cv_results = json.load(f)
    else:
        cv_results = None
        print(f"Warning: cv_results.json not found in {experiment_dir}")

    # Process each fold
    for fold_id in range(5):
        fold_dir = experiment_dir / f"fold_{fold_id}"
        if not fold_dir.exists():
            continue

        try:
            preds, labels = load_predictions(fold_dir)
        except FileNotFoundError:
            print(f"Skipping fold {fold_id}: predictions.npz not found")
            continue

        # Recompute metrics
        recomputed = recompute_metrics_from_predictions(preds, labels)

        # Compare with reported
        if cv_results:
            reported = extract_reported_metrics(cv_results, fold_id)
            comparisons = compare_metrics(reported, recomputed, tolerance)
        else:
            comparisons = []

        fold_result = {
            "fold_id": fold_id,
            "n_samples": recomputed.n_samples,
            "n_positive": recomputed.n_positive,
            "recomputed": asdict(recomputed),
            "comparisons": [asdict(c) for c in comparisons],
            "all_match": all(c.match for c in comparisons) if comparisons else True,
        }

        results["folds"].append(fold_result)

        # Update summary
        for c in comparisons:
            results["summary"]["total_comparisons"] += 1
            if c.match:
                results["summary"]["matches"] += 1
            else:
                results["summary"]["mismatches"] += 1
                results["summary"]["overall_pass"] = False

    return results


def print_verification_report(results: Dict[str, Any]):
    """Print a human-readable verification report."""
    print("\n" + "=" * 70)
    print("INDEPENDENT METRIC RECOMPUTATION REPORT")
    print("=" * 70)
    print(f"\nExperiment: {results['experiment_dir']}")

    for fold_result in results["folds"]:
        fold_id = fold_result["fold_id"]
        n_samples = fold_result["n_samples"]
        n_positive = fold_result["n_positive"]

        print(f"\n--- Fold {fold_id} ---")
        print(f"  Samples: {n_samples} (pos={n_positive}, neg={n_samples - n_positive})")

        recomputed = fold_result["recomputed"]
        print(f"  AUROC:        {recomputed['auroc']:.6f}")
        print(f"  AUPRC:        {recomputed['auprc']:.6f}")
        print(f"  TPR@5%FPR:    {recomputed['tpr_at_5pct_fpr']:.6f}")
        print(f"  TPR@10%FPR:   {recomputed['tpr_at_10pct_fpr']:.6f}")

        if fold_result["comparisons"]:
            print("\n  Comparisons with reported values:")
            for comp in fold_result["comparisons"]:
                status = "✅" if comp["match"] else "❌"
                print(f"    {status} {comp['metric_name']}: "
                      f"reported={comp['reported']:.6f}, "
                      f"recomputed={comp['recomputed']:.6f}, "
                      f"diff={comp['difference']:.2e}")

    # Summary
    summary = results["summary"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total comparisons: {summary['total_comparisons']}")
    print(f"  Matches: {summary['matches']}")
    print(f"  Mismatches: {summary['mismatches']}")

    if summary["overall_pass"]:
        print("\n  ✅ VERIFICATION PASSED - All metrics match within tolerance")
    else:
        print("\n  ❌ VERIFICATION FAILED - Some metrics do not match")


def main():
    parser = argparse.ArgumentParser(description="Independent Metric Recomputation")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="Tolerance for metric comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return 1

    # Run verification
    results = verify_experiment(experiment_dir, args.tolerance)

    # Print report
    print_verification_report(results)

    # Save results
    output_path = args.output or (experiment_dir / "metric_verification.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    exit(main())
