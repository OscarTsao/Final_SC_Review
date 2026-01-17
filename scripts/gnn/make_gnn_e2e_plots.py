#!/usr/bin/env python3
"""Generate visualization plots for GNN E2E evaluation.

Generates the following plots:
1. ROC curves (5-fold + mean curve)
2. PR curves (5-fold + mean curve)
3. Calibration diagram with ECE
4. Fold metrics comparison bar chart
5. Aggregated summary bar chart
6. Dynamic-K distribution histograms

Usage:
    python scripts/gnn/make_gnn_e2e_plots.py \
        --experiment_dir outputs/gnn_research/20260117_004627/p1_ne_gate \
        --output_dir outputs/gnn_e2e_report/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


def load_fold_predictions(experiment_dir: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load predictions from all folds.

    Returns:
        Dict mapping fold_id to (preds, labels)
    """
    fold_data = {}
    for fold_id in range(5):
        fold_dir = experiment_dir / f"fold_{fold_id}"
        npz_path = fold_dir / "predictions.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            fold_data[fold_id] = (data["preds"], data["labels"])
    return fold_data


def plot_roc_curves(
    fold_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str = "ROC Curves - 5-Fold Cross-Validation",
):
    """Plot ROC curves for all folds with mean curve."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_id, (preds, labels) in sorted(fold_data.items()):
        fpr, tpr, _ = roc_curve(labels, preds)
        auc = roc_auc_score(labels, preds)
        aucs.append(auc)

        # Interpolate for mean calculation
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        ax.plot(fpr, tpr, color=colors[fold_id], alpha=0.5, lw=1.5,
                label=f'Fold {fold_id} (AUROC = {auc:.4f})')

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='navy', lw=2.5,
            label=f'Mean ROC (AUROC = {mean_auc:.4f} ± {std_auc:.4f})')

    # Confidence band
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='navy', alpha=0.1)

    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pr_curves(
    fold_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str = "Precision-Recall Curves - 5-Fold Cross-Validation",
):
    """Plot Precision-Recall curves for all folds with mean curve."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    precisions = []
    auprcs = []
    mean_recall = np.linspace(0, 1, 100)
    positive_rates = []

    for fold_id, (preds, labels) in sorted(fold_data.items()):
        precision, recall, _ = precision_recall_curve(labels, preds)
        auprc = average_precision_score(labels, preds)
        auprcs.append(auprc)
        positive_rates.append(labels.mean())

        # Interpolate (note: recall is decreasing)
        recall_sorted = recall[::-1]
        precision_sorted = precision[::-1]
        precision_interp = np.interp(mean_recall, recall_sorted, precision_sorted)
        precisions.append(precision_interp)

        ax.plot(recall, precision, color=colors[fold_id], alpha=0.5, lw=1.5,
                label=f'Fold {fold_id} (AUPRC = {auprc:.4f})')

    # Mean PR
    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)

    ax.plot(mean_recall, mean_precision, color='navy', lw=2.5,
            label=f'Mean PR (AUPRC = {mean_auprc:.4f} ± {std_auprc:.4f})')

    # Confidence band
    std_precision = np.std(precisions, axis=0)
    prec_upper = np.minimum(mean_precision + std_precision, 1)
    prec_lower = np.maximum(mean_precision - std_precision, 0)
    ax.fill_between(mean_recall, prec_lower, prec_upper, color='navy', alpha=0.1)

    # Baseline (positive class rate)
    baseline = np.mean(positive_rates)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
               label=f'Baseline ({baseline:.3f})')

    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_calibration(
    fold_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    n_bins: int = 10,
    title: str = "Calibration Diagram",
):
    """Plot calibration diagram with ECE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    eces = []

    # Per-fold calibration
    for fold_id, (preds, labels) in sorted(fold_data.items()):
        prob_true, prob_pred = calibration_curve(labels, preds, n_bins=n_bins, strategy='uniform')

        # ECE calculation
        bin_counts = np.histogram(preds, bins=n_bins, range=(0, 1))[0]
        total = len(preds)
        ece = 0.0
        for i in range(len(prob_true)):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total) * abs(prob_true[i] - prob_pred[i])
        eces.append(ece)

        ax1.plot(prob_pred, prob_true, color=colors[fold_id], marker='o',
                 alpha=0.7, lw=1.5, label=f'Fold {fold_id} (ECE = {ece:.4f})')

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')

    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ECE bar chart
    fold_ids = list(range(5))
    ax2.bar(fold_ids, eces, color=colors, alpha=0.7)
    ax2.axhline(y=np.mean(eces), color='navy', linestyle='--', lw=2,
                label=f'Mean ECE = {np.mean(eces):.4f}')
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    ax2.set_title('ECE by Fold', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xticks(fold_ids)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_fold_metrics_comparison(
    cv_results: Dict,
    output_path: Path,
    title: str = "Metrics Comparison Across Folds",
):
    """Plot bar chart comparing metrics across folds."""
    # Extract per-fold metrics
    if "fold_results" not in cv_results:
        print("Warning: No fold_results in cv_results")
        return

    metrics_to_plot = ["auroc", "auprc"]
    fold_ids = []
    metric_values = {m: [] for m in metrics_to_plot}

    for fold_data in cv_results["fold_results"]:
        fold_id = fold_data.get("fold_id", fold_data.get("fold"))
        fold_ids.append(fold_id)

        metrics = fold_data.get("metrics", fold_data)
        for m in metrics_to_plot:
            val = metrics.get(m, metrics.get(m.replace("_", "@")))
            metric_values[m].append(val if val is not None else 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fold_ids))
    width = 0.35
    colors = ['#2ecc71', '#3498db']

    for i, (metric, values) in enumerate(metric_values.items()):
        offset = (i - len(metrics_to_plot)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.upper(), color=colors[i], alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in fold_ids])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_aggregated_summary(
    cv_results: Dict,
    output_path: Path,
    title: str = "Aggregated Metrics Summary",
):
    """Plot aggregated metrics with error bars."""
    # Extract aggregated metrics
    aggregated = cv_results.get("aggregated", {})
    if not aggregated:
        print("Warning: No aggregated metrics found")
        return

    metrics = []
    means = []
    stds = []

    for metric_name in ["auroc", "auprc", "tpr_at_5pct_fpr", "tpr_at_10pct_fpr"]:
        if metric_name in aggregated:
            value = aggregated[metric_name]
            if isinstance(value, dict):
                metrics.append(metric_name.upper())
                means.append(value.get("mean", 0))
                stds.append(value.get("std", 0))
            elif isinstance(value, str) and "±" in value:
                parts = value.split("±")
                metrics.append(metric_name.upper())
                means.append(float(parts[0].strip()))
                stds.append(float(parts[1].strip()))

    if not metrics:
        print("Warning: Could not parse aggregated metrics")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.annotate(f'{mean:.4f}±{std:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(means) * 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_prediction_distribution(
    fold_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str = "Prediction Score Distribution",
):
    """Plot histogram of prediction scores by class."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (fold_id, (preds, labels)) in enumerate(sorted(fold_data.items())):
        ax = axes[idx]

        pos_preds = preds[labels == 1]
        neg_preds = preds[labels == 0]

        ax.hist(neg_preds, bins=50, alpha=0.6, label=f'Negative (n={len(neg_preds)})',
                color='blue', density=True)
        ax.hist(pos_preds, bins=50, alpha=0.6, label=f'Positive (n={len(pos_preds)})',
                color='red', density=True)

        ax.axvline(x=0.5, color='black', linestyle='--', lw=1, label='Threshold=0.5')
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Fold {fold_id}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Combined histogram in last subplot
    ax = axes[5]
    all_preds = np.concatenate([preds for preds, _ in fold_data.values()])
    all_labels = np.concatenate([labels for _, labels in fold_data.values()])

    pos_preds = all_preds[all_labels == 1]
    neg_preds = all_preds[all_labels == 0]

    ax.hist(neg_preds, bins=50, alpha=0.6, label=f'Negative (n={len(neg_preds)})',
            color='blue', density=True)
    ax.hist(pos_preds, bins=50, alpha=0.6, label=f'Positive (n={len(pos_preds)})',
            color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', lw=1, label='Threshold=0.5')
    ax.set_xlabel('Predicted Probability', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('All Folds Combined', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate GNN E2E Visualization Plots")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GNN E2E Visualization Plot Generator")
    print("=" * 70)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")

    # Load predictions
    fold_data = load_fold_predictions(experiment_dir)
    if not fold_data:
        print("Error: No fold predictions found")
        return 1

    print(f"Loaded {len(fold_data)} folds")

    # Load cv_results if available
    cv_results_path = experiment_dir / "cv_results.json"
    if cv_results_path.exists():
        with open(cv_results_path) as f:
            cv_results = json.load(f)
    else:
        cv_results = None

    # Generate plots
    print("\nGenerating plots...")

    # 1. ROC curves
    plot_roc_curves(fold_data, output_dir / "roc_curves.png")

    # 2. PR curves
    plot_pr_curves(fold_data, output_dir / "pr_curves.png")

    # 3. Calibration diagram
    plot_calibration(fold_data, output_dir / "calibration_diagram.png")

    # 4. Prediction distribution
    plot_prediction_distribution(fold_data, output_dir / "prediction_distribution.png")

    # 5. Fold metrics comparison (if cv_results available)
    if cv_results:
        plot_fold_metrics_comparison(cv_results, output_dir / "fold_metrics_comparison.png")
        plot_aggregated_summary(cv_results, output_dir / "aggregated_summary.png")

    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
