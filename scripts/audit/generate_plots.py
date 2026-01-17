#!/usr/bin/env python3
"""Generate audit visualization plots.

This script generates:
1. ROC curves for each fold and aggregated
2. PR curves for each fold
3. Metric comparison across folds (bar chart with error bars)
4. Calibration curves

Usage:
    python scripts/audit/generate_plots.py --output_dir outputs/audit_full_eval/<ts>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_fold_predictions(gnn_results_dir: Path, experiment: str = "p1_ne_gate") -> List[Dict]:
    """Load predictions from all folds."""
    folds = []

    # Find all prediction files for this experiment
    pred_files = list(gnn_results_dir.rglob(f"**/{experiment}/fold_*/predictions.npz"))

    if not pred_files:
        print(f"  No prediction files found for {experiment}")
        return []

    # Get the experiment directory from the first file
    exp_dir = pred_files[0].parent.parent
    print(f"  Found experiment dir: {exp_dir}")

    for i in range(5):
        fold_dir = exp_dir / f"fold_{i}"
        pred_path = fold_dir / "predictions.npz"

        if pred_path.exists():
            data = np.load(pred_path)
            if "preds" in data:
                folds.append({
                    "fold_id": i,
                    "y_prob": data["preds"],
                    "y_true": data["labels"],
                })
            elif "y_prob" in data:
                folds.append({
                    "fold_id": i,
                    "y_prob": data["y_prob"],
                    "y_true": data["y_true"],
                })
        else:
            print(f"  Warning: No predictions for fold {i}")

    return folds


def plot_roc_curves(folds: List[Dict], output_path: Path) -> None:
    """Plot ROC curves for all folds and mean curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot individual fold curves
    all_fpr = []
    all_tpr = []
    all_auroc = []

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for fold_data in folds:
        fold_id = fold_data["fold_id"]
        y_true = fold_data["y_true"]
        y_prob = fold_data["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auroc.append(auroc)

        ax.plot(fpr, tpr, color=colors[fold_id], alpha=0.5,
                label=f"Fold {fold_id} (AUC={auroc:.3f})")

    # Compute mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    for fpr, tpr in zip(all_fpr, all_tpr):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr /= len(folds)

    mean_auroc = np.mean(all_auroc)
    std_auroc = np.std(all_auroc)

    ax.plot(mean_fpr, mean_tpr, color="black", linewidth=2,
            label=f"Mean (AUC={mean_auroc:.3f}±{std_auroc:.3f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    # Mark TPR at 5% FPR
    idx_5pct = np.argmin(np.abs(mean_fpr - 0.05))
    tpr_at_5pct = mean_tpr[idx_5pct]
    ax.scatter([0.05], [tpr_at_5pct], color="red", s=100, zorder=5)
    ax.annotate(f"TPR@5%FPR={tpr_at_5pct:.3f}",
                xy=(0.05, tpr_at_5pct), xytext=(0.15, tpr_at_5pct - 0.1),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - NE Gate (5-Fold CV)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curves to {output_path}")


def plot_pr_curves(folds: List[Dict], output_path: Path) -> None:
    """Plot Precision-Recall curves for all folds."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    all_auprc = []

    for fold_data in folds:
        fold_id = fold_data["fold_id"]
        y_true = fold_data["y_true"]
        y_prob = fold_data["y_prob"]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        all_auprc.append(auprc)

        ax.plot(recall, precision, color=colors[fold_id], alpha=0.6,
                label=f"Fold {fold_id} (AP={auprc:.3f})")

    mean_auprc = np.mean(all_auprc)
    std_auprc = np.std(all_auprc)

    # Baseline (random classifier)
    baseline = np.mean([f["y_true"].mean() for f in folds])
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves - NE Gate (5-Fold CV)\nMean AP={mean_auprc:.3f}±{std_auprc:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved PR curves to {output_path}")


def plot_fold_metrics_comparison(cv_results_path: Path, output_path: Path) -> None:
    """Plot bar chart comparing metrics across folds."""
    with open(cv_results_path) as f:
        cv_data = json.load(f)

    fold_results = cv_data.get("fold_results", [])
    if not fold_results:
        print("  No fold results found")
        return

    # Extract metrics
    metrics_to_plot = ["auroc", "auprc", "tpr_at_fpr_5pct", "tpr_at_fpr_10pct"]
    n_folds = len(fold_results)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_folds)
    width = 0.2
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics_to_plot):
        values = []
        for fold_result in fold_results:
            m = fold_result.get("metrics", {})
            values.append(m.get(metric, 0))

        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics Comparison Across Folds - NE Gate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fold comparison to {output_path}")


def plot_aggregated_summary(cv_results_path: Path, output_path: Path) -> None:
    """Plot aggregated metrics with error bars."""
    with open(cv_results_path) as f:
        cv_data = json.load(f)

    agg = cv_data.get("aggregated", {})
    if not agg:
        print("  No aggregated results found")
        return

    metrics = ["auroc", "auprc", "tpr_at_fpr_3pct", "tpr_at_fpr_5pct", "tpr_at_fpr_10pct"]
    means = []
    stds = []
    labels = []

    for metric in metrics:
        if metric in agg and isinstance(agg[metric], dict):
            means.append(agg[metric]["mean"])
            stds.append(agg[metric]["std"])
            labels.append(metric.replace("_", "\n"))

    if not means:
        print("  No valid aggregated metrics")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f"{mean:.3f}±{std:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02),
                    ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Aggregated NE Gate Metrics (5-Fold CV Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved aggregated summary to {output_path}")


def plot_calibration_curve(folds: List[Dict], output_path: Path, n_bins: int = 10) -> None:
    """Plot calibration curve (reliability diagram)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Aggregate all predictions
    all_probs = np.concatenate([f["y_prob"] for f in folds])
    all_labels = np.concatenate([f["y_true"] for f in folds])

    # Compute calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (all_probs >= bin_edges[i]) & (all_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(all_labels[mask].mean())
            bin_confidences.append(all_probs[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Plot calibration curve
    valid_mask = ~np.isnan(bin_accuracies)
    ax.bar(bin_centers[valid_mask], bin_counts[valid_mask] / bin_counts.sum(),
           width=0.08, alpha=0.3, color="blue", label="Sample distribution")

    ax2 = ax.twinx()
    ax2.plot(bin_confidences[valid_mask], bin_accuracies[valid_mask],
             "o-", color="red", linewidth=2, markersize=8, label="Calibration")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Compute ECE
    ece = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if not np.isnan(acc):
            ece += (count / len(all_probs)) * abs(acc - conf)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Sample Fraction", color="blue")
    ax2.set_ylabel("Accuracy (Fraction Positive)", color="red")
    ax.set_title(f"Calibration Curve - NE Gate\nECE = {ece:.4f}")
    ax2.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate audit plots")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gnn_results", type=str, default="outputs/gnn_research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gnn_results_dir = Path(args.gnn_results)

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Audit Visualization Plots")
    print("=" * 60)

    # Load P1 NE Gate predictions
    print("\nLoading fold predictions...")
    folds = load_fold_predictions(gnn_results_dir, "p1_ne_gate")

    if folds:
        print(f"  Loaded {len(folds)} folds")

        # 1. ROC curves
        print("\n1. Generating ROC curves...")
        plot_roc_curves(folds, plots_dir / "roc_curves.png")

        # 2. PR curves
        print("\n2. Generating PR curves...")
        plot_pr_curves(folds, plots_dir / "pr_curves.png")

        # 3. Calibration curve
        print("\n3. Generating calibration curve...")
        plot_calibration_curve(folds, plots_dir / "calibration_curve.png")
    else:
        print("  No fold predictions found")

    # 4. Fold metrics comparison
    print("\n4. Generating fold metrics comparison...")
    cv_files = list(gnn_results_dir.rglob("p1_ne_gate/cv_results.json"))
    if cv_files:
        plot_fold_metrics_comparison(cv_files[0], plots_dir / "fold_metrics_comparison.png")
        plot_aggregated_summary(cv_files[0], plots_dir / "aggregated_summary.png")

    print("\n" + "=" * 60)
    print(f"All plots saved to: {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
