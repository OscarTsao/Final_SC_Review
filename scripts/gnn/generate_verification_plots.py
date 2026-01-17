#!/usr/bin/env python3
"""Generate verification visualizations for GNN components.

Creates:
- ROC/PR curves for P1 and P4
- Calibration plots for P1 and P4
- Performance comparison plots
- Output: docs/verification/figures/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = (10, 6)


def load_component_predictions(
    component: str, base_dir: Path
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load predictions for all folds of a component.

    Returns:
        Dict mapping fold_id -> (preds, labels)
    """
    results = {}

    # Find the latest run directory
    component_dir = base_dir / component
    run_dirs = sorted([d for d in component_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        logger.warning(f"No run directories found for {component}")
        return results

    latest_run = run_dirs[-1]
    logger.info(f"Loading {component} from {latest_run.name}")

    for fold_id in range(5):
        pred_file = latest_run / component / f"fold_{fold_id}" / "predictions.npz"
        if pred_file.exists():
            data = np.load(pred_file)
            results[fold_id] = (data["preds"], data["labels"])
            logger.info(f"  Fold {fold_id}: {len(data['preds'])} predictions")
        else:
            logger.warning(f"  Fold {fold_id}: predictions not found at {pred_file}")

    return results


def plot_roc_pr_curves(
    all_preds: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
):
    """Generate ROC and PR curves for all components."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"p1_ne_gate": "#e74c3c", "p4_hetero": "#2ecc71"}
    labels_map = {"p1_ne_gate": "P1 NE Gate", "p4_hetero": "P4 Criterion-Aware"}

    for component, fold_preds in all_preds.items():
        if not fold_preds:
            continue

        # Aggregate across folds
        all_probs = []
        all_labels = []
        for fold_id in sorted(fold_preds.keys()):
            preds, labels = fold_preds[fold_id]
            all_probs.extend(preds)
            all_labels.extend(labels)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        axes[0].plot(
            fpr, tpr,
            color=colors.get(component, "#3498db"),
            linewidth=2,
            label=f"{labels_map.get(component, component)} (AUC={roc_auc:.3f})",
        )

        # PR curve
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)

        axes[1].plot(
            recall, precision,
            color=colors.get(component, "#3498db"),
            linewidth=2,
            label=f"{labels_map.get(component, component)} (AUC={pr_auc:.3f})",
        )

    # ROC plot styling
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curve: No-Evidence Detection", fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # PR plot styling
    baseline_prevalence = np.mean(all_labels)
    axes[1].axhline(
        baseline_prevalence, color="k", linestyle="--", linewidth=1,
        label=f"Baseline (prevalence={baseline_prevalence:.3f})"
    )
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "roc_pr_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved ROC/PR curves to {output_path}")
    plt.close()


def plot_calibration(
    all_preds: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
):
    """Generate calibration plots."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"p1_ne_gate": "#e74c3c", "p4_hetero": "#2ecc71"}
    labels_map = {"p1_ne_gate": "P1 NE Gate", "p4_hetero": "P4 Criterion-Aware"}

    for idx, (component, fold_preds) in enumerate(all_preds.items()):
        if not fold_preds:
            continue

        # Aggregate across folds
        all_probs = []
        all_labels = []
        for fold_id in sorted(fold_preds.keys()):
            preds, labels = fold_preds[fold_id]
            all_probs.extend(preds)
            all_labels.extend(labels)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Calibration curve
        frac_pos, mean_pred = calibration_curve(all_labels, all_probs, n_bins=10)

        # Expected Calibration Error
        ece = np.mean(np.abs(frac_pos - mean_pred))

        # Plot
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
        ax.plot(
            mean_pred, frac_pos,
            marker="o",
            color=colors.get(component, "#3498db"),
            linewidth=2,
            markersize=8,
            label=f"{labels_map.get(component, component)}\n(ECE={ece:.4f})",
        )

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(
            f"Calibration: {labels_map.get(component, component)}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / "calibration_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved calibration plots to {output_path}")
    plt.close()


def plot_component_comparison(
    base_dir: Path,
    output_dir: Path,
):
    """Generate component performance comparison plot."""

    metrics_data = []

    # P1: NE Gate - AUROC from aggregated section
    p1_file = list(Path(base_dir / "p1_ne_gate").glob("*/*/cv_results.json"))
    if p1_file:
        with open(p1_file[0]) as f:
            data = json.load(f)
        metrics_data.append({
            "component": "P1 NE Gate",
            "metric": "AUROC",
            "mean": data["aggregated"]["auroc"]["mean"],
            "std": data["aggregated"]["auroc"]["std"],
        })

    # P2: Dynamic-K - Evidence Recall from mass_0.8 policy (parse string format)
    p2_file = list(Path(base_dir / "p2_dynamic_k").glob("*/*/cv_results.json"))
    if p2_file:
        with open(p2_file[0]) as f:
            data = json.load(f)
        recall_str = data["aggregated"]["mass_0.8"]["recall"]  # "0.9132 +/- 0.0162"
        parts = recall_str.split(" +/- ")
        metrics_data.append({
            "component": "P2 Dynamic-K",
            "metric": "Evidence Recall",
            "mean": float(parts[0]),
            "std": float(parts[1]),
        })

    # P3: Graph Reranker - Recall@5 from refined (parse string format)
    p3_file = list(Path(base_dir / "p3_graph_reranker").glob("*/*/cv_results.json"))
    if p3_file:
        with open(p3_file[0]) as f:
            data = json.load(f)
        recall_str = data["aggregated"]["refined"]["recall@5"]  # "0.7280 +/- 0.0469"
        parts = recall_str.split(" +/- ")
        metrics_data.append({
            "component": "P3 Graph Reranker",
            "metric": "Recall@5",
            "mean": float(parts[0]),
            "std": float(parts[1]),
        })

    # P4: Criterion-Aware - AUROC (parse string format)
    p4_file = list(Path(base_dir / "p4_hetero").glob("*/*/cv_results.json"))
    if p4_file:
        with open(p4_file[0]) as f:
            data = json.load(f)
        auroc_str = data["aggregated"]["auroc"]  # "0.9053 +/- 0.0108"
        parts = auroc_str.split(" +/- ")
        metrics_data.append({
            "component": "P4 Criterion-Aware",
            "metric": "AUROC",
            "mean": float(parts[0]),
            "std": float(parts[1]),
        })

    if not metrics_data:
        logger.warning("No metrics data to plot")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_data))
    means = [d["mean"] for d in metrics_data]
    stds = [d["std"] for d in metrics_data]
    labels = [d["component"] for d in metrics_data]

    colors_list = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_list[:len(x)], alpha=0.8)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.01,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Performance", fontsize=12)
    ax.set_title("GNN Component Performance (nv-embed-v2)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = output_dir / "component_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved component comparison to {output_path}")
    plt.close()


def main():
    """Generate all verification plots."""

    base_dir = Path("outputs/gnn_research_nvembed")
    output_dir = Path("docs/verification/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GNN Verification Visualization Generator")
    logger.info("=" * 60)

    # Load predictions for components that save them
    all_preds = {}

    for component in ["p1_ne_gate"]:  # P1 has predictions.npz
        preds = load_component_predictions(component, base_dir)
        if preds:
            all_preds[component] = preds

    # Note: P4 doesn't have saved predictions.npz, only model checkpoints
    # We'll use CV results instead for comparison plot

    logger.info("\n" + "=" * 60)
    logger.info("Generating Plots")
    logger.info("=" * 60)

    if all_preds:
        logger.info("\n1. ROC/PR Curves")
        plot_roc_pr_curves(all_preds, output_dir)

        logger.info("\n2. Calibration Plots")
        plot_calibration(all_preds, output_dir)
    else:
        logger.warning("No predictions found for ROC/PR/calibration plots")

    logger.info("\n3. Component Comparison")
    plot_component_comparison(base_dir, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Visualization Complete")
    logger.info(f"📁 Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
