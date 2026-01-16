#!/usr/bin/env python3
"""
HPO for No-Evidence (NE) Gate Detection.

This script optimizes the no-evidence detection threshold to meet deployment targets:
- max_fpr ≤ 5% (false positive rate / negative coverage)
- min_tpr ≥ 90% (true positive rate / positive coverage)

Methods evaluated:
1. score_gap: threshold on (max_score - second_max_score)
2. max_score: threshold on max reranker score
3. ne_rank: predict no-evidence if NO_EVIDENCE ranks #1
4. margin: threshold on (best_real_score - NE_score)
5. calibrated_prob: threshold on sigmoid(max_score)
6. combined: weighted combination of signals

Usage:
    python scripts/hpo_ne_gate.py \
        --oof_cache outputs/oof_cache/oof_predictions.parquet \
        --targets configs/deployment_targets.yaml \
        --profile high_recall_low_hallucination \
        --outdir outputs/hpo_ne_gate

Output structure:
    outputs/hpo_ne_gate/
    ├── results.json          # Best config per method
    ├── grid_search.csv       # Full grid search results
    ├── report.md             # Human-readable report
    └── curves/               # ROC/PR curves
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
np.random.seed(SEED)


@dataclass
class NEGateResult:
    """Result for a single NE gate configuration."""
    method: str
    threshold: float
    tpr: float  # True positive rate (sensitivity/recall)
    fpr: float  # False positive rate
    tnr: float  # True negative rate (specificity)
    precision: float
    f1: float
    mcc: float
    auroc: float
    auprc: float
    accuracy: float
    balanced_acc: float
    meets_targets: bool
    extra: Dict[str, Any] = None


def compute_ne_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute binary classification metrics for NE detection.

    Args:
        y_true: Ground truth (1 = has evidence, 0 = no evidence)
        y_pred: Predictions (1 = predict has evidence, 0 = predict no evidence)
        y_score: Continuous scores (higher = more likely has evidence)

    Returns:
        Dictionary of metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Basic metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False positive rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False negative rate

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2

    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUROC and AUPRC
    auroc = 0.0
    auprc = 0.0
    if len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        except Exception:
            pass

    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def apply_ne_method(
    df: pd.DataFrame,
    method: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply NE detection method to get predictions and scores.

    Args:
        df: DataFrame with OOF predictions
        method: NE detection method
        threshold: Decision threshold

    Returns:
        (y_pred, y_score) arrays
    """
    n = len(df)
    y_score = np.zeros(n)
    y_pred = np.zeros(n, dtype=int)

    if method == "max_score":
        # Threshold on max reranker score
        y_score = df["max_score"].values
        y_pred = (y_score >= threshold).astype(int)

    elif method == "score_gap":
        # Threshold on top1 - top2 gap
        y_score = df["top1_minus_top2"].values
        y_pred = (y_score >= threshold).astype(int)

    elif method == "score_gap_mean":
        # Threshold on top1 - mean gap
        y_score = df["top1_minus_mean"].values
        y_pred = (y_score >= threshold).astype(int)

    elif method == "ne_rank":
        # NO_EVIDENCE rank-based: predict no-evidence if NE ranks #1
        # We need to compute NE rank from scores
        y_score = np.zeros(n)
        for i, (_, row) in enumerate(df.iterrows()):
            reranker_scores = json.loads(row["reranker_scores"])
            ne_score = row.get("ne_score", None)
            if ne_score is None or not reranker_scores:
                # Fallback: use max_score
                y_score[i] = row["max_score"]
            else:
                # Count how many real candidates score above NE
                n_above_ne = sum(1 for s in reranker_scores if s > ne_score)
                # Score = number of candidates above NE (higher = more likely has evidence)
                y_score[i] = n_above_ne
        y_pred = (y_score >= threshold).astype(int)

    elif method == "margin":
        # Threshold on (best_real - NE_score) margin
        y_score = df["best_minus_ne"].fillna(0).values
        y_pred = (y_score >= threshold).astype(int)

    elif method == "calibrated_prob":
        # Threshold on sigmoid(max_score)
        y_score = expit(df["max_score"].values)
        y_pred = (y_score >= threshold).astype(int)

    elif method == "std_score":
        # Threshold on score standard deviation
        y_score = df["std_score"].values
        y_pred = (y_score >= threshold).astype(int)

    else:
        raise ValueError(f"Unknown method: {method}")

    return y_pred, y_score


def grid_search_threshold(
    df: pd.DataFrame,
    method: str,
    threshold_range: Tuple[float, float],
    n_steps: int = 200,
    max_fpr: float = 0.05,
    min_tpr: float = 0.90,
) -> List[Dict[str, Any]]:
    """Grid search over thresholds for a given method.

    Args:
        df: DataFrame with OOF predictions
        method: NE detection method
        threshold_range: (min, max) threshold range
        n_steps: Number of threshold steps
        max_fpr: Maximum allowed FPR
        min_tpr: Minimum required TPR

    Returns:
        List of results for each threshold
    """
    y_true = df["gt_has_evidence"].astype(int).values

    # Generate threshold grid
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

    results = []
    for threshold in thresholds:
        y_pred, y_score = apply_ne_method(df, method, threshold)
        metrics = compute_ne_metrics(y_true, y_pred, y_score)

        # Check if meets targets
        meets_targets = (metrics["fpr"] <= max_fpr) and (metrics["tpr"] >= min_tpr)

        result = {
            "method": method,
            "threshold": float(threshold),
            "meets_targets": meets_targets,
            **metrics,
        }
        results.append(result)

    return results


def find_best_threshold(
    results: List[Dict[str, Any]],
    optimization_metric: str = "f1",
    require_targets: bool = True,
) -> Optional[Dict[str, Any]]:
    """Find best threshold from grid search results.

    Uses lexicographic optimization:
    1. First filter by target constraints (if required)
    2. Then maximize the optimization metric

    Args:
        results: Grid search results
        optimization_metric: Metric to maximize
        require_targets: Only consider results meeting targets

    Returns:
        Best result or None
    """
    if require_targets:
        valid_results = [r for r in results if r["meets_targets"]]
    else:
        valid_results = results

    if not valid_results:
        return None

    # Sort by optimization metric (descending)
    valid_results.sort(key=lambda x: x.get(optimization_metric, 0), reverse=True)
    return valid_results[0]


def run_cross_validated_hpo(
    df: pd.DataFrame,
    method: str,
    targets: Dict[str, float],
    n_steps: int = 200,
) -> Dict[str, Any]:
    """Run cross-validated HPO for a method.

    For each fold:
    1. Use other folds for threshold tuning
    2. Evaluate on held-out fold

    Args:
        df: Full OOF DataFrame
        method: NE detection method
        targets: Deployment targets
        n_steps: Grid search steps

    Returns:
        Cross-validated results
    """
    max_fpr = targets.get("max_fpr", 0.05)
    min_tpr = targets.get("min_tpr", 0.90)

    fold_ids = sorted(df["fold_id"].unique())
    fold_results = []

    # Determine threshold range based on method
    if method == "max_score":
        threshold_range = (-10.0, 10.0)
    elif method == "score_gap":
        threshold_range = (-5.0, 10.0)
    elif method == "score_gap_mean":
        threshold_range = (-5.0, 10.0)
    elif method == "ne_rank":
        threshold_range = (0, 20)
    elif method == "margin":
        threshold_range = (-10.0, 15.0)
    elif method == "calibrated_prob":
        threshold_range = (0.0, 1.0)
    elif method == "std_score":
        threshold_range = (0.0, 5.0)
    else:
        threshold_range = (-10.0, 10.0)

    for fold_id in fold_ids:
        # Split: tune on other folds, eval on this fold
        tune_df = df[df["fold_id"] != fold_id]
        eval_df = df[df["fold_id"] == fold_id]

        # Grid search on tune set
        tune_results = grid_search_threshold(
            tune_df, method, threshold_range, n_steps, max_fpr, min_tpr
        )

        # Find best threshold
        best_tune = find_best_threshold(tune_results, "precision", require_targets=True)
        if best_tune is None:
            # Fallback: find threshold closest to meeting targets
            best_tune = find_best_threshold(tune_results, "balanced_acc", require_targets=False)

        if best_tune is None:
            continue

        best_threshold = best_tune["threshold"]

        # Evaluate on held-out fold
        y_true = eval_df["gt_has_evidence"].astype(int).values
        y_pred, y_score = apply_ne_method(eval_df, method, best_threshold)
        eval_metrics = compute_ne_metrics(y_true, y_pred, y_score)

        fold_results.append({
            "fold_id": fold_id,
            "threshold": best_threshold,
            "tune_metrics": {
                "fpr": best_tune["fpr"],
                "tpr": best_tune["tpr"],
                "precision": best_tune["precision"],
                "f1": best_tune["f1"],
            },
            "eval_metrics": eval_metrics,
        })

    # Aggregate results
    if not fold_results:
        return {"method": method, "success": False, "error": "No valid thresholds found"}

    # Average metrics across folds
    avg_threshold = np.mean([r["threshold"] for r in fold_results])

    metric_keys = ["tpr", "fpr", "tnr", "precision", "f1", "mcc", "auroc", "balanced_acc"]
    avg_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [r["eval_metrics"].get(key, 0) for r in fold_results]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    meets_targets = (avg_metrics["fpr"] <= max_fpr) and (avg_metrics["tpr"] >= min_tpr)

    return {
        "method": method,
        "success": True,
        "threshold": float(avg_threshold),
        "threshold_std": float(np.std([r["threshold"] for r in fold_results])),
        "meets_targets": meets_targets,
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "fold_results": fold_results,
    }


def generate_roc_curve(
    df: pd.DataFrame,
    methods: List[str],
    output_dir: Path,
    targets: Dict[str, float],
):
    """Generate ROC curves for all methods."""
    y_true = df["gt_has_evidence"].astype(int).values

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, colors):
        try:
            # Get scores for this method
            _, y_score = apply_ne_method(df, method, threshold=0)

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auroc = roc_auc_score(y_true, y_score)

            ax.plot(fpr, tpr, label=f'{method} (AUC={auroc:.3f})', color=color, linewidth=2)
        except Exception as e:
            print(f"  Warning: Could not compute ROC for {method}: {e}")

    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Target region
    max_fpr = targets.get("max_fpr", 0.05)
    min_tpr = targets.get("min_tpr", 0.90)
    ax.axvline(max_fpr, color='red', linestyle=':', linewidth=2, label=f'Max FPR ({max_fpr})')
    ax.axhline(min_tpr, color='green', linestyle=':', linewidth=2, label=f'Min TPR ({min_tpr})')

    # Shade target region
    ax.fill_between([0, max_fpr], [min_tpr, min_tpr], [1, 1], alpha=0.1, color='green')

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title('NE Gate ROC Curves by Method', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.3])  # Zoom into relevant region
    ax.set_ylim([0.7, 1.0])

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()


def generate_report(
    results: Dict[str, Any],
    output_dir: Path,
    targets: Dict[str, float],
) -> str:
    """Generate markdown report of HPO results."""
    lines = [
        "# No-Evidence Gate HPO Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Deployment Targets",
        "",
        f"- Max FPR: {targets.get('max_fpr', 0.05):.2%}",
        f"- Min TPR: {targets.get('min_tpr', 0.90):.2%}",
        "",
        "---",
        "",
        "## Method Comparison",
        "",
        "| Method | Threshold | TPR | FPR | Precision | F1 | MCC | Meets Targets |",
        "|--------|-----------|-----|-----|-----------|----|----|---------------|",
    ]

    method_results = results.get("method_results", {})
    for method, result in method_results.items():
        if not result.get("success", False):
            lines.append(f"| {method} | - | - | - | - | - | - | FAILED |")
            continue

        avg = result.get("avg_metrics", {})
        threshold = result.get("threshold", 0)
        meets = "✅" if result.get("meets_targets", False) else "❌"

        lines.append(
            f"| {method} | {threshold:.4f} | "
            f"{avg.get('tpr', 0):.2%} | {avg.get('fpr', 0):.2%} | "
            f"{avg.get('precision', 0):.2%} | {avg.get('f1', 0):.3f} | "
            f"{avg.get('mcc', 0):.3f} | {meets} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Best Configuration",
        "",
    ])

    best = results.get("best_method", {})
    if best:
        lines.extend([
            f"**Method:** `{best.get('method', 'N/A')}`",
            f"**Threshold:** {best.get('threshold', 0):.4f}",
            "",
            "### Performance (Mean ± Std across folds)",
            "",
        ])

        avg = best.get("avg_metrics", {})
        std = best.get("std_metrics", {})

        for metric in ["tpr", "fpr", "tnr", "precision", "f1", "mcc", "auroc"]:
            avg_val = avg.get(metric, 0)
            std_val = std.get(metric, 0)
            lines.append(f"- **{metric.upper()}:** {avg_val:.4f} ± {std_val:.4f}")
    else:
        lines.append("**No method meets all deployment targets.**")

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Per-Fold Results",
        "",
    ])

    if best and best.get("fold_results"):
        lines.append("| Fold | Threshold | TPR | FPR | Precision | F1 |")
        lines.append("|------|-----------|-----|-----|-----------|-----|")

        for fr in best["fold_results"]:
            em = fr.get("eval_metrics", {})
            lines.append(
                f"| {fr['fold_id']} | {fr['threshold']:.4f} | "
                f"{em.get('tpr', 0):.2%} | {em.get('fpr', 0):.2%} | "
                f"{em.get('precision', 0):.2%} | {em.get('f1', 0):.3f} |"
            )

    report = "\n".join(lines)

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="HPO for No-Evidence Gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oof_cache",
        type=str,
        default="outputs/oof_cache/oof_predictions.parquet",
        help="Path to OOF cache parquet",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="configs/deployment_targets.yaml",
        help="Path to deployment targets config",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="high_recall_low_hallucination",
        help="Deployment profile to use",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/hpo_ne_gate",
        help="Output directory",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=200,
        help="Number of threshold steps for grid search",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["max_score", "score_gap", "margin", "calibrated_prob", "ne_rank"],
        help="NE detection methods to evaluate",
    )
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NO-EVIDENCE GATE HPO")
    print("=" * 70)

    # Load deployment targets
    with open(args.targets, "r") as f:
        targets_config = yaml.safe_load(f)

    profile = targets_config["profiles"].get(args.profile, {})
    targets = profile.get("targets", {})

    print(f"Profile: {args.profile}")
    print(f"Targets: max_fpr={targets.get('max_fpr', 0.05)}, min_tpr={targets.get('min_tpr', 0.90)}")

    # Load OOF cache
    print(f"\nLoading OOF cache: {args.oof_cache}")
    df = pd.read_parquet(args.oof_cache)
    print(f"  Total queries: {len(df)}")
    print(f"  Has evidence: {df['gt_has_evidence'].sum()} ({df['gt_has_evidence'].mean():.1%})")
    print(f"  Folds: {sorted(df['fold_id'].unique())}")

    # Run HPO for each method
    print("\n" + "=" * 70)
    print("Running Cross-Validated HPO")
    print("=" * 70)

    method_results = {}
    for method in args.methods:
        print(f"\n[{method}] Running HPO...")
        result = run_cross_validated_hpo(df, method, targets, args.n_steps)
        method_results[method] = result

        if result.get("success"):
            avg = result.get("avg_metrics", {})
            print(f"  Threshold: {result.get('threshold', 0):.4f}")
            print(f"  TPR: {avg.get('tpr', 0):.2%}, FPR: {avg.get('fpr', 0):.2%}")
            print(f"  Meets targets: {result.get('meets_targets', False)}")
        else:
            print(f"  FAILED: {result.get('error', 'Unknown error')}")

    # Find best method
    print("\n" + "=" * 70)
    print("Selecting Best Method")
    print("=" * 70)

    # Priority: methods that meet targets, then by precision
    valid_methods = [
        (name, result) for name, result in method_results.items()
        if result.get("success") and result.get("meets_targets")
    ]

    if valid_methods:
        # Sort by precision (maximize)
        valid_methods.sort(
            key=lambda x: x[1].get("avg_metrics", {}).get("precision", 0),
            reverse=True,
        )
        best_name, best_result = valid_methods[0]
        print(f"Best method meeting targets: {best_name}")
    else:
        # Fallback: find method with best balanced accuracy
        print("Warning: No method meets all targets. Using best balanced accuracy.")
        all_methods = [
            (name, result) for name, result in method_results.items()
            if result.get("success")
        ]
        if all_methods:
            all_methods.sort(
                key=lambda x: x[1].get("avg_metrics", {}).get("balanced_acc", 0),
                reverse=True,
            )
            best_name, best_result = all_methods[0]
            print(f"Best method (fallback): {best_name}")
        else:
            best_name, best_result = None, None
            print("ERROR: No valid methods found")

    # Generate plots
    print("\nGenerating ROC curves...")
    generate_roc_curve(df, args.methods, curves_dir, targets)

    # Compile results
    results = {
        "profile": args.profile,
        "targets": targets,
        "method_results": method_results,
        "best_method": best_result,
        "best_method_name": best_name,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results: {output_dir / 'results.json'}")

    # Generate report
    print("Generating report...")
    report = generate_report(results, output_dir, targets)
    print(f"Saved report: {output_dir / 'report.md'}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if best_result:
        avg = best_result.get("avg_metrics", {})
        print(f"Best Method: {best_name}")
        print(f"Threshold: {best_result.get('threshold', 0):.4f}")
        print(f"TPR (positive coverage): {avg.get('tpr', 0):.2%}")
        print(f"FPR (negative coverage): {avg.get('fpr', 0):.2%}")
        print(f"Precision: {avg.get('precision', 0):.2%}")
        print(f"F1: {avg.get('f1', 0):.3f}")
        print(f"Meets targets: {best_result.get('meets_targets', False)}")
    else:
        print("No valid method found")

    print("\n" + "=" * 70)
    print("NE GATE HPO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
