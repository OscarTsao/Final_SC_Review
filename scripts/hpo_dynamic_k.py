#!/usr/bin/env python3
"""
HPO for Dynamic-K Selection.

This script optimizes the dynamic-K selection strategy for evidence extraction.
Only evaluated on has-evidence queries (GT positives).

Target:
- min_evidence_recall ≥ 93% (aim near reranker@5 baseline)
- max_avg_k ≤ 5.0 (cost control)

Methods evaluated:
1. fixed_k: Fixed K value (baseline comparison)
2. score_gap: K where score gap exceeds threshold
3. score_ratio: K where score ratio exceeds threshold
4. prob_threshold: K where calibrated probability drops below threshold
5. ne_boundary: K = rank of NO_EVIDENCE - 1 (all candidates above NE)
6. margin_threshold: K = number of candidates with (score - NE_score) > margin

Usage:
    python scripts/hpo_dynamic_k.py \
        --oof_cache outputs/oof_cache/oof_predictions.parquet \
        --targets configs/deployment_targets.yaml \
        --profile high_recall_low_hallucination \
        --outdir outputs/hpo_dynamic_k

Output structure:
    outputs/hpo_dynamic_k/
    ├── results.json          # Best config per method
    ├── grid_search.csv       # Full grid search results
    ├── report.md             # Human-readable report
    └── curves/               # K distribution, recall vs K curves
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
np.random.seed(SEED)


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@k."""
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute nDCG@k."""
    if not relevances or sum(relevances) == 0:
        return 0.0
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevances: List[int], k: int) -> float:
    """Compute Recall@k."""
    total_pos = sum(relevances)
    if total_pos == 0:
        return 0.0
    return sum(relevances[:k]) / total_pos


def precision_at_k(relevances: List[int], k: int) -> float:
    """Compute Precision@k."""
    if k == 0:
        return 0.0
    return sum(relevances[:k]) / k


def compute_dynamic_k(
    reranker_scores: List[float],
    labels: List[int],
    method: str,
    threshold: float,
    ne_score: Optional[float] = None,
    calibrated_probs: Optional[List[float]] = None,
    max_k: int = 20,
) -> int:
    """Compute dynamic K for a single query.

    Args:
        reranker_scores: Reranker scores for candidates (sorted by score descending)
        labels: Binary labels (1 = relevant)
        method: K-selection method
        threshold: Decision threshold
        ne_score: NO_EVIDENCE pseudo-candidate score (if available)
        calibrated_probs: Calibrated probabilities (if available)
        max_k: Maximum K to return

    Returns:
        Selected K value
    """
    if not reranker_scores:
        return 1

    n = len(reranker_scores)

    # Sort by score descending
    sorted_indices = np.argsort(reranker_scores)[::-1]
    sorted_scores = [reranker_scores[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    if method == "fixed_k":
        # Fixed K (threshold is the K value)
        k = min(int(threshold), n)

    elif method == "score_gap":
        # K = position after largest score gap exceeding threshold
        if n < 2:
            k = 1
        else:
            gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
            # Find first gap >= threshold
            k = n  # default: return all
            for i, gap in enumerate(gaps):
                if gap >= threshold:
                    k = i + 1
                    break

    elif method == "score_ratio":
        # K = position where score ratio drops below threshold
        if n < 2:
            k = 1
        else:
            k = n
            for i in range(1, n):
                if sorted_scores[0] == 0:
                    continue
                ratio = sorted_scores[i] / (sorted_scores[0] + 1e-10)
                if ratio < threshold:
                    k = i
                    break

    elif method == "prob_threshold":
        # K = number of candidates with calibrated prob >= threshold
        if calibrated_probs is None:
            # Fallback to sigmoid of scores
            probs = expit(np.array(sorted_scores))
        else:
            sorted_probs = [calibrated_probs[i] for i in sorted_indices]
            probs = np.array(sorted_probs)

        k = sum(1 for p in probs if p >= threshold)
        k = max(k, 1)  # Return at least 1

    elif method == "ne_boundary":
        # K = number of candidates ranking above NO_EVIDENCE
        if ne_score is None:
            k = 1  # Fallback
        else:
            k = sum(1 for s in sorted_scores if s > ne_score)
            k = max(k, 1)  # Return at least 1

    elif method == "margin_threshold":
        # K = number of candidates with (score - NE_score) >= threshold
        if ne_score is None:
            k = 1  # Fallback
        else:
            k = sum(1 for s in sorted_scores if (s - ne_score) >= threshold)
            k = max(k, 1)  # Return at least 1

    elif method == "top1_gap":
        # K = 1 if top1-top2 gap > threshold, else expand based on gap
        if n < 2:
            k = 1
        else:
            gap = sorted_scores[0] - sorted_scores[1]
            if gap >= threshold:
                k = 1
            else:
                # Find natural break point
                gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
                max_gap_idx = np.argmax(gaps)
                k = max_gap_idx + 1

    else:
        raise ValueError(f"Unknown method: {method}")

    return min(max(k, 1), min(n, max_k))


def evaluate_dynamic_k_method(
    df: pd.DataFrame,
    method: str,
    threshold: float,
    max_k: int = 20,
) -> Dict[str, float]:
    """Evaluate dynamic-K method on has-evidence queries.

    Args:
        df: OOF DataFrame (filtered to has-evidence only)
        method: K-selection method
        threshold: Decision threshold
        max_k: Maximum K

    Returns:
        Dictionary of metrics
    """
    recalls = []
    ndcgs = []
    precisions = []
    k_values = []

    for _, row in df.iterrows():
        reranker_scores = json.loads(row["reranker_scores"])
        labels = json.loads(row["candidate_labels"])
        ne_score = row.get("ne_score", None)
        calibrated_probs = json.loads(row.get("calibrated_probs", "[]"))

        if not reranker_scores or sum(labels) == 0:
            continue

        # Compute K
        k = compute_dynamic_k(
            reranker_scores=reranker_scores,
            labels=labels,
            method=method,
            threshold=threshold,
            ne_score=ne_score,
            calibrated_probs=calibrated_probs if calibrated_probs else None,
            max_k=max_k,
        )

        # Sort by score and compute metrics
        sorted_indices = np.argsort(reranker_scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]

        recalls.append(recall_at_k(sorted_labels, k))
        ndcgs.append(ndcg_at_k(sorted_labels, k))
        precisions.append(precision_at_k(sorted_labels, k))
        k_values.append(k)

    if not recalls:
        return {
            "recall": 0.0,
            "ndcg": 0.0,
            "precision": 0.0,
            "avg_k": 0.0,
            "std_k": 0.0,
            "n_queries": 0,
        }

    return {
        "recall": np.mean(recalls),
        "ndcg": np.mean(ndcgs),
        "precision": np.mean(precisions),
        "avg_k": np.mean(k_values),
        "std_k": np.std(k_values),
        "n_queries": len(recalls),
        "k_values": k_values,  # For distribution analysis
    }


def grid_search_dynamic_k(
    df: pd.DataFrame,
    method: str,
    threshold_range: Tuple[float, float],
    n_steps: int = 100,
    min_recall: float = 0.93,
    max_avg_k: float = 5.0,
) -> List[Dict[str, Any]]:
    """Grid search over thresholds for a K-selection method.

    Args:
        df: DataFrame (has-evidence queries only)
        method: K-selection method
        threshold_range: (min, max) threshold range
        n_steps: Number of steps
        min_recall: Minimum recall target
        max_avg_k: Maximum avg K target

    Returns:
        List of results
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

    results = []
    for threshold in thresholds:
        metrics = evaluate_dynamic_k_method(df, method, threshold)

        # Check if meets targets
        meets_targets = (
            metrics["recall"] >= min_recall and metrics["avg_k"] <= max_avg_k
        )

        result = {
            "method": method,
            "threshold": float(threshold),
            "meets_targets": meets_targets,
            "recall": metrics["recall"],
            "ndcg": metrics["ndcg"],
            "precision": metrics["precision"],
            "avg_k": metrics["avg_k"],
            "std_k": metrics["std_k"],
        }
        results.append(result)

    return results


def run_cross_validated_dynamic_k_hpo(
    df: pd.DataFrame,
    method: str,
    targets: Dict[str, float],
    n_steps: int = 100,
) -> Dict[str, Any]:
    """Run cross-validated HPO for dynamic-K method.

    Args:
        df: Full OOF DataFrame
        method: K-selection method
        targets: Deployment targets
        n_steps: Grid search steps

    Returns:
        Cross-validated results
    """
    min_recall = targets.get("min_evidence_recall", 0.93)
    max_avg_k = targets.get("max_avg_k", 5.0)

    # Filter to has-evidence queries only
    has_ev_df = df[df["gt_has_evidence"] == True].copy()

    if len(has_ev_df) == 0:
        return {"method": method, "success": False, "error": "No has-evidence queries"}

    fold_ids = sorted(has_ev_df["fold_id"].unique())
    fold_results = []

    # Determine threshold range based on method
    if method == "fixed_k":
        threshold_range = (1, 20)
        n_steps = 20  # Integer K values
    elif method == "score_gap":
        threshold_range = (0.0, 5.0)
    elif method == "score_ratio":
        threshold_range = (0.1, 0.99)
    elif method == "prob_threshold":
        threshold_range = (0.1, 0.9)
    elif method == "ne_boundary":
        # No threshold for ne_boundary, just evaluate once
        threshold_range = (0, 1)
        n_steps = 1
    elif method == "margin_threshold":
        threshold_range = (-5.0, 10.0)
    elif method == "top1_gap":
        threshold_range = (0.0, 5.0)
    else:
        threshold_range = (0.0, 5.0)

    for fold_id in fold_ids:
        # Split: tune on other folds, eval on this fold
        tune_df = has_ev_df[has_ev_df["fold_id"] != fold_id]
        eval_df = has_ev_df[has_ev_df["fold_id"] == fold_id]

        if len(tune_df) == 0 or len(eval_df) == 0:
            continue

        # Grid search on tune set
        tune_results = grid_search_dynamic_k(
            tune_df, method, threshold_range, n_steps, min_recall, max_avg_k
        )

        # Find best threshold (lexicographic: meet targets → minimize avg_k → maximize recall)
        valid_results = [r for r in tune_results if r["meets_targets"]]

        if valid_results:
            # Sort by avg_k (minimize), then by recall (maximize)
            valid_results.sort(key=lambda x: (x["avg_k"], -x["recall"]))
            best_tune = valid_results[0]
        else:
            # Fallback: maximize recall
            tune_results.sort(key=lambda x: x["recall"], reverse=True)
            best_tune = tune_results[0] if tune_results else None

        if best_tune is None:
            continue

        best_threshold = best_tune["threshold"]

        # Evaluate on held-out fold
        eval_metrics = evaluate_dynamic_k_method(eval_df, method, best_threshold)

        fold_results.append({
            "fold_id": fold_id,
            "threshold": best_threshold,
            "tune_metrics": {
                "recall": best_tune["recall"],
                "avg_k": best_tune["avg_k"],
            },
            "eval_metrics": eval_metrics,
        })

    # Aggregate results
    if not fold_results:
        return {"method": method, "success": False, "error": "No valid thresholds found"}

    avg_threshold = np.mean([r["threshold"] for r in fold_results])

    metric_keys = ["recall", "ndcg", "precision", "avg_k", "std_k"]
    avg_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [r["eval_metrics"].get(key, 0) for r in fold_results]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    meets_targets = (
        avg_metrics["recall"] >= min_recall and avg_metrics["avg_k"] <= max_avg_k
    )

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


def evaluate_fixed_k_baselines(
    df: pd.DataFrame,
    k_values: List[int] = [1, 2, 3, 5, 10, 20],
) -> Dict[int, Dict[str, float]]:
    """Evaluate fixed-K baselines.

    Args:
        df: OOF DataFrame
        k_values: K values to evaluate

    Returns:
        Results per K
    """
    has_ev_df = df[df["gt_has_evidence"] == True]

    results = {}
    for k in k_values:
        metrics = evaluate_dynamic_k_method(has_ev_df, "fixed_k", k)
        results[k] = {
            "recall": metrics["recall"],
            "ndcg": metrics["ndcg"],
            "precision": metrics["precision"],
            "avg_k": k,
        }

    return results


def generate_recall_vs_k_curve(
    df: pd.DataFrame,
    output_dir: Path,
    targets: Dict[str, float],
):
    """Generate recall vs avg_k curves for different methods."""
    has_ev_df = df[df["gt_has_evidence"] == True]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Fixed K baseline
    fixed_k_results = evaluate_fixed_k_baselines(has_ev_df, list(range(1, 21)))
    k_vals = list(fixed_k_results.keys())
    recalls = [fixed_k_results[k]["recall"] for k in k_vals]
    ax.plot(k_vals, recalls, 'ko-', label='Fixed K', linewidth=2, markersize=6)

    # Score gap method at different thresholds
    score_gap_results = grid_search_dynamic_k(
        has_ev_df, "score_gap", (0.0, 5.0), 50,
        targets.get("min_evidence_recall", 0.93),
        targets.get("max_avg_k", 5.0),
    )
    sg_avg_k = [r["avg_k"] for r in score_gap_results]
    sg_recall = [r["recall"] for r in score_gap_results]
    ax.plot(sg_avg_k, sg_recall, 'b.-', label='Score Gap', alpha=0.7)

    # Margin method at different thresholds
    if "ne_score" in df.columns and df["ne_score"].notna().any():
        margin_results = grid_search_dynamic_k(
            has_ev_df, "margin_threshold", (-5.0, 10.0), 50,
            targets.get("min_evidence_recall", 0.93),
            targets.get("max_avg_k", 5.0),
        )
        m_avg_k = [r["avg_k"] for r in margin_results]
        m_recall = [r["recall"] for r in margin_results]
        ax.plot(m_avg_k, m_recall, 'r.-', label='Margin Threshold', alpha=0.7)

    # Target lines
    min_recall = targets.get("min_evidence_recall", 0.93)
    max_avg_k = targets.get("max_avg_k", 5.0)
    ax.axhline(min_recall, color='green', linestyle='--', linewidth=2, label=f'Min Recall ({min_recall:.0%})')
    ax.axvline(max_avg_k, color='red', linestyle='--', linewidth=2, label=f'Max Avg K ({max_avg_k})')

    # Shade target region
    ax.fill_between([0, max_avg_k], [min_recall, min_recall], [1, 1], alpha=0.1, color='green')

    ax.set_xlabel('Average K', fontsize=12)
    ax.set_ylabel('Evidence Recall', fontsize=12)
    ax.set_title('Evidence Recall vs Average K', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 15])
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(output_dir / "recall_vs_k.png", dpi=150)
    plt.close()


def generate_report(
    results: Dict[str, Any],
    output_dir: Path,
    targets: Dict[str, float],
    fixed_k_baselines: Dict[int, Dict[str, float]],
) -> str:
    """Generate markdown report."""
    lines = [
        "# Dynamic-K HPO Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Deployment Targets",
        "",
        f"- Min Evidence Recall: {targets.get('min_evidence_recall', 0.93):.2%}",
        f"- Max Avg K: {targets.get('max_avg_k', 5.0)}",
        "",
        "---",
        "",
        "## Fixed-K Baselines",
        "",
        "| K | Recall | nDCG | Precision |",
        "|---|--------|------|-----------|",
    ]

    for k in sorted(fixed_k_baselines.keys()):
        m = fixed_k_baselines[k]
        lines.append(f"| {k} | {m['recall']:.2%} | {m['ndcg']:.3f} | {m['precision']:.2%} |")

    lines.extend([
        "",
        "---",
        "",
        "## Method Comparison",
        "",
        "| Method | Threshold | Recall | nDCG | Precision | Avg K | Meets Targets |",
        "|--------|-----------|--------|------|-----------|-------|---------------|",
    ])

    method_results = results.get("method_results", {})
    for method, result in method_results.items():
        if not result.get("success", False):
            lines.append(f"| {method} | - | - | - | - | - | FAILED |")
            continue

        avg = result.get("avg_metrics", {})
        threshold = result.get("threshold", 0)
        meets = "✅" if result.get("meets_targets", False) else "❌"

        lines.append(
            f"| {method} | {threshold:.4f} | "
            f"{avg.get('recall', 0):.2%} | {avg.get('ndcg', 0):.3f} | "
            f"{avg.get('precision', 0):.2%} | {avg.get('avg_k', 0):.2f} | {meets} |"
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

        for metric in ["recall", "ndcg", "precision", "avg_k"]:
            avg_val = avg.get(metric, 0)
            std_val = std.get(metric, 0)
            lines.append(f"- **{metric}:** {avg_val:.4f} ± {std_val:.4f}")
    else:
        lines.append("**No method meets all deployment targets.**")

    report = "\n".join(lines)

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="HPO for Dynamic-K Selection",
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
        default="outputs/hpo_dynamic_k",
        help="Output directory",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of threshold steps for grid search",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["score_gap", "margin_threshold", "prob_threshold", "ne_boundary"],
        help="K-selection methods to evaluate",
    )
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("DYNAMIC-K HPO")
    print("=" * 70)

    # Load deployment targets
    with open(args.targets, "r") as f:
        targets_config = yaml.safe_load(f)

    profile = targets_config["profiles"].get(args.profile, {})
    targets = profile.get("targets", {})

    print(f"Profile: {args.profile}")
    print(f"Targets: min_recall={targets.get('min_evidence_recall', 0.93)}, max_avg_k={targets.get('max_avg_k', 5.0)}")

    # Load OOF cache
    print(f"\nLoading OOF cache: {args.oof_cache}")
    df = pd.read_parquet(args.oof_cache)
    has_ev_df = df[df["gt_has_evidence"] == True]
    print(f"  Total queries: {len(df)}")
    print(f"  Has-evidence queries: {len(has_ev_df)}")
    print(f"  Folds: {sorted(df['fold_id'].unique())}")

    # Fixed-K baselines
    print("\n" + "=" * 70)
    print("Fixed-K Baselines")
    print("=" * 70)

    fixed_k_baselines = evaluate_fixed_k_baselines(df)
    for k, metrics in sorted(fixed_k_baselines.items()):
        print(f"  K={k}: Recall={metrics['recall']:.2%}, nDCG={metrics['ndcg']:.3f}, Precision={metrics['precision']:.2%}")

    # Run HPO for each method
    print("\n" + "=" * 70)
    print("Running Cross-Validated HPO")
    print("=" * 70)

    method_results = {}
    for method in args.methods:
        print(f"\n[{method}] Running HPO...")
        result = run_cross_validated_dynamic_k_hpo(df, method, targets, args.n_steps)
        method_results[method] = result

        if result.get("success"):
            avg = result.get("avg_metrics", {})
            print(f"  Threshold: {result.get('threshold', 0):.4f}")
            print(f"  Recall: {avg.get('recall', 0):.2%}, Avg K: {avg.get('avg_k', 0):.2f}")
            print(f"  Meets targets: {result.get('meets_targets', False)}")
        else:
            print(f"  FAILED: {result.get('error', 'Unknown error')}")

    # Find best method
    print("\n" + "=" * 70)
    print("Selecting Best Method")
    print("=" * 70)

    # Priority: methods that meet targets, then by avg_k (minimize)
    valid_methods = [
        (name, result) for name, result in method_results.items()
        if result.get("success") and result.get("meets_targets")
    ]

    if valid_methods:
        # Sort by avg_k (minimize), then by recall (maximize)
        valid_methods.sort(
            key=lambda x: (
                x[1].get("avg_metrics", {}).get("avg_k", 100),
                -x[1].get("avg_metrics", {}).get("recall", 0),
            )
        )
        best_name, best_result = valid_methods[0]
        print(f"Best method meeting targets: {best_name}")
    else:
        # Fallback: find method with highest recall
        print("Warning: No method meets all targets. Using best recall.")
        all_methods = [
            (name, result) for name, result in method_results.items()
            if result.get("success")
        ]
        if all_methods:
            all_methods.sort(
                key=lambda x: x[1].get("avg_metrics", {}).get("recall", 0),
                reverse=True,
            )
            best_name, best_result = all_methods[0]
            print(f"Best method (fallback): {best_name}")
        else:
            best_name, best_result = None, None
            print("ERROR: No valid methods found")

    # Generate plots
    print("\nGenerating recall vs K curve...")
    generate_recall_vs_k_curve(df, curves_dir, targets)

    # Compile results
    results = {
        "profile": args.profile,
        "targets": targets,
        "fixed_k_baselines": fixed_k_baselines,
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
    report = generate_report(results, output_dir, targets, fixed_k_baselines)
    print(f"Saved report: {output_dir / 'report.md'}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if best_result:
        avg = best_result.get("avg_metrics", {})
        print(f"Best Method: {best_name}")
        print(f"Threshold: {best_result.get('threshold', 0):.4f}")
        print(f"Evidence Recall: {avg.get('recall', 0):.2%}")
        print(f"nDCG: {avg.get('ndcg', 0):.3f}")
        print(f"Precision: {avg.get('precision', 0):.2%}")
        print(f"Avg K: {avg.get('avg_k', 0):.2f}")
        print(f"Meets targets: {best_result.get('meets_targets', False)}")
    else:
        print("No valid method found")

    print("\n" + "=" * 70)
    print("DYNAMIC-K HPO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
