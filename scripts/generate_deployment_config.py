#!/usr/bin/env python3
"""
Generate Unified Deployment Configuration.

This script combines results from NE gate HPO and Dynamic-K HPO into a single
deployment configuration, then runs final E2E assessment.

Usage:
    python scripts/generate_deployment_config.py \
        --ne_hpo outputs/hpo_ne_gate/results.json \
        --dk_hpo outputs/hpo_dynamic_k/results.json \
        --oof_cache outputs/oof_cache/oof_predictions.parquet \
        --targets configs/deployment_targets.yaml \
        --profile high_recall_low_hallucination \
        --outdir outputs/deployment_optimized

Output:
    outputs/deployment_optimized/
    ├── deployment_config.yaml    # Unified deployment config
    ├── e2e_assessment.json       # Final E2E metrics
    ├── comparison_report.md      # Before/after comparison
    └── curves/                   # Performance curves
"""

import argparse
import json
import sys
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
    confusion_matrix,
    matthews_corrcoef,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
np.random.seed(SEED)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


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


def apply_ne_detection(
    row: pd.Series,
    ne_config: Dict[str, Any],
) -> bool:
    """Apply NE detection to a single query."""
    method = ne_config.get("method", "max_score")
    threshold = ne_config.get("threshold", 0.0)

    if method == "max_score":
        score = row["max_score"]
        return score >= threshold

    elif method == "score_gap":
        score = row["top1_minus_top2"]
        return score >= threshold

    elif method == "margin":
        score = row.get("best_minus_ne", 0)
        if pd.isna(score):
            score = 0
        return score >= threshold

    elif method == "calibrated_prob":
        score = expit(row["max_score"])
        return score >= threshold

    elif method == "ne_rank":
        reranker_scores = json.loads(row["reranker_scores"])
        ne_score = row.get("ne_score", None)
        if ne_score is None or not reranker_scores:
            return row["max_score"] >= threshold
        n_above_ne = sum(1 for s in reranker_scores if s > ne_score)
        return n_above_ne >= threshold

    else:
        # Default: max_score
        return row["max_score"] >= threshold


def compute_dynamic_k(
    row: pd.Series,
    dk_config: Dict[str, Any],
    max_k: int = 20,
) -> int:
    """Compute dynamic K for a single query."""
    method = dk_config.get("method", "score_gap")
    threshold = dk_config.get("threshold", 0.5)

    reranker_scores = json.loads(row["reranker_scores"])
    ne_score = row.get("ne_score", None)

    if not reranker_scores:
        return 1

    n = len(reranker_scores)
    sorted_scores = sorted(reranker_scores, reverse=True)

    if method == "fixed_k":
        k = min(int(threshold), n)

    elif method == "score_gap":
        if n < 2:
            k = 1
        else:
            gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
            k = n
            for i, gap in enumerate(gaps):
                if gap >= threshold:
                    k = i + 1
                    break

    elif method == "margin_threshold":
        if ne_score is None:
            k = 1
        else:
            k = sum(1 for s in sorted_scores if (s - ne_score) >= threshold)
            k = max(k, 1)

    elif method == "ne_boundary":
        if ne_score is None:
            k = 1
        else:
            k = sum(1 for s in sorted_scores if s > ne_score)
            k = max(k, 1)

    elif method == "prob_threshold":
        probs = expit(np.array(sorted_scores))
        k = sum(1 for p in probs if p >= threshold)
        k = max(k, 1)

    else:
        k = min(5, n)  # Default

    return min(max(k, 1), min(n, max_k))


def run_e2e_assessment(
    df: pd.DataFrame,
    ne_config: Dict[str, Any],
    dk_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run E2E assessment with given configuration.

    Args:
        df: OOF DataFrame
        ne_config: NE detection config {"method": ..., "threshold": ...}
        dk_config: Dynamic-K config {"method": ..., "threshold": ...}

    Returns:
        E2E assessment results
    """
    # Apply NE detection
    y_true = df["gt_has_evidence"].astype(int).values
    y_pred = np.array([
        1 if apply_ne_detection(row, ne_config) else 0
        for _, row in df.iterrows()
    ])

    # NE detection metrics
    ne_metrics = compute_binary_metrics(y_true, y_pred)

    # Dynamic-K on predicted positives with GT positive
    has_ev_df = df[df["gt_has_evidence"] == True]
    predicted_positive_df = df[(y_pred == 1) & (df["gt_has_evidence"] == True)]

    recalls = []
    ndcgs = []
    precisions = []
    k_values = []

    for _, row in has_ev_df.iterrows():
        # Only compute recall if we would predict positive
        pred_has_ev = apply_ne_detection(row, ne_config)

        if pred_has_ev:
            k = compute_dynamic_k(row, dk_config)
        else:
            k = 0  # No evidence returned

        reranker_scores = json.loads(row["reranker_scores"])
        labels = json.loads(row["candidate_labels"])

        if not reranker_scores or sum(labels) == 0:
            continue

        sorted_indices = np.argsort(reranker_scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]

        if k > 0:
            recalls.append(recall_at_k(sorted_labels, k))
            ndcgs.append(ndcg_at_k(sorted_labels, k))
            precisions.append(precision_at_k(sorted_labels, k))
            k_values.append(k)
        else:
            recalls.append(0.0)
            ndcgs.append(0.0)
            precisions.append(0.0)
            k_values.append(0)

    # All queries: compute avg K only on predicted positives
    all_k_values = []
    for _, row in df.iterrows():
        pred_has_ev = apply_ne_detection(row, ne_config)
        if pred_has_ev:
            k = compute_dynamic_k(row, dk_config)
            all_k_values.append(k)
        else:
            all_k_values.append(0)

    # Compute coverage metrics
    positive_coverage = ne_metrics["tpr"]  # Same as TPR
    negative_coverage = ne_metrics["fpr"]  # Same as FPR

    results = {
        "ne_detection": {
            "method": ne_config.get("method"),
            "threshold": ne_config.get("threshold"),
            **ne_metrics,
        },
        "dynamic_k": {
            "method": dk_config.get("method"),
            "threshold": dk_config.get("threshold"),
        },
        "coverage": {
            "positive_coverage": positive_coverage,
            "negative_coverage": negative_coverage,
        },
        "evidence_extraction": {
            "recall": np.mean(recalls) if recalls else 0.0,
            "ndcg": np.mean(ndcgs) if ndcgs else 0.0,
            "precision": np.mean(precisions) if precisions else 0.0,
        },
        "cost": {
            "avg_k_predicted_positive": np.mean([k for k in all_k_values if k > 0]) if any(k > 0 for k in all_k_values) else 0.0,
            "avg_k_overall": np.mean(all_k_values) if all_k_values else 0.0,
            "avg_k_has_evidence": np.mean(k_values) if k_values else 0.0,
        },
        "n_queries": {
            "total": len(df),
            "has_evidence": int(df["gt_has_evidence"].sum()),
            "no_evidence": int((~df["gt_has_evidence"]).sum()),
        },
    }

    return results


def run_cross_validated_assessment(
    df: pd.DataFrame,
    ne_config: Dict[str, Any],
    dk_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run cross-validated E2E assessment."""
    fold_ids = sorted(df["fold_id"].unique())
    fold_results = []

    for fold_id in fold_ids:
        fold_df = df[df["fold_id"] == fold_id]
        fold_result = run_e2e_assessment(fold_df, ne_config, dk_config)
        fold_result["fold_id"] = fold_id
        fold_results.append(fold_result)

    # Aggregate
    metrics_to_average = [
        ("ne_detection", "tpr"),
        ("ne_detection", "fpr"),
        ("ne_detection", "precision"),
        ("ne_detection", "f1"),
        ("coverage", "positive_coverage"),
        ("coverage", "negative_coverage"),
        ("evidence_extraction", "recall"),
        ("evidence_extraction", "ndcg"),
        ("evidence_extraction", "precision"),
        ("cost", "avg_k_has_evidence"),
    ]

    avg_metrics = {}
    std_metrics = {}

    for section, metric in metrics_to_average:
        values = [fr[section][metric] for fr in fold_results]
        avg_metrics[f"{section}.{metric}"] = np.mean(values)
        std_metrics[f"{section}.{metric}"] = np.std(values)

    return {
        "ne_config": ne_config,
        "dk_config": dk_config,
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "fold_results": fold_results,
    }


def generate_deployment_config(
    ne_hpo_results: Dict[str, Any],
    dk_hpo_results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate unified deployment config from HPO results."""
    # Extract best configs
    ne_best = ne_hpo_results.get("best_method", {})
    dk_best = dk_hpo_results.get("best_method", {})

    config = {
        "deployment_config": {
            "generated_at": datetime.now().isoformat(),
            "profile": ne_hpo_results.get("profile", "high_recall_low_hallucination"),
        },
        "ne_detection": {
            "method": ne_hpo_results.get("best_method_name", "max_score"),
            "threshold": ne_best.get("threshold", 0.0) if ne_best else 0.0,
        },
        "dynamic_k": {
            "method": dk_hpo_results.get("best_method_name", "score_gap"),
            "threshold": dk_best.get("threshold", 0.5) if dk_best else 0.5,
            "max_k": 20,
        },
        "targets": ne_hpo_results.get("targets", {}),
    }

    # Save YAML config
    config_path = output_dir / "deployment_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config


def generate_comparison_report(
    baseline: Dict[str, float],
    optimized: Dict[str, Any],
    targets: Dict[str, float],
    output_dir: Path,
) -> str:
    """Generate comparison report between baseline and optimized."""
    lines = [
        "# Deployment Optimization Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
    ]

    # Key metrics comparison
    opt_avg = optimized.get("avg_metrics", {})

    # Check if targets are met
    targets_met = []
    targets_missed = []

    max_fpr = targets.get("max_fpr", 0.05)
    opt_fpr = opt_avg.get("coverage.negative_coverage", 1.0)
    if opt_fpr <= max_fpr:
        targets_met.append(f"FPR: {opt_fpr:.2%} ≤ {max_fpr:.2%}")
    else:
        targets_missed.append(f"FPR: {opt_fpr:.2%} > {max_fpr:.2%}")

    min_tpr = targets.get("min_tpr", 0.90)
    opt_tpr = opt_avg.get("coverage.positive_coverage", 0.0)
    if opt_tpr >= min_tpr:
        targets_met.append(f"TPR: {opt_tpr:.2%} ≥ {min_tpr:.2%}")
    else:
        targets_missed.append(f"TPR: {opt_tpr:.2%} < {min_tpr:.2%}")

    min_recall = targets.get("min_evidence_recall", 0.93)
    opt_recall = opt_avg.get("evidence_extraction.recall", 0.0)
    if opt_recall >= min_recall:
        targets_met.append(f"Evidence Recall: {opt_recall:.2%} ≥ {min_recall:.2%}")
    else:
        targets_missed.append(f"Evidence Recall: {opt_recall:.2%} < {min_recall:.2%}")

    max_avg_k = targets.get("max_avg_k", 5.0)
    opt_avg_k = opt_avg.get("cost.avg_k_has_evidence", 0.0)
    if opt_avg_k <= max_avg_k:
        targets_met.append(f"Avg K: {opt_avg_k:.2f} ≤ {max_avg_k:.1f}")
    else:
        targets_missed.append(f"Avg K: {opt_avg_k:.2f} > {max_avg_k:.1f}")

    if targets_met:
        lines.append("### Targets Met ✅")
        for t in targets_met:
            lines.append(f"- {t}")
        lines.append("")

    if targets_missed:
        lines.append("### Targets Missed ❌")
        for t in targets_missed:
            lines.append(f"- {t}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Before vs After Comparison",
        "",
        "| Metric | Baseline | Optimized | Target | Status |",
        "|--------|----------|-----------|--------|--------|",
    ])

    # Comparison table
    comparisons = [
        ("FPR (neg coverage)", baseline.get("fpr", 0.145), opt_fpr, f"≤ {max_fpr:.2%}"),
        ("TPR (pos coverage)", baseline.get("tpr", 0.898), opt_tpr, f"≥ {min_tpr:.2%}"),
        ("Evidence Recall", baseline.get("evidence_recall", 0.808), opt_recall, f"≥ {min_recall:.2%}"),
        ("Avg K", baseline.get("avg_k", 1.71), opt_avg_k, f"≤ {max_avg_k:.1f}"),
    ]

    for name, base_val, opt_val, target in comparisons:
        if "≤" in target:
            target_val = float(target.split("≤")[1].strip().rstrip("%")) / 100 if "%" in target else float(target.split("≤")[1].strip())
            status = "✅" if opt_val <= target_val else "❌"
        else:
            target_val = float(target.split("≥")[1].strip().rstrip("%")) / 100 if "%" in target else float(target.split("≥")[1].strip())
            status = "✅" if opt_val >= target_val else "❌"

        if "Avg K" in name:
            lines.append(f"| {name} | {base_val:.2f} | {opt_val:.2f} | {target} | {status} |")
        else:
            lines.append(f"| {name} | {base_val:.2%} | {opt_val:.2%} | {target} | {status} |")

    lines.extend([
        "",
        "---",
        "",
        "## Optimized Configuration",
        "",
    ])

    ne_config = optimized.get("ne_config", {})
    dk_config = optimized.get("dk_config", {})

    lines.extend([
        "### NE Detection",
        f"- **Method:** `{ne_config.get('method', 'N/A')}`",
        f"- **Threshold:** {ne_config.get('threshold', 0):.4f}",
        "",
        "### Dynamic-K",
        f"- **Method:** `{dk_config.get('method', 'N/A')}`",
        f"- **Threshold:** {dk_config.get('threshold', 0):.4f}",
        "",
    ])

    report = "\n".join(lines)

    with open(output_dir / "comparison_report.md", "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate Deployment Config and Run Assessment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ne_hpo",
        type=str,
        default="outputs/hpo_ne_gate/results.json",
        help="Path to NE gate HPO results",
    )
    parser.add_argument(
        "--dk_hpo",
        type=str,
        default="outputs/hpo_dynamic_k/results.json",
        help="Path to Dynamic-K HPO results",
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
        default="outputs/deployment_optimized",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATE DEPLOYMENT CONFIG & ASSESSMENT")
    print("=" * 70)

    # Load HPO results
    print(f"\nLoading NE gate HPO: {args.ne_hpo}")
    with open(args.ne_hpo, "r") as f:
        ne_hpo_results = json.load(f)

    print(f"Loading Dynamic-K HPO: {args.dk_hpo}")
    with open(args.dk_hpo, "r") as f:
        dk_hpo_results = json.load(f)

    # Load targets
    with open(args.targets, "r") as f:
        targets_config = yaml.safe_load(f)
    profile = targets_config["profiles"].get(args.profile, {})
    targets = profile.get("targets", {})

    # Load OOF cache
    print(f"Loading OOF cache: {args.oof_cache}")
    df = pd.read_parquet(args.oof_cache)
    print(f"  Total queries: {len(df)}")

    # Generate deployment config
    print("\nGenerating deployment config...")
    config = generate_deployment_config(ne_hpo_results, dk_hpo_results, output_dir)
    print(f"  Saved: {output_dir / 'deployment_config.yaml'}")

    # Extract configs
    ne_config = config["ne_detection"]
    dk_config = config["dynamic_k"]

    print(f"\nNE Detection: {ne_config['method']} (threshold={ne_config['threshold']:.4f})")
    print(f"Dynamic-K: {dk_config['method']} (threshold={dk_config['threshold']:.4f})")

    # Run E2E assessment
    print("\n" + "=" * 70)
    print("Running Cross-Validated E2E Assessment")
    print("=" * 70)

    assessment = run_cross_validated_assessment(df, ne_config, dk_config)

    # Save assessment
    with open(output_dir / "e2e_assessment.json", "w") as f:
        json.dump(assessment, f, indent=2, default=str)
    print(f"\nSaved assessment: {output_dir / 'e2e_assessment.json'}")

    # Baseline values (from docs/optimization_plan_baseline.md)
    baseline = {
        "fpr": 0.145,  # 14.5%
        "tpr": 0.898,  # 89.8%
        "evidence_recall": 0.808,  # 80.8%
        "avg_k": 1.71,
    }

    # Generate comparison report
    print("\nGenerating comparison report...")
    report = generate_comparison_report(baseline, assessment, targets, output_dir)
    print(f"Saved report: {output_dir / 'comparison_report.md'}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg = assessment.get("avg_metrics", {})

    print(f"\nOptimized Performance (Mean across 5 folds):")
    print(f"  FPR (neg coverage): {avg.get('coverage.negative_coverage', 0):.2%}")
    print(f"  TPR (pos coverage): {avg.get('coverage.positive_coverage', 0):.2%}")
    print(f"  Evidence Recall: {avg.get('evidence_extraction.recall', 0):.2%}")
    print(f"  Avg K: {avg.get('cost.avg_k_has_evidence', 0):.2f}")

    print(f"\nTargets:")
    print(f"  Max FPR: {targets.get('max_fpr', 0.05):.2%}")
    print(f"  Min TPR: {targets.get('min_tpr', 0.90):.2%}")
    print(f"  Min Evidence Recall: {targets.get('min_evidence_recall', 0.93):.2%}")
    print(f"  Max Avg K: {targets.get('max_avg_k', 5.0):.1f}")

    # Check targets
    all_met = True
    if avg.get("coverage.negative_coverage", 1.0) > targets.get("max_fpr", 0.05):
        all_met = False
    if avg.get("coverage.positive_coverage", 0.0) < targets.get("min_tpr", 0.90):
        all_met = False
    if avg.get("evidence_extraction.recall", 0.0) < targets.get("min_evidence_recall", 0.93):
        all_met = False
    if avg.get("cost.avg_k_has_evidence", 100) > targets.get("max_avg_k", 5.0):
        all_met = False

    print(f"\nAll targets met: {'✅ YES' if all_met else '❌ NO'}")

    print("\n" + "=" * 70)
    print("DEPLOYMENT CONFIG GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
