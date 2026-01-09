#!/usr/bin/env python3
"""Evaluate postprocessing modules for deployment.

Tests:
- Calibration methods (temperature, Platt, isotonic)
- No-evidence detection (max_score, score_std, combined, rf_classifier)
- Dynamic-K selection (score_gap, threshold, elbow)
- Deployment metrics (empty detection, false evidence rate)

IMPORTANT: This script requires REAL scores from the pipeline.
If per_query.csv doesn't include scores, use --run_pipeline flag to compute them.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus, load_criteria
from final_sc_review.data.splits import split_post_ids
from final_sc_review.postprocessing.calibration import ScoreCalibrator
from final_sc_review.postprocessing.no_evidence import (
    NoEvidenceDetector,
    compute_no_evidence_metrics,
    extract_score_features,
)
from final_sc_review.postprocessing.dynamic_k import DynamicKSelector


@dataclass
class PostprocessingResults:
    """Results from postprocessing evaluation."""
    calibration: Dict
    no_evidence: Dict
    dynamic_k: Dict
    deployment: Dict


def load_per_query_with_scores(
    per_query_path: Path,
    groundtruth_path: Path,
    scores_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load per_query results and merge with groundtruth and scores.

    Args:
        per_query_path: Path to per_query.csv
        groundtruth_path: Path to groundtruth CSV
        scores_path: Optional path to scores CSV (must have query_id, uid, score columns)

    Returns:
        DataFrame with has_positive and scores columns
    """
    df = pd.read_csv(per_query_path)

    # Load groundtruth
    gt_rows = load_groundtruth(groundtruth_path)
    gt_df = pd.DataFrame([
        {
            "post_id": row.post_id,
            "criterion_id": row.criterion_id,
            "sent_uid": row.sent_uid,
            "groundtruth": row.groundtruth,
        }
        for row in gt_rows
    ])

    # Compute has_positive for each query
    has_positive = gt_df.groupby(["post_id", "criterion_id"])["groundtruth"].max().reset_index()
    has_positive.columns = ["post_id", "criterion_id", "has_positive"]

    df = df.merge(has_positive, on=["post_id", "criterion_id"], how="left")
    df["has_positive"] = df["has_positive"].fillna(0).astype(int)

    # Load scores if provided
    if scores_path and scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        # Merge scores - assumes scores_df has columns: post_id, criterion_id, scores (pipe-separated)
        df = df.merge(
            scores_df[["post_id", "criterion_id", "scores"]],
            on=["post_id", "criterion_id"],
            how="left",
        )
    else:
        # Only set scores to None if not already present in the DataFrame
        if "scores" not in df.columns:
            df["scores"] = None

    return df


def parse_scores(scores_str: str) -> List[float]:
    """Parse pipe-separated scores string to list of floats."""
    if pd.isna(scores_str) or not scores_str:
        return []
    try:
        return [float(s) for s in str(scores_str).split("|") if s]
    except (ValueError, TypeError):
        return []


def compute_calibration_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute calibration metrics: ECE, MCE, Brier, NLL."""
    if len(scores) == 0 or len(labels) == 0:
        return {"brier": 0.0, "nll": 0.0, "ece": 0.0, "mce": 0.0}

    # Normalize scores to [0, 1] if needed
    score_range = scores.max() - scores.min()
    if score_range > 0:
        scores_norm = (scores - scores.min()) / score_range
    else:
        scores_norm = np.zeros_like(scores) + 0.5

    # Brier score
    brier = brier_score_loss(labels, scores_norm)

    # Log loss (NLL)
    scores_clipped = np.clip(scores_norm, 1e-10, 1 - 1e-10)
    nll = log_loss(labels, scores_clipped, labels=[0, 1])

    # Expected Calibration Error (ECE)
    n_bins = 10
    ece = 0.0
    mce = 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (scores_norm >= bin_edges[i]) & (scores_norm < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = scores_norm[mask].mean()
            bin_size = mask.sum() / len(scores_norm)
            ece += bin_size * abs(bin_acc - bin_conf)
            mce = max(mce, abs(bin_acc - bin_conf))

    return {
        "brier": float(brier),
        "nll": float(nll),
        "ece": float(ece),
        "mce": float(mce),
    }


def evaluate_no_evidence_detection(
    df: pd.DataFrame,
    method: str = "max_score",
    threshold: float = 0.3,
    model_path: Optional[str] = None,
    use_real_scores: bool = True,
) -> Dict:
    """Evaluate no-evidence detection on per_query results.

    Args:
        df: DataFrame with has_positive and optionally scores columns
        method: Detection method
        threshold: Detection threshold
        model_path: Path to classifier model (for rf_classifier)
        use_real_scores: If True, requires real scores in df; raises error if not available

    Returns:
        Dictionary of metrics
    """
    detector = NoEvidenceDetector(
        method=method,
        max_score_threshold=threshold,
        threshold=threshold,
        model_path=model_path,
    )

    predictions = []
    ground_truth = []
    has_scores = "scores" in df.columns

    for _, row in df.iterrows():
        has_positive = bool(row["has_positive"])
        ground_truth.append(has_positive)

        # Get scores
        if has_scores and pd.notna(row.get("scores")):
            scores = parse_scores(row["scores"])
        else:
            if use_real_scores:
                raise ValueError(
                    "Real scores required but not found in per_query.csv. "
                    "Either provide --scores_csv or use --allow_synthetic_scores flag."
                )
            # Fallback to synthetic scores (with warning)
            reranked = row["reranked_topk"].split("|") if pd.notna(row.get("reranked_topk")) else []
            scores = [1.0 / (i + 1) for i in range(len(reranked))]

        result = detector.detect(scores)
        predictions.append(result.has_evidence)

    metrics = compute_no_evidence_metrics(predictions, ground_truth)
    metrics["method"] = method
    metrics["threshold"] = threshold
    metrics["used_real_scores"] = has_scores

    return metrics


def evaluate_dynamic_k(
    df: pd.DataFrame,
    method: str = "score_gap",
    min_k: int = 1,
    max_k: int = 20,
    use_real_scores: bool = True,
) -> Dict:
    """Evaluate dynamic-k selection."""
    selector = DynamicKSelector(
        method=method,
        min_k=min_k,
        max_k=max_k,
    )

    k_values = []
    optimal_k_values = []
    has_scores = "scores" in df.columns

    for _, row in df.iterrows():
        reranked = row["reranked_topk"].split("|") if pd.notna(row.get("reranked_topk")) else []
        gold_ids = set(row["gold_ids"].split("|")) if pd.notna(row.get("gold_ids")) and row["gold_ids"] else set()

        # Get scores
        if has_scores and pd.notna(row.get("scores")):
            scores = parse_scores(row["scores"])
        else:
            if use_real_scores:
                raise ValueError("Real scores required but not found in per_query.csv")
            scores = [1.0 / (i + 1) for i in range(len(reranked))]

        result = selector.select_k(scores)
        k_values.append(result.selected_k)

        # Compute optimal k (minimum k to capture all positives)
        optimal_k = 0
        for i, uid in enumerate(reranked):
            if uid in gold_ids:
                optimal_k = i + 1
        optimal_k_values.append(optimal_k)

    corr = 0.0
    if len(set(k_values)) > 1 and len(set(optimal_k_values)) > 1:
        corr = float(np.corrcoef(k_values, optimal_k_values)[0, 1])

    return {
        "method": method,
        "mean_k": float(np.mean(k_values)),
        "std_k": float(np.std(k_values)),
        "mean_optimal_k": float(np.mean(optimal_k_values)),
        "correlation": corr,
        "used_real_scores": has_scores,
    }


def compute_deployment_metrics(df: pd.DataFrame) -> Dict:
    """Compute deployment-oriented metrics."""
    # Queries with positives vs empty
    n_with_positives = (df["has_positive"] == 1).sum()
    n_empty = (df["has_positive"] == 0).sum()

    # False evidence rate: queries with no positives but we return results
    df = df.copy()
    df["returned_evidence"] = df["reranked_topk"].notna() & (df["reranked_topk"] != "")
    false_evidence = ((df["has_positive"] == 0) & df["returned_evidence"]).sum()
    false_evidence_rate = false_evidence / n_empty if n_empty > 0 else 0.0

    return {
        "n_queries": len(df),
        "n_with_positives": int(n_with_positives),
        "n_empty": int(n_empty),
        "empty_prevalence": float(n_empty / len(df)),
        "false_evidence_count": int(false_evidence),
        "false_evidence_rate": float(false_evidence_rate),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate postprocessing modules")
    parser.add_argument("--per_query", required=True, help="Path to per_query.csv")
    parser.add_argument("--scores_csv", default=None, help="Path to CSV with real scores")
    parser.add_argument("--groundtruth", default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--allow_synthetic_scores",
        action="store_true",
        help="Allow synthetic scores (1/rank) if real scores not available. NOT RECOMMENDED for paper."
    )
    parser.add_argument("--rf_model", default=None, help="Path to RF classifier model for rf_classifier method")
    args = parser.parse_args()

    print("Loading data...")
    df = load_per_query_with_scores(
        Path(args.per_query),
        Path(args.groundtruth),
        Path(args.scores_csv) if args.scores_csv else None,
    )

    print(f"Loaded {len(df)} queries")
    print(f"  With positives: {(df['has_positive'] == 1).sum()}")
    print(f"  Empty: {(df['has_positive'] == 0).sum()}")

    has_real_scores = "scores" in df.columns and df["scores"].notna().any()
    if not has_real_scores:
        if args.allow_synthetic_scores:
            print("\n  WARNING: Using SYNTHETIC scores (1/rank). Results are NOT valid for paper!")
        else:
            print("\n  ERROR: No real scores found. Either:")
            print("    1. Provide --scores_csv with real pipeline scores, OR")
            print("    2. Use --allow_synthetic_scores (NOT RECOMMENDED)")
            return 1
    else:
        print(f"  Real scores: Available")

    use_real = has_real_scores or not args.allow_synthetic_scores

    results = {}

    # Deployment metrics
    print("\nComputing deployment metrics...")
    results["deployment"] = compute_deployment_metrics(df)
    print(f"  Empty prevalence: {results['deployment']['empty_prevalence']:.2%}")
    print(f"  False evidence rate (baseline): {results['deployment']['false_evidence_rate']:.2%}")

    # No-evidence detection
    print("\nEvaluating no-evidence detection methods...")
    results["no_evidence"] = {}
    methods = ["max_score", "score_std", "combined"]
    if args.rf_model:
        methods.append("rf_classifier")

    for method in methods:
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            key = f"{method}_t{threshold}"
            try:
                metrics = evaluate_no_evidence_detection(
                    df,
                    method=method,
                    threshold=threshold,
                    model_path=args.rf_model if method == "rf_classifier" else None,
                    use_real_scores=use_real,
                )
                results["no_evidence"][key] = metrics
                print(f"  {key}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
            except ValueError as e:
                print(f"  {key}: SKIPPED - {e}")

    # Dynamic-K selection
    print("\nEvaluating dynamic-K selection methods...")
    results["dynamic_k"] = {}
    for method in ["score_gap", "elbow"]:
        try:
            metrics = evaluate_dynamic_k(df, method=method, use_real_scores=use_real)
            results["dynamic_k"][method] = metrics
            print(f"  {method}: mean_k={metrics['mean_k']:.2f}, correlation={metrics['correlation']:.3f}")
        except ValueError as e:
            print(f"  {method}: SKIPPED - {e}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
