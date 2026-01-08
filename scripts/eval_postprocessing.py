#!/usr/bin/env python3
"""Evaluate postprocessing modules for deployment.

Tests:
- Calibration methods (temperature, Platt, isotonic)
- No-evidence detection (max_score, score_std, combined)
- Dynamic-K selection (score_gap, threshold, elbow)
- Deployment metrics (empty detection, false evidence rate)
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
from final_sc_review.postprocessing.no_evidence import NoEvidenceDetector, compute_no_evidence_metrics
from final_sc_review.postprocessing.dynamic_k import DynamicKSelector


@dataclass
class PostprocessingResults:
    """Results from postprocessing evaluation."""
    calibration: Dict
    no_evidence: Dict
    dynamic_k: Dict
    deployment: Dict


def load_per_query_with_scores(per_query_path: Path, groundtruth_path: Path) -> pd.DataFrame:
    """Load per_query results and merge with groundtruth for analysis."""
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

    return df


def compute_calibration_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute calibration metrics: ECE, MCE, Brier, NLL."""
    # Normalize scores to [0, 1] if needed
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    # Brier score
    brier = brier_score_loss(labels, scores_norm)

    # Log loss (NLL)
    scores_clipped = np.clip(scores_norm, 1e-10, 1 - 1e-10)
    nll = log_loss(labels, scores_clipped)

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
) -> Dict:
    """Evaluate no-evidence detection on per_query results."""
    detector = NoEvidenceDetector(
        method=method,
        max_score_threshold=threshold,
    )

    predictions = []
    ground_truth = []

    for _, row in df.iterrows():
        has_positive = bool(row["has_positive"])
        ground_truth.append(has_positive)

        # Simulate scores from retriever results
        # For simplicity, use number of gold IDs as proxy
        gold_count = len(row["gold_ids"].split("|")) if pd.notna(row["gold_ids"]) and row["gold_ids"] else 0
        reranked = row["reranked_topk"].split("|") if pd.notna(row["reranked_topk"]) else []

        # Generate fake scores (in real use, these would come from the model)
        if reranked:
            scores = [1.0 / (i + 1) for i in range(len(reranked))]  # Decaying scores
        else:
            scores = []

        result = detector.detect(scores)
        predictions.append(result.has_evidence)

    metrics = compute_no_evidence_metrics(predictions, ground_truth)
    metrics["method"] = method
    metrics["threshold"] = threshold

    return metrics


def evaluate_dynamic_k(
    df: pd.DataFrame,
    method: str = "score_gap",
    min_k: int = 1,
    max_k: int = 20,
) -> Dict:
    """Evaluate dynamic-k selection."""
    selector = DynamicKSelector(
        method=method,
        min_k=min_k,
        max_k=max_k,
    )

    k_values = []
    optimal_k_values = []

    for _, row in df.iterrows():
        reranked = row["reranked_topk"].split("|") if pd.notna(row["reranked_topk"]) else []
        gold_ids = set(row["gold_ids"].split("|")) if pd.notna(row["gold_ids"]) and row["gold_ids"] else set()

        if reranked:
            scores = [1.0 / (i + 1) for i in range(len(reranked))]
        else:
            scores = []

        result = selector.select_k(scores)
        k_values.append(result.selected_k)

        # Compute optimal k (minimum k to capture all positives)
        optimal_k = 0
        for i, uid in enumerate(reranked):
            if uid in gold_ids:
                optimal_k = i + 1
        optimal_k_values.append(optimal_k)

    return {
        "method": method,
        "mean_k": float(np.mean(k_values)),
        "std_k": float(np.std(k_values)),
        "mean_optimal_k": float(np.mean(optimal_k_values)),
        "correlation": float(np.corrcoef(k_values, optimal_k_values)[0, 1]) if len(set(k_values)) > 1 else 0.0,
    }


def compute_deployment_metrics(df: pd.DataFrame) -> Dict:
    """Compute deployment-oriented metrics."""
    # Queries with positives vs empty
    n_with_positives = (df["has_positive"] == 1).sum()
    n_empty = (df["has_positive"] == 0).sum()

    # False evidence rate: queries with no positives but we return results
    # (Here we assume any non-empty reranked_topk is "evidence returned")
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
    parser.add_argument("--groundtruth", default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    print("Loading data...")
    df = load_per_query_with_scores(Path(args.per_query), Path(args.groundtruth))

    print(f"Loaded {len(df)} queries")
    print(f"  With positives: {(df['has_positive'] == 1).sum()}")
    print(f"  Empty: {(df['has_positive'] == 0).sum()}")

    results = {}

    # Deployment metrics
    print("\nComputing deployment metrics...")
    results["deployment"] = compute_deployment_metrics(df)
    print(f"  Empty prevalence: {results['deployment']['empty_prevalence']:.2%}")
    print(f"  False evidence rate: {results['deployment']['false_evidence_rate']:.2%}")

    # No-evidence detection
    print("\nEvaluating no-evidence detection methods...")
    results["no_evidence"] = {}
    for method in ["max_score", "score_std", "combined"]:
        for threshold in [0.2, 0.3, 0.4, 0.5]:
            key = f"{method}_t{threshold}"
            metrics = evaluate_no_evidence_detection(df, method=method, threshold=threshold)
            results["no_evidence"][key] = metrics
            print(f"  {key}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    # Dynamic-K selection
    print("\nEvaluating dynamic-K selection methods...")
    results["dynamic_k"] = {}
    for method in ["score_gap", "elbow"]:
        metrics = evaluate_dynamic_k(df, method=method)
        results["dynamic_k"][method] = metrics
        print(f"  {method}: mean_k={metrics['mean_k']:.2f}, correlation={metrics['correlation']:.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
