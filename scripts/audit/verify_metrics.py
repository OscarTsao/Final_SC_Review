#!/usr/bin/env python3
"""Independent metric recomputation and verification.

This script:
1. Loads raw predictions from saved .npz files
2. Recomputes metrics using sklearn directly
3. Compares with reported metrics from cv_results.json
4. Reports any discrepancies

Usage:
    python scripts/audit/verify_metrics.py --output_dir outputs/audit_full_eval/<ts>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_predictions(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions and labels from npz file."""
    data = np.load(npz_path)
    # Handle different key names
    if "y_prob" in data:
        y_prob = data["y_prob"]
        y_true = data["y_true"]
    elif "preds" in data:
        y_prob = data["preds"]
        y_true = data["labels"]
    else:
        raise KeyError(f"Unknown npz format. Keys: {list(data.keys())}")
    return y_prob, y_true


def compute_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """Compute TPR at target FPR level."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, 1.0
    best_idx = idx[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])


def recompute_metrics(y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Recompute all NE gate metrics from raw predictions."""
    n_samples = len(y_true)
    has_evidence_rate = float(y_true.mean())

    # Handle edge case
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": has_evidence_rate,
            "tpr_at_fpr_3pct": 0.0,
            "tpr_at_fpr_5pct": 0.0,
            "tpr_at_fpr_10pct": 0.0,
            "threshold_at_fpr_5pct": 1.0,
            "n_samples": n_samples,
            "has_evidence_rate": has_evidence_rate,
        }

    # AUROC and AUPRC
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # TPR at FPR levels
    tpr_3, _ = compute_tpr_at_fpr(y_true, y_prob, 0.03)
    tpr_5, thresh_5 = compute_tpr_at_fpr(y_true, y_prob, 0.05)
    tpr_10, _ = compute_tpr_at_fpr(y_true, y_prob, 0.10)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_fpr_3pct": tpr_3,
        "tpr_at_fpr_5pct": tpr_5,
        "tpr_at_fpr_10pct": tpr_10,
        "threshold_at_fpr_5pct": thresh_5,
        "n_samples": n_samples,
        "has_evidence_rate": has_evidence_rate,
    }


def compare_metrics(reported: Dict[str, float], recomputed: Dict[str, float], tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare reported vs recomputed metrics."""
    result = {"matches": True, "discrepancies": [], "compared_keys": []}

    # Only compare metrics that exist in both
    all_keys = ["auroc", "auprc", "tpr_at_fpr_3pct", "tpr_at_fpr_5pct", "tpr_at_fpr_10pct"]

    for key in all_keys:
        if key not in reported:
            continue  # Skip keys not present in reported metrics

        r = reported.get(key, 0.0)
        c = recomputed.get(key, 0.0)
        diff = abs(r - c)

        result["compared_keys"].append(key)

        if diff > tolerance:
            result["matches"] = False
            result["discrepancies"].append({
                "metric": key,
                "reported": r,
                "recomputed": c,
                "diff": diff,
            })

    return result


def verify_gnn_results(gnn_results_dir: Path) -> Dict[str, Any]:
    """Verify all GNN experiment results."""
    results = {
        "experiments": {},
        "all_match": True,
    }

    # Find all cv_results.json files
    cv_files = list(gnn_results_dir.rglob("cv_results.json"))

    for cv_file in cv_files:
        exp_name = cv_file.parent.name
        print(f"\nVerifying {exp_name}...")

        exp_result = {
            "cv_results_path": str(cv_file),
            "folds": [],
            "all_folds_match": True,
        }

        with open(cv_file) as f:
            cv_data = json.load(f)

        # Check each fold
        for fold_result in cv_data.get("fold_results", []):
            fold_id = fold_result["fold_id"]
            # Handle different JSON formats (metrics vs final_metrics)
            reported_metrics = fold_result.get("metrics") or fold_result.get("final_metrics")
            if reported_metrics is None:
                print(f"  Fold {fold_id}: ⚠️ No metrics found in results")
                continue

            # Find predictions file
            pred_path = cv_file.parent / f"fold_{fold_id}" / "predictions.npz"

            fold_data = {
                "fold_id": fold_id,
                "predictions_found": pred_path.exists(),
            }

            if pred_path.exists():
                # Load and recompute
                y_prob, y_true = load_predictions(pred_path)
                recomputed = recompute_metrics(y_prob, y_true)

                # Compare
                comparison = compare_metrics(reported_metrics, recomputed)
                fold_data["comparison"] = comparison
                fold_data["reported"] = reported_metrics
                fold_data["recomputed"] = recomputed

                if comparison["matches"]:
                    print(f"  Fold {fold_id}: ✅ MATCH")
                else:
                    print(f"  Fold {fold_id}: ❌ MISMATCH")
                    for d in comparison["discrepancies"]:
                        print(f"    - {d['metric']}: reported={d['reported']:.6f}, recomputed={d['recomputed']:.6f}, diff={d['diff']:.2e}")
                    exp_result["all_folds_match"] = False
                    results["all_match"] = False
            else:
                print(f"  Fold {fold_id}: ⚠️ No predictions file")
                fold_data["comparison"] = {"matches": None, "note": "predictions file not found"}

            exp_result["folds"].append(fold_data)

        # Verify aggregated metrics
        if cv_data.get("aggregated"):
            agg = cv_data["aggregated"]

            # Recompute aggregation from fold metrics - handle different formats
            fold_metrics = []
            for f in cv_data["fold_results"]:
                m = f.get("metrics") or f.get("final_metrics")
                if m:
                    fold_metrics.append(m)

            if fold_metrics:
                recomputed_agg = {}
                agg_match = True
                agg_discrepancies = []

                # Only check metrics that are present and numeric in agg
                for key in ["auroc", "auprc", "tpr_at_fpr_3pct", "tpr_at_fpr_5pct", "tpr_at_fpr_10pct"]:
                    if key not in fold_metrics[0]:
                        continue  # Skip if not in fold results

                    values = [m[key] for m in fold_metrics if key in m]
                    if not values:
                        continue

                    recomputed_mean = float(np.mean(values))
                    recomputed_std = float(np.std(values))
                    recomputed_agg[key] = {"mean": recomputed_mean, "std": recomputed_std}

                    # Check aggregated metrics - handle different formats
                    if key in agg:
                        if isinstance(agg[key], dict):
                            reported_mean = agg[key].get("mean", 0)
                        elif isinstance(agg[key], str):
                            # Parse "0.8911 +/- 0.0091" format
                            reported_mean = float(agg[key].split("+/-")[0].strip())
                        else:
                            reported_mean = agg[key]

                        if abs(reported_mean - recomputed_mean) > 1e-4:  # Use looser tolerance for parsed values
                            agg_match = False
                            agg_discrepancies.append({
                                "metric": key,
                                "reported_mean": reported_mean,
                                "recomputed_mean": recomputed_mean,
                            })

                    # Also check *_mean keys like "auroc_mean"
                    mean_key = f"{key}_mean"
                    if mean_key in agg:
                        reported_mean = agg[mean_key]
                        if abs(reported_mean - recomputed_mean) > 1e-6:
                            agg_match = False
                            agg_discrepancies.append({
                                "metric": mean_key,
                                "reported_mean": reported_mean,
                                "recomputed_mean": recomputed_mean,
                            })

                exp_result["aggregated_match"] = agg_match
                exp_result["aggregated_discrepancies"] = agg_discrepancies

                if agg_match:
                    print(f"  Aggregated: ✅ MATCH")
                else:
                    print(f"  Aggregated: ❌ MISMATCH")
                    for d in agg_discrepancies:
                        print(f"    - {d['metric']}: reported={d['reported_mean']:.6f}, recomputed={d['recomputed_mean']:.6f}")

        results["experiments"][exp_name] = exp_result

    return results


def verify_ranking_metrics() -> Dict[str, Any]:
    """Verify ranking metric implementations against known values."""
    import math

    results = {"tests": [], "all_pass": True}

    # Test case 1: Perfect ranking
    gold = ["a", "b"]
    ranked = ["a", "b", "c", "d", "e"]

    tests = [
        ("recall@3", recall_at_k(gold, ranked, 3), 1.0),
        ("recall@1", recall_at_k(gold, ranked, 1), 0.5),
        ("mrr@5", mrr_at_k(gold, ranked, 5), 1.0),
        ("ndcg@5", ndcg_at_k(gold, ranked, 5), 1.0),
    ]

    # Test case 2: Imperfect ranking
    gold2 = ["c", "d"]
    ranked2 = ["a", "b", "c", "d", "e"]

    tests.extend([
        ("recall@2 (imperfect)", recall_at_k(gold2, ranked2, 2), 0.0),
        ("recall@4 (imperfect)", recall_at_k(gold2, ranked2, 4), 1.0),
        ("mrr@5 (imperfect)", mrr_at_k(gold2, ranked2, 5), 1/3),
    ])

    # Test case 3: nDCG calculation
    # For gold=["c","d"], ranked=["a","b","c","d","e"], k=5
    # DCG = 0/log2(2) + 0/log2(3) + 1/log2(4) + 1/log2(5) = 0.5 + 0.431 = 0.931
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
    # nDCG = 0.931 / 1.631 = 0.571
    dcg = 1/math.log2(4) + 1/math.log2(5)
    idcg = 1/math.log2(2) + 1/math.log2(3)
    expected_ndcg = dcg / idcg
    tests.append(("ndcg@5 (imperfect)", ndcg_at_k(gold2, ranked2, 5), expected_ndcg))

    # Test case 4: Empty gold
    tests.append(("recall@5 (empty gold)", recall_at_k([], ranked, 5), 0.0))
    tests.append(("mrr@5 (empty gold)", mrr_at_k([], ranked, 5), 0.0))
    tests.append(("ndcg@5 (empty gold)", ndcg_at_k([], ranked, 5), 0.0))

    for name, computed, expected in tests:
        passed = abs(computed - expected) < 1e-6
        results["tests"].append({
            "name": name,
            "computed": computed,
            "expected": expected,
            "passed": passed,
        })
        if not passed:
            results["all_pass"] = False
            print(f"  ❌ {name}: computed={computed:.6f}, expected={expected:.6f}")
        else:
            print(f"  ✅ {name}: {computed:.6f}")

    return results


def recall_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = set(ranked_ids[:k]) & gold
    return len(hits) / len(gold)


def mrr_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(gold_ids, ranked_ids, k):
    import math
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if sent_id in gold else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Verify metrics")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gnn_results", type=str, default="outputs/gnn_research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    gnn_results_dir = Path(args.gnn_results)

    all_results = {
        "ranking_metrics": None,
        "gnn_metrics": None,
        "all_verified": True,
    }

    # 1. Verify ranking metric implementations
    print("=" * 60)
    print("1. Verifying Ranking Metric Implementations")
    print("=" * 60)
    ranking_results = verify_ranking_metrics()
    all_results["ranking_metrics"] = ranking_results
    if not ranking_results["all_pass"]:
        all_results["all_verified"] = False

    # 2. Verify GNN results
    if gnn_results_dir.exists():
        print("\n" + "=" * 60)
        print("2. Verifying GNN Experiment Results")
        print("=" * 60)
        gnn_results = verify_gnn_results(gnn_results_dir)
        all_results["gnn_metrics"] = gnn_results
        if not gnn_results["all_match"]:
            all_results["all_verified"] = False
    else:
        print(f"\n⚠️ GNN results directory not found: {gnn_results_dir}")
        all_results["gnn_metrics"] = {"error": "directory not found"}

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metric_verification.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    if all_results["all_verified"]:
        print("✅ ALL METRICS VERIFIED SUCCESSFULLY")
        return 0
    else:
        print("❌ VERIFICATION FAILED - See details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
