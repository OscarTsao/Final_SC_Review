#!/usr/bin/env python3
"""Quick validation of HPO-optimized post-processing parameters.

This script uses cached reranker scores to validate post-processing
without needing to train models (much faster than running full notebook).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from final_sc_review.postprocessing.no_evidence import (
    NoEvidenceDetector,
    compute_no_evidence_metrics,
)
from final_sc_review.postprocessing.dynamic_k import DynamicKSelector
from final_sc_review.postprocessing.calibration import ScoreCalibrator
from final_sc_review.metrics.ranking import ndcg_at_k, recall_at_k


def main():
    # Load HPO best results
    hpo_results_path = Path("outputs/hpo_postprocessing_reranker/best_results.json")
    with open(hpo_results_path) as f:
        hpo_results = json.load(f)

    best_params = hpo_results["best_params"]
    print("=" * 80)
    print("VALIDATING HPO-OPTIMIZED POST-PROCESSING PARAMETERS")
    print("=" * 80)
    print(f"\nBest HPO Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\nExpected metrics from HPO:")
    for k, v in hpo_results["best_metrics"].items():
        print(f"  {k}: {v:.4f}")

    # Load cached reranker scores
    cache_path = Path("outputs/hpo_postprocessing_reranker/reranker_scores_cache.json")
    with open(cache_path) as f:
        cache_data = json.load(f)

    print(f"\nLoaded {len(cache_data)} cached validation queries")

    # Prepare data
    val_results = []
    all_scores = []
    all_labels = []

    for item in cache_data:
        val_results.append({
            'query_key': item['query_key'],
            'scores': item['scores'],
            'labels': item['labels'],
            'positive_ids': set(item['positive_ids']),
            'ranked_ids': item['ranked_ids'],
            'has_evidence': item['has_evidence'],
        })
        all_scores.extend(item['scores'])
        all_labels.extend(item['labels'])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    print(f"\nScore distribution:")
    print(f"  min: {all_scores.min():.4f}")
    print(f"  max: {all_scores.max():.4f}")
    print(f"  mean: {all_scores.mean():.4f}")
    print(f"  std: {all_scores.std():.4f}")

    # Initialize post-processing with HPO params
    no_evidence_detector = NoEvidenceDetector(
        method=best_params['ne_method'],
        max_score_threshold=best_params['max_score_threshold'],
        score_std_threshold=best_params['score_std_threshold'],
        score_gap_threshold=best_params['score_gap_threshold'],
    )

    dynamic_k_selector = DynamicKSelector(
        method=best_params['dk_method'],
        min_k=best_params['min_k'],
        max_k=best_params['max_k'],
        score_gap_ratio=best_params['score_gap_ratio'],
    )

    # Fit calibrator
    calibrator = ScoreCalibrator(method=best_params['cal_method'])
    calibrator.fit(all_scores, all_labels)

    # Evaluate
    ne_predictions = []
    ne_groundtruth = []
    ndcg_scores = []
    recall_scores = []
    dynamic_k_values = []

    for result in val_results:
        scores = result['scores']
        positive_ids = result['positive_ids']
        ranked_ids = result['ranked_ids']
        has_evidence = result['has_evidence']

        if not scores:
            continue

        # Calibrate
        try:
            calibrated = calibrator.calibrate(np.array(scores)).tolist()
        except Exception:
            calibrated = scores

        # No-evidence detection
        ne_result = no_evidence_detector.detect(scores, calibrated)
        ne_predictions.append(ne_result.has_evidence)
        ne_groundtruth.append(has_evidence)

        # Dynamic-k selection (only for queries with evidence)
        if has_evidence and ne_result.has_evidence:
            try:
                dk_result = dynamic_k_selector.select_k(scores, calibrated)
                k = dk_result.selected_k
            except Exception:
                k = best_params['min_k']

            dynamic_k_values.append(k)

            # Metrics at dynamic k
            ndcg = ndcg_at_k(positive_ids, ranked_ids, k)
            recall = recall_at_k(positive_ids, ranked_ids, k)

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)

    # Compute metrics
    ne_metrics = compute_no_evidence_metrics(ne_predictions, ne_groundtruth)
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_k = np.mean(dynamic_k_values) if dynamic_k_values else 0.0

    # Combined score (same as HPO objective)
    combined = 0.3 * ne_metrics['f1'] + 0.5 * avg_ndcg + 0.2 * avg_recall

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print(f"\nNo-Evidence Detection:")
    print(f"  F1:        {ne_metrics['f1']:.4f} (expected: {hpo_results['best_metrics']['ne_f1']:.4f})")
    print(f"  Precision: {ne_metrics['precision']:.4f} (expected: {hpo_results['best_metrics']['ne_precision']:.4f})")
    print(f"  Recall:    {ne_metrics['recall']:.4f} (expected: {hpo_results['best_metrics']['ne_recall']:.4f})")

    print(f"\nDynamic-K Selection:")
    print(f"  Average K: {avg_k:.2f}")
    print(f"  nDCG:      {avg_ndcg:.4f} (expected: {hpo_results['best_metrics']['dynamic_ndcg']:.4f})")
    print(f"  Recall:    {avg_recall:.4f} (expected: {hpo_results['best_metrics']['dynamic_recall']:.4f})")

    print(f"\nCombined Score: {combined:.4f} (expected: {hpo_results['best_combined_score']:.4f})")

    # Check if results match
    ne_f1_match = abs(ne_metrics['f1'] - hpo_results['best_metrics']['ne_f1']) < 0.001
    ndcg_match = abs(avg_ndcg - hpo_results['best_metrics']['dynamic_ndcg']) < 0.001

    print("\n" + "=" * 80)
    if ne_f1_match and ndcg_match:
        print("✅ VALIDATION PASSED: Results match HPO expectations!")
    else:
        print("⚠️  VALIDATION WARNING: Results differ from HPO (may be due to calibrator randomness)")
    print("=" * 80)

    # Dynamic-K distribution
    if dynamic_k_values:
        print(f"\nDynamic-K Distribution:")
        k_counts = {}
        for k in dynamic_k_values:
            k_counts[k] = k_counts.get(k, 0) + 1
        for k in sorted(k_counts.keys()):
            print(f"  k={k}: {k_counts[k]} queries ({k_counts[k]/len(dynamic_k_values)*100:.1f}%)")


if __name__ == "__main__":
    main()
