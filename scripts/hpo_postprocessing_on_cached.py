#!/usr/bin/env python3
"""HPO for post-processing optimization using cached reranker scores.

Optimizes NoEvidenceDetector and DynamicKSelector on the best NV-Embed-v2 + jina-reranker-v3 results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import optuna
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_sc_review.postprocessing.no_evidence import (
    NoEvidenceDetector,
    compute_no_evidence_metrics,
    extract_score_features,
)
from final_sc_review.postprocessing.dynamic_k import DynamicKSelector
from final_sc_review.postprocessing.calibration import ScoreCalibrator, compute_ece
from final_sc_review.metrics.ranking import ndcg_at_k, recall_at_k, mrr_at_k, map_at_k
from final_sc_review.reranker.losses import HybridRerankerLoss

# Best hyperparameters from jina-reranker-v3 HPO (Trial 33)
BEST_RERANKER_PARAMS = {
    'batch_size': 1,
    'num_epochs': 1,
    'learning_rate': 4.447467238603695e-05,
    'weight_decay': 8.769982161626777e-05,
    'grad_accum': 2,
    'pointwise_type': 'bce',
    'pairwise_type': 'pairwise_softplus',
    'listwise_type': 'lambda',
    'w_list': 1.0755666826190335,
    'w_pair': 1.8398728897689836,
    'w_point': 0.813832693617893,
    'temperature': 0.9342605824607415,
    'sigma': 1.5735217400312576,
    'margin': 0.7247599691970003,
    'max_pairs': 100,
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
}


def load_cached_data(cache_dir: Path) -> tuple:
    """Load cached train/val data.

    Cache format (list of dicts):
    - post_id, criterion_id, criterion_text, gold_uids, is_no_evidence
    - candidates: list of {sent_uid, text, score}
    """
    with open(cache_dir / "nv-embed-v2_train_cache.json") as f:
        train_cache = json.load(f)
    with open(cache_dir / "nv-embed-v2_val_cache.json") as f:
        val_cache = json.load(f)
    return train_cache, val_cache


def prepare_val_results_from_cache(val_cache: list) -> tuple:
    """Prepare validation results directly from cache (no retraining needed).

    The cache already contains retriever scores, so we can use them directly
    for post-processing optimization without retraining the reranker.
    """
    val_results = []
    all_scores = []
    all_labels = []

    for item in val_cache:
        post_id = item['post_id']
        criterion_id = item['criterion_id']
        gold_uids = set(item['gold_uids'])
        has_evidence = not item['is_no_evidence']
        candidates = item['candidates']

        if len(candidates) < 2:
            continue

        # Extract scores and compute labels
        scores = [c['score'] for c in candidates]
        labels = [1 if c['sent_uid'] in gold_uids else 0 for c in candidates]
        sent_uids = [c['sent_uid'] for c in candidates]

        # Scores are already sorted by retriever, rank by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_scores = [scores[i] for i in ranked_indices]
        ranked_labels = [labels[i] for i in ranked_indices]
        ranked_ids = [sent_uids[i] for i in ranked_indices]

        # Get positive IDs that are in candidates
        positive_ids = {uid for uid in gold_uids if uid in sent_uids}

        query_key = f"{post_id}_{criterion_id}"

        val_results.append({
            'query_key': query_key,
            'scores': ranked_scores,
            'labels': ranked_labels,
            'positive_ids': positive_ids,
            'ranked_ids': ranked_ids,
            'has_evidence': has_evidence,
        })

        all_scores.extend(ranked_scores)
        all_labels.extend(ranked_labels)

    return val_results, np.array(all_scores), np.array(all_labels)


def load_scores_cache(cache_path: Path) -> tuple:
    """Load pre-computed scores cache if available."""
    with open(cache_path) as f:
        cache_data = json.load(f)

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

    return val_results, np.array(all_scores), np.array(all_labels)


def create_objective(
    val_results: list,
    all_scores: np.ndarray,
    all_labels: np.ndarray,
):
    """Create Optuna objective function for post-processing HPO."""

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters

        # No-evidence detection params (including combined gap+std methods)
        ne_method = trial.suggest_categorical("ne_method", [
            "max_score", "score_std", "score_gap", "score_gap_mean", "combined",
            "score_gap_and_std", "score_gap_or_std", "score_gap_std_weighted"
        ])
        max_score_threshold = trial.suggest_float("max_score_threshold", 0.1, 0.9)
        score_std_threshold = trial.suggest_float("score_std_threshold", 0.01, 0.5)
        score_gap_threshold = trial.suggest_float("score_gap_threshold", 0.01, 0.5)

        # Dynamic-k params
        dk_method = trial.suggest_categorical("dk_method", ["score_gap", "elbow"])
        min_k = trial.suggest_int("min_k", 1, 5)
        max_k = trial.suggest_int("max_k", 5, 20)
        score_gap_ratio = trial.suggest_float("score_gap_ratio", 0.1, 0.7)

        # Calibration params
        cal_method = trial.suggest_categorical("cal_method", ["platt", "isotonic", "temperature"])

        # Initialize modules
        no_evidence_detector = NoEvidenceDetector(
            method=ne_method,
            max_score_threshold=max_score_threshold,
            score_std_threshold=score_std_threshold,
            score_gap_threshold=score_gap_threshold,
        )

        dynamic_k_selector = DynamicKSelector(
            method=dk_method,
            min_k=min_k,
            max_k=max_k,
            score_gap_ratio=score_gap_ratio,
        )

        # Fit calibrator on all validation scores
        calibrator = ScoreCalibrator(method=cal_method)
        try:
            calibrator.fit(all_scores, all_labels)
        except Exception:
            # Fallback if calibration fails
            return 0.0

        # Evaluate on validation set
        ne_predictions = []
        ne_groundtruth = []

        ndcg_scores = []
        recall_scores = []

        for result in val_results:
            scores = result['scores']
            positive_ids = result['positive_ids']
            ranked_ids = result['ranked_ids']
            has_evidence = result['has_evidence']

            if not scores:
                continue

            # Calibrate scores
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
                # Use dynamic k
                try:
                    dk_result = dynamic_k_selector.select_k(scores, calibrated)
                    k = dk_result.selected_k
                except Exception:
                    k = min_k

                # Compute metrics at dynamic k
                ndcg = ndcg_at_k(positive_ids, ranked_ids, k)
                recall = recall_at_k(positive_ids, ranked_ids, k)

                ndcg_scores.append(ndcg)
                recall_scores.append(recall)

        # Compute no-evidence F1
        ne_metrics = compute_no_evidence_metrics(ne_predictions, ne_groundtruth)
        ne_f1 = ne_metrics['f1']

        # Compute average nDCG with dynamic k
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0

        # Combined objective: weighted sum of ne_f1 and dynamic_ndcg
        # Higher weight on nDCG since that's the primary metric
        combined = 0.3 * ne_f1 + 0.5 * avg_ndcg + 0.2 * avg_recall

        # Log intermediate results
        trial.set_user_attr("ne_f1", ne_f1)
        trial.set_user_attr("ne_precision", ne_metrics['precision'])
        trial.set_user_attr("ne_recall", ne_metrics['recall'])
        trial.set_user_attr("dynamic_ndcg", avg_ndcg)
        trial.set_user_attr("dynamic_recall", avg_recall)

        return combined

    return objective


def main():
    parser = argparse.ArgumentParser(description="HPO for post-processing optimization")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of HPO trials")
    parser.add_argument("--study_name", type=str, default="postprocessing_hpo", help="Study name")
    parser.add_argument("--output_dir", type=str, default="outputs/hpo_postprocessing", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_cache", action="store_true", help="Use cached scores if available")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached scores
    scores_cache_path = output_dir / "val_scores_cache.json"

    if args.use_cache and scores_cache_path.exists():
        print(f"Loading cached scores from {scores_cache_path}")
        val_results, all_scores, all_labels = load_scores_cache(scores_cache_path)
    else:
        # Load cached data from HPO fine-tuning cache
        cache_dir = Path("outputs/hpo_finetuning_with_no_evidence/nv-embed-v2_jina-reranker-v3")
        print(f"Loading cached data from {cache_dir}")
        train_cache, val_cache = load_cached_data(cache_dir)
        print(f"Train queries: {len(train_cache)}, Val queries: {len(val_cache)}")

        # Prepare validation results directly from cache (retriever scores)
        # No need to retrain reranker - we use the cached retriever scores for post-processing HPO
        print("Preparing validation results from cache...")
        val_results, all_scores, all_labels = prepare_val_results_from_cache(val_cache)

        # Save processed results for faster re-runs
        with open(scores_cache_path, 'w') as f:
            cache_data = []
            for r in val_results:
                cache_data.append({
                    'query_key': r['query_key'],
                    'scores': r['scores'],
                    'labels': r['labels'],
                    'positive_ids': list(r['positive_ids']),
                    'ranked_ids': r['ranked_ids'],
                    'has_evidence': r['has_evidence'],
                })
            json.dump(cache_data, f)
        print(f"Saved processed cache to {scores_cache_path}")

    print(f"Validation results: {len(val_results)} queries")
    print(f"Total score-label pairs: {len(all_scores)}")

    # Statistics
    n_with_evidence = sum(1 for r in val_results if r['has_evidence'])
    n_without_evidence = len(val_results) - n_with_evidence
    print(f"Queries with evidence: {n_with_evidence}, without: {n_without_evidence}")

    # Create Optuna study
    storage = f"sqlite:///{output_dir / 'optuna.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Create objective
    objective = create_objective(val_results, all_scores, all_labels)

    # Run optimization
    print(f"\nStarting HPO with {args.n_trials} trials...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: print(
                f"Trial {trial.number}: {trial.value:.4f} "
                f"(ne_f1={trial.user_attrs.get('ne_f1', 0):.4f}, "
                f"ndcg={trial.user_attrs.get('dynamic_ndcg', 0):.4f})"
            )
        ],
    )

    # Print results
    print("\n" + "=" * 70)
    print("HPO RESULTS")
    print("=" * 70)

    best_trial = study.best_trial
    print(f"\nBest Trial {best_trial.number}:")
    print(f"  Combined Score: {best_trial.value:.4f}")
    print(f"  No-Evidence F1: {best_trial.user_attrs.get('ne_f1', 0):.4f}")
    print(f"  No-Evidence Precision: {best_trial.user_attrs.get('ne_precision', 0):.4f}")
    print(f"  No-Evidence Recall: {best_trial.user_attrs.get('ne_recall', 0):.4f}")
    print(f"  Dynamic nDCG: {best_trial.user_attrs.get('dynamic_ndcg', 0):.4f}")
    print(f"  Dynamic Recall: {best_trial.user_attrs.get('dynamic_recall', 0):.4f}")

    print("\nBest Parameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # Save results
    results = {
        "best_combined_score": best_trial.value,
        "best_params": best_trial.params,
        "best_metrics": {
            "ne_f1": best_trial.user_attrs.get('ne_f1', 0),
            "ne_precision": best_trial.user_attrs.get('ne_precision', 0),
            "ne_recall": best_trial.user_attrs.get('ne_recall', 0),
            "dynamic_ndcg": best_trial.user_attrs.get('dynamic_ndcg', 0),
            "dynamic_recall": best_trial.user_attrs.get('dynamic_recall', 0),
        },
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "best_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
