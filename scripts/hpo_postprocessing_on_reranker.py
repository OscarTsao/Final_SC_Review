#!/usr/bin/env python3
"""HPO for post-processing optimization using reranker scores.

This script:
1. Loads validation data from retriever cache
2. Scores with pre-trained reranker (no fine-tuning needed for threshold HPO)
3. Runs post-processing HPO on reranker scores

This ensures thresholds are optimized for reranker score distributions,
not retriever score distributions.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import optuna
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_sc_review.postprocessing.no_evidence import (
    NoEvidenceDetector,
    compute_no_evidence_metrics,
)
from final_sc_review.postprocessing.dynamic_k import DynamicKSelector
from final_sc_review.postprocessing.calibration import ScoreCalibrator
from final_sc_review.metrics.ranking import ndcg_at_k, recall_at_k

RERANKER_MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual"
TOP_K = 20


def load_cached_data(cache_dir: Path) -> tuple:
    """Load cached train/val data from retriever cache."""
    with open(cache_dir / "nv-embed-v2_val_cache.json") as f:
        val_cache = json.load(f)
    return val_cache


def score_with_reranker(val_cache: list, model, tokenizer, device: str) -> tuple:
    """Score validation data with pre-trained reranker."""
    model.eval()
    val_results = []
    all_scores = []
    all_labels = []

    print("Scoring validation data with reranker...")
    with torch.no_grad():
        for item in tqdm(val_cache, desc="Scoring"):
            post_id = item['post_id']
            criterion_id = item['criterion_id']
            criterion_text = item['criterion_text']
            gold_uids = set(item['gold_uids'])
            is_no_evidence = item['is_no_evidence']
            candidates = item['candidates'][:TOP_K]

            if len(candidates) < 2:
                continue

            # Build query from criterion
            query_text = criterion_text

            # Score candidates with reranker
            texts = [[query_text, c['text']] for c in candidates]
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1).float().cpu().numpy()

            # Rerank by score (descending)
            ranked_indices = np.argsort(scores)[::-1]
            ranked_scores = [float(scores[i]) for i in ranked_indices]
            ranked_labels = [1 if candidates[i]['sent_uid'] in gold_uids else 0 for i in ranked_indices]
            ranked_ids = [candidates[i]['sent_uid'] for i in ranked_indices]

            # Get positive IDs that are in candidates
            candidate_uids = {c['sent_uid'] for c in candidates}
            positive_ids = gold_uids & candidate_uids

            query_key = f"{post_id}_{criterion_id}"

            val_results.append({
                'query_key': query_key,
                'scores': ranked_scores,
                'labels': ranked_labels,
                'positive_ids': positive_ids,
                'ranked_ids': ranked_ids,
                'has_evidence': not is_no_evidence,
            })

            all_scores.extend(ranked_scores)
            all_labels.extend(ranked_labels)

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
        max_score_threshold = trial.suggest_float("max_score_threshold", -5.0, 5.0)
        score_std_threshold = trial.suggest_float("score_std_threshold", 0.01, 3.0)
        score_gap_threshold = trial.suggest_float("score_gap_threshold", 0.01, 3.0)

        # Dynamic-k params
        dk_method = trial.suggest_categorical("dk_method", ["score_gap", "elbow"])
        min_k = trial.suggest_int("min_k", 1, 5)
        max_k = trial.suggest_int("max_k", 5, 20)
        score_gap_ratio = trial.suggest_float("score_gap_ratio", 0.01, 0.7)

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
    parser = argparse.ArgumentParser(description="HPO for post-processing on reranker scores")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of HPO trials")
    parser.add_argument("--study_name", type=str, default="postprocessing_reranker_hpo", help="Study name")
    parser.add_argument("--output_dir", type=str, default="outputs/hpo_postprocessing_reranker", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_cache", action="store_true", help="Use cached reranker scores if available")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached reranker scores
    scores_cache_path = output_dir / "reranker_scores_cache.json"

    if args.use_cache and scores_cache_path.exists():
        print(f"Loading cached reranker scores from {scores_cache_path}")
        with open(scores_cache_path) as f:
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

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
    else:
        # Load retrieval candidates from cache
        cache_dir = Path("outputs/hpo_finetuning_with_no_evidence/nv-embed-v2_jina-reranker-v3")
        print(f"Loading cached data from {cache_dir}")
        val_cache = load_cached_data(cache_dir)
        print(f"Val queries: {len(val_cache)}")

        # Load pre-trained reranker
        print(f"Loading pre-trained reranker: {RERANKER_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(args.device)

        # Score validation data
        val_results, all_scores, all_labels = score_with_reranker(val_cache, model, tokenizer, args.device)

        # Save cache for future runs
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
        with open(scores_cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"Saved reranker scores cache to {scores_cache_path}")

        # Clean up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\nValidation results: {len(val_results)} queries")
    print(f"Total score-label pairs: {len(all_scores)}")

    # Print score distribution
    print(f"\nReranker Score Distribution:")
    print(f"  Min: {all_scores.min():.4f}")
    print(f"  Max: {all_scores.max():.4f}")
    print(f"  Mean: {all_scores.mean():.4f}")
    print(f"  Std: {all_scores.std():.4f}")

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
    print("HPO RESULTS (Reranker Scores)")
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
        "score_distribution": {
            "min": float(all_scores.min()),
            "max": float(all_scores.max()),
            "mean": float(all_scores.mean()),
            "std": float(all_scores.std()),
        },
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "best_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
