#!/usr/bin/env python3
"""Tier A HPO: Inference-only hyperparameter optimization for rerankers.

Optimizes inference parameters without retraining:
- top_k_rerank: Number of candidates to rerank
- max_length: Max sequence length for reranker
- score_threshold: Threshold for no-evidence detection

Uses Optuna for Bayesian optimization with TPE sampler.

Note: Uses pickle for loading cached candidate sets (internal use only).

Usage:
    python scripts/reranker/hpo_inference.py \
        --candidates candidates_dev_select_k100.pkl \
        --reranker bge-reranker-v2-m3 \
        --n_trials 50
"""

import argparse
import json
import pickle  # Internal cache loading only
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.data.io import load_sentence_corpus
from final_sc_review.reranker.zoo import RerankerZoo, RerankerConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ndcg(relevances: List[int], k: int) -> float:
    """Compute nDCG@k."""
    relevances = relevances[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Compute Recall@k."""
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & gold)
    return hits / len(gold) if gold else 0.0


def compute_mrr(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Compute MRR@k."""
    for i, uid in enumerate(retrieved[:k]):
        if uid in gold:
            return 1.0 / (i + 1)
    return 0.0


class RerankerHPO:
    """Hyperparameter optimization for reranker inference."""

    def __init__(
        self,
        candidates_path: Path,
        reranker_name: str,
        generator: str = "fusion-rrf",
        device: str = "cuda",
    ):
        self.candidates_path = candidates_path
        self.reranker_name = reranker_name
        self.generator = generator
        self.device = device
        self.current_max_length = None

        # Load data
        self._load_data()

        # Initialize zoo and get reranker
        self.zoo = RerankerZoo(device=device)
        self.reranker = None  # Lazy load with config override

    def _load_data(self):
        """Load candidates and sentence corpus."""
        print("Loading candidates...")
        with open(self.candidates_path, "rb") as f:
            self.candidates_data = pickle.load(f)
        print(f"  Loaded {self.candidates_data['n_queries']} queries")

        print("Loading sentence corpus...")
        sentences = load_sentence_corpus(Path("data/groundtruth/sentence_corpus.jsonl"))
        self.sent_uid_to_text = {s.sent_uid: s.text for s in sentences}
        print(f"  Loaded {len(self.sent_uid_to_text)} sentences")

    def _get_reranker_with_config(self, max_length: int):
        """Get reranker with custom max_length."""
        # Get base config
        base_config = self.zoo.get_config(self.reranker_name)

        # Create modified config
        modified_config = RerankerConfig(
            name=base_config.name,
            model_id=base_config.model_id,
            reranker_type=base_config.reranker_type,
            max_length=max_length,
            batch_size=base_config.batch_size,
            use_fp16=base_config.use_fp16,
            trust_remote_code=base_config.trust_remote_code,
            query_instruction=base_config.query_instruction,
            doc_instruction=base_config.doc_instruction,
            listwise_max_docs=base_config.listwise_max_docs,
            cutoff_layers=base_config.cutoff_layers,
            compress_ratio=base_config.compress_ratio,
        )

        # Create new zoo with modified config
        modified_zoo = RerankerZoo(configs=[modified_config], device=self.device)
        return modified_zoo.get_reranker(self.reranker_name)

    def evaluate(
        self,
        top_k_rerank: int,
        max_length: int,
        score_threshold: float,
        k_eval: int = 10,
        sample_size: Optional[int] = None,
    ) -> Dict:
        """Evaluate reranker with given hyperparameters.

        Returns:
            Dict with nDCG@k, MRR@k, Recall@k, false_evidence_rate
        """
        # Get reranker with custom max_length if different from default
        if self.reranker is None or self.current_max_length != max_length:
            self.reranker = self._get_reranker_with_config(max_length)
            self.reranker.load_model()
            self.current_max_length = max_length

        all_candidates = self.candidates_data["candidates"]

        # Optional sampling for faster HPO
        query_keys = list(all_candidates.keys())
        if sample_size and sample_size < len(query_keys):
            np.random.seed(42)
            query_keys = np.random.choice(query_keys, sample_size, replace=False).tolist()

        # Metrics
        ndcg_scores = []
        mrr_scores = []
        recall_scores = []
        false_evidence_counts = []

        for query_key in query_keys:
            data = all_candidates[query_key]
            query = data["query"]
            cands = data["candidates"].get(self.generator, [])

            if not cands:
                continue

            # Limit candidates to rerank
            cands_to_rerank = cands[:top_k_rerank]

            # Build candidate tuples
            candidate_tuples = []
            for uid, _ in cands_to_rerank:
                text = self.sent_uid_to_text.get(uid, "")
                candidate_tuples.append((uid, text))

            # Rerank
            reranked = self.reranker.rerank(
                query=query["criterion_text"],
                candidates=candidate_tuples,
                top_k=k_eval,
            )

            retrieved_uids = [r.sent_uid for r in reranked]
            gold_uids = query["gold_uids"]

            if query["has_evidence"]:
                # Compute ranking metrics
                relevances = [1 if uid in gold_uids else 0 for uid in retrieved_uids[:k_eval]]
                ndcg_scores.append(compute_ndcg(relevances, k_eval))
                mrr_scores.append(compute_mrr(retrieved_uids, gold_uids, k_eval))
                recall_scores.append(compute_recall(retrieved_uids, gold_uids, k_eval))
            else:
                # No-evidence query: check false positive
                top_scores = [r.score for r in reranked[:k_eval]]
                false_positive = any(s > score_threshold for s in top_scores)
                false_evidence_counts.append(1.0 if false_positive else 0.0)

        return {
            "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
            "false_evidence_rate": float(np.mean(false_evidence_counts)) if false_evidence_counts else 0.0,
            "n_has_evidence": len(ndcg_scores),
            "n_no_evidence": len(false_evidence_counts),
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        top_k_rerank = trial.suggest_categorical("top_k_rerank", [10, 20, 30, 50, 75, 100])
        max_length = trial.suggest_categorical("max_length", [256, 384, 512, 768, 1024])
        score_threshold = trial.suggest_float("score_threshold", -5.0, 5.0)

        # Evaluate
        metrics = self.evaluate(
            top_k_rerank=top_k_rerank,
            max_length=max_length,
            score_threshold=score_threshold,
            k_eval=10,
            sample_size=None,  # Use full dataset
        )

        # Log metrics
        trial.set_user_attr("ndcg@10", metrics["ndcg"])
        trial.set_user_attr("mrr@10", metrics["mrr"])
        trial.set_user_attr("recall@10", metrics["recall"])
        trial.set_user_attr("false_evidence_rate", metrics["false_evidence_rate"])

        # Objective: maximize nDCG@10 while penalizing false evidence
        # Weight false evidence penalty (can be tuned)
        fe_penalty = 0.1 * metrics["false_evidence_rate"]
        objective_value = metrics["ndcg"] - fe_penalty

        return objective_value


def main():
    parser = argparse.ArgumentParser(description="Tier A HPO for reranker inference")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Path to candidates pickle file")
    parser.add_argument("--reranker", type=str, default="bge-reranker-v2-m3",
                        help="Reranker model to optimize")
    parser.add_argument("--generator", type=str, default="fusion-rrf",
                        help="Candidate generator to use")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of HPO trials")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name (default: reranker_hpo_<reranker>)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (default: sqlite in outputs)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/reranker_research/hpo"))
    args = parser.parse_args()

    print("=" * 80)
    print("TIER A HPO: INFERENCE-ONLY OPTIMIZATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Reranker: {args.reranker}")
    print(f"Candidates: {args.candidates}")
    print(f"N trials: {args.n_trials}")
    print("=" * 80)

    # Initialize HPO
    hpo = RerankerHPO(
        candidates_path=args.candidates,
        reranker_name=args.reranker,
        generator=args.generator,
    )

    # Setup Optuna study
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"reranker_hpo_{args.reranker}"
    storage = args.storage or f"sqlite:///{args.output_dir}/optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}")
    print(f"Storage: {storage}")
    print(f"Existing trials: {len(study.trials)}")

    # Run optimization
    print(f"\nRunning {args.n_trials} trials...")
    study.optimize(
        hpo.objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Report results
    print("\n" + "=" * 80)
    print("HPO COMPLETE")
    print("=" * 80)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best value (objective): {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nBest trial metrics:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "reranker": args.reranker,
        "generator": args.generator,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs,
    }

    output_path = args.output_dir / f"hpo_results_{args.reranker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print top 5 trials
    print("\n" + "-" * 80)
    print("TOP 5 TRIALS")
    print("-" * 80)
    print(f"{'Trial':>6} {'Objective':>10} {'top_k':>8} {'max_len':>8} {'threshold':>10} {'nDCG@10':>10}")
    print("-" * 80)

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    for trial in sorted_trials[:5]:
        if trial.value is not None:
            print(f"{trial.number:>6} {trial.value:>10.4f} "
                  f"{trial.params.get('top_k_rerank', 'N/A'):>8} "
                  f"{trial.params.get('max_length', 'N/A'):>8} "
                  f"{trial.params.get('score_threshold', 0):>10.2f} "
                  f"{trial.user_attrs.get('ndcg@10', 0):>10.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
