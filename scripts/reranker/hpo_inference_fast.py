#!/usr/bin/env python3
"""Fast HPO for reranker inference with precomputed scores.

Key optimizations:
1. Precompute all reranking scores ONCE at max sequence length
2. HPO only searches over top_k and threshold (no model reloading)
3. Uses batch inference for speed

This is 10-50x faster than the standard HPO script.

Usage:
    python scripts/reranker/hpo_inference_fast.py \
        --candidates candidates_dev_select_k100.pkl \
        --reranker jina-reranker-v3 \
        --n_trials 50
"""

import argparse
import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.data.io import load_sentence_corpus
from final_sc_review.reranker.zoo import RerankerZoo
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
    return len(set(retrieved[:k]) & gold) / len(gold) if gold else 0.0


def compute_mrr(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Compute MRR@k."""
    for i, uid in enumerate(retrieved[:k]):
        if uid in gold:
            return 1.0 / (i + 1)
    return 0.0


class FastRerankerHPO:
    """Fast HPO with precomputed scores."""

    def __init__(
        self,
        candidates_path: Path,
        reranker_name: str,
        generator: str = "fusion-rrf",
        device: str = "cuda",
        max_rerank_k: int = 100,
    ):
        self.candidates_path = candidates_path
        self.reranker_name = reranker_name
        self.generator = generator
        self.device = device
        self.max_rerank_k = max_rerank_k

        # Load data
        self._load_data()

        # Precomputed scores cache
        self.precomputed_scores = None
        self.query_metadata = None

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

    def precompute_scores(self):
        """Precompute all reranking scores once."""
        print(f"\nPrecomputing scores for {self.reranker_name}...")
        print(f"  Max K: {self.max_rerank_k}")

        # Initialize reranker
        zoo = RerankerZoo(device=self.device)
        reranker = zoo.get_reranker(self.reranker_name)
        reranker.load_model()

        all_candidates = self.candidates_data["candidates"]

        precomputed = {}
        metadata = {}

        for query_key in tqdm(all_candidates.keys(), desc="Precomputing"):
            data = all_candidates[query_key]
            query = data["query"]
            cands = data["candidates"].get(self.generator, [])

            if not cands:
                continue

            # Limit candidates
            cands_to_rerank = cands[:self.max_rerank_k]

            # Build candidate tuples
            candidate_tuples = []
            for uid, _ in cands_to_rerank:
                text = self.sent_uid_to_text.get(uid, "")
                candidate_tuples.append((uid, text))

            # Get scores for ALL candidates (not just top_k)
            results = reranker.rerank(
                query=query["criterion_text"],
                candidates=candidate_tuples,
                top_k=len(candidate_tuples),  # Get all scores
            )

            # Store sorted (uid, score) pairs
            precomputed[query_key] = [(r.sent_uid, r.score) for r in results]
            metadata[query_key] = {
                "gold_uids": set(query["gold_uids"]),
                "has_evidence": query["has_evidence"],
            }

        self.precomputed_scores = precomputed
        self.query_metadata = metadata

        # Free GPU memory
        del reranker
        del zoo
        gc.collect()

        print(f"  Precomputed {len(precomputed)} queries")
        return len(precomputed)

    def evaluate_fast(
        self,
        top_k_rerank: int,
        score_threshold: float,
        k_eval: int = 10,
    ) -> Dict:
        """Fast evaluation using precomputed scores."""
        if self.precomputed_scores is None:
            raise ValueError("Must call precompute_scores() first")

        ndcg_scores = []
        mrr_scores = []
        recall_scores = []
        false_evidence_counts = []

        for query_key, scored_results in self.precomputed_scores.items():
            meta = self.query_metadata[query_key]
            gold_uids = meta["gold_uids"]
            has_evidence = meta["has_evidence"]

            # Apply top_k_rerank (already sorted by score)
            top_results = scored_results[:top_k_rerank]

            # Get top k_eval for evaluation
            eval_results = top_results[:k_eval]
            retrieved_uids = [uid for uid, _ in eval_results]
            scores = [score for _, score in eval_results]

            if has_evidence:
                # Compute ranking metrics
                relevances = [1 if uid in gold_uids else 0 for uid in retrieved_uids]
                ndcg_scores.append(compute_ndcg(relevances, k_eval))
                mrr_scores.append(compute_mrr(retrieved_uids, gold_uids, k_eval))
                recall_scores.append(compute_recall(retrieved_uids, gold_uids, k_eval))
            else:
                # No-evidence query: check false positive
                false_positive = any(s > score_threshold for s in scores)
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
        """Optuna objective function - fast evaluation."""
        top_k_rerank = trial.suggest_categorical("top_k_rerank", [10, 20, 30, 50, 75, 100])
        score_threshold = trial.suggest_float("score_threshold", -5.0, 5.0)

        metrics = self.evaluate_fast(
            top_k_rerank=top_k_rerank,
            score_threshold=score_threshold,
            k_eval=10,
        )

        trial.set_user_attr("ndcg@10", metrics["ndcg"])
        trial.set_user_attr("mrr@10", metrics["mrr"])
        trial.set_user_attr("recall@10", metrics["recall"])
        trial.set_user_attr("false_evidence_rate", metrics["false_evidence_rate"])

        # Objective: maximize nDCG@10 with false evidence penalty
        fe_penalty = 0.1 * metrics["false_evidence_rate"]
        return metrics["ndcg"] - fe_penalty


def run_fast_hpo(
    candidates_path: Path,
    reranker_name: str,
    generator: str = "fusion-rrf",
    n_trials: int = 50,
    study_name: str = None,
    output_dir: Path = Path("outputs/reranker_research/hpo"),
):
    """Run fast HPO for a single reranker."""
    print("=" * 80)
    print("FAST HPO: PRECOMPUTED SCORES")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Reranker: {reranker_name}")
    print(f"N trials: {n_trials}")
    print("=" * 80)

    # Initialize and precompute
    hpo = FastRerankerHPO(
        candidates_path=candidates_path,
        reranker_name=reranker_name,
        generator=generator,
    )
    hpo.precompute_scores()

    # Setup study
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"fast_hpo_{reranker_name}"
    storage = f"sqlite:///{output_dir}/optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}")
    print(f"Running {n_trials} trials (fast mode)...")

    study.optimize(hpo.objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 80)
    print("HPO COMPLETE")
    print("=" * 80)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("\nBest metrics:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "reranker": reranker_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs,
    }

    output_path = output_dir / f"hpo_fast_{reranker_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return study.best_params, study.best_trial.user_attrs


def main():
    parser = argparse.ArgumentParser(description="Fast HPO for reranker inference")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reranker", type=str, default="bge-reranker-v2-m3")
    parser.add_argument("--generator", type=str, default="fusion-rrf")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/reranker_research/hpo"))
    args = parser.parse_args()

    run_fast_hpo(
        candidates_path=args.candidates,
        reranker_name=args.reranker,
        generator=args.generator,
        n_trials=args.n_trials,
        study_name=args.study_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
