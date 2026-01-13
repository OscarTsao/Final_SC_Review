#!/usr/bin/env python
"""HPO for each Retriever + Reranker combination.

This script runs Optuna HPO for each retriever+reranker combination,
optimizing inference parameters (top_k, threshold, etc.) to find
the best configuration for each pair.

Usage:
    # Run HPO for all combinations
    python scripts/hpo_retriever_reranker_combinations.py \
        --output_dir outputs/combination_hpo \
        --retrievers bge-m3 bge-large-en-v1.5 \
        --rerankers bge-reranker-v2-m3 jina-reranker-v2 \
        --n_trials 100

    # Run HPO for specific combination
    python scripts/hpo_retriever_reranker_combinations.py \
        --output_dir outputs/combination_hpo \
        --retrievers bge-m3 \
        --rerankers jina-reranker-v2 \
        --n_trials 200
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

# Enable GPU optimizations early (Flash Attention, TF32, bf16)
from final_sc_review.utils.gpu_optimize import enable_gpu_optimizations
enable_gpu_optimizations()

from final_sc_review.data.io import load_criteria
from final_sc_review.data.schemas import Sentence
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.ranking import (
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def compute_batch_metrics(all_retrieved: List[List[str]], all_gold: List[List[str]], k: int) -> Dict[str, float]:
    """Compute average metrics over all queries."""
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []
    map_scores = []

    for retrieved, gold in zip(all_retrieved, all_gold):
        if gold:  # Only compute for queries with gold labels
            ndcg_scores.append(ndcg_at_k(gold, retrieved, k))
            mrr_scores.append(mrr_at_k(gold, retrieved, k))
            recall_scores.append(recall_at_k(gold, retrieved, k))
            map_scores.append(map_at_k(gold, retrieved, k))

    return {
        "ndcg_at_10": np.mean(ndcg_scores) if ndcg_scores else 0.0,
        "mrr_at_10": np.mean(mrr_scores) if mrr_scores else 0.0,
        "recall_at_10": np.mean(recall_scores) if recall_scores else 0.0,
        "map_at_10": np.mean(map_scores) if map_scores else 0.0,
    }


from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.reranker.zoo import RerankerZoo
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CombinationConfig:
    """Best configuration for a retriever+reranker combination."""
    retriever: str
    reranker: str
    top_k_retriever: int
    top_k_rerank: int
    score_threshold: float
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    map_at_10: float
    false_evidence_rate: float
    n_trials: int
    timestamp: str


def build_queries(
    groundtruth: pd.DataFrame,
    criteria: Dict[str, str],
    split_post_ids: List[str],
    max_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build query list from groundtruth."""
    queries = []

    split_gt = groundtruth[groundtruth["post_id"].isin(split_post_ids)]

    for (post_id, criterion), group in split_gt.groupby(["post_id", "criterion"]):
        if criterion not in criteria:
            continue

        gold_uids = set(group[group["groundtruth"] == 1]["sent_uid"].tolist())

        queries.append({
            "post_id": post_id,
            "criterion": criterion,
            "query": criteria[criterion],
            "gold_uids": gold_uids,
            "has_evidence": len(gold_uids) > 0,
        })

    if max_queries:
        queries = queries[:max_queries]

    return queries


class CombinationHPO:
    """HPO for a specific retriever+reranker combination."""

    def __init__(
        self,
        retriever_zoo: RetrieverZoo,
        reranker_zoo: RerankerZoo,
        retriever_name: str,
        reranker_name: str,
        queries: List[Dict[str, Any]],
        output_dir: Path,
    ):
        self.retriever_zoo = retriever_zoo
        self.reranker_zoo = reranker_zoo
        self.retriever_name = retriever_name
        self.reranker_name = reranker_name
        self.queries = queries
        self.output_dir = output_dir

        # Load models
        self.retriever = retriever_zoo.get_retriever(retriever_name)
        self.retriever.encode_corpus()
        self.reranker = reranker_zoo.get_reranker(reranker_name)

        # Cache retrieval results for efficiency
        self._retrieval_cache: Dict[str, Dict[int, List]] = {}

    def _get_retrieval_results(self, top_k: int) -> Dict[str, List]:
        """Get cached retrieval results for given top_k."""
        cache_key = f"{self.retriever_name}_{top_k}"

        if cache_key not in self._retrieval_cache:
            results = {}
            for q in tqdm(self.queries, desc=f"Retrieving (K={top_k})", leave=False):
                candidates = self.retriever.retrieve_within_post(
                    query=q["query"],
                    post_id=q["post_id"],
                    top_k=top_k,
                )
                results[f"{q['post_id']}_{q['criterion']}"] = candidates
            self._retrieval_cache[cache_key] = results

        return self._retrieval_cache[cache_key]

    def _evaluate(
        self,
        top_k_retriever: int,
        top_k_rerank: int,
        score_threshold: float,
    ) -> Dict[str, float]:
        """Evaluate with given parameters using batched reranking."""
        retrieval_results = self._get_retrieval_results(top_k_retriever)

        # Build all queries and candidates for batch reranking
        queries_and_candidates = []
        query_indices = []  # Track which queries have candidates

        for i, q in enumerate(self.queries):
            query_key = f"{q['post_id']}_{q['criterion']}"
            candidates = retrieval_results.get(query_key, [])

            if candidates:
                rerank_candidates = [(c.sent_uid, c.text) for c in candidates[:top_k_rerank]]
                queries_and_candidates.append((q["query"], rerank_candidates))
                query_indices.append(i)

        # Batch rerank all queries at once
        batch_results = self.reranker.rerank_batch(queries_and_candidates)

        # Map results back to queries
        rerank_results_map = {}
        for batch_idx, query_idx in enumerate(query_indices):
            rerank_results_map[query_idx] = batch_results[batch_idx]

        # Process results
        all_retrieved = []
        all_gold = []
        false_evidence_count = 0
        no_evidence_queries = 0

        for i, q in enumerate(self.queries):
            all_gold.append(list(q["gold_uids"]))

            if i not in rerank_results_map:
                all_retrieved.append([])
            else:
                rerank_results = rerank_results_map[i]
                ranked_uids = [r.sent_uid for r in rerank_results if r.score >= score_threshold]
                all_retrieved.append(ranked_uids)

                # Track false evidence on no-evidence queries
                if not q["has_evidence"]:
                    no_evidence_queries += 1
                    if ranked_uids:
                        false_evidence_count += 1

            # Also count no-evidence queries without candidates
            if i not in rerank_results_map and not q["has_evidence"]:
                no_evidence_queries += 1

        # Compute metrics
        metrics = compute_batch_metrics(all_retrieved, all_gold, k=10)

        false_evidence_rate = (
            false_evidence_count / no_evidence_queries
            if no_evidence_queries > 0 else 0.0
        )

        metrics["false_evidence_rate"] = false_evidence_rate
        return metrics

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters (only top_k params - threshold disabled for ranking metrics)
        top_k_retriever = trial.suggest_categorical("top_k_retriever", [50, 100, 200, 400])
        top_k_rerank = trial.suggest_categorical("top_k_rerank", [10, 20, 50, 100])

        # Fixed threshold - disabled for ranking-focused HPO
        # Threshold only matters for precision/FER, not recall/nDCG
        score_threshold = -10.0

        # Skip invalid combinations
        if top_k_rerank > top_k_retriever:
            raise optuna.TrialPruned()

        metrics = self._evaluate(top_k_retriever, top_k_rerank, score_threshold)

        # Store all metrics
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("score_threshold", score_threshold)

        # Optimize nDCG@10 directly (no FER penalty for ranking-focused HPO)
        return metrics["ndcg_at_10"]

    def run(self, n_trials: int) -> CombinationConfig:
        """Run HPO and return best configuration."""
        study_name = f"{self.retriever_name}_{self.reranker_name}_hpo"
        db_path = self.output_dir / f"{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.NopPruner(),  # No pruning for fast evaluation
        )

        logger.info(f"Starting HPO for {self.retriever_name} + {self.reranker_name}")
        study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_trial = study.best_trial

        return CombinationConfig(
            retriever=self.retriever_name,
            reranker=self.reranker_name,
            top_k_retriever=best_trial.params["top_k_retriever"],
            top_k_rerank=best_trial.params["top_k_rerank"],
            score_threshold=best_trial.user_attrs.get("score_threshold", -10.0),
            ndcg_at_10=best_trial.user_attrs["ndcg_at_10"],
            mrr_at_10=best_trial.user_attrs["mrr_at_10"],
            recall_at_10=best_trial.user_attrs["recall_at_10"],
            map_at_10=best_trial.user_attrs["map_at_10"],
            false_evidence_rate=best_trial.user_attrs["false_evidence_rate"],
            n_trials=len(study.trials),
            timestamp=datetime.now().isoformat(),
        )


def run_all_combinations(
    retrievers: List[str],
    rerankers: List[str],
    n_trials: int,
    output_dir: Path,
    split: str = "val",
    max_queries: Optional[int] = None,
) -> List[CombinationConfig]:
    """Run HPO for all retriever+reranker combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_dir = Path("data")
    groundtruth_df = pd.read_csv(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    logger.info(f"Loaded {len(groundtruth_df)} groundtruth rows")
    criteria_list = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    # Convert list of Criterion objects to dict {criterion_id: text}
    criteria = {c.criterion_id: c.text for c in criteria_list}
    logger.info(f"Criteria dict keys: {list(criteria.keys())}")

    # Load sentence corpus
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            data = json.loads(line)
            sentences.append(Sentence(
                post_id=data["post_id"],
                sid=data["sid"],
                sent_uid=data["sent_uid"],
                text=data["text"],
            ))

    # Create splits from unique post_ids
    unique_post_ids = groundtruth_df["post_id"].unique().tolist()
    splits = split_post_ids(unique_post_ids, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    selected_post_ids = splits[split]

    # Build queries
    queries = build_queries(groundtruth_df, criteria, selected_post_ids, max_queries)
    logger.info(f"Built {len(queries)} queries")

    # Initialize zoos
    cache_dir = data_dir / "cache"
    retriever_zoo = RetrieverZoo(sentences, cache_dir)
    reranker_zoo = RerankerZoo()

    # Run HPO for each combination
    results = []
    total_combinations = len(retrievers) * len(rerankers)

    for i, retriever_name in enumerate(retrievers):
        for j, reranker_name in enumerate(rerankers):
            combo_idx = i * len(rerankers) + j + 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Combination {combo_idx}/{total_combinations}")
            logger.info(f"Retriever: {retriever_name}")
            logger.info(f"Reranker: {reranker_name}")
            logger.info(f"{'=' * 60}")

            try:
                hpo = CombinationHPO(
                    retriever_zoo=retriever_zoo,
                    reranker_zoo=reranker_zoo,
                    retriever_name=retriever_name,
                    reranker_name=reranker_name,
                    queries=queries,
                    output_dir=output_dir,
                )

                config = hpo.run(n_trials)
                results.append(config)

                # Save intermediate results
                save_results(results, output_dir)

                logger.info(f"Best config: K_ret={config.top_k_retriever}, K_rerank={config.top_k_rerank}, "
                           f"thresh={config.score_threshold:.2f}")
                logger.info(f"nDCG@10={config.ndcg_at_10:.4f}, MRR@10={config.mrr_at_10:.4f}, "
                           f"FalseEvid={config.false_evidence_rate:.4f}")

            except Exception as e:
                logger.error(f"Failed: {retriever_name} + {reranker_name}: {e}")
                continue

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def save_results(results: List[CombinationConfig], output_dir: Path):
    """Save results to CSV and JSON."""
    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    df = df.sort_values("ndcg_at_10", ascending=False)

    # Save CSV
    df.to_csv(output_dir / "combination_hpo_results.csv", index=False)

    # Save JSON summary
    best_overall = df.iloc[0].to_dict()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_combinations": len(results),
        "best_overall": best_overall,
        "all_results": [asdict(r) for r in results],
    }

    with open(output_dir / "combination_hpo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print leaderboard
    print("\n" + "=" * 80)
    print("RETRIEVER + RERANKER HPO LEADERBOARD")
    print("=" * 80)
    print(df[["retriever", "reranker", "top_k_retriever", "top_k_rerank",
              "score_threshold", "ndcg_at_10", "mrr_at_10", "false_evidence_rate"]].to_string())
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="HPO for retriever+reranker combinations")
    parser.add_argument("--output_dir", type=str, default="outputs/combination_hpo")
    parser.add_argument(
        "--retrievers",
        type=str,
        nargs="+",
        default=["bge-m3", "bge-large-en-v1.5", "e5-large-v2"],
        help="Retriever names"
    )
    parser.add_argument(
        "--rerankers",
        type=str,
        nargs="+",
        default=["bge-reranker-v2-m3", "jina-reranker-v2"],
        help="Reranker names"
    )
    parser.add_argument("--n_trials", type=int, default=100, help="HPO trials per combination")
    parser.add_argument("--max_queries", type=int, default=None, help="Limit queries")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])

    args = parser.parse_args()

    print("=" * 80)
    print("RETRIEVER + RERANKER COMBINATION HPO")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Retrievers: {args.retrievers}")
    print(f"Rerankers: {args.rerankers}")
    print(f"N trials per combination: {args.n_trials}")
    print(f"Total combinations: {len(args.retrievers) * len(args.rerankers)}")
    print("=" * 80)

    results = run_all_combinations(
        retrievers=args.retrievers,
        rerankers=args.rerankers,
        n_trials=args.n_trials,
        output_dir=Path(args.output_dir),
        split=args.split,
        max_queries=args.max_queries,
    )

    logger.info(f"\nHPO complete! {len(results)} combinations evaluated")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
