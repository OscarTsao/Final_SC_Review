#!/usr/bin/env python3
"""
HPO for retriever with frozen embeddings.

Optimizes fusion weights and parameters using pre-computed embeddings.
Uses Optuna for hyperparameter optimization on dev_tune split.

Usage:
    python scripts/retriever/hpo_frozen.py --n_trials 100
    python scripts/retriever/hpo_frozen.py --study_name my_study --n_trials 200
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import yaml
from tqdm import tqdm


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def create_objective(retriever, queries, groundtruth, k_values):
    """Create Optuna objective function."""

    def objective(trial):
        # Sample hyperparameters
        top_k_retriever = trial.suggest_int("top_k_retriever", 20, 100)
        top_k_final = trial.suggest_int("top_k_final", 5, 20)
        use_sparse = trial.suggest_categorical("use_sparse", [True, False])
        use_colbert = trial.suggest_categorical("use_colbert", [True, False])
        fusion_method = trial.suggest_categorical("fusion_method", ["weighted_sum", "rrf"])
        score_norm = trial.suggest_categorical(
            "score_normalization",
            ["none", "minmax_per_query", "zscore_per_query"]
        )

        # Weights
        w_dense = trial.suggest_float("w_dense", 0.1, 1.0)
        w_sparse = trial.suggest_float("w_sparse", 0.0, 1.0) if use_sparse else 0.0
        w_colbert = trial.suggest_float("w_colbert", 0.0, 1.0) if use_colbert else 0.0

        # Evaluate
        ndcg_scores = []

        for post_id, criterion_id, gold_uids, query_text in queries:
            if not gold_uids:
                continue

            results = retriever.retrieve_within_post(
                query=query_text,
                post_id=post_id,
                top_k_retriever=top_k_retriever,
                use_dense=True,
                use_sparse=use_sparse,
                use_colbert=use_colbert,
                dense_weight=w_dense,
                sparse_weight=w_sparse,
                colbert_weight=w_colbert,
                fusion_method=fusion_method,
                score_normalization=score_norm,
            )

            ranked_uids = [r[0] for r in results[:top_k_final]]
            gold_set = set(gold_uids)

            # Calculate nDCG@10
            dcg = 0.0
            for i, uid in enumerate(ranked_uids[:10]):
                if uid in gold_set:
                    dcg += 1.0 / np.log2(i + 2)

            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_uids), 10)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    return objective


def main():
    parser = argparse.ArgumentParser(description="HPO for retriever with frozen embeddings")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--study_name", type=str, default="retriever_frozen_hpo", help="Study name")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth, load_criteria
    from final_sc_review.data.splits import split_post_ids
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever

    # Load config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    paths = config.get("paths", {})
    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("="*60)
    print("RETRIEVER HPO (FROZEN EMBEDDINGS)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"N trials: {args.n_trials}")
    print(f"Study: {args.study_name}")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Build queries
    from collections import defaultdict
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}

    # Split posts
    print("\n[2/4] Splitting posts...")
    splits = split_post_ids(list(all_posts), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    dev_tune_posts = set(splits["train"])

    # Filter to dev_tune
    queries = []
    for (post_id, criterion_id), gold_uids in query_gold.items():
        if post_id in dev_tune_posts:
            query_text = criterion_text.get(criterion_id, criterion_id)
            queries.append((post_id, criterion_id, list(gold_uids), query_text))

    print(f"  Dev tune queries: {len(queries)}")

    # Initialize retriever
    print("\n[3/4] Initializing retriever...")
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        rebuild_cache=False,
    )

    # Run HPO
    print(f"\n[4/4] Running HPO with {args.n_trials} trials...")
    output_dir = repo_root / "outputs" / "retriever" / "hpo"
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_dir / 'optuna.db'}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = create_objective(retriever, queries, query_gold, k_values=[5, 10, 20])
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Save results
    best_params = study.best_params
    best_value = study.best_value

    results_path = output_dir / f"{args.study_name}_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "study_name": args.study_name,
        "n_trials": args.n_trials,
        "best_ndcg10": best_value,
        "best_params": best_params,
    }
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n  Best nDCG@10: {best_value:.4f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")
    print(f"  Saved to: {results_path}")

    print("\n" + "="*60)
    print("[SUCCESS] HPO complete")
    print("="*60)


if __name__ == "__main__":
    main()
