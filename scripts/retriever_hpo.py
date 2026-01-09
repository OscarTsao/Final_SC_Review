#!/usr/bin/env python3
"""Retriever HPO Training Pipeline (Phase 4).

For each retriever model:
1. Screen phase: 30 trials, short budget
2. Full phase: top 20% promoted to full budget
3. Evaluate on dev_tune, select on dev_select
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
import torch
import yaml

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_time_tracker import GPUTimeTracker
from create_splits import load_split

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.metrics.ranking import recall_at_k, ndcg_at_k, mrr_at_k
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrieverHPOConfig:
    """HPO configuration for retriever training."""
    model_name: str
    model_id: str
    training_mode: str  # full, lora, none
    n_trials_screen: int = 30
    n_trials_full: int = 10
    promote_ratio: float = 0.2
    k_primary: list = None
    output_dir: str = "outputs/retriever_hpo"

    def __post_init__(self):
        if self.k_primary is None:
            self.k_primary = [3, 5, 10]


def compute_retrieval_metrics(
    model,
    groundtruth_rows,
    criteria,
    sentences,
    post_ids: set,
    ks: list = [3, 5, 10],
) -> Dict:
    """Compute retrieval metrics on a split."""
    from sentence_transformers import SentenceTransformer

    # Build indices
    criterion_text = {c.criterion_id: c.text for c in criteria}
    post_to_sentences = defaultdict(list)
    sent_uid_to_text = {}
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)
        sent_uid_to_text[sent.sent_uid] = sent.text

    # Group groundtruth
    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth_rows:
        if row.post_id not in post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    # Metrics accumulators
    metrics = {f"oracle@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"ndcg@{k}": [] for k in ks})
    metrics.update({f"mrr@{k}": [] for k in ks})

    n_positive = 0
    n_empty = 0

    for (post_id, criterion_id), data in groups.items():
        gold_uids = data["gold_uids"]

        if not gold_uids:
            n_empty += 1
            continue

        n_positive += 1
        query_text = criterion_text.get(criterion_id)
        if not query_text:
            continue

        post_sents = post_to_sentences.get(post_id, [])
        if not post_sents:
            continue

        candidate_texts = [s.text for s in post_sents]
        candidate_uids = [s.sent_uid for s in post_sents]

        # Encode and score
        query_emb = model.encode([query_text], normalize_embeddings=True)
        cand_embs = model.encode(candidate_texts, normalize_embeddings=True)
        scores = np.dot(query_emb, cand_embs.T)[0]

        # Rank
        ranked_indices = np.argsort(scores)[::-1]
        ranked_uids = [candidate_uids[i] for i in ranked_indices]

        n_candidates = len(ranked_uids)

        for k in ks:
            k_eff = min(k, n_candidates)
            top_k_uids = set(ranked_uids[:k_eff])

            # Oracle
            oracle = 1.0 if top_k_uids & gold_uids else 0.0
            metrics[f"oracle@{k}"].append(oracle)

            # Recall, nDCG, MRR
            metrics[f"recall@{k}"].append(recall_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"mrr@{k}"].append(mrr_at_k(gold_uids, ranked_uids, k_eff))

    # Aggregate
    result = {
        "n_positive": n_positive,
        "n_empty": n_empty,
    }
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    return result


def compute_bm25_metrics(
    groundtruth_rows,
    criteria,
    sentences,
    post_ids: set,
    ks: list = [3, 5, 10],
) -> Dict:
    """Compute BM25 retrieval metrics on a split."""
    from final_sc_review.retriever.zoo import BM25Retriever, RetrieverConfig
    from final_sc_review.data.schemas import Sentence

    # Build BM25 retriever
    bm25_config = RetrieverConfig(
        name="bm25",
        model_id="bm25",
        retriever_type="lexical",
    )
    cache_dir = Path("data/cache")
    bm25_retriever = BM25Retriever(bm25_config, sentences, cache_dir)
    bm25_retriever.encode_corpus()

    # Build indices
    criterion_text = {c.criterion_id: c.text for c in criteria}
    post_to_sentences = defaultdict(list)
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)

    # Group groundtruth
    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth_rows:
        if row.post_id not in post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    # Metrics accumulators
    metrics = {f"oracle@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"ndcg@{k}": [] for k in ks})
    metrics.update({f"mrr@{k}": [] for k in ks})

    n_positive = 0
    n_empty = 0

    for (post_id, criterion_id), data in groups.items():
        gold_uids = data["gold_uids"]

        if not gold_uids:
            n_empty += 1
            continue

        n_positive += 1
        query_text = criterion_text.get(criterion_id)
        if not query_text:
            continue

        # Use BM25 retrieval
        results = bm25_retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k=max(ks) * 10,  # Retrieve enough candidates
        )

        ranked_uids = [r.sent_uid for r in results]
        n_candidates = len(ranked_uids)

        for k in ks:
            k_eff = min(k, n_candidates)
            top_k_uids = set(ranked_uids[:k_eff])

            # Oracle
            oracle = 1.0 if top_k_uids & gold_uids else 0.0
            metrics[f"oracle@{k}"].append(oracle)

            # Recall, nDCG, MRR
            metrics[f"recall@{k}"].append(recall_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"mrr@{k}"].append(mrr_at_k(gold_uids, ranked_uids, k_eff))

    # Aggregate
    result = {
        "n_positive": n_positive,
        "n_empty": n_empty,
    }
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    return result


def train_retriever_trial(
    trial: optuna.Trial,
    config: RetrieverHPOConfig,
    groundtruth,
    criteria,
    sentences,
    dev_tune_ids: set,
    gpu_tracker: GPUTimeTracker,
    is_full_budget: bool = False,
) -> float:
    """Run a single HPO trial for retriever training."""
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers
    from datasets import Dataset

    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    temperature = trial.suggest_float("temperature", 0.02, 0.1)
    max_length = trial.suggest_categorical("max_length", [128, 256])
    hard_neg_count = trial.suggest_int("hard_neg_count", 1, 5)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])  # Smaller batches for OOM safety

    # Epochs based on budget
    num_epochs = 3 if not is_full_budget else trial.suggest_int("num_epochs", 3, 7)

    logger.info(f"Trial {trial.number}: lr={lr:.2e}, epochs={num_epochs}, batch={batch_size}")

    # Build training pairs
    criterion_text = {c.criterion_id: c.text for c in criteria}
    post_to_sentences = defaultdict(list)
    sent_uid_to_text = {}
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)
        sent_uid_to_text[sent.sent_uid] = sent.text

    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth:
        if row.post_id not in dev_tune_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    # Build training data
    train_anchors = []
    train_positives = []

    for (post_id, criterion_id), data in groups.items():
        gold_uids = data["gold_uids"]
        if not gold_uids:
            continue

        query = criterion_text.get(criterion_id)
        if not query:
            continue

        for gold_uid in gold_uids:
            positive_text = sent_uid_to_text.get(gold_uid)
            if positive_text:
                train_anchors.append(query)
                train_positives.append(positive_text)

    if len(train_anchors) < 10:
        return float("inf")

    train_dataset = Dataset.from_dict({
        "anchor": train_anchors,
        "positive": train_positives,
    })

    # Load model with memory optimization
    model = SentenceTransformer(config.model_id, trust_remote_code=True)
    # Enable gradient checkpointing if possible
    try:
        model[0].auto_model.gradient_checkpointing_enable()
    except:
        pass

    # Training args
    output_dir = Path(config.output_dir) / f"trial_{trial.number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=50,
    )

    loss = losses.MultipleNegativesRankingLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    # Track GPU time
    session_id = gpu_tracker.start(
        phase="retriever_hpo",
        description=f"{config.model_name} trial {trial.number}"
    )

    try:
        trainer.train()
    finally:
        gpu_tracker.stop()

    # Evaluate on dev_tune
    metrics = compute_retrieval_metrics(
        model, groundtruth, criteria, sentences,
        dev_tune_ids, ks=config.k_primary
    )

    # Primary objective: oracle@10
    objective = -metrics["oracle@10"]  # Minimize negative

    logger.info(f"Trial {trial.number}: oracle@10={metrics['oracle@10']:.4f}, ndcg@10={metrics['ndcg@10']:.4f}")

    # Save trial results
    trial_result = {
        "trial": trial.number,
        "params": trial.params,
        "metrics": metrics,
        "objective": -objective,
    }
    with open(output_dir / "trial_result.json", "w") as f:
        json.dump(trial_result, f, indent=2)

    return objective


def run_retriever_hpo(config: RetrieverHPOConfig):
    """Run full HPO pipeline for a retriever."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RETRIEVER HPO: {config.model_name}")
    logger.info(f"{'='*60}")

    # Load data
    data_dir = Path("data")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")

    # Load splits
    dev_tune_ids = set(load_split("dev_tune"))
    dev_select_ids = set(load_split("dev_select"))

    logger.info(f"dev_tune posts: {len(dev_tune_ids)}")
    logger.info(f"dev_select posts: {len(dev_select_ids)}")

    # GPU tracker
    gpu_tracker = GPUTimeTracker(output_dir="outputs/system")

    # Output dir
    output_dir = Path(config.output_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.training_mode == "none":
        # Inference-only evaluation
        logger.info("Inference-only mode - no training")

        # Special handling for BM25 (lexical baseline)
        if config.model_id == "bm25":
            logger.info("Using BM25 lexical retriever")
            metrics_tune = compute_bm25_metrics(
                groundtruth, criteria, sentences,
                dev_tune_ids, ks=config.k_primary
            )
            metrics_select = compute_bm25_metrics(
                groundtruth, criteria, sentences,
                dev_select_ids, ks=config.k_primary
            )
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.model_id, trust_remote_code=True)

            metrics_tune = compute_retrieval_metrics(
                model, groundtruth, criteria, sentences,
                dev_tune_ids, ks=config.k_primary
            )
            metrics_select = compute_retrieval_metrics(
                model, groundtruth, criteria, sentences,
                dev_select_ids, ks=config.k_primary
            )

        result = {
            "model_name": config.model_name,
            "model_id": config.model_id,
            "training_mode": "none",
            "dev_tune_metrics": metrics_tune,
            "dev_select_metrics": metrics_select,
            "best_trial": None,
        }

        with open(output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"retriever_{config.model_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Screen phase
    logger.info(f"\n--- Screen Phase: {config.n_trials_screen} trials ---")

    def objective_screen(trial):
        return train_retriever_trial(
            trial, config, groundtruth, criteria, sentences,
            dev_tune_ids, gpu_tracker, is_full_budget=False
        )

    study.optimize(objective_screen, n_trials=config.n_trials_screen, show_progress_bar=True)

    # Get top trials for full budget
    n_promote = max(1, int(config.n_trials_screen * config.promote_ratio))
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else float("inf"))[:n_promote]

    logger.info(f"\n--- Full Budget Phase: {len(top_trials)} promoted trials ---")

    # Run full budget for top trials
    best_result = None
    best_value = float("inf")

    for i, trial in enumerate(top_trials):
        logger.info(f"Full budget trial {i+1}/{len(top_trials)}")

        # Create new trial with same params
        full_trial = study.ask()
        for key, value in trial.params.items():
            full_trial.set_user_attr(f"screen_{key}", value)

        value = train_retriever_trial(
            full_trial, config, groundtruth, criteria, sentences,
            dev_tune_ids, gpu_tracker, is_full_budget=True
        )
        study.tell(full_trial, value)

        if value < best_value:
            best_value = value
            best_result = {
                "trial": full_trial.number,
                "params": full_trial.params,
                "value": -value,
            }

    # Final evaluation on dev_select with best model
    logger.info("\n--- Final Evaluation on dev_select ---")

    # Retrain best model and evaluate
    from sentence_transformers import SentenceTransformer
    best_model_path = output_dir / f"trial_{best_result['trial']}"

    if (best_model_path / "model.safetensors").exists():
        model = SentenceTransformer(str(best_model_path), trust_remote_code=True)
    else:
        model = SentenceTransformer(config.model_id, trust_remote_code=True)

    metrics_select = compute_retrieval_metrics(
        model, groundtruth, criteria, sentences,
        dev_select_ids, ks=config.k_primary
    )

    result = {
        "model_name": config.model_name,
        "model_id": config.model_id,
        "training_mode": config.training_mode,
        "best_trial": best_result,
        "dev_select_metrics": metrics_select,
        "n_trials_total": len(study.trials),
        "gpu_hours": gpu_tracker.total_gpu_hours,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Best oracle@10 on dev_select: {metrics_select['oracle@10']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_lists.yaml")
    parser.add_argument("--model", type=str, default=None, help="Run single model only")
    args = parser.parse_args()

    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    retrievers = model_config["retrievers_to_tune"]
    hpo_config = model_config["retriever_hpo"]

    results = []

    for retriever in retrievers:
        if args.model and retriever["name"] != args.model:
            continue

        config = RetrieverHPOConfig(
            model_name=retriever["name"],
            model_id=retriever["model_id"],
            training_mode=retriever.get("training_mode", "full"),
            n_trials_screen=hpo_config["n_trials_screen"],
            n_trials_full=hpo_config.get("n_trials_full", 10),
            promote_ratio=hpo_config["top_k_promote"],
            k_primary=model_config["k_policy"]["k_primary"],
        )

        try:
            result = run_retriever_hpo(config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to run HPO for {retriever['name']}: {e}")
            results.append({
                "model_name": retriever["name"],
                "error": str(e),
            })

    # Save all results
    output_dir = Path("outputs/retriever_hpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "hpo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Select best and top3
    valid_results = [r for r in results if "error" not in r and r.get("dev_select_metrics")]
    if valid_results:
        sorted_results = sorted(
            valid_results,
            key=lambda r: -r["dev_select_metrics"]["oracle@10"]
        )

        best = sorted_results[0]
        top3 = sorted_results[:3]

        with open(output_dir / "best_retriever.json", "w") as f:
            json.dump(best, f, indent=2)

        with open(output_dir / "top3_retrievers.json", "w") as f:
            json.dump(top3, f, indent=2)

        print(f"\nBest Retriever: {best['model_name']} (oracle@10={best['dev_select_metrics']['oracle@10']:.4f})")


if __name__ == "__main__":
    main()
