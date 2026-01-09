#!/usr/bin/env python3
"""Reranker HPO Training Pipeline (Phase 5).

For reranker model:
1. Screen phase: 30 trials, short budget (3 epochs)
2. Full phase: top 20% promoted to full budget (4-7 epochs)
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
class RerankerHPOConfig:
    """HPO configuration for reranker training."""
    model_name: str = "jina-reranker-v3"
    model_id: str = "jinaai/jina-reranker-v3"
    retriever_model: str = "intfloat/e5-large-v2"
    n_trials_screen: int = 30
    n_trials_full: int = 10
    promote_ratio: float = 0.2
    k_primary: list = None
    output_dir: str = "outputs/reranker_hpo"
    max_candidates: int = 16

    def __post_init__(self):
        if self.k_primary is None:
            self.k_primary = [3, 5, 10]


def build_reranker_examples(groundtruth, criteria, sentences, post_ids, max_candidates=16, seed=42):
    """Build reranker training examples with hard negatives from retriever."""
    from sentence_transformers import SentenceTransformer

    criterion_text = {c.criterion_id: c.text for c in criteria}
    post_to_sentences = defaultdict(list)
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)

    retriever = SentenceTransformer("intfloat/e5-large-v2")

    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth:
        if row.post_id not in post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    examples = []
    np.random.seed(seed)

    for (post_id, criterion_id), data in groups.items():
        gold_uids = data["gold_uids"]
        all_uids = data["all_uids"]

        if not gold_uids:
            continue

        query_text = criterion_text.get(criterion_id)
        if not query_text:
            continue

        post_sents = post_to_sentences.get(post_id, [])
        if not post_sents:
            continue

        sent_texts = [s.text for s in post_sents]
        sent_uids = [s.sent_uid for s in post_sents]

        query_emb = retriever.encode(["query: " + query_text], normalize_embeddings=True)
        sent_embs = retriever.encode(["passage: " + t for t in sent_texts], normalize_embeddings=True)
        scores = np.dot(query_emb, sent_embs.T)[0]

        ranked_indices = np.argsort(scores)[::-1]

        positives = [(sent_uids[i], sent_texts[i], 1) for i in range(len(sent_uids)) if sent_uids[i] in gold_uids]
        negatives = [(sent_uids[i], sent_texts[i], 0) for i in ranked_indices if sent_uids[i] not in gold_uids]

        n_neg = max_candidates - len(positives)
        if n_neg > 0:
            negatives = negatives[:n_neg]

        candidates = positives + negatives
        if len(candidates) < 2:
            continue

        examples.append({
            "query": query_text,
            "post_id": post_id,
            "criterion_id": criterion_id,
            "candidates": candidates,
        })

    return examples


def compute_reranker_metrics(
    model,
    tokenizer,
    examples,
    ks: list = [3, 5, 10],
    device: str = "cuda",
) -> Dict:
    """Compute reranker metrics on examples."""
    metrics = {f"oracle@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"ndcg@{k}": [] for k in ks})
    metrics.update({f"mrr@{k}": [] for k in ks})

    n_positive = 0
    n_empty = 0

    model.eval()

    for example in examples:
        query = example["query"]
        candidates = example["candidates"]

        gold_uids = {uid for uid, text, label in candidates if label == 1}

        if not gold_uids:
            n_empty += 1
            continue

        n_positive += 1

        pairs = [(query, text) for uid, text, label in candidates]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            scores = model(**inputs).logits.squeeze(-1).cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1]
        ranked_uids = [candidates[i][0] for i in ranked_indices]

        n_candidates = len(ranked_uids)

        for k in ks:
            k_eff = min(k, n_candidates)
            top_k_uids = set(ranked_uids[:k_eff])

            oracle = 1.0 if top_k_uids & gold_uids else 0.0
            metrics[f"oracle@{k}"].append(oracle)

            metrics[f"recall@{k}"].append(recall_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"mrr@{k}"].append(mrr_at_k(gold_uids, ranked_uids, k_eff))

    result = {
        "n_positive": n_positive,
        "n_empty": n_empty,
    }
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    return result


def train_reranker_trial(
    trial: optuna.Trial,
    config: RerankerHPOConfig,
    train_examples,
    val_examples,
    gpu_tracker: GPUTimeTracker,
    is_full_budget: bool = False,
) -> float:
    """Run a single HPO trial for reranker training."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from final_sc_review.reranker.losses import hybrid_loss

    lr = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    temperature = trial.suggest_float("temperature", 0.5, 2.0)
    w_list = trial.suggest_float("w_list", 0.5, 2.0)
    w_pair = trial.suggest_float("w_pair", 0.5, 2.0)
    w_point = trial.suggest_float("w_point", 0.0, 1.0)
    max_length = trial.suggest_categorical("max_length", [128, 256])

    num_epochs = 3 if not is_full_budget else trial.suggest_int("num_epochs", 3, 7)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=1,
        trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    session_id = gpu_tracker.start(
        phase="reranker_hpo",
        description=f"{config.model_name} trial {trial.number}"
    )

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for example in train_examples:
                query = example["query"]
                candidates = example["candidates"]

                if len(candidates) < 2:
                    continue

                pairs = [(query, text) for uid, text, label in candidates]
                labels = torch.tensor([label for uid, text, label in candidates], dtype=torch.float32).to(device)

                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                optimizer.zero_grad()
                logits = model(**inputs).logits.squeeze(-1)

                loss = hybrid_loss(
                    logits, labels, [len(candidates)],
                    w_list=w_list, w_pair=w_pair, w_point=w_point,
                    temperature=temperature
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if epoch > 0:
                trial.report(avg_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        val_metrics = compute_reranker_metrics(model, tokenizer, val_examples, config.k_primary, device)

        return -val_metrics["oracle@10"]

    finally:
        gpu_tracker.stop()
        del model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_reranker_hpo(config: RerankerHPOConfig):
    """Run complete reranker HPO pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RERANKER HPO: {config.model_name}")
    logger.info(f"{'='*60}")

    data_dir = Path("data")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")

    dev_tune_ids = set(load_split("dev_tune"))
    dev_select_ids = set(load_split("dev_select"))

    logger.info(f"dev_tune posts: {len(dev_tune_ids)}")
    logger.info(f"dev_select posts: {len(dev_select_ids)}")

    logger.info("Building training examples with hard negatives...")
    train_examples = build_reranker_examples(
        groundtruth, criteria, sentences, dev_tune_ids,
        max_candidates=config.max_candidates, seed=42
    )
    val_examples = build_reranker_examples(
        groundtruth, criteria, sentences, dev_select_ids,
        max_candidates=config.max_candidates, seed=42
    )

    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Val examples: {len(val_examples)}")

    gpu_tracker = GPUTimeTracker(output_dir="outputs/system")

    output_dir = Path(config.output_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"reranker_{config.model_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logger.info(f"\n--- Screen Phase: {config.n_trials_screen} trials ---")

    def objective_screen(trial):
        return train_reranker_trial(
            trial, config, train_examples, val_examples,
            gpu_tracker, is_full_budget=False
        )

    study.optimize(objective_screen, n_trials=config.n_trials_screen, show_progress_bar=True)

    n_promote = max(1, int(config.n_trials_screen * config.promote_ratio))
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else float("inf"))[:n_promote]

    logger.info(f"\n--- Full Budget Phase: {len(top_trials)} promoted trials ---")

    best_result = None
    best_value = float("inf")

    for i, trial in enumerate(top_trials):
        logger.info(f"Full budget trial {i+1}/{len(top_trials)}")

        full_trial = study.ask()
        for key, value in trial.params.items():
            full_trial.set_user_attr(f"screen_{key}", value)

        try:
            value = train_reranker_trial(
                full_trial, config, train_examples, val_examples,
                gpu_tracker, is_full_budget=True
            )
            study.tell(full_trial, value)

            if value < best_value:
                best_value = value
                best_result = {
                    "trial": full_trial.number,
                    "params": full_trial.params,
                    "value": -value,
                }
        except optuna.TrialPruned:
            logger.info(f"Full budget trial {i+1} was pruned")
            study.tell(full_trial, float("inf"))

    logger.info("\n--- Final Metrics ---")
    best_trial = study.best_trial

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=1,
        trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dev_select_metrics = compute_reranker_metrics(model, tokenizer, val_examples, config.k_primary, device)

    result = {
        "model_name": config.model_name,
        "model_id": config.model_id,
        "retriever_model": config.retriever_model,
        "best_trial": best_result,
        "dev_select_metrics": dev_select_metrics,
        "n_trials_total": len(study.trials),
        "gpu_hours": gpu_tracker.total_gpu_hours,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    parent_output = Path(config.output_dir)
    parent_output.mkdir(parents=True, exist_ok=True)

    with open(parent_output / "best_reranker.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nBest Reranker: {config.model_name} (oracle@10={dev_select_metrics['oracle@10']:.4f})")
    logger.info(f"Total GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_lists.yaml")
    parser.add_argument("--model", type=str, default="jina-reranker-v3")
    parser.add_argument("--n_trials_screen", type=int, default=30)
    parser.add_argument("--n_trials_full", type=int, default=10)
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        rerankers = yaml_config.get("rerankers_to_tune", yaml_config.get("rerankers", []))
        reranker_cfg = None
        for r in rerankers:
            if r.get("name") == args.model:
                reranker_cfg = r
                break

        if reranker_cfg:
            config = RerankerHPOConfig(
                model_name=reranker_cfg.get("name", args.model),
                model_id=reranker_cfg.get("model_id", "jinaai/jina-reranker-v3"),
                n_trials_screen=args.n_trials_screen,
                n_trials_full=args.n_trials_full,
            )
        else:
            config = RerankerHPOConfig(
                model_name=args.model,
                n_trials_screen=args.n_trials_screen,
                n_trials_full=args.n_trials_full,
            )
    else:
        config = RerankerHPOConfig(
            model_name=args.model,
            n_trials_screen=args.n_trials_screen,
            n_trials_full=args.n_trials_full,
        )

    result = run_reranker_hpo(config)
    print(f"\nBest Reranker: {result['model_name']} (oracle@10={result['dev_select_metrics']['oracle@10']:.4f})")


if __name__ == "__main__":
    main()
