"""Training-stage HPO objective (optional)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import optuna
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import GroupedRerankerDataset, build_grouped_examples
from final_sc_review.reranker.losses import hybrid_loss
from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    model_name: str
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    w_list: float
    w_pair: float
    w_point: float
    temperature: float
    max_pairs_per_group: int | None
    seed: int


def run_training_hpo(trial: optuna.Trial, cfg: Dict) -> float:
    """Run one training trial and return dev metric (negative loss)."""
    seed = cfg["split"]["seed"] + trial.number
    set_seed(seed)

    train_cfg = _sample_training_params(trial, cfg)
    train_cfg.seed = seed

    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["criteria"]))

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    train_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["train"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=seed,
    )
    val_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["val"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=seed,
    )

    train_dataset = GroupedRerankerDataset(train_examples)
    val_dataset = GroupedRerankerDataset(val_examples)

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.model_name,
        num_labels=1,
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_batch(b, tokenizer, train_cfg.max_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_batch(b, tokenizer, train_cfg.max_length),
    )

    best_val = float("inf")
    for epoch in range(train_cfg.num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _compute_loss(batch, model, train_cfg, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = _evaluate(val_loader, model, train_cfg, device)
        best_val = min(best_val, val_loss)
        trial.report(-val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return -best_val


def _compute_loss(batch: Dict, model, cfg: TrainingConfig, device: str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    group_sizes = batch["group_sizes"].tolist()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
    return hybrid_loss(
        logits=logits,
        labels=labels,
        group_sizes=group_sizes,
        w_list=cfg.w_list,
        w_pair=cfg.w_pair,
        w_point=cfg.w_point,
        temperature=cfg.temperature,
        max_pairs_per_group=cfg.max_pairs_per_group,
    )


def _evaluate(loader: DataLoader, model, cfg: TrainingConfig, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for batch in loader:
            loss = _compute_loss(batch, model, cfg, device)
            total += loss.item()
    return total / max(len(loader), 1)


def _collate_batch(batch: List[Dict], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    group_sizes: List[int] = []
    pairs: List[List[str]] = []
    labels: List[float] = []
    for ex in batch:
        query = ex["query"]
        sentences = ex["sentences"]
        ex_labels = ex["labels"]
        group_sizes.append(len(sentences))
        for sent, label in zip(sentences, ex_labels):
            pairs.append([query, sent])
            labels.append(float(label))
    enc = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float),
        "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
    }


def _sample_training_params(trial: optuna.Trial, cfg: Dict) -> TrainingConfig:
    space = cfg["search_space"]
    return TrainingConfig(
        model_name=cfg["models"]["jina_v3"],
        max_length=trial.suggest_categorical("max_length", space["max_length"]),
        batch_size=trial.suggest_categorical("batch_size", space["batch_size"]),
        num_epochs=trial.suggest_int("num_epochs", space["num_epochs"][0], space["num_epochs"][1]),
        learning_rate=trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True),
        weight_decay=trial.suggest_float("weight_decay", space["weight_decay"][0], space["weight_decay"][1], log=True),
        w_list=trial.suggest_float("w_list", 0.0, 1.0),
        w_pair=trial.suggest_float("w_pair", 0.0, 1.0),
        w_point=trial.suggest_float("w_point", 0.0, 1.0),
        temperature=trial.suggest_float("temperature", 0.5, 2.0),
        max_pairs_per_group=space.get("max_pairs_per_group"),
        seed=cfg["split"]["seed"],
    )
