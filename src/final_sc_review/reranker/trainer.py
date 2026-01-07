"""Hybrid reranker training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from final_sc_review.reranker.losses import hybrid_loss
from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    w_list: float
    w_pair: float
    w_point: float
    temperature: float
    max_pairs_per_group: Optional[int]
    seed: int
    early_stopping_patience: int


def _collate_batch(batch: Sequence[Dict], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """Collate grouped examples into a flattened batch with group sizes."""
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


class HybridRerankerTrainer:
    """Minimal trainer for hybrid loss fine-tuning."""

    def __init__(self, config: TrainConfig):
        set_seed(config.seed)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=1,
            trust_remote_code=True,
        )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: _collate_batch(b, self.tokenizer, self.config.max_length),
        )

    def train(self, train_dataset, val_dataset) -> None:
        cfg = self.config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        best_val = float("inf")
        patience = 0
        rng = torch.Generator(device=self.device)
        rng.manual_seed(cfg.seed)

        for epoch in range(cfg.num_epochs):
            self.model.train()
            train_loader = self._make_loader(train_dataset, shuffle=True)
            running = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                group_sizes = batch["group_sizes"].tolist()

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss = hybrid_loss(
                    logits=logits,
                    labels=labels,
                    group_sizes=group_sizes,
                    w_list=cfg.w_list,
                    w_pair=cfg.w_pair,
                    w_point=cfg.w_point,
                    temperature=cfg.temperature,
                    max_pairs_per_group=cfg.max_pairs_per_group,
                    rng=rng,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                running += loss.item()
            avg_train = running / max(len(train_loader), 1)

            val_loss = self.evaluate(val_dataset)
            logger.info("Epoch %d | train=%.4f | val=%.4f", epoch + 1, avg_train, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                self.save(cfg.output_dir)
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

    def evaluate(self, dataset) -> float:
        self.model.eval()
        loader = self._make_loader(dataset, shuffle=False)
        total = 0.0
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                group_sizes = batch["group_sizes"].tolist()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss = hybrid_loss(
                    logits=logits,
                    labels=labels,
                    group_sizes=group_sizes,
                    w_list=self.config.w_list,
                    w_pair=self.config.w_pair,
                    w_point=self.config.w_point,
                    temperature=self.config.temperature,
                    max_pairs_per_group=self.config.max_pairs_per_group,
                )
                total += loss.item()
        return total / max(len(loader), 1)

    def save(self, output_dir: str) -> None:
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
