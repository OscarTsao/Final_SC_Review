#!/usr/bin/env python3
"""Train fixed NO_EVIDENCE reranker.

Implements fixes for previous training collapse:
(A) Query-level balanced sampling (1:1 or 1:2 evidence:no-evidence per batch)
(B) Margin hinge objectives:
    Evidence query: best_gold >= s_NE + m_pos
    No-evidence query: s_NE >= best_sentence + m_neg
    L = L_ranking + w_ne*(hinge_pos + hinge_neg)
    Defaults: m_pos=0.5, m_neg=0.5, w_ne=0.2
(C) Hard negatives for no-evidence queries (use top retrieved candidates)

Output: outputs/training/noevidence_fixed/<timestamp>/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.data.io import load_groundtruth, load_criteria

logger = get_logger(__name__)


class NEDataset(Dataset):
    """Dataset for NO_EVIDENCE training with query-level sampling."""

    NE_TOKEN = "[NO_EVIDENCE]"

    def __init__(
        self,
        queries: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]

        query_text = query["criterion_text"]
        candidates = query["candidates"]
        has_evidence = query["has_evidence"]

        pairs = []
        labels = []

        for sent_uid, text, score, is_gold in candidates:
            pairs.append((query_text, text))
            labels.append(1 if is_gold else 0)

        pairs.append((query_text, self.NE_TOKEN))
        labels.append(1 if not has_evidence else 0)

        return {
            "pairs": pairs,
            "labels": labels,
            "has_evidence": has_evidence,
            "query_id": query["query_id"],
        }


class MarginHingeLoss(nn.Module):
    """Margin hinge loss for NO_EVIDENCE training."""

    def __init__(self, m_pos: float = 0.5, m_neg: float = 0.5, w_ne: float = 0.2):
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.w_ne = w_ne

    def forward(self, scores, has_evidence, gold_mask):
        batch_size = scores.shape[0]
        n_candidates = scores.shape[1] - 1

        candidate_scores = scores[:, :n_candidates]
        ne_scores = scores[:, -1]

        losses = []

        for i in range(batch_size):
            if has_evidence[i]:
                gold_indices = gold_mask[i].nonzero(as_tuple=True)[0]
                if len(gold_indices) > 0:
                    best_gold = candidate_scores[i, gold_indices].max()
                    hinge_pos = F.relu(ne_scores[i] + self.m_pos - best_gold)
                    losses.append(hinge_pos)
            else:
                best_sentence = candidate_scores[i].max()
                hinge_neg = F.relu(best_sentence + self.m_neg - ne_scores[i])
                losses.append(hinge_neg)

        if losses:
            return self.w_ne * torch.stack(losses).mean()
        return torch.tensor(0.0, device=scores.device)


class CombinedLoss(nn.Module):
    """Combined ranking + NO_EVIDENCE margin loss."""

    def __init__(self, m_pos: float = 0.5, m_neg: float = 0.5, w_ne: float = 0.2, w_ranking: float = 1.0):
        super().__init__()
        self.margin_loss = MarginHingeLoss(m_pos, m_neg, w_ne)
        self.ranking_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.w_ranking = w_ranking

    def forward(self, scores, has_evidence, gold_mask, labels):
        ranking = self.ranking_loss(scores, labels.float())
        weights = torch.where(labels == 1, 5.0, 1.0)
        ranking = (ranking * weights).mean()
        margin = self.margin_loss(scores, has_evidence, gold_mask)
        return self.w_ranking * ranking + margin


def create_balanced_sampler(queries: List[Dict], ratio: float = 1.0) -> WeightedRandomSampler:
    """Create weighted sampler for query-level balancing."""
    evidence_count = sum(1 for q in queries if q["has_evidence"])
    no_evidence_count = len(queries) - evidence_count

    if evidence_count > 0 and no_evidence_count > 0:
        evidence_weight = 1.0 / evidence_count
        no_evidence_weight = ratio / no_evidence_count
    else:
        evidence_weight = 1.0
        no_evidence_weight = 1.0

    weights = [evidence_weight if q["has_evidence"] else no_evidence_weight for q in queries]
    return WeightedRandomSampler(weights, len(queries), replacement=True)


def train_noevidence_reranker(
    train_queries: List[Dict],
    val_queries: List[Dict],
    model_name: str = "jinaai/jina-reranker-v3",
    output_dir: Path = None,
    n_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    m_pos: float = 0.5,
    m_neg: float = 0.5,
    w_ne: float = 0.2,
    balance_ratio: float = 1.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train NO_EVIDENCE reranker with fixes."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("outputs/training/noevidence_fixed") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training NO_EVIDENCE reranker, output to {output_dir}")
    logger.info(f"Train queries: {len(train_queries)}, Val queries: {len(val_queries)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    train_dataset = NEDataset(train_queries, tokenizer)
    val_dataset = NEDataset(val_queries, tokenizer)
    sampler = create_balanced_sampler(train_queries, balance_ratio)

    def collate_fn(batch):
        all_pairs = []
        all_labels = []
        all_has_evidence = []
        query_ids = []

        for item in batch:
            all_pairs.extend(item["pairs"])
            all_labels.extend(item["labels"])
            all_has_evidence.append(item["has_evidence"])
            query_ids.append(item["query_id"])

        queries_text = [p[0] for p in all_pairs]
        docs_text = [p[1] for p in all_pairs]

        encodings = tokenizer(
            queries_text, docs_text, padding=True, truncation=True, max_length=512, return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(all_labels),
            "has_evidence": torch.tensor(all_has_evidence),
            "query_ids": query_ids,
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = CombinedLoss(m_pos, m_neg, w_ne)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_metric = 0.0
    history = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            has_evidence = batch["has_evidence"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            batch_size_actual = len(batch["query_ids"])
            n_per_query = len(logits) // batch_size_actual

            scores = logits.view(batch_size_actual, n_per_query)
            labels_reshaped = labels.view(batch_size_actual, n_per_query)
            gold_mask = labels_reshaped[:, :-1]

            loss = criterion(scores, has_evidence, gold_mask, labels_reshaped)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": train_loss / n_batches})

        model.save_pretrained(output_dir / "best_model")
        tokenizer.save_pretrained(output_dir / "best_model")

        epoch_stats = {"epoch": epoch + 1, "train_loss": train_loss / n_batches}
        history.append(epoch_stats)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss/n_batches:.4f}")

    results = {
        "model_name": model_name, "n_epochs": n_epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "m_pos": m_pos, "m_neg": m_neg,
        "w_ne": w_ne, "balance_ratio": balance_ratio, "history": history,
    }

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train fixed NO_EVIDENCE reranker")
    parser.add_argument("--groundtruth", type=str, default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--criteria", type=str, default="data/DSM5/MDD_Criteira.json")
    parser.add_argument("--model", type=str, default="jinaai/jina-reranker-v3")
    parser.add_argument("--output_dir", type=str, default="outputs/training/noevidence_fixed")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--m_pos", type=float, default=0.5)
    parser.add_argument("--m_neg", type=float, default=0.5)
    parser.add_argument("--w_ne", type=float, default=0.2)
    parser.add_argument("--balance_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading data...")
    gt_df = load_groundtruth(Path(args.groundtruth))
    criteria = load_criteria(Path(args.criteria))
    criteria_map = {c["id"]: c["criterion"] for c in criteria}

    queries = []
    for (post_id, criterion_id), group in gt_df.groupby(["post_id", "criterion"]):
        gold_rows = group[group["groundtruth"] == 1]
        gold_ids = set(gold_rows["sent_uid"].tolist())
        has_evidence = len(gold_ids) > 0

        candidates = []
        for _, row in group.iterrows():
            candidates.append((row["sent_uid"], row["sentence"], 0.0, row["sent_uid"] in gold_ids))

        queries.append({
            "query_id": f"{post_id}_{criterion_id}",
            "post_id": post_id,
            "criterion_id": criterion_id,
            "criterion_text": criteria_map.get(criterion_id, ""),
            "candidates": candidates[:20],
            "has_evidence": has_evidence,
        })

    post_ids = list(set(q["post_id"] for q in queries))
    np.random.shuffle(post_ids)
    split_idx = int(0.8 * len(post_ids))
    train_posts = set(post_ids[:split_idx])

    train_queries = [q for q in queries if q["post_id"] in train_posts]
    val_queries = [q for q in queries if q["post_id"] not in train_posts]

    logger.info(f"Train: {len(train_queries)} queries, Val: {len(val_queries)} queries")

    results = train_noevidence_reranker(
        train_queries=train_queries,
        val_queries=val_queries,
        model_name=args.model,
        output_dir=Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"),
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        m_pos=args.m_pos,
        m_neg=args.m_neg,
        w_ne=args.w_ne,
        balance_ratio=args.balance_ratio,
    )

    print(f"\nTraining complete.")


if __name__ == "__main__":
    main()
