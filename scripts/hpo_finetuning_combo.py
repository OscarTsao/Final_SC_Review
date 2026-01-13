#!/usr/bin/env python3
"""HPO Fine-tuning for Retriever+Reranker combinations.

This script performs hyperparameter optimization for reranker training
using candidates from a specified retriever. It optimizes:
- Loss function types (pointwise, pairwise, listwise)
- Loss weights
- LoRA parameters
- Learning rate, epochs, batch size

Usage:
    python scripts/hpo_finetuning_combo.py \
        --retriever nv-embed-v2 \
        --reranker jina-reranker-v3 \
        --n_trials 50

For NV-Embed-v2 retriever, run in the nv-embed-v2 conda env:
    mamba run -n nv-embed-v2 python scripts/hpo_finetuning_combo.py \
        --retriever nv-embed-v2 --reranker jina-reranker-v3 --stage cache

Then run training in main env:
    python scripts/hpo_finetuning_combo.py \
        --retriever nv-embed-v2 --reranker jina-reranker-v3 --stage train
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.seed import set_seed

logger = get_logger(__name__)

# Lazy imports for training dependencies
PEFT_AVAILABLE = False


def _import_training_deps():
    """Import training dependencies (torch, optuna, peft)."""
    global torch, DataLoader, optuna, PEFT_AVAILABLE
    global LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    import optuna as _optuna

    torch = _torch
    DataLoader = _DataLoader
    optuna = _optuna

    # Try to import PEFT
    try:
        from peft import LoraConfig as _LoraConfig
        from peft import get_peft_model as _get_peft_model
        from peft import TaskType as _TaskType
        from peft import prepare_model_for_kbit_training as _prepare_model_for_kbit_training
        LoraConfig = _LoraConfig
        get_peft_model = _get_peft_model
        TaskType = _TaskType
        prepare_model_for_kbit_training = _prepare_model_for_kbit_training
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False
        logger.warning("PEFT not available, LoRA training disabled")


def _import_loss_deps():
    """Import loss function dependencies."""
    from final_sc_review.reranker.losses import LossConfig, compute_loss, LossType
    return LossConfig, compute_loss, LossType


@dataclass
class TrainingConfig:
    """Training configuration for reranker fine-tuning."""
    # Model
    reranker_model_id: str
    max_length: int = 512

    # Training
    batch_size: int = 2
    num_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4

    # Loss function selection
    pointwise_type: str = "bce"  # bce, focal
    pairwise_type: str = "ranknet"  # ranknet, margin_ranking, pairwise_softplus
    listwise_type: str = "listnet"  # listnet, listmle, plistmle, lambda, approx_ndcg

    # Loss weights
    w_list: float = 1.0
    w_pair: float = 0.5
    w_point: float = 0.1

    # Loss hyperparameters
    temperature: float = 1.0
    sigma: float = 1.0
    margin: float = 1.0
    max_pairs_per_group: int = 50

    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Memory
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True

    # Seed
    seed: int = 42


def load_sentences(corpus_path: Path) -> List[Dict]:
    """Load sentence corpus."""
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            sent = json.loads(line)
            sentences.append(sent)
    return sentences


def build_retriever_cache(
    retriever_name: str,
    sentences: List[Dict],
    groundtruth,
    criteria,
    split_post_ids: List[str],
    output_dir: Path,
    top_k: int = 100,
) -> Path:
    """Build retrieval cache for a retriever.

    Returns path to the cache file.
    """
    from final_sc_review.data.schemas import Sentence

    cache_path = output_dir / f"{retriever_name}_retrieval_cache.json"

    if cache_path.exists():
        logger.info(f"Cache exists: {cache_path}")
        return cache_path

    logger.info(f"Building retrieval cache for {retriever_name}...")

    # Build sentence objects
    sentence_objs = [
        Sentence(
            post_id=s["post_id"],
            sid=s["sid"],
            sent_uid=s["sent_uid"],
            text=s["text"],
        )
        for s in sentences
    ]

    # Get retriever
    from final_sc_review.retriever.zoo import RetrieverZoo

    zoo = RetrieverZoo(
        sentences=sentence_objs,
        cache_dir=PROJECT_ROOT / "data" / "cache",
    )
    retriever = zoo.get_retriever(retriever_name)
    retriever.encode_corpus()

    # Build post_to_indices
    post_to_indices = defaultdict(list)
    for idx, sent in enumerate(sentences):
        post_to_indices[sent["post_id"]].append(idx)

    # Build groundtruth map
    gt_map = defaultdict(set)
    for row in groundtruth:
        if row.groundtruth == 1:
            gt_map[(row.post_id, row.criterion_id)].add(row.sent_uid)

    # Build criteria map
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Build queries
    queries = []
    allowed_posts = set(split_post_ids)
    for (post_id, cid), gold_uids in gt_map.items():
        if post_id not in allowed_posts:
            continue
        if cid not in criteria_map:
            continue
        queries.append({
            "post_id": post_id,
            "criterion_id": cid,
            "criterion_text": criteria_map[cid],
            "gold_uids": list(gold_uids),
        })

    # Also add no_evidence queries (queries with no positives)
    for post_id in allowed_posts:
        if post_id not in post_to_indices:
            continue
        for cid, ctext in criteria_map.items():
            key = (post_id, cid)
            if key not in gt_map:
                # This is a no_evidence query
                queries.append({
                    "post_id": post_id,
                    "criterion_id": cid,
                    "criterion_text": ctext,
                    "gold_uids": [],  # No positives
                    "is_no_evidence": True,
                })

    logger.info(f"Total queries: {len(queries)}")

    # Run retrieval
    cache_data = []
    for q in tqdm(queries, desc="Retrieving"):
        results = retriever.retrieve_within_post(
            query=q["criterion_text"],
            post_id=q["post_id"],
            top_k=top_k,
        )

        candidates = []
        for r in results:
            candidates.append({
                "sent_uid": r.sent_uid,
                "text": r.text,
                "score": r.score,
            })

        cache_data.append({
            "post_id": q["post_id"],
            "criterion_id": q["criterion_id"],
            "criterion_text": q["criterion_text"],
            "gold_uids": q["gold_uids"],
            "is_no_evidence": q.get("is_no_evidence", False),
            "candidates": candidates,
        })

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)

    logger.info(f"Saved retrieval cache to {cache_path}")
    return cache_path


def load_retrieval_cache(cache_path: Path) -> List[Dict]:
    """Load retrieval cache."""
    with open(cache_path) as f:
        return json.load(f)


def build_training_examples(
    cache_data: List[Dict],
    max_candidates: int = 32,
    include_no_evidence: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """Build training examples from retrieval cache.

    Each example has:
    - query: criterion text
    - sentences: list of candidate texts
    - labels: list of 0/1 labels
    - sent_uids: list of sent_uids
    """
    import random
    rng = random.Random(seed)

    examples = []
    for item in cache_data:
        gold_uids = set(item["gold_uids"])
        candidates = item["candidates"]

        # Skip no_evidence if not included
        if not gold_uids and not include_no_evidence:
            continue

        # Limit candidates
        if len(candidates) > max_candidates:
            # Prioritize positives, then sample negatives
            pos_candidates = [c for c in candidates if c["sent_uid"] in gold_uids]
            neg_candidates = [c for c in candidates if c["sent_uid"] not in gold_uids]

            remaining = max_candidates - len(pos_candidates)
            if remaining > 0:
                sampled_negs = rng.sample(neg_candidates, min(remaining, len(neg_candidates)))
                candidates = pos_candidates + sampled_negs
            else:
                candidates = pos_candidates[:max_candidates]

            rng.shuffle(candidates)

        sentences = [c["text"] for c in candidates]
        sent_uids = [c["sent_uid"] for c in candidates]
        labels = [1.0 if uid in gold_uids else 0.0 for uid in sent_uids]

        examples.append({
            "query": item["criterion_text"],
            "sentences": sentences,
            "labels": labels,
            "sent_uids": sent_uids,
            "post_id": item["post_id"],
            "criterion_id": item["criterion_id"],
            "is_no_evidence": item.get("is_no_evidence", False),
        })

    return examples


def collate_batch(
    batch: List[Dict],
    tokenizer,
    max_length: int,
) -> Dict[str, "torch.Tensor"]:
    """Collate batch for training."""
    group_sizes = []
    pairs = []
    labels = []

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


def compute_ranking_metrics(
    retrieved_uids: List[str],
    gold_uids: set,
    k: int = 10,
) -> Dict[str, float]:
    """Compute ranking metrics at K."""
    retrieved_k = set(retrieved_uids[:k])
    recall = len(retrieved_k & gold_uids) / len(gold_uids) if gold_uids else 0.0

    mrr = 0.0
    for i, uid in enumerate(retrieved_uids[:k]):
        if uid in gold_uids:
            mrr = 1.0 / (i + 1)
            break

    relevances = [1 if uid in gold_uids else 0 for uid in retrieved_uids[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    precisions = []
    hits = 0
    for i, uid in enumerate(retrieved_uids[:k]):
        if uid in gold_uids:
            hits += 1
            precisions.append(hits / (i + 1))
    map_k = np.mean(precisions) if precisions else 0.0

    return {
        "recall_at_10": recall,
        "mrr_at_10": mrr,
        "ndcg_at_10": ndcg,
        "map_at_10": map_k,
    }


def create_model_with_lora(
    model_id: str,
    config: TrainingConfig,
    device: str,
):
    """Create model with LoRA."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing (disabled for jina-reranker-v2 due to custom model)
    if config.use_gradient_checkpointing and "jina-reranker-v2" not in model_id.lower():
        try:
            model.gradient_checkpointing_enable()
        except (NotImplementedError, AttributeError):
            logger.warning("Gradient checkpointing not supported for this model, skipping")

    # Apply LoRA
    if config.use_lora and PEFT_AVAILABLE:
        # Detect model architecture and use appropriate target modules
        model_type = type(model).__name__.lower()
        if "xlmroberta" in model_type or "bge" in model_id.lower():
            if "jina-reranker-v2" in model_id.lower():
                # jina-reranker-v2 uses custom XLMRoBERTa with fused QKV that returns tuple
                # Only target MLP layers (fc1, fc2) and out_proj which are standard Linear
                target_modules = ["out_proj", "fc1", "fc2"]
            else:
                # Standard XLMRoBERTa (bge-reranker-v2-m3)
                target_modules = ["query", "key", "value"]
        else:
            # Qwen-based models (jina-reranker-v3)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )

        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    model.to(device)
    return model, tokenizer


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    config: TrainingConfig,
    device: str,
) -> float:
    """Train for one epoch."""
    LossConfig, compute_loss, _ = _import_loss_deps()

    model.train()
    total_loss = 0.0
    accum_steps = config.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)

    loss_config = LossConfig(
        pointwise_type=config.pointwise_type,
        pairwise_type=config.pairwise_type,
        listwise_type=config.listwise_type,
        w_list=config.w_list,
        w_pair=config.w_pair,
        w_point=config.w_point,
        temperature=config.temperature,
        sigma=config.sigma,
        margin=config.margin,
        max_pairs_per_group=config.max_pairs_per_group,
    )

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        group_sizes = batch["group_sizes"].tolist()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

        loss, _ = compute_loss(logits, labels, group_sizes, loss_config)
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps

    return total_loss / len(dataloader)


def compute_val_loss(
    model,
    dataloader,
    config: TrainingConfig,
    device: str,
) -> float:
    """Compute validation loss."""
    LossConfig, compute_loss, _ = _import_loss_deps()

    model.eval()
    total_loss = 0.0

    loss_config = LossConfig(
        pointwise_type=config.pointwise_type,
        pairwise_type=config.pairwise_type,
        listwise_type=config.listwise_type,
        w_list=config.w_list,
        w_pair=config.w_pair,
        w_point=config.w_point,
        temperature=config.temperature,
        sigma=config.sigma,
        margin=config.margin,
        max_pairs_per_group=config.max_pairs_per_group,
    )

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            group_sizes = batch["group_sizes"].tolist()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

            loss, _ = compute_loss(logits, labels, group_sizes, loss_config)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_val_ranking_metrics(
    model,
    tokenizer,
    examples: List[Dict],
    config: TrainingConfig,
    device: str,
) -> Dict[str, float]:
    """Compute ranking metrics on validation set."""
    model.eval()
    all_metrics = []

    with torch.inference_mode():
        for ex in tqdm(examples, desc="Computing metrics", leave=False):
            if not ex["labels"] or sum(ex["labels"]) == 0:
                continue  # Skip no-evidence examples

            query = ex["query"]
            sentences = ex["sentences"]
            sent_uids = ex["sent_uids"]
            gold_uids = set(uid for uid, lbl in zip(sent_uids, ex["labels"]) if lbl > 0)

            # Score all candidates
            pairs = [[query, sent] for sent in sentences]
            enc = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**enc).logits.squeeze(-1)
            scores = logits.float().cpu().numpy()

            # Rank by score
            ranked_indices = np.argsort(-scores)
            ranked_uids = [sent_uids[i] for i in ranked_indices]

            m = compute_ranking_metrics(ranked_uids, gold_uids, k=10)
            all_metrics.append(m)

    if not all_metrics:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr_at_10": 0.0, "map_at_10": 0.0}

    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def run_hpo_trial(
    trial: optuna.Trial,
    train_examples: List[Dict],
    val_examples: List[Dict],
    reranker_model_id: str,
    device: str,
    base_seed: int = 42,
) -> float:
    """Run a single HPO trial."""
    seed = base_seed + trial.number
    set_seed(seed)

    # Sample hyperparameters
    config = TrainingConfig(
        reranker_model_id=reranker_model_id,
        max_length=512,
        batch_size=trial.suggest_categorical("batch_size", [1, 2, 4]),
        num_epochs=trial.suggest_int("num_epochs", 1, 3),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        gradient_accumulation_steps=trial.suggest_categorical("grad_accum", [2, 4, 8]),

        # Loss function types
        pointwise_type=trial.suggest_categorical("pointwise_type", ["bce", "focal"]),
        pairwise_type=trial.suggest_categorical("pairwise_type", ["ranknet", "margin_ranking", "pairwise_softplus"]),
        listwise_type=trial.suggest_categorical("listwise_type", ["listnet", "listmle", "plistmle", "lambda", "approx_ndcg"]),

        # Loss weights
        w_list=trial.suggest_float("w_list", 0.0, 2.0),
        w_pair=trial.suggest_float("w_pair", 0.0, 2.0),
        w_point=trial.suggest_float("w_point", 0.0, 1.0),

        # Loss hyperparameters
        temperature=trial.suggest_float("temperature", 0.5, 2.0),
        sigma=trial.suggest_float("sigma", 0.5, 2.0),
        margin=trial.suggest_float("margin", 0.5, 2.0),
        max_pairs_per_group=trial.suggest_categorical("max_pairs", [32, 50, 100]),

        # LoRA config
        use_lora=True,
        lora_r=trial.suggest_categorical("lora_r", [8, 16, 32]),
        lora_alpha=trial.suggest_categorical("lora_alpha", [16, 32, 64]),
        lora_dropout=trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1]),

        seed=seed,
    )

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Create model
    model, tokenizer = create_model_with_lora(
        model_id=reranker_model_id,
        config=config,
        device=device,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer, config.max_length),
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer, config.max_length),
    )

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Training loop
    best_ndcg = 0.0
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        val_loss = compute_val_loss(model, val_loader, config, device)

        # Compute ranking metrics on validation
        val_metrics = compute_val_ranking_metrics(model, tokenizer, val_examples, config, device)
        ndcg = val_metrics["ndcg_at_10"]
        best_ndcg = max(best_ndcg, ndcg)

        logger.info(f"Trial {trial.number} Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, nDCG@10={ndcg:.4f}")

        trial.report(ndcg, step=epoch)
        if trial.should_prune():
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return best_ndcg


def get_reranker_model_id(reranker_name: str) -> str:
    """Get model ID for a reranker name."""
    model_ids = {
        "jina-reranker-v3": "jinaai/jina-reranker-v3",
        "jina-reranker-v2": "jinaai/jina-reranker-v2-base-multilingual",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    }
    if reranker_name not in model_ids:
        raise ValueError(f"Unknown reranker: {reranker_name}")
    return model_ids[reranker_name]


def main():
    parser = argparse.ArgumentParser(description="HPO Fine-tuning for Retriever+Reranker")
    parser.add_argument("--retriever", type=str, required=True,
                        help="Retriever name (e.g., nv-embed-v2, qwen3-embed-4b)")
    parser.add_argument("--reranker", type=str, required=True,
                        help="Reranker name (e.g., jina-reranker-v3, bge-reranker-v2-m3)")
    parser.add_argument("--stage", type=str, default="all", choices=["cache", "train", "all"],
                        help="Stage to run: cache (build retrieval cache), train (run HPO), all (both)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/hpo_finetuning",
                        help="Output directory")
    parser.add_argument("--top_k_retriever", type=int, default=100,
                        help="Top-k candidates from retriever")
    parser.add_argument("--max_candidates", type=int, default=32,
                        help="Max candidates per training example")
    parser.add_argument("--include_no_evidence", action="store_true",
                        help="Include no-evidence examples in training")
    args = parser.parse_args()

    set_seed(args.seed)

    # Paths
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / args.output_dir / f"{args.retriever}_{args.reranker}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    sentences = load_sentences(data_dir / "groundtruth" / "sentence_corpus.jsonl")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")

    # Split post_ids
    all_post_ids = sorted(set(s["post_id"] for s in sentences))
    splits = split_post_ids(all_post_ids, seed=args.seed, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    logger.info(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Stage 1: Build retrieval cache
    if args.stage in ["cache", "all"]:
        logger.info(f"Building retrieval cache for {args.retriever}...")

        # Build cache for train and val splits
        for split_name in ["train", "val"]:
            cache_path = output_dir / f"{args.retriever}_{split_name}_cache.json"
            if not cache_path.exists():
                build_retriever_cache(
                    retriever_name=args.retriever,
                    sentences=sentences,
                    groundtruth=groundtruth,
                    criteria=criteria,
                    split_post_ids=splits[split_name],
                    output_dir=output_dir,
                    top_k=args.top_k_retriever,
                )
                # Rename to include split name
                temp_cache = output_dir / f"{args.retriever}_retrieval_cache.json"
                if temp_cache.exists():
                    temp_cache.rename(cache_path)
            else:
                logger.info(f"Cache exists: {cache_path}")

        if args.stage == "cache":
            logger.info("Cache stage complete. Run with --stage train to continue.")
            return

    # Stage 2: Run HPO training
    if args.stage in ["train", "all"]:
        # Import training dependencies
        _import_training_deps()

        logger.info(f"Running HPO fine-tuning for {args.retriever} + {args.reranker}...")

        # Load caches
        train_cache_path = output_dir / f"{args.retriever}_train_cache.json"
        val_cache_path = output_dir / f"{args.retriever}_val_cache.json"

        if not train_cache_path.exists() or not val_cache_path.exists():
            logger.error("Retrieval caches not found. Run with --stage cache first.")
            return

        train_cache = load_retrieval_cache(train_cache_path)
        val_cache = load_retrieval_cache(val_cache_path)

        # Build training examples
        train_examples = build_training_examples(
            train_cache,
            max_candidates=args.max_candidates,
            include_no_evidence=args.include_no_evidence,
            seed=args.seed,
        )
        val_examples = build_training_examples(
            val_cache,
            max_candidates=args.max_candidates,
            include_no_evidence=False,  # Don't include no-evidence in validation
            seed=args.seed,
        )

        logger.info(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")

        # Get reranker model ID
        reranker_model_id = get_reranker_model_id(args.reranker)

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create Optuna study
        study_name = f"hpo_finetune_{args.retriever}_{args.reranker}"
        storage = f"sqlite:///{output_dir}/optuna.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        # Run optimization
        def objective(trial):
            return run_hpo_trial(
                trial=trial,
                train_examples=train_examples,
                val_examples=val_examples,
                reranker_model_id=reranker_model_id,
                device=device,
                base_seed=args.seed,
            )

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        # Report results
        logger.info("=" * 80)
        logger.info(f"BEST TRIAL: {study.best_trial.number}")
        logger.info(f"Best nDCG@10: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        logger.info("=" * 80)

        # Save results
        results = {
            "retriever": args.retriever,
            "reranker": args.reranker,
            "best_ndcg_at_10": study.best_value,
            "best_params": study.best_params,
            "n_trials": args.n_trials,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "best_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save to CSV for comparison
        results_csv = PROJECT_ROOT / args.output_dir / "hpo_finetuning_results.csv"
        row = {
            "retriever": args.retriever,
            "reranker": args.reranker,
            "best_ndcg_at_10": study.best_value,
            "n_trials": args.n_trials,
            "timestamp": datetime.now().isoformat(),
            **{f"best_{k}": v for k, v in study.best_params.items()},
        }

        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(results_csv, index=False)
        logger.info(f"Results saved to {results_csv}")


if __name__ == "__main__":
    main()
