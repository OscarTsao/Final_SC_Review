#!/usr/bin/env python3
"""Train/finetune retriever with contrastive learning.

Uses sentence-transformers with in-batch negatives and hard negative mining.
GPU time tracked for paper experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml

# Add scripts directory to path for GPU tracker import
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gpu_time_tracker import GPUTimeTracker

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrieverTrainConfig:
    """Configuration for retriever finetuning."""
    model_name: str = "BAAI/bge-m3"
    output_dir: str = "outputs/retriever_finetuned"
    max_length: int = 256
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    temperature: float = 0.05
    hard_neg_count: int = 3
    seed: int = 42
    use_fp16: bool = True
    gradient_checkpointing: bool = True


def build_training_pairs(
    groundtruth_rows,
    criteria,
    sentences,
    split_post_ids: Set[str],
    hard_neg_count: int = 3,
) -> List[Dict]:
    """Build training pairs with hard negatives.

    Returns:
        List of dicts with:
        - query: criterion text
        - positive: gold evidence sentence
        - negatives: hard negative sentences from same post
    """
    # Index sentences by post
    post_to_sentences = defaultdict(list)
    sent_uid_to_text = {}
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)
        sent_uid_to_text[sent.sent_uid] = sent.text

    # Index criteria
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Group groundtruth by (post_id, criterion_id)
    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": set()})
    for row in groundtruth_rows:
        if row.post_id not in split_post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].add(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    # Build training pairs
    training_pairs = []
    for (post_id, criterion_id), data in groups.items():
        gold_uids = data["gold_uids"]
        if not gold_uids:
            continue  # Skip groups with no positives

        query = criterion_text.get(criterion_id)
        if not query:
            continue

        # Get all sentences in post as potential negatives
        post_sents = post_to_sentences.get(post_id, [])
        negative_uids = [s.sent_uid for s in post_sents if s.sent_uid not in gold_uids]

        # Create pairs for each positive
        for gold_uid in gold_uids:
            positive_text = sent_uid_to_text.get(gold_uid)
            if not positive_text:
                continue

            # Sample hard negatives
            neg_sample = negative_uids[:hard_neg_count] if len(negative_uids) <= hard_neg_count else \
                         list(np.random.choice(negative_uids, hard_neg_count, replace=False))
            neg_texts = [sent_uid_to_text.get(uid, "") for uid in neg_sample]
            neg_texts = [t for t in neg_texts if t]  # Filter empty

            training_pairs.append({
                "query": query,
                "positive": positive_text,
                "negatives": neg_texts,
                "post_id": post_id,
                "criterion_id": criterion_id,
            })

    return training_pairs


def train_retriever(
    config: RetrieverTrainConfig,
    train_pairs: List[Dict],
    val_pairs: List[Dict],
) -> str:
    """Train retriever using sentence-transformers.

    Returns:
        Path to saved model
    """
    try:
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.training_args import BatchSamplers
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "sentence-transformers>=3.0 and datasets required. "
            "Install with: pip install sentence-transformers datasets"
        )

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {config.model_name}")
    model = SentenceTransformer(config.model_name)

    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        try:
            model[0].auto_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    # Prepare dataset for MultipleNegativesRankingLoss
    # Format: (anchor, positive) pairs - in-batch negatives are used automatically
    train_data = {
        "anchor": [p["query"] for p in train_pairs],
        "positive": [p["positive"] for p in train_pairs],
    }
    train_dataset = Dataset.from_dict(train_data)

    val_data = {
        "anchor": [p["query"] for p in val_pairs],
        "positive": [p["positive"] for p in val_pairs],
    }
    val_dataset = Dataset.from_dict(val_data)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Loss function: MultipleNegativesRankingLoss uses in-batch negatives
    loss = losses.MultipleNegativesRankingLoss(model)

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.use_fp16 and torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        seed=config.seed,
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    model.save(str(final_path))
    logger.info(f"Model saved to: {final_path}")

    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="Train/finetune retriever")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3", help="Base model")
    parser.add_argument("--output_dir", type=str, default="outputs/retriever_finetuned")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--hard_neg_count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config from file if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        config = RetrieverTrainConfig(
            model_name=str(cfg.get("model_name", args.model_name)),
            output_dir=str(cfg.get("output_dir", args.output_dir)),
            num_epochs=int(cfg.get("num_epochs", args.num_epochs)),
            batch_size=int(cfg.get("batch_size", args.batch_size)),
            learning_rate=float(cfg.get("learning_rate", args.learning_rate)),
            hard_neg_count=int(cfg.get("hard_neg_count", args.hard_neg_count)),
            seed=int(cfg.get("seed", args.seed)),
            max_length=int(cfg.get("max_length", 256)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
            temperature=float(cfg.get("temperature", 0.05)),
            use_fp16=bool(cfg.get("use_fp16", True)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
    else:
        config = RetrieverTrainConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hard_neg_count=args.hard_neg_count,
            seed=args.seed,
        )

    # Paths
    data_dir = Path("data")
    groundtruth_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

    # Load data
    logger.info("Loading data...")
    groundtruth = load_groundtruth(groundtruth_path)
    criteria = load_criteria(criteria_path)
    sentences = load_sentence_corpus(corpus_path)

    # Split
    all_post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(all_post_ids, seed=config.seed)

    logger.info(f"Train posts: {len(splits['train'])}")
    logger.info(f"Val posts: {len(splits['val'])}")

    # Build training pairs
    logger.info("Building training pairs...")
    train_pairs = build_training_pairs(
        groundtruth, criteria, sentences,
        set(splits["train"]),
        hard_neg_count=config.hard_neg_count,
    )
    val_pairs = build_training_pairs(
        groundtruth, criteria, sentences,
        set(splits["val"]),
        hard_neg_count=config.hard_neg_count,
    )

    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")

    # Initialize GPU time tracking
    gpu_tracker = GPUTimeTracker(output_dir="outputs/system")
    session_id = gpu_tracker.start(
        phase="retriever_training",
        description=f"Finetuning {config.model_name}"
    )

    try:
        # Train
        model_path = train_retriever(config, train_pairs, val_pairs)
    finally:
        # Stop GPU tracking
        session = gpu_tracker.stop()
        logger.info(f"GPU training time: {session.duration_hours:.2f}h")
        logger.info(f"Total GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")

    # Save training metadata
    output_dir = Path(config.output_dir)
    metadata = {
        "model_name": config.model_name,
        "model_path": model_path,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "seed": config.seed,
        "timestamp": datetime.now().isoformat(),
        "gpu_tracking": {
            "session_id": session_id,
            "duration_hours": session.duration_hours,
            "avg_utilization": session.avg_utilization,
            "peak_memory_gb": session.peak_memory_gb,
            "total_gpu_hours": gpu_tracker.total_gpu_hours,
        },
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Model saved to: {model_path}")
    print(gpu_tracker.format_summary())


if __name__ == "__main__":
    main()
