#!/usr/bin/env python3
"""Generate training data with hard negatives for reranker training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.training.hard_negative_miner import (
    HardNegativeConfig,
    HardNegativeMiner,
    InBatchNegativeMiner,
    create_training_examples,
)
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_data.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    # Load data
    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    criteria = load_criteria(Path(cfg["paths"]["criteria"]))
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Split data
    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    train_posts = set(splits["train"])

    # Group groundtruth by (post_id, criterion_id)
    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in train_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    # Prepare batch for mining - INCLUDE EMPTY GROUPS for no-evidence learning
    batch = []
    empty_batch = []  # Separate list for empty (no-evidence) groups
    for (post_id, criterion_id), rows in sorted(grouped.items()):
        query = criteria_map.get(criterion_id)
        if query is None:
            continue
        gold_ids = [r.sent_uid for r in rows if r.groundtruth == 1]
        group_data = {
            "query": query,
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gold_ids": gold_ids,
            "has_evidence": len(gold_ids) > 0,  # Track has_evidence flag
        }
        if gold_ids:
            batch.append(group_data)
        else:
            empty_batch.append(group_data)

    logger.info("Prepared %d queries with positives for training", len(batch))
    logger.info("Prepared %d empty queries (no evidence) for training", len(empty_batch))

    # Include empty groups in training for no-evidence detection
    include_empty = cfg.get("include_empty_groups", True)
    if include_empty:
        batch.extend(empty_batch)
        logger.info("Total training queries (with empty): %d", len(batch))

    # Initialize retriever for hard negative mining
    hard_neg_cfg = cfg.get("hard_negative", {})
    if hard_neg_cfg.get("enabled", False):
        cache_dir = Path(cfg["paths"]["cache_dir"])
        retriever = BgeM3Retriever(
            sentences=sentences,
            cache_dir=cache_dir,
            model_name=cfg["models"]["bge_m3"],
            device=cfg.get("device"),
            query_max_length=cfg["models"].get("bge_query_max_length", 128),
            passage_max_length=cfg["models"].get("bge_passage_max_length", 256),
            use_fp16=cfg["models"].get("bge_use_fp16", True),
            batch_size=cfg["models"].get("bge_batch_size", 64),
            rebuild_cache=False,
        )

        miner_config = HardNegativeConfig(
            enabled=True,
            method=hard_neg_cfg.get("method", "top_k_hard"),
            k_neg=hard_neg_cfg.get("k_neg", 3),
            rank_low=hard_neg_cfg.get("rank_low", 5),
            rank_high=hard_neg_cfg.get("rank_high", 50),
            retriever_top_k=hard_neg_cfg.get("retriever_top_k", 100),
        )

        miner = HardNegativeMiner(retriever, miner_config)
        batch_with_negatives = miner.mine_batch_negatives(batch)
        logger.info("Mined hard negatives using method: %s", miner_config.method)
    else:
        batch_with_negatives = batch
        logger.info("Hard negative mining disabled, using random negatives")

    # Optionally add in-batch negatives
    include_in_batch = hard_neg_cfg.get("include_in_batch", False)
    if include_in_batch:
        in_batch_miner = InBatchNegativeMiner(
            sentences, k_neg=hard_neg_cfg.get("in_batch_k_neg", 2)
        )
        batch_with_negatives = in_batch_miner.mine_in_batch(
            batch_with_negatives, seed=cfg["split"]["seed"]
        )
        logger.info("Added in-batch negatives")

    # Create training examples
    examples = create_training_examples(
        batch_with_negatives, sentences, include_in_batch=include_in_batch
    )
    logger.info("Created %d training examples", len(examples))

    # Save output
    output_path = Path(args.output) if args.output else Path(cfg["paths"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Saved training data to %s", output_path)

    # Save statistics
    num_with_evidence = sum(1 for b in batch if b.get("has_evidence", True))
    num_empty = sum(1 for b in batch if not b.get("has_evidence", True))
    stats = {
        "num_queries": len(batch),
        "num_queries_with_evidence": num_with_evidence,
        "num_queries_empty": num_empty,
        "empty_included": include_empty,
        "num_examples": len(examples),
        "hard_negative_enabled": hard_neg_cfg.get("enabled", False),
        "hard_negative_method": hard_neg_cfg.get("method", "none"),
        "k_neg": hard_neg_cfg.get("k_neg", 0),
        "include_in_batch": include_in_batch,
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved statistics to %s", stats_path)


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
