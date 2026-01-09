#!/usr/bin/env python3
"""Train Jina-v3 reranker with hybrid loss (listwise + pairwise + pointwise)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add scripts directory to path for GPU tracker import
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gpu_time_tracker import GPUTimeTracker

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import GroupedRerankerDataset, build_grouped_examples
from final_sc_review.reranker.trainer import HybridRerankerTrainer, TrainConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reranker_hybrid.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    gt_path = Path(cfg["paths"]["groundtruth"])
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

    groundtruth = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)

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
        seed=cfg["split"]["seed"],
    )
    val_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["val"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=cfg["split"]["seed"],
    )

    train_dataset = GroupedRerankerDataset(train_examples)
    val_dataset = GroupedRerankerDataset(val_examples)

    train_cfg = TrainConfig(
        model_name=cfg["models"]["jina_v3"],
        output_dir=cfg["paths"]["output_dir"],
        max_length=cfg["models"]["max_length"],
        batch_size=cfg["train"]["batch_size"],
        num_epochs=cfg["train"]["num_epochs"],
        learning_rate=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        w_list=cfg["loss"]["w_list"],
        w_pair=cfg["loss"]["w_pair"],
        w_point=cfg["loss"]["w_point"],
        temperature=cfg["loss"]["temperature"],
        max_pairs_per_group=cfg["loss"].get("max_pairs_per_group"),
        seed=cfg["split"]["seed"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
    )

    trainer = HybridRerankerTrainer(train_cfg)

    # Initialize GPU time tracking
    gpu_tracker = GPUTimeTracker(output_dir="outputs/system")
    session_id = gpu_tracker.start(
        phase="reranker_training",
        description=f"Training {cfg['models']['jina_v3']} with hybrid loss"
    )

    try:
        trainer.train(train_dataset, val_dataset)
    finally:
        # Always stop GPU tracking
        session = gpu_tracker.stop()
        logger.info(f"GPU training time: {session.duration_hours:.2f}h")
        logger.info(f"Total GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")

    # Save split metadata for reproducibility
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "split_post_ids.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Save GPU tracking info to output directory
    with open(out_dir / "gpu_tracking.json", "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session_id,
            "duration_hours": session.duration_hours,
            "duration_seconds": session.duration_seconds,
            "avg_utilization": session.avg_utilization,
            "peak_memory_gb": session.peak_memory_gb,
            "total_gpu_hours": gpu_tracker.total_gpu_hours,
        }, f, indent=2)

    logger.info("Training complete; model saved to %s", out_dir)


if __name__ == "__main__":
    main()
