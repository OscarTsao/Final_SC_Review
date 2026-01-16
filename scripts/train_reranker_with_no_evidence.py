#!/usr/bin/env python3
"""
Train Jina-v3 reranker with NO_EVIDENCE pseudo-candidate.

This script trains the reranker to properly discriminate between has-evidence
and no-evidence queries by including NO_EVIDENCE as a pseudo-candidate.

For has-evidence queries: NO_EVIDENCE is a negative (should score low)
For no-evidence queries: NO_EVIDENCE is the positive (should score highest)
"""

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

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import GroupedRerankerDataset, build_grouped_examples
from final_sc_review.reranker.trainer import HybridRerankerTrainer, TrainConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train reranker with NO_EVIDENCE pseudo-candidate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reranker_with_no_evidence.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    gt_path = Path(cfg["paths"]["groundtruth"])
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

    groundtruth = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)

    # Get NO_EVIDENCE settings from config
    add_no_evidence = cfg["data"].get("add_no_evidence", False)
    include_no_evidence_queries = cfg["data"].get("include_no_evidence_queries", False)

    logger.info("NO_EVIDENCE training settings:")
    logger.info("  add_no_evidence: %s", add_no_evidence)
    logger.info("  include_no_evidence_queries: %s", include_no_evidence_queries)

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )

    # Build examples with NO_EVIDENCE
    train_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["train"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=cfg["split"]["seed"],
        add_no_evidence=add_no_evidence,
        include_no_evidence_queries=include_no_evidence_queries,
    )
    val_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["val"],
        max_candidates=cfg["data"]["max_candidates"],
        seed=cfg["split"]["seed"],
        add_no_evidence=add_no_evidence,
        include_no_evidence_queries=include_no_evidence_queries,
    )

    # Count query types
    n_train_has_evidence = sum(1 for ex in train_examples if not ex.get("is_no_evidence", False))
    n_train_no_evidence = sum(1 for ex in train_examples if ex.get("is_no_evidence", False))
    n_val_has_evidence = sum(1 for ex in val_examples if not ex.get("is_no_evidence", False))
    n_val_no_evidence = sum(1 for ex in val_examples if ex.get("is_no_evidence", False))

    logger.info("Training set: %d has-evidence, %d no-evidence queries",
                n_train_has_evidence, n_train_no_evidence)
    logger.info("Validation set: %d has-evidence, %d no-evidence queries",
                n_val_has_evidence, n_val_no_evidence)

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

    # Try to use GPU tracker if available
    try:
        from gpu_time_tracker import GPUTimeTracker
        gpu_tracker = GPUTimeTracker(output_dir="outputs/system")
        session_id = gpu_tracker.start(
            phase="reranker_with_no_evidence_training",
            description=f"Training {cfg['models']['jina_v3']} with NO_EVIDENCE"
        )
        use_tracker = True
    except ImportError:
        logger.warning("GPU time tracker not available")
        use_tracker = False

    try:
        trainer.train(train_dataset, val_dataset)
    finally:
        if use_tracker:
            session = gpu_tracker.stop()
            logger.info(f"GPU training time: {session.duration_hours:.2f}h")
            logger.info(f"Total GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")

    # Save split metadata and training info
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "split_post_ids.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Save training metadata
    metadata = {
        "add_no_evidence": add_no_evidence,
        "include_no_evidence_queries": include_no_evidence_queries,
        "n_train_has_evidence": n_train_has_evidence,
        "n_train_no_evidence": n_train_no_evidence,
        "n_val_has_evidence": n_val_has_evidence,
        "n_val_no_evidence": n_val_no_evidence,
        "loss_weights": {
            "w_list": cfg["loss"]["w_list"],
            "w_pair": cfg["loss"]["w_pair"],
            "w_point": cfg["loss"]["w_point"],
        },
    }
    with open(out_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if use_tracker:
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
