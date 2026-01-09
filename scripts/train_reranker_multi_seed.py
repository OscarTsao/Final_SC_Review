#!/usr/bin/env python3
"""Multi-seed reranker training for robustness and GPU time accumulation.

Runs multiple training runs with different seeds to:
1. Accumulate GPU time toward 12h target
2. Compute variance/confidence intervals for paper
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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
from final_sc_review.reranker.dataset import build_grouped_examples
from final_sc_review.reranker.trainer import train_reranker, RerankerConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def run_seed_experiment(
    seed: int,
    base_config: dict,
    groundtruth,
    criteria,
    sentences,
    output_dir: Path,
    gpu_tracker: GPUTimeTracker,
) -> dict:
    """Run a single seed experiment and return results."""

    logger.info(f"="*60)
    logger.info(f"SEED {seed} EXPERIMENT")
    logger.info(f"="*60)

    # Update config for this seed
    config = RerankerConfig(
        model_name=str(base_config["models"]["jina_v3"]),
        max_length=int(base_config["models"].get("max_length", 256)),
        batch_size=int(base_config["train"]["batch_size"]),
        num_epochs=int(base_config["train"]["num_epochs"]),
        learning_rate=float(base_config["train"]["learning_rate"]),
        weight_decay=float(base_config["train"].get("weight_decay", 0.01)),
        w_list=float(base_config["loss"]["w_list"]),
        w_pair=float(base_config["loss"]["w_pair"]),
        w_point=float(base_config["loss"]["w_point"]),
        temperature=float(base_config["loss"].get("temperature", 1.0)),
        max_pairs_per_group=int(base_config["loss"].get("max_pairs_per_group", 32)),
        early_stopping_patience=int(base_config["train"].get("early_stopping_patience", 3)),
    )

    # Split with this seed
    all_post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(all_post_ids, seed=seed)

    # Build datasets
    max_candidates = int(base_config["data"].get("max_candidates", 16))
    train_groups = build_grouped_examples(
        groundtruth, criteria,
        list(splits["train"]),
        max_candidates=max_candidates,
        seed=seed
    )
    val_groups = build_grouped_examples(
        groundtruth, criteria,
        list(splits["val"]),
        max_candidates=max_candidates,
        seed=seed
    )

    logger.info(f"Train groups: {len(train_groups)}, Val groups: {len(val_groups)}")

    # Output directory for this seed
    seed_output_dir = output_dir / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    # Start GPU tracking
    session_id = gpu_tracker.start(
        phase="reranker_multiseed",
        description=f"Training seed={seed}"
    )

    try:
        # Train
        model, metrics = train_reranker(
            config=config,
            train_groups=train_groups,
            val_groups=val_groups,
            output_dir=str(seed_output_dir),
        )
    finally:
        session = gpu_tracker.stop()
        logger.info(f"Seed {seed} training time: {session.duration_hours:.2f}h")

    # Save results
    result = {
        "seed": seed,
        "best_val_loss": metrics.get("best_val_loss", float("inf")),
        "train_loss_final": metrics.get("train_loss_final", float("inf")),
        "epochs_trained": metrics.get("epochs_trained", 0),
        "gpu_hours": session.duration_hours,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }

    with open(seed_output_dir / "seed_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-seed reranker training")
    parser.add_argument("--config", type=str, default="configs/reranker_extended.yaml")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                       help="Comma-separated seeds")
    parser.add_argument("--output_dir", type=str, default="outputs/reranker_multiseed")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Paths
    data_dir = Path(config["paths"]["data_dir"])
    groundtruth_path = Path(config["paths"]["groundtruth"])
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

    # Load data
    logger.info("Loading data...")
    groundtruth = load_groundtruth(groundtruth_path)
    criteria = load_criteria(criteria_path)
    sentences = load_sentence_corpus(corpus_path)

    # GPU tracker
    gpu_tracker = GPUTimeTracker(output_dir="outputs/system")

    # Run experiments
    all_results = []
    for seed in seeds:
        result = run_seed_experiment(
            seed=seed,
            base_config=config,
            groundtruth=groundtruth,
            criteria=criteria,
            sentences=sentences,
            output_dir=output_dir,
            gpu_tracker=gpu_tracker,
        )
        all_results.append(result)

        # Report progress
        logger.info(f"\nCumulative GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")

    # Aggregate results
    import numpy as np

    val_losses = [r["best_val_loss"] for r in all_results]
    gpu_hours = [r["gpu_hours"] for r in all_results]

    summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "val_loss_mean": float(np.mean(val_losses)),
        "val_loss_std": float(np.std(val_losses)),
        "val_loss_min": float(np.min(val_losses)),
        "val_loss_max": float(np.max(val_losses)),
        "total_gpu_hours": float(sum(gpu_hours)),
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("MULTI-SEED TRAINING SUMMARY")
    print("="*60)
    print(f"Seeds tested: {seeds}")
    print(f"Val loss: {summary['val_loss_mean']:.4f} ± {summary['val_loss_std']:.4f}")
    print(f"Range: [{summary['val_loss_min']:.4f}, {summary['val_loss_max']:.4f}]")
    print(f"Total GPU hours: {summary['total_gpu_hours']:.2f}h")
    print(f"Cumulative GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")
    print(gpu_tracker.format_summary())


if __name__ == "__main__":
    main()
