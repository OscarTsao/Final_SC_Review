#!/usr/bin/env python3
"""Train reranker with maximum GPU utilization.

Implements all GPU optimizations:
- Mixed precision (BF16/FP16)
- Gradient accumulation
- Multi-worker DataLoader with prefetching
- torch.compile (optional)
- Flash Attention 2 (when available)
- Gradient checkpointing (optional)
- Parallel HPO with Optuna

Usage:
    # Single training run
    python scripts/reranker/train_maxout.py \
        --model BAAI/bge-reranker-v2-m3 \
        --batch_size 16 \
        --gradient_accumulation 4 \
        --epochs 5

    # HPO mode
    python scripts/reranker/train_maxout.py \
        --model BAAI/bge-reranker-v2-m3 \
        --hpo \
        --n_trials 50 \
        --n_jobs 2

    # Benchmark GPU utilization
    python scripts/reranker/train_maxout.py \
        --model BAAI/bge-reranker-v2-m3 \
        --benchmark
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import GroupedRerankerDataset, build_grouped_examples
from final_sc_review.reranker.maxout_trainer import (
    MaxoutConfig,
    MaxoutHPO,
    MaxoutHPOConfig,
    MaxoutRerankerTrainer,
    benchmark_gpu_utilization,
)
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_datasets(config_path: str, max_candidates: int = 16):
    """Load train and validation datasets."""
    with open(config_path, "r", encoding="utf-8") as f:
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
        max_candidates=max_candidates,
        seed=cfg["split"]["seed"],
    )
    val_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["val"],
        max_candidates=max_candidates,
        seed=cfg["split"]["seed"],
    )

    train_dataset = GroupedRerankerDataset(train_examples)
    val_dataset = GroupedRerankerDataset(val_examples)

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val examples")
    return train_dataset, val_dataset, splits


def run_training(args):
    """Run single training with maxout config."""
    train_dataset, val_dataset, splits = load_datasets(args.config, args.max_candidates)

    config = MaxoutConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        # Loss types (Section 7)
        listwise_type=args.listwise_type,
        pairwise_type=args.pairwise_type,
        pointwise_type=args.pointwise_type,
        # Loss weights
        w_list=args.w_list,
        w_pair=args.w_pair,
        w_point=args.w_point,
        temperature=args.temperature,
        # Loss-specific hyperparameters
        sigma=args.sigma,
        margin=args.margin,
        ndcg_k=args.ndcg_k,
        # Curriculum training (R4)
        curriculum_enabled=args.curriculum,
        curriculum_warmup_epochs=args.curriculum_warmup_epochs,
        curriculum_final_loss=args.curriculum_final_loss,
        # GPU Optimizations
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_compile=args.use_compile,
        use_flash_attention=args.use_flash_attention,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("=" * 80)
    print("MAXOUT RERANKER TRAINING")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {config.model_name}")
    print(f"Effective batch size: {config.effective_batch_size()}")
    print(f"Loss types: list={config.listwise_type}, pair={config.pairwise_type}, point={config.pointwise_type}")
    print(f"Loss weights: w_list={config.w_list}, w_pair={config.w_pair}, w_point={config.w_point}")
    if config.curriculum_enabled:
        print(f"Curriculum: BCE warmup ({config.curriculum_warmup_epochs} epochs) -> {config.curriculum_final_loss}")
    print(f"AMP: {config.use_amp} ({config.amp_dtype})")
    print(f"Compile: {config.use_compile}")
    print(f"Flash Attention: {config.use_flash_attention}")
    print("=" * 80)

    trainer = MaxoutRerankerTrainer(config)
    stats = trainer.train(train_dataset, val_dataset)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "split_post_ids.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {stats['best_val_loss']:.4f}")
    print(f"Total time: {stats['total_time_seconds']:.1f}s")
    print(f"Steps/second: {stats['steps_per_second']:.2f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 80)

    trainer.cleanup()
    return stats


def run_hpo(args):
    """Run parallel HPO."""
    train_dataset, val_dataset, _ = load_datasets(args.config, args.max_candidates)

    base_config = MaxoutConfig(
        model_name=args.model,
        max_length=args.max_length,
        num_epochs=args.epochs,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_compile=args.use_compile,
        use_flash_attention=args.use_flash_attention,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    hpo_config = MaxoutHPOConfig(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        storage=args.storage,
    )

    print("=" * 80)
    print("MAXOUT HPO")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {base_config.model_name}")
    print(f"N trials: {hpo_config.n_trials}")
    print(f"N jobs (parallel): {hpo_config.n_jobs}")
    print("=" * 80)

    hpo = MaxoutHPO(
        base_config=base_config,
        hpo_config=hpo_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    results = hpo.run()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "hpo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("HPO COMPLETE")
    print(f"Best trial: #{results['best_trial_number']}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"Completed: {results['n_complete']} / {results['n_trials']}")
    print("=" * 80)

    return results


def run_single_benchmark(model_name, batch_size, use_amp, amp_dtype, num_workers, n_batches, config_path, max_candidates):
    """Run a single benchmark config in isolation."""
    import gc

    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Load data
    train_dataset, _, _ = load_datasets(config_path, max_candidates)

    config = MaxoutConfig(
        model_name=model_name,
        batch_size=batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        num_workers=num_workers,
    )

    result = benchmark_gpu_utilization(config, train_dataset, n_batches=n_batches)
    return result


def run_benchmark(args):
    """Benchmark GPU utilization."""
    import gc

    print("=" * 80)
    print("GPU UTILIZATION BENCHMARK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Clear GPU memory first
    torch.cuda.empty_cache()
    gc.collect()

    # Load data once
    train_dataset, _, _ = load_datasets(args.config, args.max_candidates)

    configs_to_test = [
        {"use_amp": True, "amp_dtype": "bfloat16", "num_workers": 0, "name": "AMP BF16"},
        {"use_amp": True, "amp_dtype": "bfloat16", "num_workers": 4, "name": "AMP BF16 + 4 Workers"},
    ]

    results = []
    for cfg in configs_to_test:
        # Force cleanup before each config
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"\n--- {cfg['name']} ---")
        print(f"AMP: {cfg['use_amp']}, Workers: {cfg['num_workers']}")

        try:
            config = MaxoutConfig(
                model_name=args.model,
                batch_size=args.batch_size,
                use_amp=cfg["use_amp"],
                amp_dtype=cfg["amp_dtype"],
                num_workers=cfg["num_workers"],
            )

            bench = benchmark_gpu_utilization(config, train_dataset, n_batches=args.n_batches)
            bench["name"] = cfg["name"]
            results.append(bench)
            print(f"Batches/sec: {bench['batches_per_second']:.2f}")
            print(f"GPU Util: {bench['gpu_utilization']}%")
            print(f"Memory: {bench['memory_used_mb']} / {bench['memory_total_mb']} MB")
        except Exception as e:
            print(f"Failed: {e}")
            results.append({"error": str(e), "name": cfg["name"]})

        # Force cleanup after each config
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"gpu_benchmark_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Config':<40} {'Batches/s':>12} {'GPU%':>8}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            print(f"{r['name']:<40} {r['batches_per_second']:>12.2f} {r['gpu_utilization']:>8}%")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train reranker with maximum GPU utilization")

    # Mode
    parser.add_argument("--hpo", action="store_true", help="Run HPO mode")
    parser.add_argument("--benchmark", action="store_true", help="Run GPU benchmark")

    # Model
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="Model name or path")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")

    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    # Loss function types (Section 7 of research plan)
    parser.add_argument("--listwise_type", type=str, default="listnet",
                        choices=["listnet", "listmle", "plistmle", "lambda", "approx_ndcg"],
                        help="Listwise loss function")
    parser.add_argument("--pairwise_type", type=str, default="ranknet",
                        choices=["ranknet", "margin_ranking", "pairwise_softplus"],
                        help="Pairwise loss function")
    parser.add_argument("--pointwise_type", type=str, default="bce",
                        choices=["bce", "focal"],
                        help="Pointwise loss function")

    # Loss weights
    parser.add_argument("--w_list", type=float, default=1.0, help="Listwise loss weight")
    parser.add_argument("--w_pair", type=float, default=0.5, help="Pairwise loss weight")
    parser.add_argument("--w_point", type=float, default=0.1, help="Pointwise loss weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")

    # Loss-specific hyperparameters
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for RankNet/Lambda")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for margin ranking")
    parser.add_argument("--ndcg_k", type=int, default=10, help="K for Lambda/ApproxNDCG")

    # Curriculum training (R4)
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum training (BCE warmup -> final loss)")
    parser.add_argument("--curriculum_warmup_epochs", type=int, default=1,
                        help="Epochs of BCE warmup before switching")
    parser.add_argument("--curriculum_final_loss", type=str, default="lambda",
                        choices=["lambda", "listmle", "plistmle"],
                        help="Final loss for curriculum training")

    # GPU Optimizations
    parser.add_argument("--use_amp", type=bool, default=True, help="Use mixed precision")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"], help="AMP dtype")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--use_flash_attention", type=bool, default=True,
                        help="Use Flash Attention 2")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # HPO
    parser.add_argument("--n_trials", type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel HPO jobs")
    parser.add_argument("--study_name", type=str, default="reranker_training_hpo")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage (e.g., sqlite:///hpo.db)")

    # Benchmark
    parser.add_argument("--n_batches", type=int, default=50, help="Batches for benchmark")

    # Data
    parser.add_argument("--config", type=str, default="configs/reranker_hybrid.yaml",
                        help="Config file for data paths")
    parser.add_argument("--max_candidates", type=int, default=50,
                        help="Max candidates per query")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/training/maxout")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")
    else:
        print("WARNING: No GPU available, training will be slow!")

    # Run appropriate mode
    if args.benchmark:
        run_benchmark(args)
    elif args.hpo:
        run_hpo(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
