#!/usr/bin/env python3
"""Train all rerankers with maximum GPU utilization.

Runs each model in a separate subprocess to ensure clean GPU memory.
Uses optimal settings for maximum GPU utilization:
- AMP BF16 for memory efficiency
- Gradient accumulation for larger effective batch sizes
- Multi-worker DataLoader with prefetching
- Automatic batch size tuning per model

Usage:
    python scripts/reranker/train_all_rerankers.py --epochs 5
    python scripts/reranker/train_all_rerankers.py --models bge-reranker-v2-m3 jina-reranker-v2
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Model configurations with recommended training settings
MODEL_CONFIGS = {
    "bge-reranker-v2-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "batch_size": 8,  # Base batch size per GPU
        "gradient_accumulation": 8,  # Effective batch = 64
        "max_length": 512,
        "learning_rate": 2e-5,
    },
    "jina-reranker-v2": {
        "model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "batch_size": 8,
        "gradient_accumulation": 8,
        "max_length": 512,
        "learning_rate": 2e-5,
    },
    "mxbai-rerank-base-v2": {
        "model_id": "mixedbread-ai/mxbai-rerank-base-v2",
        "batch_size": 8,
        "gradient_accumulation": 8,
        "max_length": 512,
        "learning_rate": 2e-5,
    },
    "mxbai-rerank-large-v2": {
        "model_id": "mixedbread-ai/mxbai-rerank-large-v2",
        "batch_size": 4,  # Smaller for large model
        "gradient_accumulation": 16,
        "max_length": 512,
        "learning_rate": 1e-5,
    },
    "qwen3-reranker-0.6b": {
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
        "batch_size": 4,
        "gradient_accumulation": 16,
        "max_length": 512,
        "learning_rate": 1e-5,
    },
}


def train_single_model(
    model_name: str,
    config: Dict,
    epochs: int,
    output_dir: Path,
) -> Dict:
    """Train a single model in a subprocess."""
    model_output = output_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/reranker/train_maxout.py",
        "--model", config["model_id"],
        "--batch_size", str(config["batch_size"]),
        "--gradient_accumulation", str(config["gradient_accumulation"]),
        "--max_length", str(config["max_length"]),
        "--learning_rate", str(config["learning_rate"]),
        "--epochs", str(epochs),
        "--output_dir", str(model_output),
        "--num_workers", "4",
        "--use_amp",
        "--amp_dtype", "bfloat16",
    ]

    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"Model: {config['model_id']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")
    print(f"Output: {model_output}")
    print(f"{'='*80}\n")

    # Set environment for clean GPU
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per model
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            print(f"SUCCESS: {model_name} trained in {elapsed:.1f}s")
            # Try to parse training stats
            stats_file = model_output / "training_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                return {
                    "model": model_name,
                    "status": "success",
                    "elapsed_seconds": elapsed,
                    "best_val_loss": stats.get("best_val_loss"),
                }
            return {
                "model": model_name,
                "status": "success",
                "elapsed_seconds": elapsed,
            }
        else:
            print(f"FAILED: {model_name}")
            print(f"STDERR: {result.stderr[-1000:]}")
            return {
                "model": model_name,
                "status": "failed",
                "error": result.stderr[-500:],
            }

    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "status": "timeout",
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Train all rerankers")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to train (default: all)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/training/rerankers"))
    args = parser.parse_args()

    models_to_train = args.models or list(MODEL_CONFIGS.keys())

    print("=" * 80)
    print("RERANKER TRAINING - MAXIMUM GPU UTILIZATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Models: {', '.join(models_to_train)}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    results = []
    for model_name in models_to_train:
        if model_name not in MODEL_CONFIGS:
            print(f"WARNING: Unknown model {model_name}, skipping")
            continue

        config = MODEL_CONFIGS[model_name]
        result = train_single_model(
            model_name=model_name,
            config=config,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
        results.append(result)

    # Save summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "epochs": args.epochs,
            "results": results,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for r in results:
        status = r["status"]
        model = r["model"]
        if status == "success":
            elapsed = r.get("elapsed_seconds", 0)
            val_loss = r.get("best_val_loss", "N/A")
            print(f"{model}: SUCCESS - {elapsed:.1f}s, val_loss={val_loss}")
        else:
            print(f"{model}: {status.upper()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
