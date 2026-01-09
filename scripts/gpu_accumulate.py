#!/usr/bin/env python3
"""GPU time accumulation through multiple training runs.

Runs various training experiments to accumulate GPU time toward 12h target:
1. Reranker training with different seeds
2. Retriever finetuning with more epochs
3. Different hyperparameter configurations
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from gpu_time_tracker import GPUTimeTracker


def run_command(cmd: list, description: str, gpu_tracker: GPUTimeTracker) -> bool:
    """Run a command with GPU tracking."""
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    session_id = gpu_tracker.start(
        phase="gpu_accumulation",
        description=description
    )

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        success = result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        success = False
    finally:
        session = gpu_tracker.stop()
        print(f"\nCompleted: {description}")
        print(f"Duration: {session.duration_hours:.2f}h")
        print(f"Total GPU hours: {gpu_tracker.total_gpu_hours:.2f}h")

    return success


def main():
    gpu_tracker = GPUTimeTracker(output_dir="outputs/system")

    print(f"\n{'#'*60}")
    print("GPU TIME ACCUMULATION")
    print(f"Current total: {gpu_tracker.total_gpu_hours:.2f}h")
    print(f"Target: 12.0h")
    print(f"Remaining: {max(0, 12.0 - gpu_tracker.total_gpu_hours):.2f}h")
    print(f"{'#'*60}\n")

    experiments = [
        # Reranker with different seeds
        {
            "cmd": ["python", "scripts/train_reranker_hybrid.py",
                   "--config", "configs/reranker_hybrid.yaml"],
            "description": "Reranker seed=42 (baseline)",
        },
        # Retriever finetuning with more epochs
        {
            "cmd": ["python", "scripts/train_retriever.py",
                   "--config", "configs/retriever_finetune.yaml"],
            "description": "Retriever finetuning 10 epochs",
        },
        # Extended reranker
        {
            "cmd": ["python", "scripts/train_reranker_hybrid.py",
                   "--config", "configs/reranker_extended.yaml"],
            "description": "Extended reranker 10 epochs",
        },
    ]

    for i, exp in enumerate(experiments):
        print(f"\n[Experiment {i+1}/{len(experiments)}]")

        # Check if we've reached target
        if gpu_tracker.total_gpu_hours >= 12.0:
            print("Target reached! Stopping experiments.")
            break

        run_command(exp["cmd"], exp["description"], gpu_tracker)

        print(f"\nProgress: {gpu_tracker.total_gpu_hours:.2f}h / 12.0h")

    # Final summary
    print(f"\n{'#'*60}")
    print("FINAL SUMMARY")
    print(f"{'#'*60}")
    print(gpu_tracker.format_summary())


if __name__ == "__main__":
    main()
