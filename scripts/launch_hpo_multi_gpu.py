#!/usr/bin/env python3
"""Launch multi-GPU HPO workers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from final_sc_review.hpo.cache_builder import build_cache
from final_sc_review.hpo.multi_gpu import detect_gpus, launch_workers
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hpo_inference.yaml")
    parser.add_argument("--study_name", type=str, default="sc_inference")
    parser.add_argument("--storage", type=str, default="sqlite:///outputs/hpo/optuna.db")
    parser.add_argument("--n_trials_per_worker", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pruner", type=str, default="hyperband")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    build_cache(Path(args.config), force_rebuild=False)

    gpu_count = detect_gpus()
    if gpu_count <= 0:
        raise RuntimeError("No GPUs detected for multi-GPU HPO")
    num_workers = args.num_workers or gpu_count
    num_workers = min(num_workers, gpu_count)

    commands = []
    envs = []
    for worker_id in range(num_workers):
        cmd = [
            sys.executable,
            "scripts/hpo_inference.py",
            "--config",
            args.config,
            "--study_name",
            args.study_name,
            "--storage",
            args.storage,
            "--n_trials",
            str(args.n_trials_per_worker),
            "--seed",
            str(args.seed + worker_id),
            "--pruner",
            args.pruner,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        commands.append(cmd)
        envs.append(env)

    exit_code = launch_workers(commands, envs)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
