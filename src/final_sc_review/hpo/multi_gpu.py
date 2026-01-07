"""Multi-GPU worker helpers."""

from __future__ import annotations

import os
import subprocess
from typing import List


def detect_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def launch_workers(commands: List[List[str]], envs: List[dict]) -> int:
    processes = []
    for cmd, env in zip(commands, envs):
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
    exit_codes = [p.wait() for p in processes]
    return max(exit_codes) if exit_codes else 0
