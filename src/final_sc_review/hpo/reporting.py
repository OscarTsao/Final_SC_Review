"""HPO reporting utilities."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_sha(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_gpu_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = "true"
            info["cuda_device_count"] = str(torch.cuda.device_count())
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda or "unknown"
        else:
            info["cuda_available"] = "false"
    except Exception:
        info["cuda_available"] = "unknown"
    return info


def get_env_info() -> Dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        import transformers
        import optuna

        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["optuna"] = optuna.__version__
    except Exception:
        pass
    info.update(get_gpu_info())
    return info


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@dataclass
class TrialLogWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["trial_number", "metric", "status", "duration_sec", "params", "timestamp"]
                )

    def append(self, trial_number: int, metric: float, status: str, duration_sec: float, params: Dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    trial_number,
                    f"{metric:.6f}",
                    status,
                    f"{duration_sec:.2f}",
                    json.dumps(params, sort_keys=True),
                    datetime.utcnow().isoformat() + "Z",
                ]
            )


def build_run_manifest(
    repo_dir: Path,
    config_snapshot: Dict,
    dataset_checksums: Dict[str, str],
    cache_dir: Optional[Path] = None,
) -> Dict:
    manifest = {
        "git_sha": get_git_sha(repo_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "env": get_env_info(),
        "dataset_checksums": dataset_checksums,
        "config": config_snapshot,
    }
    if cache_dir is not None:
        manifest["cache_dir"] = str(cache_dir)
    return manifest
