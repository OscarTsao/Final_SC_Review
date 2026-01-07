"""Seeding utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional in some contexts
    torch = None


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = cudnn_benchmark
