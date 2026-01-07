"""Leakage-free splits by post_id."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple


def split_post_ids(
    post_ids: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    """Split post_ids into train/val/test with no overlap."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    unique_ids = sorted(set(post_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train : n_train + n_val]
    test_ids = unique_ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def k_fold_post_ids(post_ids: Sequence[str], k: int = 5, seed: int = 42) -> List[Dict[str, List[str]]]:
    """Create k-fold splits by post_id."""
    if k < 2:
        raise ValueError("k must be >= 2")
    unique_ids = sorted(set(post_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    folds: List[List[str]] = [[] for _ in range(k)]
    for idx, pid in enumerate(unique_ids):
        folds[idx % k].append(pid)
    splits = []
    for i in range(k):
        test_ids = folds[i]
        train_ids = [pid for j, fold in enumerate(folds) if j != i for pid in fold]
        splits.append({"train": train_ids, "test": test_ids})
    return splits
