#!/usr/bin/env python3
"""Create dev_tune/dev_select/test splits per R2.

Ensures post-disjoint splits:
- dev_tune: for HPO and threshold tuning
- dev_select: for final model selection among tuned candidates
- test: untouched until final
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def create_paper_splits(
    groundtruth_path: str,
    output_dir: str = "data/splits",
    seed: int = 42,
    dev_tune_ratio: float = 0.6,
    dev_select_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict:
    """Create post-disjoint splits for paper experiments.

    Returns:
        dict with split info and checksums
    """
    assert abs(dev_tune_ratio + dev_select_ratio + test_ratio - 1.0) < 1e-6

    # Load groundtruth
    groundtruth = load_groundtruth(groundtruth_path)
    all_post_ids = sorted({row.post_id for row in groundtruth})

    logger.info(f"Total posts: {len(all_post_ids)}")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_post_ids).tolist()

    n_total = len(shuffled)
    n_dev_tune = int(n_total * dev_tune_ratio)
    n_dev_select = int(n_total * dev_select_ratio)

    dev_tune_ids = shuffled[:n_dev_tune]
    dev_select_ids = shuffled[n_dev_tune:n_dev_tune + n_dev_select]
    test_ids = shuffled[n_dev_tune + n_dev_select:]

    logger.info(f"dev_tune: {len(dev_tune_ids)} posts ({len(dev_tune_ids)/n_total:.1%})")
    logger.info(f"dev_select: {len(dev_select_ids)} posts ({len(dev_select_ids)/n_total:.1%})")
    logger.info(f"test: {len(test_ids)} posts ({len(test_ids)/n_total:.1%})")

    # Verify disjoint
    assert len(set(dev_tune_ids) & set(dev_select_ids)) == 0
    assert len(set(dev_tune_ids) & set(test_ids)) == 0
    assert len(set(dev_select_ids) & set(test_ids)) == 0

    # Compute checksums
    def checksum(ids):
        return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]

    splits = {
        "dev_tune": dev_tune_ids,
        "dev_select": dev_select_ids,
        "test": test_ids,
    }

    checksums = {
        "dev_tune": checksum(dev_tune_ids),
        "dev_select": checksum(dev_select_ids),
        "test": checksum(test_ids),
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():
        with open(output_path / f"{split_name}_post_ids.json", "w") as f:
            json.dump(ids, f, indent=2)

    # Save metadata
    metadata = {
        "seed": seed,
        "n_total": n_total,
        "n_dev_tune": len(dev_tune_ids),
        "n_dev_select": len(dev_select_ids),
        "n_test": len(test_ids),
        "checksums": checksums,
        "ratios": {
            "dev_tune": dev_tune_ratio,
            "dev_select": dev_select_ratio,
            "test": test_ratio,
        },
    }

    with open(output_path / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Splits saved to {output_path}")

    return metadata


def load_split(split_name: str, splits_dir: str = "data/splits") -> list:
    """Load a split by name."""
    path = Path(splits_dir) / f"{split_name}_post_ids.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    metadata = create_paper_splits(
        groundtruth_path="data/groundtruth/evidence_sentence_groundtruth.csv",
        output_dir="data/splits",
        seed=42,
        dev_tune_ratio=0.6,
        dev_select_ratio=0.2,
        test_ratio=0.2,
    )

    print("\n" + "="*60)
    print("SPLIT CREATION COMPLETE")
    print("="*60)
    print(f"dev_tune: {metadata['n_dev_tune']} posts (checksum: {metadata['checksums']['dev_tune']})")
    print(f"dev_select: {metadata['n_dev_select']} posts (checksum: {metadata['checksums']['dev_select']})")
    print(f"test: {metadata['n_test']} posts (checksum: {metadata['checksums']['test']})")
