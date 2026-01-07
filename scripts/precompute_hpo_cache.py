#!/usr/bin/env python3
"""Precompute HPO cache for inference stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from final_sc_review.hpo.cache_builder import build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hpo_inference.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_cache(Path(args.config), force_rebuild=args.force)


if __name__ == "__main__":
    main()
