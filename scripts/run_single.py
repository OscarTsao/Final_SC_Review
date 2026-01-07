#!/usr/bin/env python3
"""Run a single post_id + criterion_id through the pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from final_sc_review.data.io import load_criteria
from final_sc_review.pipeline.run import load_pipeline_from_config
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--post_id", type=str, required=True)
    parser.add_argument("--criterion_id", type=str, required=True)
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}
    query = criteria_map.get(args.criterion_id)
    if query is None:
        raise ValueError(f"Unknown criterion_id: {args.criterion_id}")

    pipeline = load_pipeline_from_config(Path(args.config), rebuild_cache=args.rebuild_cache)
    results = pipeline.retrieve(query=query, post_id=args.post_id)

    for sent_id, sent_text, score in results:
        print(f"{sent_id}\t{score:.4f}\t{sent_text}")


if __name__ == "__main__":
    main()
