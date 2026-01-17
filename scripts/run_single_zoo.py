#!/usr/bin/env python3
"""Run a single post_id + criterion_id through the zoo pipeline.

This uses the zoo-based pipeline which supports the best HPO model combo
(nv-embed-v2 + jina-reranker-v3).

Usage:
    python scripts/run_single_zoo.py --post_id POST123 --criterion_id A1
    python scripts/run_single_zoo.py --config configs/default.yaml --post_id POST123 --criterion_id A1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from final_sc_review.data.io import load_criteria
from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single query through zoo pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--post_id", type=str, required=True)
    parser.add_argument("--criterion_id", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=None, help="Number of results to return")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load criteria
    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}

    query = criteria_map.get(args.criterion_id)
    if query is None:
        raise ValueError(f"Unknown criterion_id: {args.criterion_id}")

    # Load pipeline
    logger.info("Loading zoo pipeline...")
    pipeline = load_zoo_pipeline_from_config(Path(args.config))

    # Get model info
    retriever_name = cfg.get("models", {}).get("retriever_name", "unknown")
    reranker_name = cfg.get("models", {}).get("reranker_name", "unknown")

    print(f"\n{'=' * 60}")
    print(f"Pipeline: {retriever_name} + {reranker_name}")
    print(f"Post ID: {args.post_id}")
    print(f"Criterion: {args.criterion_id}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    # Run inference
    results = pipeline.retrieve(query=query, post_id=args.post_id, top_k=args.top_k)

    print("Results:")
    print("-" * 60)
    for i, (sent_uid, sent_text, score) in enumerate(results, 1):
        print(f"{i}. [{sent_uid}] (score: {score:.4f})")
        print(f"   {sent_text}")
        print()


if __name__ == "__main__":
    main()
