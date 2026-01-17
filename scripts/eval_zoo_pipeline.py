#!/usr/bin/env python3
"""Evaluate retriever+reranker performance using the zoo pipeline.

This script uses the zoo-based pipeline which supports dynamic selection
of retrievers and rerankers from the model zoo.

Usage:
    python scripts/eval_zoo_pipeline.py --config configs/default.yaml
    python scripts/eval_zoo_pipeline.py --config configs/locked_best_config.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.retrieval_eval import evaluate_rankings
from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zoo-based pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test", "train"],
        help="Evaluation split (default: test)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for evaluation metrics",
    )
    parser.add_argument(
        "--skip_no_positives",
        action="store_true",
        help="Skip queries with no positive examples",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load data
    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Split data
    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    eval_posts = set(splits[args.split])

    # Load pipeline
    logger.info("Loading zoo pipeline...")
    pipeline = load_zoo_pipeline_from_config(Path(args.config))

    # Get model info for logging
    retriever_name = cfg.get("models", {}).get("retriever_name", "unknown")
    reranker_name = cfg.get("models", {}).get("reranker_name", "unknown")
    logger.info(f"Retriever: {retriever_name}")
    logger.info(f"Reranker: {reranker_name}")

    # Group groundtruth by (post_id, criterion)
    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in eval_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    # Run evaluation
    rankings_retrieval = []
    rankings_reranked = []

    total_queries = len(grouped)
    logger.info(f"Evaluating {total_queries} queries on {args.split} split...")

    for i, ((post_id, criterion_id), rows) in enumerate(sorted(grouped.items())):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing query {i + 1}/{total_queries}")

        query_text = criteria_map.get(criterion_id)
        if query_text is None:
            continue

        gold_uids = {r.sent_uid for r in rows if r.groundtruth == 1}

        # Skip if no positives and flag is set
        if args.skip_no_positives and not gold_uids:
            continue

        # Run pipeline
        results = pipeline.retrieve(query_text, post_id)

        # Extract rankings
        reranked_uids = [r[0] for r in results]

        rankings_reranked.append({
            "query_id": f"{post_id}_{criterion_id}",
            "ranked_uids": reranked_uids,
            "gold_uids": gold_uids,
        })

    # Compute metrics
    logger.info("Computing metrics...")
    metrics_reranked = evaluate_rankings(rankings_reranked, ks=args.ks)

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({args.split} split)")
    print(f"Retriever: {retriever_name}")
    print(f"Reranker: {reranker_name}")
    print("=" * 60)
    print(f"\nQueries evaluated: {len(rankings_reranked)}")
    print("\nReranked Results:")
    for metric, value in sorted(metrics_reranked.items()):
        print(f"  {metric}: {value:.4f}")

    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/eval_zoo")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_{args.split}_{timestamp}.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "split": args.split,
        "retriever": retriever_name,
        "reranker": reranker_name,
        "n_queries": len(rankings_reranked),
        "metrics_reranked": metrics_reranked,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
