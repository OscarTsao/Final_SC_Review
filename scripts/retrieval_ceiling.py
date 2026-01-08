#!/usr/bin/env python3
"""Retrieval ceiling analysis for paper-grade research.

Computes:
- Oracle recall at various K values
- Per-criterion oracle recall
- Identifies never-retrieved positives
- Error categorization
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus, load_criteria
from final_sc_review.data.splits import split_post_ids
from final_sc_review.pipeline.run import load_pipeline_from_config


def compute_oracle_recall(
    pipeline,
    groundtruth: pd.DataFrame,
    split_post_ids: List[str],
    criteria: List[dict],
    k_values: List[int] = [50, 100, 200, 400, 800],
) -> Dict:
    """Compute oracle recall at various K values.

    Oracle recall = fraction of positives that appear in top-K retriever results
    (assuming a perfect reranker that knows which are positive).
    """
    results = {k: {"hits": 0, "total": 0, "per_criterion": defaultdict(lambda: {"hits": 0, "total": 0})} for k in k_values}
    never_retrieved = []  # Track positives never in any pool

    # Filter groundtruth to split
    gt_split = groundtruth[groundtruth["post_id"].isin(split_post_ids)]

    # Get positive examples
    positives = gt_split[gt_split["groundtruth"] == 1]

    # Group by (post_id, criterion)
    for (post_id, criterion), group in positives.groupby(["post_id", "criterion"]):
        positive_sent_uids = set(group["sent_uid"].tolist())

        if not positive_sent_uids:
            continue

        # Get criterion text
        crit_text = None
        for c in criteria:
            if c["id"] == criterion:
                crit_text = c["text"]
                break

        if crit_text is None:
            continue

        # Get retriever candidates at max K
        max_k = max(k_values)
        try:
            candidates = pipeline.retrieve(query=crit_text, post_id=post_id, top_k=max_k)
            candidate_uids = {c[0] for c in candidates}  # sent_uid is first element
        except Exception as e:
            print(f"Error retrieving for {post_id}/{criterion}: {e}")
            continue

        # Check recall at each K
        for k in k_values:
            top_k_uids = {c[0] for c in candidates[:k]}
            hits = len(positive_sent_uids & top_k_uids)
            total = len(positive_sent_uids)

            results[k]["hits"] += hits
            results[k]["total"] += total
            results[k]["per_criterion"][criterion]["hits"] += hits
            results[k]["per_criterion"][criterion]["total"] += total

        # Track never retrieved positives
        for uid in positive_sent_uids:
            if uid not in candidate_uids:
                never_retrieved.append({
                    "post_id": post_id,
                    "criterion": criterion,
                    "sent_uid": uid,
                    "sentence": group[group["sent_uid"] == uid]["sentence"].iloc[0] if len(group[group["sent_uid"] == uid]) > 0 else "",
                })

    # Compute recall values
    summary = {}
    for k in k_values:
        if results[k]["total"] > 0:
            summary[f"oracle_recall@{k}"] = results[k]["hits"] / results[k]["total"]
        else:
            summary[f"oracle_recall@{k}"] = 0.0

        # Per-criterion breakdown
        summary[f"per_criterion@{k}"] = {}
        for crit, counts in results[k]["per_criterion"].items():
            if counts["total"] > 0:
                summary[f"per_criterion@{k}"][crit] = {
                    "recall": counts["hits"] / counts["total"],
                    "hits": counts["hits"],
                    "total": counts["total"],
                }

    summary["never_retrieved"] = never_retrieved
    summary["never_retrieved_count"] = len(never_retrieved)
    summary["total_positives"] = results[max(k_values)]["total"]

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval ceiling analysis")
    parser.add_argument("--config", default="configs/default_val_optimized.yaml", help="Config file")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Split to analyze")
    parser.add_argument("--output", default="outputs/stageA/retrieval_ceiling.json", help="Output file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load data
    print("Loading data...")
    groundtruth_rows = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    criteria_objs = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")

    # Convert to DataFrame for easier processing
    groundtruth = pd.DataFrame([
        {
            "post_id": row.post_id,
            "criterion": row.criterion_id,
            "sid": row.sid,
            "sent_uid": row.sent_uid,
            "sentence": row.sentence_text,
            "groundtruth": row.groundtruth,
        }
        for row in groundtruth_rows
    ])

    # Convert criteria to list of dicts
    criteria = [{"id": c.criterion_id, "text": c.text} for c in criteria_objs]

    # Get split post IDs
    all_post_ids = groundtruth["post_id"].unique().tolist()
    splits = split_post_ids(
        all_post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    split_pids = splits[args.split]

    print(f"Analyzing {len(split_pids)} posts in {args.split} split...")

    # Load pipeline (retriever only mode)
    print("Loading pipeline...")
    pipeline = load_pipeline_from_config(config_path)

    # Compute ceiling analysis
    print("Computing oracle recall...")
    k_values = [25, 50, 100, 200, 400]
    results = compute_oracle_recall(
        pipeline=pipeline,
        groundtruth=groundtruth,
        split_post_ids=split_pids,
        criteria=criteria,
        k_values=k_values,
    )

    # Add metadata
    results["config"] = str(config_path)
    results["split"] = args.split
    results["n_posts"] = len(split_pids)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL CEILING ANALYSIS")
    print("=" * 60)
    print(f"Split: {args.split} ({len(split_pids)} posts)")
    print(f"Total positives: {results['total_positives']}")
    print(f"Never retrieved: {results['never_retrieved_count']}")
    print()
    for k in k_values:
        print(f"Oracle Recall@{k}: {results.get(f'oracle_recall@{k}', 0):.4f}")

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
