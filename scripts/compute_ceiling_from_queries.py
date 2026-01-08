#!/usr/bin/env python3
"""Compute oracle recall from per_query.csv results."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def compute_oracle_recall(per_query_path: Path, k_values: list = [5, 10, 20, 50]) -> dict:
    """Compute oracle recall at various K values from per_query.csv."""
    df = pd.read_csv(per_query_path)

    results = {f"oracle_recall@{k}": 0.0 for k in k_values}
    per_criterion = defaultdict(lambda: {f"recall@{k}": {"hits": 0, "total": 0} for k in k_values})

    # Filter to queries with positives
    df_pos = df[df["gold_ids"].notna() & (df["gold_ids"] != "")]

    total_hits = {k: 0 for k in k_values}
    total_positives = 0

    for _, row in df_pos.iterrows():
        gold_ids = set(row["gold_ids"].split("|")) if pd.notna(row["gold_ids"]) else set()
        if not gold_ids:
            continue

        retriever_ids = row["retriever_topk"].split("|") if pd.notna(row["retriever_topk"]) else []
        criterion = row["criterion_id"]

        total_positives += len(gold_ids)
        per_criterion[criterion]["total"] = per_criterion[criterion].get("total", 0) + len(gold_ids)

        for k in k_values:
            topk_ids = set(retriever_ids[:k])
            hits = len(gold_ids & topk_ids)
            total_hits[k] += hits
            per_criterion[criterion][f"recall@{k}"]["hits"] += hits
            per_criterion[criterion][f"recall@{k}"]["total"] += len(gold_ids)

    # Compute final recall values
    for k in k_values:
        if total_positives > 0:
            results[f"oracle_recall@{k}"] = total_hits[k] / total_positives

    results["total_queries"] = len(df)
    results["queries_with_positives"] = len(df_pos)
    results["total_positives"] = total_positives
    results["per_criterion"] = dict(per_criterion)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="per_query.csv path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    results = compute_oracle_recall(Path(args.input), k_values=[1, 5, 10, 20, 50])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("ORACLE RECALL ANALYSIS")
    print("=" * 60)
    print(f"Total queries: {results['total_queries']}")
    print(f"Queries with positives: {results['queries_with_positives']}")
    print(f"Total positives: {results['total_positives']}")
    print()
    for k in [1, 5, 10, 20, 50]:
        print(f"Oracle Recall@{k}: {results.get(f'oracle_recall@{k}', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
