#!/usr/bin/env python3
"""
Evaluate retriever on dev_select split.

Uses best HPO config to evaluate on held-out dev_select split.

Usage:
    python scripts/retriever/dev_select_eval.py
    python scripts/retriever/dev_select_eval.py --config configs/best.yaml
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def compute_metrics(ranked_uids, gold_uids, k):
    """Compute ranking metrics at K."""
    gold_set = set(gold_uids)
    ranked_k = ranked_uids[:k]

    # Recall@K
    hits = sum(1 for uid in ranked_k if uid in gold_set)
    recall = hits / len(gold_uids) if gold_uids else 0.0

    # MRR@K
    mrr = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@K
    dcg = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_uids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever on dev_select")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--hpo_results", type=str, default=None, help="HPO results JSON")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth, load_criteria
    from final_sc_review.data.splits import split_post_ids
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("="*60)
    print("DEV_SELECT EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # Load HPO results if available
    hpo_params = {}
    if args.hpo_results:
        hpo_path = Path(args.hpo_results)
    else:
        hpo_path = repo_root / "outputs" / "retriever" / "hpo" / "retriever_frozen_hpo_results.json"

    if hpo_path.exists():
        with open(hpo_path) as f:
            hpo_data = json.load(f)
            hpo_params = hpo_data.get("best_params", {})
        print(f"[INFO] Loaded HPO params from {hpo_path}")
    else:
        print("[INFO] No HPO results found, using defaults")

    # Load data
    print("\n[1/4] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Build queries
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}

    # Split posts
    print("\n[2/4] Splitting posts...")
    splits = split_post_ids(list(all_posts), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    dev_select_posts = set(splits["val"])

    # Filter to dev_select
    queries = []
    for (post_id, criterion_id), gold_uids in query_gold.items():
        if post_id in dev_select_posts:
            query_text = criterion_text.get(criterion_id, criterion_id)
            queries.append((post_id, criterion_id, list(gold_uids), query_text))

    print(f"  Dev select queries: {len(queries)}")

    # Initialize retriever
    print("\n[3/4] Initializing retriever...")
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        rebuild_cache=False,
    )

    # Evaluate
    print("\n[4/4] Evaluating...")
    k_values = [1, 5, 10, 20]
    results_by_k = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    # Extract HPO params
    top_k_retriever = hpo_params.get("top_k_retriever", 50)
    top_k_final = hpo_params.get("top_k_final", 10)
    use_sparse = hpo_params.get("use_sparse", True)
    use_colbert = hpo_params.get("use_colbert", False)
    fusion_method = hpo_params.get("fusion_method", "weighted_sum")
    score_norm = hpo_params.get("score_normalization", "zscore_per_query")
    w_dense = hpo_params.get("w_dense", 0.7)
    w_sparse = hpo_params.get("w_sparse", 0.3)
    w_colbert = hpo_params.get("w_colbert", 0.0)

    for post_id, criterion_id, gold_uids, query_text in queries:
        if not gold_uids:
            continue

        results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k_retriever=top_k_retriever,
            use_dense=True,
            use_sparse=use_sparse,
            use_colbert=use_colbert,
            dense_weight=w_dense,
            sparse_weight=w_sparse,
            colbert_weight=w_colbert,
            fusion_method=fusion_method,
            score_normalization=score_norm,
        )

        ranked_uids = [r[0] for r in results]

        for k in k_values:
            metrics = compute_metrics(ranked_uids, gold_uids, k)
            for metric_name, value in metrics.items():
                results_by_k[k][metric_name].append(value)

    # Aggregate results
    summary = {"timestamp": datetime.now().isoformat(), "split": "dev_select", "n_queries": len(queries)}
    summary["hpo_params"] = hpo_params

    print("\n  Results:")
    for k in k_values:
        summary[f"recall@{k}"] = np.mean(results_by_k[k]["recall"])
        summary[f"mrr@{k}"] = np.mean(results_by_k[k]["mrr"])
        summary[f"ndcg@{k}"] = np.mean(results_by_k[k]["ndcg"])
        print(f"    @{k}: Recall={summary[f'recall@{k}']:.3f}, MRR={summary[f'mrr@{k}']:.3f}, nDCG={summary[f'ndcg@{k}']:.3f}")

    # Save results
    output_dir = repo_root / "outputs" / "retriever"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "dev_select_results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved to: {results_path}")

    print("\n" + "="*60)
    print("[SUCCESS] Dev select evaluation complete")
    print("="*60)


if __name__ == "__main__":
    main()
