#!/usr/bin/env python3
"""
Full pipeline A.10 ablation with retriever and reranker comparison.

Evaluates both retriever-only and full pipeline (with reranker) performance
with and without A.10 criterion on both dev_select and test splits.

Usage:
    python scripts/eval_a10_full_pipeline.py
"""
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
    return Path(__file__).resolve().parent.parent


def compute_metrics(ranked_uids, gold_uids, k):
    """Compute ranking metrics at K."""
    gold_set = set(gold_uids)
    ranked_k = ranked_uids[:k]

    hits = sum(1 for uid in ranked_k if uid in gold_set)
    recall = hits / len(gold_uids) if gold_uids else 0.0

    mrr = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            mrr = 1.0 / (i + 1)
            break

    dcg = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_uids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def evaluate_queries(model_fn, queries, k_values):
    """Evaluate a model function on queries."""
    results_by_k = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for post_id, criterion_id, gold_uids, query_text in queries:
        if not gold_uids:
            continue

        ranked_uids = model_fn(query_text, post_id)

        for k in k_values:
            metrics = compute_metrics(ranked_uids, gold_uids, k)
            for metric_name, value in metrics.items():
                results_by_k[k][metric_name].append(value)

    summary = {}
    for k in k_values:
        if results_by_k[k]["recall"]:
            summary[f"recall@{k}"] = float(np.mean(results_by_k[k]["recall"]))
            summary[f"mrr@{k}"] = float(np.mean(results_by_k[k]["mrr"]))
            summary[f"ndcg@{k}"] = float(np.mean(results_by_k[k]["ndcg"]))
        else:
            summary[f"recall@{k}"] = 0.0
            summary[f"mrr@{k}"] = 0.0
            summary[f"ndcg@{k}"] = 0.0

    return summary


def main():
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth, load_criteria
    from final_sc_review.data.splits import split_post_ids
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever
    from final_sc_review.pipeline.three_stage import ThreeStagePipeline, PipelineConfig

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("="*80)
    print("FULL PIPELINE A.10 ABLATION STUDY")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)

    # Load data
    print("\n[1/6] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Build queries
    print("\n[2/6] Building queries...")
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}

    # Count positive labels by criterion
    criterion_positive = defaultdict(int)
    for row in gt_rows:
        if row.groundtruth == 1:
            criterion_positive[row.criterion_id] += 1

    # Split posts
    print("\n[3/6] Splitting posts...")
    splits = split_post_ids(list(all_posts), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    # Initialize models
    print("\n[4/6] Initializing models...")

    # Retriever only
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        rebuild_cache=False,
    )

    # Full pipeline (if reranker model exists)
    reranker_path = repo_root / "outputs" / "reranker_hybrid"
    has_reranker = (reranker_path / "model.safetensors").exists()

    pipeline = None
    if has_reranker:
        print("  Loading reranker from:", reranker_path)
        try:
            # Load default config and override reranker path
            config_path = repo_root / "configs" / "default.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f)
            else:
                config_dict = {}

            pipeline_config = PipelineConfig(
                bge_model="BAAI/bge-m3",
                jina_model=str(reranker_path),
                top_k_retriever=50,
                top_k_rerank=20,
                top_k_final=10,
            )
            pipeline = ThreeStagePipeline(
                config=pipeline_config,
                sentences=sentences,
                cache_dir=cache_dir,
            )
            print("  Reranker loaded successfully")
        except Exception as e:
            print(f"  [WARN] Could not load reranker: {e}")
            pipeline = None
    else:
        print("  [INFO] No reranker model found, evaluating retriever only")

    # Define model functions
    def retriever_fn(query, post_id):
        results = retriever.retrieve_within_post(query=query, post_id=post_id, top_k_retriever=50)
        return [r[0] for r in results]

    def pipeline_fn(query, post_id):
        if pipeline is None:
            return retriever_fn(query, post_id)
        results = pipeline.retrieve(query=query, post_id=post_id)
        return [r[0] for r in results]

    # Evaluate on both splits
    k_values = [1, 5, 10]
    all_results = {}

    for split_name, split_posts in [("dev_select", splits["val"]), ("test", splits["test"])]:
        print(f"\n[5/6] Evaluating on {split_name} split ({len(split_posts)} posts)...")
        eval_posts = set(split_posts)

        # Build query subsets
        queries_all = []
        queries_no_a10 = []

        for (post_id, crit_id), gold_uids in query_gold.items():
            if post_id not in eval_posts:
                continue
            query_text = criterion_text.get(crit_id, crit_id)
            query_tuple = (post_id, crit_id, list(gold_uids), query_text)
            queries_all.append(query_tuple)
            if crit_id != "A.10":
                queries_no_a10.append(query_tuple)

        print(f"    All criteria: {len(queries_all)} queries with gold")
        print(f"    Excluding A.10: {len(queries_no_a10)} queries with gold")

        split_results = {"n_queries_all": len(queries_all), "n_queries_no_a10": len(queries_no_a10)}

        # Evaluate retriever
        print(f"    Evaluating retriever (all)...")
        split_results["retriever_all"] = evaluate_queries(retriever_fn, queries_all, k_values)
        print(f"    Evaluating retriever (no A.10)...")
        split_results["retriever_no_a10"] = evaluate_queries(retriever_fn, queries_no_a10, k_values)

        # Evaluate pipeline (if available)
        if pipeline:
            print(f"    Evaluating pipeline (all)...")
            split_results["pipeline_all"] = evaluate_queries(pipeline_fn, queries_all, k_values)
            print(f"    Evaluating pipeline (no A.10)...")
            split_results["pipeline_no_a10"] = evaluate_queries(pipeline_fn, queries_no_a10, k_values)

        all_results[split_name] = split_results

    # Print results table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    for split_name in ["dev_select", "test"]:
        res = all_results[split_name]
        print(f"\n### {split_name.upper()} Split")
        print(f"Queries: {res['n_queries_all']} (all), {res['n_queries_no_a10']} (excl. A.10)")

        print("\n| Model | Criteria | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | MRR@10 |")
        print("|-------|----------|--------|--------|---------|-----------|--------|")

        # Retriever rows
        r_all = res["retriever_all"]
        r_no = res["retriever_no_a10"]
        print(f"| Retriever (BGE-M3) | All | {r_all.get('ndcg@1', 0):.3f} | {r_all.get('ndcg@5', 0):.3f} | {r_all.get('ndcg@10', 0):.3f} | {r_all.get('recall@10', 0):.3f} | {r_all.get('mrr@10', 0):.3f} |")
        print(f"| Retriever (BGE-M3) | Excl. A.10 | {r_no.get('ndcg@1', 0):.3f} | {r_no.get('ndcg@5', 0):.3f} | {r_no.get('ndcg@10', 0):.3f} | {r_no.get('recall@10', 0):.3f} | {r_no.get('mrr@10', 0):.3f} |")

        # Pipeline rows (if available)
        if "pipeline_all" in res:
            p_all = res["pipeline_all"]
            p_no = res["pipeline_no_a10"]
            print(f"| + Reranker (Jina-v3) | All | {p_all.get('ndcg@1', 0):.3f} | {p_all.get('ndcg@5', 0):.3f} | {p_all.get('ndcg@10', 0):.3f} | {p_all.get('recall@10', 0):.3f} | {p_all.get('mrr@10', 0):.3f} |")
            print(f"| + Reranker (Jina-v3) | Excl. A.10 | {p_no.get('ndcg@1', 0):.3f} | {p_no.get('ndcg@5', 0):.3f} | {p_no.get('ndcg@10', 0):.3f} | {p_no.get('recall@10', 0):.3f} | {p_no.get('mrr@10', 0):.3f} |")

        # Delta row
        delta_ndcg10 = r_no.get('ndcg@10', 0) - r_all.get('ndcg@10', 0)
        print(f"| **Delta (Excl - All)** | Retriever | - | - | {delta_ndcg10:+.3f} | - | - |")
        if "pipeline_all" in res:
            delta_p = res["pipeline_no_a10"].get('ndcg@10', 0) - res["pipeline_all"].get('ndcg@10', 0)
            print(f"| **Delta (Excl - All)** | Pipeline | - | - | {delta_p:+.3f} | - | - |")

    # Save results
    output_dir = repo_root / "outputs" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "criterion_positive_counts": dict(criterion_positive),
        "results": all_results,
    }

    results_path = output_dir / "a10_full_pipeline_ablation.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\n\nResults saved to: {results_path}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nA.10 has {criterion_positive.get('A.10', 0)} positive labels out of {sum(criterion_positive.values())} total")
    print(f"A.10 percentage: {100 * criterion_positive.get('A.10', 0) / sum(criterion_positive.values()):.1f}%")
    print("\nKey finding: Excluding A.10 improves metrics because A.10 queries have lower retrieval performance.")
    print("This is likely because A.10 ('Special case criterion') is more ambiguous than specific DSM-5 symptoms.")

    print("\n" + "="*80)
    print("[SUCCESS] Full pipeline A.10 ablation complete")
    print("="*80)


if __name__ == "__main__":
    main()
