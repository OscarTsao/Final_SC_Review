#!/usr/bin/env python3
"""
Step 2: Rerank candidates using jina-reranker-v3.

This script runs in the jina-reranker conda environment (transformers 4.51+).
It loads candidates from step 1 and reranks them using jina-reranker-v3.

Usage:
    conda activate jina-reranker
    python scripts/step2_rerank_jina.py --input data/cache/retrieval_candidates.json --output outputs/reranked_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_metrics(gold_uids: set, ranked_ids: list, k: int) -> dict:
    """Compute ranking metrics at k."""
    # nDCG@k
    dcg = 0.0
    for i, uid in enumerate(ranked_ids[:k]):
        if uid in gold_uids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    n_relevant = min(len(gold_uids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Recall@k
    hits = sum(1 for uid in ranked_ids[:k] if uid in gold_uids)
    recall = hits / len(gold_uids) if gold_uids else 0.0

    # MRR@k
    mrr = 0.0
    for i, uid in enumerate(ranked_ids[:k]):
        if uid in gold_uids:
            mrr = 1.0 / (i + 1)
            break

    return {'ndcg': ndcg, 'recall': recall, 'mrr': mrr}


def main():
    parser = argparse.ArgumentParser(description="Step 2: Rerank using jina-reranker-v3")
    parser.add_argument("--input", type=str, default="data/cache/retrieval_candidates.json",
                        help="Input file with retrieval candidates from step 1")
    parser.add_argument("--output", type=str, default="outputs/reranked_results.json",
                        help="Output file for reranked results")
    parser.add_argument("--model", type=str, default="jinaai/jina-reranker-v3",
                        help="Reranker model name")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of queries to process in each batch")
    args = parser.parse_args()

    input_path = project_root / args.input
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load candidates from step 1
    print(f"Loading candidates from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    queries = data['queries']
    print(f"Loaded {len(queries)} queries")
    print(f"Retriever: {data['retriever']}, Top-K: {data['top_k']}")

    # Load jina-reranker-v3
    print(f"\nLoading reranker: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    )
    print("Reranker loaded!")

    # Process each query
    print(f"\nReranking candidates...")
    all_results = []
    all_metrics = {k: {'ret': [], 'rerank': []} for k in [1, 5, 10, 20]}

    for query_data in tqdm(queries, desc="Reranking"):
        criterion_text = query_data['criterion_text']
        candidates = query_data['candidates']
        gold_uids = set(query_data['gold_uids'])

        if len(candidates) < 2:
            continue

        # Get document texts
        doc_texts = [c['text'] for c in candidates]

        # Rerank using jina model
        try:
            rerank_results = model.rerank(criterion_text, doc_texts, top_n=len(doc_texts))
        except Exception as e:
            print(f"Error reranking {query_data['post_id']}/{query_data['criterion_id']}: {e}")
            continue

        # Map back to candidates with scores
        reranked_candidates = []
        for r in rerank_results:
            orig_idx = r['index']
            reranked_candidates.append({
                'sent_uid': candidates[orig_idx]['sent_uid'],
                'text': candidates[orig_idx]['text'],
                'retriever_score': candidates[orig_idx]['retriever_score'],
                'reranker_score': float(r['relevance_score']),
            })

        # Compute metrics (only for queries with gold labels)
        query_metrics = {}
        if gold_uids:
            retriever_ranking = [c['sent_uid'] for c in candidates]
            reranker_ranking = [c['sent_uid'] for c in reranked_candidates]

            for k in [1, 5, 10, 20]:
                ret_metrics = compute_metrics(gold_uids, retriever_ranking, k)
                rerank_metrics = compute_metrics(gold_uids, reranker_ranking, k)

                all_metrics[k]['ret'].append(ret_metrics)
                all_metrics[k]['rerank'].append(rerank_metrics)

                query_metrics[f'retriever@{k}'] = ret_metrics
                query_metrics[f'reranker@{k}'] = rerank_metrics

        all_results.append({
            'post_id': query_data['post_id'],
            'criterion_id': query_data['criterion_id'],
            'criterion_text': criterion_text,
            'gold_uids': query_data['gold_uids'],
            'is_no_evidence': query_data['is_no_evidence'],
            'retriever_ranking': candidates,
            'reranker_ranking': reranked_candidates,
            'metrics': query_metrics,
        })

    # Aggregate metrics
    print("\n" + "=" * 70)
    print("RERANKING RESULTS")
    print("=" * 70)

    aggregated_metrics = {}
    n_queries_with_evidence = len(all_metrics[10]['ret'])
    print(f"\nQueries with evidence: {n_queries_with_evidence}")

    print(f"\n{'Metric':<15} {'@1':>12} {'@5':>12} {'@10':>12} {'@20':>12}")
    print("-" * 65)

    for metric_name in ['ndcg', 'recall', 'mrr']:
        # Retriever
        row_ret = f"Ret {metric_name.upper():<10}"
        for k in [1, 5, 10, 20]:
            values = [m[metric_name] for m in all_metrics[k]['ret']]
            mean = np.mean(values) if values else 0.0
            aggregated_metrics[f'ret_{metric_name}@{k}'] = mean
            row_ret += f" {mean:>12.4f}"
        print(row_ret)

        # Reranker
        row_rerank = f"Rerank {metric_name.upper():<7}"
        for k in [1, 5, 10, 20]:
            values = [m[metric_name] for m in all_metrics[k]['rerank']]
            mean = np.mean(values) if values else 0.0
            aggregated_metrics[f'rerank_{metric_name}@{k}'] = mean
            row_rerank += f" {mean:>12.4f}"
        print(row_rerank)

    # Improvement
    print("-" * 65)
    print("\nImprovement (Reranker - Retriever):")
    for metric_name in ['ndcg', 'recall', 'mrr']:
        row = f"{metric_name.upper():<12}"
        for k in [1, 5, 10, 20]:
            ret_mean = aggregated_metrics[f'ret_{metric_name}@{k}']
            rerank_mean = aggregated_metrics[f'rerank_{metric_name}@{k}']
            diff = rerank_mean - ret_mean
            pct = (diff / ret_mean * 100) if ret_mean > 0 else 0
            row += f" {diff:+.4f} ({pct:+.1f}%)"
        print(row)

    # Save results
    print(f"\nSaving results to {output_path}")
    output_data = {
        'retriever': data['retriever'],
        'reranker': args.model,
        'n_queries': len(all_results),
        'n_queries_with_evidence': n_queries_with_evidence,
        'aggregated_metrics': aggregated_metrics,
        'queries': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
