#!/usr/bin/env python3
"""Evaluate finetuned retriever against baseline.

Compares the finetuned BGE-M3 model against the original baseline
using paper-standard K values.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.ranking import ndcg_at_k, mrr_at_k
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_retriever_model(
    model_path: str,
    model_name: str,
    groundtruth_rows,
    criteria,
    sentences,
    val_post_ids: set,
    ks: list = [1, 3, 5, 10, 20],
) -> dict:
    """Evaluate a retriever model on validation set."""
    from collections import defaultdict

    logger.info(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    # Build indices
    criterion_text = {c.criterion_id: c.text for c in criteria}
    post_to_sentences = defaultdict(list)
    sent_uid_to_text = {}
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)
        sent_uid_to_text[sent.sent_uid] = sent.text

    # Group groundtruth by (post_id, criterion_id)
    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth_rows:
        if row.post_id not in val_post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    # Evaluate each query
    all_oracle_recalls = {k: [] for k in ks}
    all_ndcgs = {k: [] for k in ks}
    all_mrrs = {k: [] for k in ks}

    queries_with_positives = 0

    for (post_id, criterion_id), data in groups.items():
        gold_uids = data["gold_uids"]
        if not gold_uids:
            continue

        queries_with_positives += 1
        query_text = criterion_text.get(criterion_id)
        if not query_text:
            continue

        # Get candidate sentences for this post
        post_sents = post_to_sentences.get(post_id, [])
        if not post_sents:
            continue

        candidate_texts = [s.text for s in post_sents]
        candidate_uids = [s.sent_uid for s in post_sents]

        # Encode and score
        query_emb = model.encode([query_text], normalize_embeddings=True)
        cand_embs = model.encode(candidate_texts, normalize_embeddings=True)

        scores = np.dot(query_emb, cand_embs.T)[0]

        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_uids = [candidate_uids[i] for i in ranked_indices]

        n_candidates = len(ranked_uids)

        # Compute metrics at each K
        for k in ks:
            k_eff = min(k, n_candidates)

            # Oracle recall: did we find at least one positive in top-k?
            top_k_uids = ranked_uids[:k_eff]
            hits = len(set(top_k_uids) & gold_uids)
            oracle = 1.0 if hits > 0 else 0.0
            all_oracle_recalls[k].append(oracle)

            # nDCG (uses gold_ids, ranked_ids API)
            ndcg = ndcg_at_k(gold_uids, ranked_uids, k_eff)
            all_ndcgs[k].append(ndcg)

            # MRR (uses gold_ids, ranked_ids API)
            mrr = mrr_at_k(gold_uids, ranked_uids, k_eff)
            all_mrrs[k].append(mrr)

    # Aggregate
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "queries_with_positives": queries_with_positives,
    }

    for k in ks:
        results[f"Oracle@{k}"] = np.mean(all_oracle_recalls[k]) if all_oracle_recalls[k] else 0.0
        results[f"nDCG@{k}"] = np.mean(all_ndcgs[k]) if all_ndcgs[k] else 0.0
        results[f"MRR@{k}"] = np.mean(all_mrrs[k]) if all_mrrs[k] else 0.0

    return results


def main():
    # Paths
    data_dir = Path("data")
    groundtruth_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

    # Load data
    logger.info("Loading data...")
    groundtruth = load_groundtruth(groundtruth_path)
    criteria = load_criteria(criteria_path)
    sentences = load_sentence_corpus(corpus_path)

    # Split
    all_post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(all_post_ids, seed=42)
    val_post_ids = set(splits["val"])

    logger.info(f"Validation posts: {len(val_post_ids)}")

    # Models to compare
    models = [
        ("BAAI/bge-m3", "bge-m3-baseline"),
        ("outputs/retriever_finetuned/final", "bge-m3-finetuned"),
    ]

    ks = [1, 3, 5, 10, 20]

    print("\n" + "="*80)
    print("FINETUNED RETRIEVER EVALUATION")
    print("="*80)

    results_all = []
    for model_path, model_name in models:
        logger.info(f"\nEvaluating: {model_name}")
        results = evaluate_retriever_model(
            model_path, model_name,
            groundtruth, criteria, sentences,
            val_post_ids, ks
        )
        results_all.append(results)

        print(f"\n{model_name}:")
        for k in ks:
            print(f"  @{k}: Oracle={results[f'Oracle@{k}']:.4f}, nDCG={results[f'nDCG@{k}']:.4f}, MRR={results[f'MRR@{k}']:.4f}")

    # Comparison table
    print("\n" + "-"*80)
    print("COMPARISON TABLE")
    print("-"*80)
    print(f"{'Model':<20} | {'Oracle@20':<10} | {'nDCG@10':<10} | {'MRR@10':<10}")
    print("-"*60)
    for r in results_all:
        print(f"{r['model_name']:<20} | {r['Oracle@20']:<10.4f} | {r['nDCG@10']:<10.4f} | {r['MRR@10']:<10.4f}")

    # Delta
    if len(results_all) == 2:
        baseline = results_all[0]
        finetuned = results_all[1]
        print("\n" + "-"*80)
        print("IMPROVEMENT (finetuned - baseline)")
        print("-"*80)
        for k in ks:
            delta_oracle = finetuned[f"Oracle@{k}"] - baseline[f"Oracle@{k}"]
            delta_ndcg = finetuned[f"nDCG@{k}"] - baseline[f"nDCG@{k}"]
            delta_mrr = finetuned[f"MRR@{k}"] - baseline[f"MRR@{k}"]
            print(f"  @{k}: Oracle={delta_oracle:+.4f}, nDCG={delta_ndcg:+.4f}, MRR={delta_mrr:+.4f}")


if __name__ == "__main__":
    main()
