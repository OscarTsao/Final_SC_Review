#!/usr/bin/env python3
"""Standalone NV-Embed-v2 HPO script.

Must be run in the nv-embed-v2 conda environment:
    mamba run -n nv-embed-v2 python scripts/hpo_nv_embed_retriever.py

This script bypasses the zoo architecture and directly uses the NV-Embed-v2
model for retrieval, combined with rerankers from the main environment.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_criteria_from_json(path: Path) -> dict:
    """Load criteria from JSON file."""
    with open(path) as f:
        data = json.load(f)
    criteria = {}
    for item in data["criteria"]:
        cid = item["id"]
        criteria[cid] = item["text"]
    return criteria


def load_sentences(corpus_path: Path) -> list:
    """Load sentence corpus."""
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            sent = json.loads(line)
            sentences.append(sent)
    return sentences


def load_groundtruth(gt_path: Path) -> dict:
    """Load groundtruth labels."""
    import csv
    gt = defaultdict(set)
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["groundtruth"] == "1":
                key = (row["post_id"], row["criterion"])
                gt[key].add(row["sent_uid"])
    return gt


def split_post_ids(post_ids: list, seed: int = 42, train_ratio: float = 0.6,
                   val_ratio: float = 0.2) -> dict:
    """Split post_ids into train/val/test."""
    import random
    unique_ids = sorted(set(post_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": unique_ids[:n_train],
        "val": unique_ids[n_train:n_train + n_val],
        "test": unique_ids[n_train + n_val:],
    }


def compute_metrics(retrieved_uids: list, gold_uids: set, k: int = 10) -> dict:
    """Compute ranking metrics at K."""
    # Recall@K
    retrieved_k = set(retrieved_uids[:k])
    recall = len(retrieved_k & gold_uids) / len(gold_uids) if gold_uids else 0.0

    # MRR@K
    mrr = 0.0
    for i, uid in enumerate(retrieved_uids[:k]):
        if uid in gold_uids:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@K
    relevances = [1 if uid in gold_uids else 0 for uid in retrieved_uids[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MAP@K
    precisions = []
    hits = 0
    for i, uid in enumerate(retrieved_uids[:k]):
        if uid in gold_uids:
            hits += 1
            precisions.append(hits / (i + 1))
    map_k = np.mean(precisions) if precisions else 0.0

    return {
        "recall_at_10": recall,
        "mrr_at_10": mrr,
        "ndcg_at_10": ndcg,
        "map_at_10": map_k,
    }


def main():
    parser = argparse.ArgumentParser(description="NV-Embed-v2 Retriever-Only HPO")
    parser.add_argument("--output_dir", type=str, default="outputs/nv_embed_retriever_hpo")
    parser.add_argument("--k_values", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    print("=" * 80)
    print("NV-EMBED-V2 RETRIEVER ASSESSMENT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Paths
    data_dir = PROJECT_ROOT / "data"
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"
    cache_dir = data_dir / "cache" / "nv-embed-v2"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] Loading data...")
    sentences = load_sentences(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    groundtruth = load_groundtruth(gt_path)
    print(f"  Loaded groundtruth for {len(groundtruth)} query pairs")

    criteria = load_criteria_from_json(criteria_path)
    print(f"  Loaded {len(criteria)} criteria")

    # Get split
    all_post_ids = list(set(s["post_id"] for s in sentences))
    splits = split_post_ids(all_post_ids, seed=42)
    split_posts = set(splits[args.split])
    print(f"  Using {args.split} split: {len(split_posts)} posts")

    # Build indices
    print("\n[2/5] Building indices...")
    post_to_indices = defaultdict(list)
    for idx, sent in enumerate(sentences):
        post_to_indices[sent["post_id"]].append(idx)

    # Load model
    print("\n[3/5] Loading NV-Embed-v2 model...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", **model_kwargs)
    model = model.cuda()
    model.train(False)
    print("  Model loaded successfully!")

    # Load or encode corpus
    print("\n[4/5] Loading/encoding corpus...")
    embeddings_path = cache_dir / "embeddings.npy"

    query_instruction = "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

    if embeddings_path.exists():
        print(f"  Loading cached embeddings from {embeddings_path}")
        corpus_embeddings = np.load(embeddings_path)
    else:
        texts = [s["text"] for s in sentences]
        print(f"  Encoding {len(texts)} sentences...")
        corpus_embeddings = model._do_encode(
            texts,
            batch_size=8,
            instruction="",
            max_length=512,
            num_workers=0,
        )
        corpus_embeddings = corpus_embeddings.cpu().numpy()
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_embeddings = corpus_embeddings / (norms + 1e-8)
        np.save(embeddings_path, corpus_embeddings)
        print(f"  Saved embeddings to {embeddings_path}")

    # Build queries
    print("\n[5/5] Running assessment at different K values...")
    queries = []
    for post_id in split_posts:
        if post_id not in post_to_indices:
            continue
        for cid, ctext in criteria.items():
            gold_uids = groundtruth.get((post_id, cid), set())
            if gold_uids:
                queries.append({
                    "post_id": post_id,
                    "criterion_id": cid,
                    "criterion_text": ctext,
                    "gold_uids": gold_uids,
                })

    print(f"  Total queries: {len(queries)}")

    # Assess at different K values
    results = []
    for top_k in args.k_values:
        print(f"\n  Assessing K={top_k}...")
        metrics_list = []

        for q in tqdm(queries, desc=f"K={top_k}", leave=False):
            # Encode query
            query_emb = model.encode(
                [q["criterion_text"]],
                instruction=query_instruction,
                max_length=512,
            )
            query_emb = query_emb.cpu().numpy()[0]
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

            # Get within-post candidates
            indices = post_to_indices[q["post_id"]]
            candidate_embs = corpus_embeddings[indices]
            scores = candidate_embs @ query_emb

            # Rank and get top-K
            ranked_indices = np.argsort(-scores)[:top_k]
            retrieved_uids = [sentences[indices[i]]["sent_uid"] for i in ranked_indices]

            # Compute metrics
            m = compute_metrics(retrieved_uids, q["gold_uids"], k=10)
            metrics_list.append(m)

        # Aggregate
        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        avg_metrics["top_k_retriever"] = top_k
        results.append(avg_metrics)

        print(f"    nDCG@10: {avg_metrics['ndcg_at_10']:.4f}, Recall@10: {avg_metrics['recall_at_10']:.4f}")

    # Save results
    df = pd.DataFrame(results)
    df["retriever"] = "nv-embed-v2"
    df["timestamp"] = datetime.now().isoformat()

    output_path = output_dir / "nv_embed_retriever_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Retriever-only, before reranking)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
