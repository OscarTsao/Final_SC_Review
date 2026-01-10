#!/usr/bin/env python3
"""Standalone evaluation script for NV-Embed-v2.

Run with: mamba run -n nv-embed-v2 python scripts/retriever/eval_nv_embed_v2.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModel
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def load_criteria(criteria_path: Path) -> dict:
    """Load criteria definitions."""
    with open(criteria_path) as f:
        data = json.load(f)
    criteria = {}
    for item in data["criteria"]:
        cid = item["id"]
        criteria[cid] = item["text"]
    return criteria


def split_post_ids(post_ids: list, seed: int = 42, train_ratio: float = 0.6,
                   val_ratio: float = 0.2, test_ratio: float = 0.2) -> dict:
    """Split post_ids into train/val/test with no overlap."""
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


def compute_ndcg(relevances: list, k: int) -> float:
    """Compute nDCG@k."""
    relevances = relevances[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(retrieved: list, gold: set, k: int) -> float:
    """Compute Recall@k."""
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & gold)
    return hits / len(gold) if gold else 0.0


def compute_mrr(retrieved: list, gold: set, k: int) -> float:
    """Compute MRR@k."""
    for i, uid in enumerate(retrieved[:k]):
        if uid in gold:
            return 1.0 / (i + 1)
    return 0.0


def main():
    print("=" * 80)
    print("NV-EMBED-V2 EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Paths
    data_dir = PROJECT_ROOT / "data"
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"
    cache_dir = data_dir / "cache" / "nv-embed-v2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] Loading data...")
    sentences = load_sentences(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    groundtruth = load_groundtruth(gt_path)
    print(f"  Loaded groundtruth for {len(groundtruth)} query pairs")

    criteria = load_criteria(criteria_path)
    print(f"  Loaded {len(criteria)} criteria")

    # Get all unique post IDs and compute split
    all_post_ids = list(set(s["post_id"] for s in sentences))
    splits = split_post_ids(all_post_ids, seed=42, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    dev_select_posts = set(splits["val"])  # dev_select = val split
    print(f"  Computed split: {len(dev_select_posts)} post IDs for dev_select")

    # Build indices
    print("\n[2/5] Building indices...")
    post_to_indices = defaultdict(list)
    sent_uid_to_idx = {}
    for idx, sent in enumerate(sentences):
        post_id = sent["post_id"]
        post_to_indices[post_id].append(idx)
        sent_uid_to_idx[sent["sent_uid"]] = idx

    # Load model
    print("\n[3/5] Loading NV-Embed-v2 model...")

    # Use BF16 without quantization - 32GB VRAM should be enough for 7B model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    print("  Using BF16 precision (no quantization)")

    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", **model_kwargs)
    model = model.cuda()
    # Set to evaluation mode (disables dropout)
    model.train(False)
    print("  Model loaded successfully!")

    # Encode corpus
    print("\n[4/5] Encoding corpus...")
    embeddings_path = cache_dir / "embeddings.npy"

    # Query instruction for evidence retrieval
    query_instruction = "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

    if embeddings_path.exists():
        print(f"  Loading cached embeddings from {embeddings_path}")
        corpus_embeddings = np.load(embeddings_path)
    else:
        texts = [s["text"] for s in sentences]
        print(f"  Encoding {len(texts)} sentences...")

        # NV-Embed-v2 uses _do_encode for batch encoding
        # For passages, no instruction prefix is needed
        corpus_embeddings = model._do_encode(
            texts,
            batch_size=8,
            instruction="",  # No instruction for passages
            max_length=512,
            num_workers=0,
        )
        corpus_embeddings = corpus_embeddings.cpu().numpy()
        # Normalize
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_embeddings = corpus_embeddings / (norms + 1e-8)

        np.save(embeddings_path, corpus_embeddings)
        print(f"  Saved embeddings to {embeddings_path}")

    # Build queries
    print("\n[5/5] Evaluating...")
    queries = []
    for post_id in dev_select_posts:
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

    # Evaluate
    k_values = [1, 5, 10]
    metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
    metrics_no_a10 = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for q in tqdm(queries, desc="Evaluating"):
        # Encode query with instruction
        query_emb = model.encode(
            [q["criterion_text"]],
            instruction=query_instruction,
            max_length=512,
        )
        query_emb = query_emb.cpu().numpy()[0]
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Get within-post candidates
        indices = post_to_indices[q["post_id"]]
        candidate_embs = corpus_embeddings[indices]
        scores = candidate_embs @ query_emb

        # Rank
        ranked_indices = np.argsort(-scores)
        retrieved_uids = [sentences[indices[i]]["sent_uid"] for i in ranked_indices]

        # Compute metrics
        gold = q["gold_uids"]
        for k in k_values:
            relevances = [1 if uid in gold else 0 for uid in retrieved_uids[:k]]

            metrics[k]["recall"].append(compute_recall(retrieved_uids, gold, k))
            metrics[k]["mrr"].append(compute_mrr(retrieved_uids, gold, k))
            metrics[k]["ndcg"].append(compute_ndcg(relevances, k))

            # Exclude A.10
            if q["criterion_id"] != "A.10":
                metrics_no_a10[k]["recall"].append(compute_recall(retrieved_uids, gold, k))
                metrics_no_a10[k]["mrr"].append(compute_mrr(retrieved_uids, gold, k))
                metrics_no_a10[k]["ndcg"].append(compute_ndcg(relevances, k))

    # Aggregate results
    results = {
        "all_criteria": {
            "n_queries": len(queries),
        },
        "excluding_a10": {
            "n_queries": len([q for q in queries if q["criterion_id"] != "A.10"]),
        },
    }

    for k in k_values:
        results["all_criteria"][f"recall@{k}"] = float(np.mean(metrics[k]["recall"]))
        results["all_criteria"][f"mrr@{k}"] = float(np.mean(metrics[k]["mrr"]))
        results["all_criteria"][f"ndcg@{k}"] = float(np.mean(metrics[k]["ndcg"]))

        results["excluding_a10"][f"recall@{k}"] = float(np.mean(metrics_no_a10[k]["recall"]))
        results["excluding_a10"][f"mrr@{k}"] = float(np.mean(metrics_no_a10[k]["mrr"]))
        results["excluding_a10"][f"ndcg@{k}"] = float(np.mean(metrics_no_a10[k]["ndcg"]))

    results["delta_ndcg10"] = results["excluding_a10"]["ndcg@10"] - results["all_criteria"]["ndcg@10"]

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nAll Criteria (n={results['all_criteria']['n_queries']}):")
    for k in k_values:
        print(f"  @{k}: Recall={results['all_criteria'][f'recall@{k}']:.4f}, "
              f"MRR={results['all_criteria'][f'mrr@{k}']:.4f}, "
              f"nDCG={results['all_criteria'][f'ndcg@{k}']:.4f}")

    print(f"\nExcluding A.10 (n={results['excluding_a10']['n_queries']}):")
    for k in k_values:
        print(f"  @{k}: Recall={results['excluding_a10'][f'recall@{k}']:.4f}, "
              f"MRR={results['excluding_a10'][f'mrr@{k}']:.4f}, "
              f"nDCG={results['excluding_a10'][f'ndcg@{k}']:.4f}")

    print(f"\nDelta nDCG@10: {results['delta_ndcg10']:.4f}")

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "retriever_comparison" / "nv_embed_v2_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "nvidia/NV-Embed-v2",
            "split": "dev_select",
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
