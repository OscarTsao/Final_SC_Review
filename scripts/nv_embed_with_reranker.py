#!/usr/bin/env python3
"""NV-Embed-v2 + Reranker evaluation script.

Two-stage execution:
1. Run in nv-embed-v2 env to cache retrieval results:
   mamba run -n nv-embed-v2 python scripts/nv_embed_with_reranker.py --stage retrieve

2. Run in main env to apply reranking:
   python scripts/nv_embed_with_reranker.py --stage rerank --reranker jina-reranker-v2
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
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


def load_criteria(path: Path) -> dict:
    """Load criteria from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item["text"] for item in data["criteria"]}


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
    retrieved_k = set(retrieved_uids[:k])
    recall = len(retrieved_k & gold_uids) / len(gold_uids) if gold_uids else 0.0

    mrr = 0.0
    for i, uid in enumerate(retrieved_uids[:k]):
        if uid in gold_uids:
            mrr = 1.0 / (i + 1)
            break

    relevances = [1 if uid in gold_uids else 0 for uid in retrieved_uids[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    precisions = []
    hits = 0
    for i, uid in enumerate(retrieved_uids[:k]):
        if uid in gold_uids:
            hits += 1
            precisions.append(hits / (i + 1))
    map_k = np.mean(precisions) if precisions else 0.0

    return {"recall_at_10": recall, "mrr_at_10": mrr, "ndcg_at_10": ndcg, "map_at_10": map_k}


def stage_retrieve(args):
    """Stage 1: Run NV-Embed-v2 retrieval and cache results."""
    import torch
    from transformers import AutoModel

    print("=" * 80)
    print("STAGE 1: NV-EMBED-V2 RETRIEVAL")
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
    groundtruth = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    print(f"  Loaded {len(sentences)} sentences, {len(criteria)} criteria")

    # Get split
    all_post_ids = list(set(s["post_id"] for s in sentences))
    splits = split_post_ids(all_post_ids, seed=42)
    split_posts = set(splits[args.split])
    print(f"  Using {args.split} split: {len(split_posts)} posts")

    # Build indices
    post_to_indices = defaultdict(list)
    for idx, sent in enumerate(sentences):
        post_to_indices[sent["post_id"]].append(idx)

    # Load model
    print("\n[2/5] Loading NV-Embed-v2 model...")
    model = AutoModel.from_pretrained(
        "nvidia/NV-Embed-v2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.cuda()
    model.train(False)
    print("  Model loaded!")

    # Load embeddings
    print("\n[3/5] Loading corpus embeddings...")
    embeddings_path = cache_dir / "embeddings.npy"
    corpus_embeddings = np.load(embeddings_path)
    print(f"  Loaded {len(corpus_embeddings)} embeddings")

    # Build queries
    print("\n[4/5] Building queries...")
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

    # Run retrieval
    print(f"\n[5/5] Running retrieval (top_k={args.top_k_retriever})...")
    query_instruction = "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

    retrieval_cache = []
    for q in tqdm(queries, desc="Retrieving"):
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
        ranked_indices = np.argsort(-scores)[:args.top_k_retriever]
        candidates = []
        for ri in ranked_indices:
            idx = indices[ri]
            candidates.append({
                "sent_uid": sentences[idx]["sent_uid"],
                "text": sentences[idx]["text"],
                "score": float(scores[ri]),
            })

        retrieval_cache.append({
            "post_id": q["post_id"],
            "criterion_id": q["criterion_id"],
            "criterion_text": q["criterion_text"],
            "gold_uids": list(q["gold_uids"]),
            "candidates": candidates,
        })

    # Save cache as JSON (safe serialization)
    cache_path = output_dir / "nv_embed_retrieval_cache.json"
    with open(cache_path, "w") as f:
        json.dump(retrieval_cache, f)
    print(f"\nRetrieval cache saved to: {cache_path}")

    # Compute retrieval-only metrics
    metrics_list = []
    for item in retrieval_cache:
        retrieved = [c["sent_uid"] for c in item["candidates"]]
        m = compute_metrics(retrieved, set(item["gold_uids"]), k=10)
        metrics_list.append(m)

    avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print(f"\nRetrieval-only metrics (before reranking):")
    print(f"  nDCG@10: {avg['ndcg_at_10']:.4f}")
    print(f"  Recall@10: {avg['recall_at_10']:.4f}")
    print("=" * 80)


def stage_rerank(args):
    """Stage 2: Apply reranker to cached retrieval results."""
    print("=" * 80)
    print(f"STAGE 2: RERANKING WITH {args.reranker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    output_dir = PROJECT_ROOT / args.output_dir

    # Load retrieval cache
    cache_path = output_dir / "nv_embed_retrieval_cache.json"
    print(f"\n[1/3] Loading retrieval cache from {cache_path}...")
    with open(cache_path) as f:
        retrieval_cache = json.load(f)
    print(f"  Loaded {len(retrieval_cache)} queries")

    # Load reranker
    print(f"\n[2/3] Loading reranker: {args.reranker}...")
    from final_sc_review.reranker.zoo import RerankerZoo
    zoo = RerankerZoo()
    reranker = zoo.get_reranker(args.reranker)
    print("  Reranker loaded!")

    # Rerank
    print(f"\n[3/3] Reranking (top_k_rerank={args.top_k_rerank})...")
    results = []

    for item in tqdm(retrieval_cache, desc="Reranking"):
        query = item["criterion_text"]
        candidates = item["candidates"][:args.top_k_rerank]
        gold_uids = set(item["gold_uids"])

        if not candidates:
            continue

        # Rerank - API expects List[Tuple[sent_uid, text]]
        candidate_tuples = [(c["sent_uid"], c["text"]) for c in candidates]

        reranked = reranker.rerank(query, candidate_tuples)
        reranked_uids = [r.sent_uid for r in reranked]

        # Compute metrics
        m = compute_metrics(reranked_uids, gold_uids, k=10)
        results.append(m)

    # Aggregate
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}

    print(f"\n{'='*80}")
    print("RESULTS: NV-Embed-v2 + " + args.reranker)
    print(f"{'='*80}")
    print(f"  nDCG@10:   {avg['ndcg_at_10']:.4f}")
    print(f"  Recall@10: {avg['recall_at_10']:.4f}")
    print(f"  MRR@10:    {avg['mrr_at_10']:.4f}")
    print(f"  MAP@10:    {avg['map_at_10']:.4f}")
    print(f"{'='*80}")

    # Save results
    result_df = pd.DataFrame([{
        "retriever": "nv-embed-v2",
        "reranker": args.reranker,
        "top_k_retriever": args.top_k_retriever,
        "top_k_rerank": args.top_k_rerank,
        **avg,
        "timestamp": datetime.now().isoformat(),
    }])

    result_path = output_dir / f"nv_embed_{args.reranker}_results.csv"
    result_df.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="NV-Embed-v2 + Reranker")
    parser.add_argument("--stage", type=str, required=True, choices=["retrieve", "rerank"])
    parser.add_argument("--output_dir", type=str, default="outputs/nv_embed_reranker")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--top_k_retriever", type=int, default=100)
    parser.add_argument("--top_k_rerank", type=int, default=10)
    parser.add_argument("--reranker", type=str, default="jina-reranker-v2")

    args = parser.parse_args()

    if args.stage == "retrieve":
        stage_retrieve(args)
    else:
        stage_rerank(args)


if __name__ == "__main__":
    main()
