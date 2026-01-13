#!/usr/bin/env python3
"""Build retrieval cache for a retriever (standalone script).

This script can run in any conda environment with the retriever's dependencies.
It builds retrieval caches that can be used by hpo_finetuning_combo.py.

Usage:
    # For NV-Embed-v2 (run in nv-embed-v2 env):
    mamba run -n nv-embed-v2 python scripts/build_retrieval_cache.py \
        --retriever nv-embed-v2 --output_dir outputs/hpo_finetuning/nv-embed-v2_jina-reranker-v3

    # For other retrievers (run in main env):
    python scripts/build_retrieval_cache.py \
        --retriever qwen3-embed-4b --output_dir outputs/hpo_finetuning/qwen3-embed-4b_jina-reranker-v3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Sentence:
    """Sentence data."""
    post_id: str
    sid: str
    sent_uid: str
    text: str


@dataclass
class Criterion:
    """Criterion data."""
    criterion_id: str
    text: str


@dataclass
class GroundTruthRow:
    """Groundtruth row."""
    post_id: str
    criterion_id: str
    sent_uid: str
    groundtruth: int


def load_sentences(path: Path) -> List[Sentence]:
    """Load sentence corpus."""
    sentences = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            sentences.append(Sentence(
                post_id=data["post_id"],
                sid=data["sid"],
                sent_uid=data["sent_uid"],
                text=data["text"],
            ))
    return sentences


def load_groundtruth(path: Path) -> List[GroundTruthRow]:
    """Load groundtruth labels."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(GroundTruthRow(
                post_id=row["post_id"],
                criterion_id=row["criterion"],
                sent_uid=row["sent_uid"],
                groundtruth=int(row["groundtruth"]),
            ))
    return rows


def load_criteria(path: Path) -> List[Criterion]:
    """Load criteria from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [Criterion(criterion_id=item["id"], text=item["text"]) for item in data["criteria"]]


def split_post_ids(
    post_ids: List[str],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Dict[str, List[str]]:
    """Split post_ids into train/val/test."""
    rng = random.Random(seed)
    ids = sorted(set(post_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def build_cache_with_nv_embed(
    sentences: List[Sentence],
    queries: List[Dict],
    cache_dir: Path,
    top_k: int = 100,
) -> List[Dict]:
    """Build retrieval cache using NV-Embed-v2."""
    import torch
    from transformers import AutoModel

    print("Loading NV-Embed-v2 model...")
    model = AutoModel.from_pretrained(
        "nvidia/NV-Embed-v2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.cuda()
    model.train(False)
    print("  Model loaded!")

    # Build embeddings cache path
    embeddings_path = cache_dir / "embeddings.npy"

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        corpus_embeddings = np.load(embeddings_path)
    else:
        print(f"Encoding {len(sentences)} sentences...")
        texts = [s.text for s in sentences]

        # Encode in batches
        batch_size = 8
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            embs = model._do_encode(
                batch,
                batch_size=batch_size,
                instruction="",
                max_length=512,
                num_workers=0,
            )
            all_embeddings.append(embs.cpu().numpy())

        corpus_embeddings = np.vstack(all_embeddings)

        # Normalize
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_embeddings = corpus_embeddings / (norms + 1e-8)

        np.save(embeddings_path, corpus_embeddings)
        print(f"Saved embeddings to {embeddings_path}")

    # Build post_to_indices
    post_to_indices = defaultdict(list)
    for idx, sent in enumerate(sentences):
        post_to_indices[sent.post_id].append(idx)

    # Query instruction
    query_instruction = "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

    # Run retrieval
    cache_data = []
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
        indices = post_to_indices.get(q["post_id"], [])
        if not indices:
            cache_data.append({
                "post_id": q["post_id"],
                "criterion_id": q["criterion_id"],
                "criterion_text": q["criterion_text"],
                "gold_uids": q["gold_uids"],
                "is_no_evidence": q.get("is_no_evidence", False),
                "candidates": [],
            })
            continue

        candidate_embs = corpus_embeddings[indices]
        scores = candidate_embs @ query_emb

        # Rank and get top-K
        ranked_indices = np.argsort(-scores)[:top_k]
        candidates = []
        for ri in ranked_indices:
            idx = indices[ri]
            candidates.append({
                "sent_uid": sentences[idx].sent_uid,
                "text": sentences[idx].text,
                "score": float(scores[ri]),
            })

        cache_data.append({
            "post_id": q["post_id"],
            "criterion_id": q["criterion_id"],
            "criterion_text": q["criterion_text"],
            "gold_uids": q["gold_uids"],
            "is_no_evidence": q.get("is_no_evidence", False),
            "candidates": candidates,
        })

    return cache_data


def build_cache_with_dense_retriever(
    retriever_name: str,
    sentences: List[Sentence],
    queries: List[Dict],
    cache_dir: Path,
    top_k: int = 100,
) -> List[Dict]:
    """Build retrieval cache using a dense retriever from RetrieverZoo."""
    from final_sc_review.retriever.zoo import RetrieverZoo

    zoo = RetrieverZoo(
        sentences=sentences,
        cache_dir=cache_dir,
    )
    retriever = zoo.get_retriever(retriever_name)
    retriever.encode_corpus()

    cache_data = []
    for q in tqdm(queries, desc="Retrieving"):
        results = retriever.retrieve_within_post(
            query=q["criterion_text"],
            post_id=q["post_id"],
            top_k=top_k,
        )

        candidates = []
        for r in results:
            candidates.append({
                "sent_uid": r.sent_uid,
                "text": r.text,
                "score": r.score,
            })

        cache_data.append({
            "post_id": q["post_id"],
            "criterion_id": q["criterion_id"],
            "criterion_text": q["criterion_text"],
            "gold_uids": q["gold_uids"],
            "is_no_evidence": q.get("is_no_evidence", False),
            "candidates": candidates,
        })

    return cache_data


def main():
    parser = argparse.ArgumentParser(description="Build retrieval cache")
    parser.add_argument("--retriever", type=str, required=True,
                        help="Retriever name (e.g., nv-embed-v2, qwen3-embed-4b)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cache files")
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "test", "both"],
                        help="Split to cache (default: both train and val)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Top-k candidates from retriever")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Paths
    data_dir = PROJECT_ROOT / "data"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache dir for embeddings
    cache_dir = data_dir / "cache" / args.retriever
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    sentences = load_sentences(data_dir / "groundtruth" / "sentence_corpus.jsonl")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")

    # Split post_ids
    all_post_ids = sorted(set(s.post_id for s in sentences))
    splits = split_post_ids(all_post_ids, seed=args.seed)

    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Build criteria and groundtruth maps
    criteria_map = {c.criterion_id: c.text for c in criteria}
    gt_map = defaultdict(set)
    for row in groundtruth:
        if row.groundtruth == 1:
            gt_map[(row.post_id, row.criterion_id)].add(row.sent_uid)

    # Build post_to_indices
    post_to_indices = defaultdict(list)
    for idx, sent in enumerate(sentences):
        post_to_indices[sent.post_id].append(idx)

    # Determine splits to process
    if args.split == "both":
        splits_to_process = ["train", "val"]
    else:
        splits_to_process = [args.split]

    for split_name in splits_to_process:
        cache_path = output_dir / f"{args.retriever}_{split_name}_cache.json"

        if cache_path.exists():
            print(f"Cache exists: {cache_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Building cache for {split_name} split...")
        print(f"{'='*60}")

        # Build queries for this split
        allowed_posts = set(splits[split_name])
        queries = []

        # Add positive evidence queries
        for (post_id, cid), gold_uids in gt_map.items():
            if post_id not in allowed_posts:
                continue
            if cid not in criteria_map:
                continue
            queries.append({
                "post_id": post_id,
                "criterion_id": cid,
                "criterion_text": criteria_map[cid],
                "gold_uids": list(gold_uids),
            })

        # Add no_evidence queries
        for post_id in allowed_posts:
            if post_id not in post_to_indices:
                continue
            for cid, ctext in criteria_map.items():
                key = (post_id, cid)
                if key not in gt_map:
                    queries.append({
                        "post_id": post_id,
                        "criterion_id": cid,
                        "criterion_text": ctext,
                        "gold_uids": [],
                        "is_no_evidence": True,
                    })

        print(f"Total queries for {split_name}: {len(queries)}")

        # Build cache
        if args.retriever == "nv-embed-v2":
            cache_data = build_cache_with_nv_embed(
                sentences=sentences,
                queries=queries,
                cache_dir=cache_dir,
                top_k=args.top_k,
            )
        else:
            cache_data = build_cache_with_dense_retriever(
                retriever_name=args.retriever,
                sentences=sentences,
                queries=queries,
                cache_dir=cache_dir,
                top_k=args.top_k,
            )

        # Save cache
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
        print(f"Saved cache to {cache_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
