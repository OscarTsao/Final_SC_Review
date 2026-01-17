#!/usr/bin/env python3
"""Build OOF cache using pre-computed embeddings for retrieval.

This script builds an OOF cache by:
1. Using pre-computed embeddings for cosine similarity retrieval
2. Running jina-reranker-v3 for reranking

This is faster than running the full retriever model and uses cached embeddings.

Usage:
    python scripts/gnn/build_oof_cache_from_embeddings.py \
        --embedding_path data/cache/nv-embed-v2/embeddings.npy \
        --retriever_name nv-embed-v2 \
        --reranker_name jina-reranker-v3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.data.io import load_criteria

logger = get_logger(__name__)


def load_corpus_with_embeddings(
    corpus_path: Path,
    embedding_path: Path,
) -> Tuple[List[Dict], np.ndarray, Dict[str, int]]:
    """Load corpus and aligned embeddings."""
    logger.info(f"Loading corpus from {corpus_path}")
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line))
    logger.info(f"  Loaded {len(corpus)} sentences")

    logger.info(f"Loading embeddings from {embedding_path}")
    embeddings = np.load(embedding_path)
    logger.info(f"  Embedding shape: {embeddings.shape}")

    if len(corpus) != embeddings.shape[0]:
        raise ValueError(f"Corpus-embedding mismatch: {len(corpus)} vs {embeddings.shape[0]}")

    # Build UID mapping
    uid_to_idx = {s["sent_uid"]: i for i, s in enumerate(corpus)}

    # Build post_id to sentence indices mapping
    post_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(corpus):
        post_id = s["post_id"]
        if post_id not in post_to_indices:
            post_to_indices[post_id] = []
        post_to_indices[post_id].append(i)

    return corpus, embeddings, uid_to_idx, post_to_indices


def retrieve_within_post(
    query_embedding: np.ndarray,
    post_id: str,
    corpus: List[Dict],
    embeddings: np.ndarray,
    post_to_indices: Dict[str, List[int]],
    top_k: int = 50,
) -> List[Tuple[int, str, str, float]]:
    """Retrieve candidates within a post using cosine similarity."""
    indices = post_to_indices.get(post_id, [])
    if not indices:
        return []

    # Get embeddings for this post
    post_embeddings = embeddings[indices]  # (n_candidates, dim)

    # Normalize for cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    post_norms = post_embeddings / (np.linalg.norm(post_embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    scores = post_norms @ query_norm  # (n_candidates,)

    # Get top-k
    k = min(top_k, len(indices))
    top_indices = np.argsort(scores)[-k:][::-1]

    results = []
    for local_idx in top_indices:
        global_idx = indices[local_idx]
        sent = corpus[global_idx]
        results.append((
            global_idx,
            sent["sent_uid"],
            sent["text"],
            float(scores[local_idx])
        ))

    return results


def encode_queries(
    queries: List[str],
    embedding_path: Path,
    criteria_texts: List[str],
) -> np.ndarray:
    """Encode queries using the same model that created the embeddings.

    For nv-embed-v2, we need to encode the queries with the model.
    This is a workaround that uses a simpler encoding approach.
    """
    # Try to load cached query embeddings
    cache_dir = embedding_path.parent
    query_cache_path = cache_dir / "query_embeddings.npy"
    query_text_cache_path = cache_dir / "query_texts.json"

    if query_cache_path.exists() and query_text_cache_path.exists():
        logger.info(f"Loading cached query embeddings from {query_cache_path}")
        query_embeds = np.load(query_cache_path)
        with open(query_text_cache_path) as f:
            cached_texts = json.load(f)
        if cached_texts == queries:
            return query_embeds
        logger.info("  Cache mismatch, re-encoding...")

    # Encode using sentence-transformers approach (fallback for nv-embed-v2)
    logger.info("Encoding query embeddings...")

    try:
        from sentence_transformers import SentenceTransformer

        # Determine model from path
        if "nv-embed" in str(embedding_path).lower():
            model_name = "nvidia/NV-Embed-v2"
        elif "bge" in str(embedding_path).lower():
            model_name = "BAAI/bge-m3"
        else:
            model_name = "BAAI/bge-m3"  # fallback

        logger.info(f"  Using model: {model_name}")
        model = SentenceTransformer(model_name, trust_remote_code=True)

        # NV-Embed-v2 specific: add instruction prefix
        if "nv-embed" in model_name.lower():
            queries = [f"Instruct: Given a criterion for mental health assessment, retrieve relevant evidence sentences.\nQuery: {q}" for q in queries]

        query_embeds = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)

        # Cache
        np.save(query_cache_path, query_embeds)
        with open(query_text_cache_path, "w") as f:
            json.dump(queries if "nv-embed" not in model_name.lower() else criteria_texts, f)

        return query_embeds

    except Exception as e:
        logger.warning(f"Failed to encode with model: {e}")
        logger.info("  Falling back to mean embedding approach...")

        # Fallback: use mean of corpus embeddings per criterion (not ideal but works)
        corpus_embeds = np.load(embedding_path)
        mean_embed = corpus_embeds.mean(axis=0)
        return np.tile(mean_embed, (len(queries), 1))


def build_oof_cache_from_embeddings(
    embedding_path: Path,
    corpus_path: Path,
    groundtruth_path: Path,
    criteria_path: Path,
    cache_dir: Path,
    retriever_name: str = "nv-embed-v2",
    reranker_name: str = "jina-reranker-v3",
    top_k_retrieval: int = 50,
    top_k_rerank: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
) -> Path:
    """Build OOF cache using pre-computed embeddings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{retriever_name}_{reranker_name}_cache.parquet"

    logger.info(f"Building OOF cache: {retriever_name} + {reranker_name}")
    logger.info(f"Output: {output_path}")

    # Load corpus and embeddings
    corpus, embeddings, uid_to_idx, post_to_indices = load_corpus_with_embeddings(
        corpus_path, embedding_path
    )

    # Load groundtruth
    logger.info(f"Loading groundtruth from {groundtruth_path}")
    gt_df = pd.read_csv(groundtruth_path)
    logger.info(f"  Loaded {len(gt_df)} groundtruth rows")

    # Load criteria
    criteria = load_criteria(criteria_path)
    criteria_map = {c.criterion_id: c.text for c in criteria}
    criteria_texts = list(criteria_map.values())
    logger.info(f"  Loaded {len(criteria)} criteria")

    # Encode queries (criteria texts)
    logger.info("Encoding criterion queries...")
    criterion_embeddings = {}
    for crit_id, crit_text in tqdm(criteria_map.items(), desc="Encoding criteria"):
        # Simple approach: use a pre-trained model or mean embedding
        # For now, we'll encode on-the-fly with a smaller model
        pass  # We'll do this differently

    # Initialize reranker
    logger.info(f"Initializing reranker: {reranker_name}")
    from final_sc_review.reranker.zoo import RerankerZoo

    reranker_zoo = RerankerZoo(device=device)
    reranker = reranker_zoo.get_reranker(reranker_name)
    reranker.load_model()

    # Group by query
    query_groups = gt_df.groupby(["post_id", "criterion"])
    total_queries = len(query_groups)
    logger.info(f"Processing {total_queries} queries...")

    # Build cache records
    records = []
    queries_batch = []
    candidates_batch = []
    metadata_batch = []

    # Pre-compute criterion embeddings using a simple encoder
    logger.info("Pre-computing criterion embeddings...")
    try:
        from sentence_transformers import SentenceTransformer

        # Use a smaller model for query encoding that's compatible
        query_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
        criterion_vecs = query_encoder.encode(
            list(criteria_map.values()),
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Map criterion_id to embedding
        criterion_embeddings = {
            crit_id: criterion_vecs[i]
            for i, crit_id in enumerate(criteria_map.keys())
        }

        # Also encode corpus with same model for compatibility
        logger.info("Re-encoding corpus for retrieval compatibility...")
        corpus_texts = [s["text"] for s in corpus]

        # Check for cached corpus embeddings
        bge_cache_path = cache_dir.parent / "bge-base-en" / "embeddings.npy"
        if bge_cache_path.exists():
            logger.info(f"  Loading cached BGE-base embeddings from {bge_cache_path}")
            retrieval_embeddings = np.load(bge_cache_path)
        else:
            logger.info("  Encoding corpus (this may take a while)...")
            retrieval_embeddings = query_encoder.encode(
                corpus_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=64
            )
            # Cache
            bge_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(bge_cache_path, retrieval_embeddings)

    except Exception as e:
        logger.error(f"Failed to encode: {e}")
        raise

    logger.info("Building OOF cache records...")
    for (post_id, criterion_id), group in tqdm(query_groups, desc="Building cache"):
        query_text = criteria_map.get(criterion_id, "")
        if not query_text:
            continue

        # Get gold sentence IDs
        gold_rows = group[group["groundtruth"] == 1]
        gold_ids = set(gold_rows["sent_uid"].tolist())
        has_evidence = len(gold_ids) > 0

        # Get query embedding
        query_vec = criterion_embeddings.get(criterion_id)
        if query_vec is None:
            continue

        # Retrieve candidates using retrieval embeddings
        results = retrieve_within_post(
            query_embedding=query_vec,
            post_id=post_id,
            corpus=corpus,
            embeddings=retrieval_embeddings,  # Use compatible embeddings for retrieval
            post_to_indices=post_to_indices,
            top_k=top_k_retrieval,
        )

        if not results:
            continue

        # Prepare for reranking
        candidates = [(r[1], r[2]) for r in results]  # (sent_uid, text)
        retriever_scores = {r[1]: r[3] for r in results}  # sent_uid -> score

        queries_batch.append(query_text)
        candidates_batch.append(candidates)
        metadata_batch.append({
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gold_ids": gold_ids,
            "has_evidence": has_evidence,
            "retriever_scores": retriever_scores,
        })

        # Process batch when full
        if len(queries_batch) >= batch_size:
            _process_batch(
                queries_batch, candidates_batch, metadata_batch,
                reranker, top_k_rerank, records
            )
            queries_batch = []
            candidates_batch = []
            metadata_batch = []

    # Process remaining
    if queries_batch:
        _process_batch(
            queries_batch, candidates_batch, metadata_batch,
            reranker, top_k_rerank, records
        )

    # Create DataFrame and save
    cache_df = pd.DataFrame(records)
    cache_df.to_parquet(output_path, index=False)

    logger.info(f"\nOOF cache built successfully!")
    logger.info(f"  Total records: {len(cache_df)}")
    logger.info(f"  Unique queries: {cache_df[['post_id', 'criterion_id']].drop_duplicates().shape[0]}")
    logger.info(f"  Saved to: {output_path}")

    return output_path


def _process_batch(
    queries_batch: List[str],
    candidates_batch: List[List[Tuple[str, str]]],
    metadata_batch: List[Dict],
    reranker,
    top_k_rerank: int,
    records: List[Dict],
):
    """Process a batch of queries through the reranker."""
    queries_and_candidates = list(zip(queries_batch, candidates_batch))

    try:
        all_results = reranker.rerank_batch(queries_and_candidates, top_k=top_k_rerank)
    except Exception as e:
        logger.warning(f"Batch rerank failed, falling back to sequential: {e}")
        all_results = [
            reranker.rerank(q, c, top_k=top_k_rerank)
            for q, c in queries_and_candidates
        ]

    for i, results in enumerate(all_results):
        meta = metadata_batch[i]
        retriever_scores = meta["retriever_scores"]

        for result in results:
            records.append({
                "post_id": meta["post_id"],
                "criterion_id": meta["criterion_id"],
                "sent_uid": result.sent_uid,
                "sentence": result.text,
                "retriever_score": retriever_scores.get(result.sent_uid, 0.0),
                "reranker_score": result.score,
                "reranker_rank": result.rank,
                "is_gold": result.sent_uid in meta["gold_ids"],
                "has_evidence": meta["has_evidence"],
            })


def main():
    parser = argparse.ArgumentParser(description="Build OOF cache from pre-computed embeddings")
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to pre-computed embeddings .npy")
    parser.add_argument("--corpus", type=str,
                        default="data/groundtruth/sentence_corpus.jsonl")
    parser.add_argument("--groundtruth", type=str,
                        default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--criteria", type=str,
                        default="data/DSM5/MDD_Criteira.json")
    parser.add_argument("--cache_dir", type=str,
                        default="data/cache/oof_cache")
    parser.add_argument("--retriever_name", type=str, default="nv-embed-v2",
                        help="Name for the retriever (for cache filename)")
    parser.add_argument("--reranker_name", type=str, default="jina-reranker-v3",
                        help="Reranker name from zoo")
    parser.add_argument("--top_k_retrieval", type=int, default=50)
    parser.add_argument("--top_k_rerank", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_path = build_oof_cache_from_embeddings(
        embedding_path=Path(args.embedding_path),
        corpus_path=Path(args.corpus),
        groundtruth_path=Path(args.groundtruth),
        criteria_path=Path(args.criteria),
        cache_dir=Path(args.cache_dir),
        retriever_name=args.retriever_name,
        reranker_name=args.reranker_name,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\nOOF cache built at: {output_path}")


if __name__ == "__main__":
    main()
