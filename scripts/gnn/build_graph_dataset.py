#!/usr/bin/env python3
"""Build PyG graph dataset from feature store / OOF cache.

This script:
1. Loads the OOF cache with reranker scores and candidate info
2. Loads pre-computed BGE-M3 embeddings
3. Builds PyG Data objects for each query (candidates as nodes)
4. Saves datasets per fold for 5-fold CV

Output: data/cache/gnn/<timestamp>/
    - fold_0.pt, fold_1.pt, ..., fold_4.pt
    - metadata.json
    - uid_to_idx.json (embedding index mapping)

Usage:
    python scripts/gnn/build_graph_dataset.py \
        --cache_path data/cache/oof_cache/bge-m3_jina-reranker-v3_cache.parquet \
        --embedding_path data/cache/bge_m3/dense.npy \
        --output_dir data/cache/gnn
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
from final_sc_review.gnn.config import GraphConstructionConfig, EdgeType
from final_sc_review.gnn.graphs.features import check_leakage, assert_no_leakage

logger = get_logger(__name__)


def load_oof_cache(cache_path: Path) -> pd.DataFrame:
    """Load OOF cache with reranker scores."""
    logger.info(f"Loading OOF cache from {cache_path}")
    df = pd.read_parquet(cache_path)
    logger.info(f"  Loaded {len(df)} candidate records")
    logger.info(f"  Columns: {list(df.columns)}")
    return df


def build_uid_mapping(corpus_path: Path) -> Dict[str, int]:
    """Build UID to embedding index mapping from sentence corpus."""
    logger.info(f"Building UID mapping from {corpus_path}")
    uid_to_idx = {}

    with open(corpus_path) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            uid_to_idx[data["sent_uid"]] = idx

    logger.info(f"  Mapped {len(uid_to_idx)} UIDs")
    return uid_to_idx


def create_fold_assignments(
    post_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, int]:
    """Assign post IDs to folds (post-id disjoint)."""
    np.random.seed(seed)
    shuffled = list(post_ids)
    np.random.shuffle(shuffled)

    fold_size = len(shuffled) // n_folds
    post_to_fold = {}

    for i, post_id in enumerate(shuffled):
        fold = min(i // fold_size, n_folds - 1)
        post_to_fold[post_id] = fold

    return post_to_fold


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Verify no label-derived columns are used as features."""
    # Check column names
    leaked = check_leakage({col: 0 for col in df.columns})

    # Filter out columns that are labels (stored separately, not features)
    label_cols = {"has_evidence", "is_gold", "groundtruth", "n_gold_sentences"}
    leaked_features = [l for l in leaked if l not in label_cols]

    if leaked_features:
        raise ValueError(
            f"Label leakage detected in cache columns: {leaked_features}. "
            f"These cannot be used as features."
        )

    logger.info("  Leakage check passed")


def build_graph_dataset(
    cache_path: Path,
    embedding_path: Path,
    corpus_path: Path,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
    knn_k: int = 5,
    knn_threshold: float = 0.5,
    top_k_candidates: int = 20,
) -> Path:
    """Build PyG graph dataset from OOF cache.

    Args:
        cache_path: Path to OOF cache parquet
        embedding_path: Path to dense embeddings .npy
        corpus_path: Path to sentence corpus JSONL
        output_dir: Output directory for graphs
        n_folds: Number of CV folds
        seed: Random seed for fold assignment
        knn_k: k for semantic kNN edges
        knn_threshold: Minimum cosine similarity for kNN edges
        top_k_candidates: Max candidates per query

    Returns:
        Path to output directory
    """
    # Import PyG
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "torch-geometric required. Install with: "
            "pip install torch torch-geometric torch-scatter torch-sparse"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / timestamp
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building graph dataset in {out_path}")

    # Load data
    cache_df = load_oof_cache(cache_path)
    verify_no_leakage(cache_df)

    # Load embeddings
    logger.info(f"Loading embeddings from {embedding_path}")
    embeddings = np.load(embedding_path)
    logger.info(f"  Embedding shape: {embeddings.shape}")

    # Build UID mapping
    uid_to_idx = build_uid_mapping(corpus_path)

    # Save UID mapping
    uid_map_path = out_path / "uid_to_idx.json"
    with open(uid_map_path, "w") as f:
        json.dump(uid_to_idx, f)
    logger.info(f"  Saved UID mapping to {uid_map_path}")

    # Get unique post IDs and assign folds
    post_ids = cache_df["post_id"].unique().tolist()
    post_to_fold = create_fold_assignments(post_ids, n_folds, seed)
    logger.info(f"  Assigned {len(post_ids)} posts to {n_folds} folds")

    # Group by query
    query_groups = cache_df.groupby(["post_id", "criterion_id"])
    logger.info(f"  Processing {len(query_groups)} queries")

    # Build graphs per fold
    fold_graphs: Dict[int, List] = {i: [] for i in range(n_folds)}
    fold_stats: Dict[int, Dict] = {i: {"n_queries": 0, "n_has_evidence": 0, "n_candidates": 0} for i in range(n_folds)}

    from final_sc_review.gnn.config import GraphConstructionConfig
    from final_sc_review.gnn.graphs.builder import GraphBuilder

    config = GraphConstructionConfig(
        embedding_path=embedding_path,
        embedding_dim=embeddings.shape[1],
        edge_types=[EdgeType.SEMANTIC_KNN, EdgeType.ADJACENCY],
        knn_k=knn_k,
        knn_threshold=knn_threshold,
        use_embedding=True,
        use_reranker_score=True,
        use_rank_percentile=True,
        use_score_gaps=True,
        use_score_stats=True,
    )

    builder = GraphBuilder(config)
    builder._embedding_cache = embeddings
    builder._uid_to_idx = uid_to_idx

    for (post_id, criterion_id), group in tqdm(query_groups, desc="Building graphs"):
        fold_id = post_to_fold[post_id]

        # Get candidates (top-k by reranker score)
        sorted_group = group.sort_values("reranker_score", ascending=False).head(top_k_candidates)

        candidate_uids = sorted_group["sent_uid"].tolist()
        reranker_scores = sorted_group["reranker_score"].values.astype(np.float32)

        # Extract sentence IDs from UIDs
        sentence_ids = []
        for uid in candidate_uids:
            try:
                sid = int(uid.split("_")[-1])
            except ValueError:
                sid = 0
            sentence_ids.append(sid)

        # Determine has_evidence label
        has_evidence = sorted_group["has_evidence"].iloc[0] if "has_evidence" in sorted_group.columns else False

        # Build per-node labels (for Dynamic-K training)
        node_labels = None
        if "is_gold" in sorted_group.columns:
            node_labels = sorted_group["is_gold"].values.astype(np.float32)

        query_id = f"{post_id}_{criterion_id}"

        try:
            graph = builder.build_graph(
                candidate_uids=candidate_uids,
                reranker_scores=reranker_scores,
                sentence_ids=sentence_ids,
                query_id=query_id,
                has_evidence=bool(has_evidence),
                node_labels=node_labels,
            )

            # Add fold info
            graph.fold_id = fold_id
            graph.post_id = post_id
            graph.criterion_id = criterion_id

            fold_graphs[fold_id].append(graph)

            # Update stats
            fold_stats[fold_id]["n_queries"] += 1
            fold_stats[fold_id]["n_candidates"] += len(candidate_uids)
            if has_evidence:
                fold_stats[fold_id]["n_has_evidence"] += 1

        except Exception as e:
            logger.warning(f"Failed to build graph for {query_id}: {e}")
            continue

    # Save per-fold datasets
    for fold_id, graphs in fold_graphs.items():
        fold_path = out_path / f"fold_{fold_id}.pt"

        # Collate graphs
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(graphs)

        torch.save({
            "data": batch,
            "graphs": graphs,
            "n_graphs": len(graphs),
        }, fold_path)

        logger.info(
            f"  Fold {fold_id}: {len(graphs)} graphs, "
            f"{fold_stats[fold_id]['n_has_evidence']} has_evidence "
            f"({fold_stats[fold_id]['n_has_evidence']/max(len(graphs),1)*100:.1f}%)"
        )

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "n_folds": n_folds,
        "seed": seed,
        "total_queries": sum(s["n_queries"] for s in fold_stats.values()),
        "total_candidates": sum(s["n_candidates"] for s in fold_stats.values()),
        "has_evidence_rate": sum(s["n_has_evidence"] for s in fold_stats.values()) / max(sum(s["n_queries"] for s in fold_stats.values()), 1),
        "top_k_candidates": top_k_candidates,
        "knn_k": knn_k,
        "knn_threshold": knn_threshold,
        "embedding_dim": embeddings.shape[1],
        "fold_stats": fold_stats,
        "config": config.to_dict(),
        "source_cache": str(cache_path),
        "source_embeddings": str(embedding_path),
    }

    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total graphs: {metadata['total_queries']}")
    logger.info(f"Has evidence rate: {metadata['has_evidence_rate']:.2%}")
    logger.info(f"Avg candidates/query: {metadata['total_candidates']/max(metadata['total_queries'],1):.1f}")
    logger.info(f"Output: {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build PyG graph dataset")
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/cache/oof_cache/bge-m3_jina-reranker-v3_cache.parquet",
        help="Path to OOF cache parquet",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="data/cache/bge_m3/dense.npy",
        help="Path to dense embeddings .npy",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/groundtruth/sentence_corpus.jsonl",
        help="Path to sentence corpus JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/gnn",
        help="Output directory for graphs",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--knn_k", type=int, default=5, help="k for semantic kNN edges")
    parser.add_argument("--knn_threshold", type=float, default=0.5, help="Min cosine sim for kNN")
    parser.add_argument("--top_k", type=int, default=20, help="Max candidates per query")
    args = parser.parse_args()

    out_path = build_graph_dataset(
        cache_path=Path(args.cache_path),
        embedding_path=Path(args.embedding_path),
        corpus_path=Path(args.corpus_path),
        output_dir=Path(args.output_dir),
        n_folds=args.n_folds,
        seed=args.seed,
        knn_k=args.knn_k,
        knn_threshold=args.knn_threshold,
        top_k_candidates=args.top_k,
    )

    print(f"\nGraph dataset created at: {out_path}")


if __name__ == "__main__":
    main()
