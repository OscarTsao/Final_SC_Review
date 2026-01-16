"""Graph construction from embeddings and scores.

Builds PyG Data objects for GNN training and inference.
Each query becomes a graph with candidates as nodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None
    InMemoryDataset = None

from final_sc_review.gnn.config import EdgeType, GraphConstructionConfig
from final_sc_review.gnn.graphs.features import (
    NodeFeatureExtractor,
    EdgeFeatureExtractor,
    GraphStatsExtractor,
    assert_no_leakage,
    check_leakage,
)
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """Build PyG graphs from embeddings and reranker scores.

    Each query (post_id, criterion_id) becomes a graph where:
    - Nodes: candidate sentences
    - Node features: embeddings + scores + rank features (NO gold labels)
    - Edges: semantic kNN + adjacency edges
    - Edge features: cosine similarity + sequence distance

    IMPORTANT: No gold labels are used in graph construction.
    Labels (has_evidence) are only attached for training, not as node features.
    """

    def __init__(self, config: GraphConstructionConfig):
        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required for GNN. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )

        self.config = config
        self.node_extractor = NodeFeatureExtractor(
            use_embedding=config.use_embedding,
            use_reranker_score=config.use_reranker_score,
            use_rank_percentile=config.use_rank_percentile,
            use_score_gaps=config.use_score_gaps,
            use_score_stats=config.use_score_stats,
            embedding_dim=config.embedding_dim,
        )
        self.edge_extractor = EdgeFeatureExtractor()
        self.stats_extractor = GraphStatsExtractor()

        # Embedding cache
        self._embedding_cache: Optional[np.ndarray] = None
        self._uid_to_idx: Optional[Dict[str, int]] = None

    def load_embeddings(
        self,
        embedding_path: Optional[Path] = None,
        uid_mapping_path: Optional[Path] = None,
    ) -> None:
        """Load precomputed embeddings from disk.

        Args:
            embedding_path: Path to embedding .npy file
            uid_mapping_path: Path to UID-to-index mapping JSON
        """
        if embedding_path is None:
            embedding_path = self.config.embedding_path

        if uid_mapping_path is None:
            # Default: same directory as embeddings
            uid_mapping_path = embedding_path.parent / "uid_to_idx.json"

        logger.info(f"Loading embeddings from {embedding_path}")
        self._embedding_cache = np.load(embedding_path)
        logger.info(f"  Loaded embeddings: {self._embedding_cache.shape}")

        if uid_mapping_path.exists():
            logger.info(f"Loading UID mapping from {uid_mapping_path}")
            with open(uid_mapping_path) as f:
                self._uid_to_idx = json.load(f)
            logger.info(f"  Loaded {len(self._uid_to_idx)} UID mappings")
        else:
            logger.warning(f"No UID mapping found at {uid_mapping_path}")
            self._uid_to_idx = {}

    def get_embeddings(self, uids: List[str]) -> np.ndarray:
        """Get embeddings for a list of UIDs.

        Args:
            uids: List of sentence UIDs

        Returns:
            Embeddings array [n_uids, embedding_dim]
        """
        if self._embedding_cache is None:
            raise RuntimeError("Embeddings not loaded. Call load_embeddings() first.")

        indices = []
        for uid in uids:
            if uid in self._uid_to_idx:
                indices.append(self._uid_to_idx[uid])
            else:
                # Return zero embedding for unknown UIDs
                indices.append(-1)

        embeddings = []
        for idx in indices:
            if idx >= 0:
                embeddings.append(self._embedding_cache[idx])
            else:
                embeddings.append(np.zeros(self.config.embedding_dim))

        return np.stack(embeddings)

    def build_edges_semantic_knn(
        self,
        embeddings: np.ndarray,
        k: int,
        threshold: float,
    ) -> np.ndarray:
        """Build semantic kNN edges based on cosine similarity.

        Args:
            embeddings: Node embeddings [n_nodes, dim]
            k: Number of nearest neighbors
            threshold: Minimum cosine similarity

        Returns:
            Edge index array [2, n_edges]
        """
        n = len(embeddings)
        if n <= 1:
            return np.array([[], []], dtype=np.int64)

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Build kNN edges
        edges_src = []
        edges_dst = []

        for i in range(n):
            # Get top-k neighbors (excluding self)
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf  # Exclude self

            # Get indices of top-k
            top_k_idx = np.argsort(-sims)[:k]

            for j in top_k_idx:
                if sims[j] >= threshold:
                    edges_src.append(i)
                    edges_dst.append(j)

        return np.array([edges_src, edges_dst], dtype=np.int64)

    def build_edges_adjacency(
        self,
        sentence_ids: List[int],
    ) -> np.ndarray:
        """Build adjacency edges based on sentence order.

        Connects consecutive sentences within the same post.

        Args:
            sentence_ids: Sentence indices for each candidate

        Returns:
            Edge index array [2, n_edges]
        """
        n = len(sentence_ids)
        if n <= 1:
            return np.array([[], []], dtype=np.int64)

        # Sort by sentence ID and connect consecutive
        sorted_idx = np.argsort(sentence_ids)

        edges_src = []
        edges_dst = []

        for i in range(len(sorted_idx) - 1):
            curr_idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]

            # Only connect if actually adjacent (sid difference = 1)
            if abs(sentence_ids[curr_idx] - sentence_ids[next_idx]) <= 1:
                # Bidirectional edges
                edges_src.extend([curr_idx, next_idx])
                edges_dst.extend([next_idx, curr_idx])

        return np.array([edges_src, edges_dst], dtype=np.int64)

    def build_graph(
        self,
        candidate_uids: List[str],
        reranker_scores: np.ndarray,
        sentence_ids: List[int],
        query_id: str,
        has_evidence: Optional[bool] = None,  # Label for training (NOT used as feature)
        node_labels: Optional[np.ndarray] = None,  # Per-node labels for training
    ) -> "Data":
        """Build a PyG Data object for a single query.

        Args:
            candidate_uids: List of candidate sentence UIDs
            reranker_scores: Reranker scores for each candidate
            sentence_ids: Sentence indices within post
            query_id: Unique query identifier
            has_evidence: Graph-level label (for training only, NOT a feature)
            node_labels: Per-node labels (for training only, NOT features)

        Returns:
            PyG Data object
        """
        n_nodes = len(candidate_uids)
        if n_nodes == 0:
            raise ValueError(f"Cannot build graph with 0 candidates for query {query_id}")

        # Get embeddings
        embeddings = self.get_embeddings(candidate_uids)

        # Compute ranks from scores
        ranks = np.argsort(np.argsort(-reranker_scores))

        # Extract node features (NO gold labels!)
        node_features = self.node_extractor.extract(
            embeddings=embeddings,
            reranker_scores=reranker_scores,
            ranks=ranks,
        )

        # Build edges
        edge_indices = []

        if EdgeType.SEMANTIC_KNN in self.config.edge_types:
            knn_edges = self.build_edges_semantic_knn(
                embeddings, self.config.knn_k, self.config.knn_threshold
            )
            edge_indices.append(knn_edges)

        if EdgeType.ADJACENCY in self.config.edge_types:
            adj_edges = self.build_edges_adjacency(sentence_ids)
            edge_indices.append(adj_edges)

        if EdgeType.FULL in self.config.edge_types:
            # Full graph (all pairs)
            src = []
            dst = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            full_edges = np.array([src, dst], dtype=np.int64)
            edge_indices.append(full_edges)

        # Combine edge indices
        if edge_indices:
            edge_index = np.hstack(edge_indices)
            # Remove duplicates
            edge_set = set(zip(edge_index[0], edge_index[1]))
            edge_index = np.array([[s, d] for s, d in edge_set], dtype=np.int64).T
            if edge_index.size == 0:
                edge_index = np.array([[], []], dtype=np.int64)
        else:
            edge_index = np.array([[], []], dtype=np.int64)

        # Extract edge features
        edge_features = self.edge_extractor.extract_batch(
            embeddings, edge_index, sentence_ids
        )

        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32) if len(edge_features) > 0 else None,
        )

        # Store metadata
        data.query_id = query_id
        data.candidate_uids = candidate_uids
        data.reranker_scores = torch.tensor(reranker_scores, dtype=torch.float32)
        data.n_candidates = n_nodes

        # Add labels for training (NOT used as features)
        if has_evidence is not None:
            data.y = torch.tensor([int(has_evidence)], dtype=torch.float32)

        if node_labels is not None:
            data.node_labels = torch.tensor(node_labels, dtype=torch.float32)

        return data

    def build_graphs_from_feature_store(
        self,
        feature_df,  # pandas DataFrame from feature store
        include_labels: bool = True,
    ) -> List["Data"]:
        """Build graphs for all queries in a feature store DataFrame.

        Args:
            feature_df: DataFrame with columns:
                - query_id, post_id, criterion_id
                - candidate_ids (JSON list)
                - candidate_scores (JSON list)
                - has_evidence (label, optional)
            include_labels: Whether to include labels in graphs

        Returns:
            List of PyG Data objects
        """
        graphs = []

        for _, row in feature_df.iterrows():
            query_id = row["query_id"]
            candidate_uids = json.loads(row["candidate_ids"])
            scores = np.array(json.loads(row["candidate_scores"]))

            # Extract sentence IDs from UIDs (format: post_id_sid)
            sentence_ids = []
            for uid in candidate_uids:
                try:
                    sid = int(uid.split("_")[-1])
                except ValueError:
                    sid = 0
                sentence_ids.append(sid)

            has_evidence = row.get("has_evidence") if include_labels else None

            try:
                graph = self.build_graph(
                    candidate_uids=candidate_uids,
                    reranker_scores=scores,
                    sentence_ids=sentence_ids,
                    query_id=query_id,
                    has_evidence=has_evidence,
                )
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to build graph for {query_id}: {e}")
                continue

        return graphs

    def extract_graph_stats(
        self,
        graph: "Data",
    ) -> Dict[str, float]:
        """Extract graph statistics for baseline models.

        Args:
            graph: PyG Data object

        Returns:
            Dictionary of graph statistics
        """
        # Convert to numpy
        edge_index = graph.edge_index.numpy() if graph.edge_index.numel() > 0 else np.array([[], []])
        n_nodes = graph.x.shape[0]
        scores = graph.reranker_scores.numpy()

        # Get embeddings if available (first embedding_dim features)
        if self.config.use_embedding:
            embeddings = graph.x[:, :self.config.embedding_dim].numpy()
        else:
            embeddings = None

        return self.stats_extractor.extract(
            n_nodes=n_nodes,
            edge_index=edge_index,
            node_scores=scores,
            embeddings=embeddings,
        )


class QueryGraphDataset(InMemoryDataset if HAS_PYG else object):
    """PyG InMemoryDataset for query graphs.

    Stores all graphs in memory for efficient training.
    Supports fold-based splitting for cross-validation.
    """

    def __init__(
        self,
        root: str,
        graphs: Optional[List["Data"]] = None,
        transform=None,
        pre_transform=None,
    ):
        self._graphs = graphs or []
        super().__init__(root, transform, pre_transform)

        if self._graphs:
            self.data, self.slices = self.collate(self._graphs)
        else:
            # Try to load processed data
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        if self._graphs:
            data, slices = self.collate(self._graphs)
            torch.save((data, slices), self.processed_paths[0])

    def save(self, path: Optional[Path] = None):
        """Save dataset to disk."""
        if path is None:
            path = Path(self.processed_paths[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((self.data, self.slices), path)
        logger.info(f"Saved dataset with {len(self)} graphs to {path}")

    @classmethod
    def load(cls, path: Path, transform=None) -> "QueryGraphDataset":
        """Load dataset from disk."""
        data, slices = torch.load(path)
        dataset = cls(str(path.parent), transform=transform)
        dataset.data = data
        dataset.slices = slices
        return dataset

    def get_fold_indices(self, fold_ids: np.ndarray, fold: int) -> Tuple[List[int], List[int]]:
        """Get train/val indices for a fold.

        Args:
            fold_ids: Fold ID for each graph
            fold: Target fold (used as validation)

        Returns:
            (train_indices, val_indices)
        """
        val_mask = fold_ids == fold
        train_mask = ~val_mask

        train_idx = np.where(train_mask)[0].tolist()
        val_idx = np.where(val_mask)[0].tolist()

        return train_idx, val_idx
