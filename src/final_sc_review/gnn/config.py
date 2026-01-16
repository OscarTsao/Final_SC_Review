"""GNN configuration dataclasses.

All hyperparameters for graph construction, GNN architectures, and training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GNNType(str, Enum):
    """GNN layer type."""
    GCN = "gcn"
    SAGE = "sage"
    GAT = "gat"


class PoolingType(str, Enum):
    """Graph pooling type."""
    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"
    SET2SET = "set2set"


class EdgeType(str, Enum):
    """Edge construction strategy."""
    SEMANTIC_KNN = "semantic_knn"
    ADJACENCY = "adjacency"
    FULL = "full"


class DynamicKPolicy(str, Enum):
    """Dynamic-K selection policy."""
    THRESHOLD = "threshold"  # DK-A: p_i >= tau
    MASS = "mass"  # DK-B: cumsum(p) >= gamma
    COMBINED = "combined"  # Both constraints


@dataclass
class GraphConstructionConfig:
    """Configuration for graph construction from embeddings and scores."""

    # Embedding source
    embedding_path: Path = Path("data/cache/bge_m3/dense.npy")
    embedding_dim: int = 1024

    # Edge construction
    edge_types: List[EdgeType] = field(default_factory=lambda: [EdgeType.SEMANTIC_KNN, EdgeType.ADJACENCY])

    # Semantic kNN edges
    knn_k: int = 5
    knn_threshold: float = 0.5  # Minimum cosine similarity

    # Node features (inference-time only - NO gold labels)
    use_embedding: bool = True
    use_reranker_score: bool = True
    use_rank_percentile: bool = True
    use_score_gaps: bool = True
    use_score_stats: bool = True

    # Node feature dimensions (derived)
    @property
    def node_feature_dim(self) -> int:
        dim = 0
        if self.use_embedding:
            dim += self.embedding_dim
        if self.use_reranker_score:
            dim += 1
        if self.use_rank_percentile:
            dim += 1
        if self.use_score_gaps:
            dim += 2  # gap_to_prev, gap_to_next
        if self.use_score_stats:
            dim += 4  # score_zscore, score_minmax, above_mean, below_median
        return dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_path": str(self.embedding_path),
            "embedding_dim": self.embedding_dim,
            "edge_types": [e.value for e in self.edge_types],
            "knn_k": self.knn_k,
            "knn_threshold": self.knn_threshold,
            "use_embedding": self.use_embedding,
            "use_reranker_score": self.use_reranker_score,
            "use_rank_percentile": self.use_rank_percentile,
            "use_score_gaps": self.use_score_gaps,
            "use_score_stats": self.use_score_stats,
            "node_feature_dim": self.node_feature_dim,
        }


@dataclass
class GNNModelConfig:
    """Configuration for GNN model architecture."""

    # Architecture
    gnn_type: GNNType = GNNType.GAT
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3

    # GAT-specific
    num_heads: int = 4
    concat_heads: bool = True

    # Pooling
    pooling_type: PoolingType = PoolingType.ATTENTION

    # Output heads
    ne_gate_output_dim: int = 1  # Binary: has_evidence
    dynamic_k_output_dim: int = 1  # Per-node: P(select)
    reranker_output_dim: int = 1  # Per-node: score adjustment

    # Regularization
    layer_norm: bool = True
    residual: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gnn_type": self.gnn_type.value,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "concat_heads": self.concat_heads,
            "pooling_type": self.pooling_type.value,
            "ne_gate_output_dim": self.ne_gate_output_dim,
            "dynamic_k_output_dim": self.dynamic_k_output_dim,
            "reranker_output_dim": self.reranker_output_dim,
            "layer_norm": self.layer_norm,
            "residual": self.residual,
        }


@dataclass
class GNNTrainingConfig:
    """Configuration for GNN training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Loss
    loss_type: str = "focal"  # focal, bce, weighted_bce
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    pos_weight: Optional[float] = None  # Auto-computed if None

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5

    # Gradient clipping
    grad_clip: float = 1.0

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "loss_type": self.loss_type,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "pos_weight": self.pos_weight,
            "use_scheduler": self.use_scheduler,
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": self.warmup_epochs,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass
class DynamicKConfig:
    """Configuration for Dynamic-K selection."""

    # Hard constraints
    k_min: int = 2
    k_max: int = 10  # Hard cap
    k_max_ratio: float = 0.5  # Max fraction of candidates

    # Policy
    policy: DynamicKPolicy = DynamicKPolicy.THRESHOLD

    # Threshold policy (DK-A)
    threshold_tau: float = 0.5

    # Mass policy (DK-B)
    mass_gamma: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_min": self.k_min,
            "k_max": self.k_max,
            "k_max_ratio": self.k_max_ratio,
            "policy": self.policy.value,
            "threshold_tau": self.threshold_tau,
            "mass_gamma": self.mass_gamma,
        }


@dataclass
class GNNConfig:
    """Master configuration combining all GNN settings."""

    # Sub-configs
    graph: GraphConstructionConfig = field(default_factory=GraphConstructionConfig)
    model: GNNModelConfig = field(default_factory=GNNModelConfig)
    training: GNNTrainingConfig = field(default_factory=GNNTrainingConfig)
    dynamic_k: DynamicKConfig = field(default_factory=DynamicKConfig)

    # Paths
    output_dir: Path = Path("outputs/gnn_research")
    cache_dir: Path = Path("data/cache/gnn")

    # CV settings
    n_folds: int = 5
    inner_train_ratio: float = 0.7  # For nested tuning

    # Experiment tracking
    experiment_name: str = "gnn_ne_dynk"
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "dynamic_k": self.dynamic_k.to_dict(),
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "n_folds": self.n_folds,
            "inner_train_ratio": self.inner_train_ratio,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GNNConfig":
        """Create config from dictionary."""
        graph = GraphConstructionConfig(
            embedding_path=Path(d.get("graph", {}).get("embedding_path", "data/cache/bge_m3/dense.npy")),
            embedding_dim=d.get("graph", {}).get("embedding_dim", 1024),
            edge_types=[EdgeType(e) for e in d.get("graph", {}).get("edge_types", ["semantic_knn", "adjacency"])],
            knn_k=d.get("graph", {}).get("knn_k", 5),
            knn_threshold=d.get("graph", {}).get("knn_threshold", 0.5),
            use_embedding=d.get("graph", {}).get("use_embedding", True),
            use_reranker_score=d.get("graph", {}).get("use_reranker_score", True),
            use_rank_percentile=d.get("graph", {}).get("use_rank_percentile", True),
            use_score_gaps=d.get("graph", {}).get("use_score_gaps", True),
            use_score_stats=d.get("graph", {}).get("use_score_stats", True),
        )

        model = GNNModelConfig(
            gnn_type=GNNType(d.get("model", {}).get("gnn_type", "gat")),
            hidden_dim=d.get("model", {}).get("hidden_dim", 256),
            num_layers=d.get("model", {}).get("num_layers", 3),
            dropout=d.get("model", {}).get("dropout", 0.3),
            num_heads=d.get("model", {}).get("num_heads", 4),
            concat_heads=d.get("model", {}).get("concat_heads", True),
            pooling_type=PoolingType(d.get("model", {}).get("pooling_type", "attention")),
            layer_norm=d.get("model", {}).get("layer_norm", True),
            residual=d.get("model", {}).get("residual", True),
        )

        training = GNNTrainingConfig(
            learning_rate=d.get("training", {}).get("learning_rate", 1e-4),
            weight_decay=d.get("training", {}).get("weight_decay", 1e-5),
            batch_size=d.get("training", {}).get("batch_size", 32),
            max_epochs=d.get("training", {}).get("max_epochs", 100),
            patience=d.get("training", {}).get("patience", 10),
            loss_type=d.get("training", {}).get("loss_type", "focal"),
            focal_alpha=d.get("training", {}).get("focal_alpha", 0.25),
            focal_gamma=d.get("training", {}).get("focal_gamma", 2.0),
            seed=d.get("training", {}).get("seed", 42),
            device=d.get("training", {}).get("device", "cuda"),
        )

        dynamic_k = DynamicKConfig(
            k_min=d.get("dynamic_k", {}).get("k_min", 2),
            k_max=d.get("dynamic_k", {}).get("k_max", 10),
            k_max_ratio=d.get("dynamic_k", {}).get("k_max_ratio", 0.5),
            policy=DynamicKPolicy(d.get("dynamic_k", {}).get("policy", "threshold")),
            threshold_tau=d.get("dynamic_k", {}).get("threshold_tau", 0.5),
            mass_gamma=d.get("dynamic_k", {}).get("mass_gamma", 0.9),
        )

        return cls(
            graph=graph,
            model=model,
            training=training,
            dynamic_k=dynamic_k,
            output_dir=Path(d.get("output_dir", "outputs/gnn_research")),
            cache_dir=Path(d.get("cache_dir", "data/cache/gnn")),
            n_folds=d.get("n_folds", 5),
            inner_train_ratio=d.get("inner_train_ratio", 0.7),
            experiment_name=d.get("experiment_name", "gnn_ne_dynk"),
            run_id=d.get("run_id"),
        )
