"""GNN-based NE detection and Dynamic-K selection module.

This module implements graph neural network approaches for:
1. No-Evidence (NE) detection: Graph-level classification
2. Dynamic-K selection: Node-level scoring for adaptive output size
3. Graph reranking: Score refinement using graph structure
4. Heterogeneous graphs: Cross-criterion reasoning

Architecture:
- Graph per query: nodes = candidates, edges = semantic/adjacency
- Node features: embeddings + reranker scores + rank (NO gold labels)
- Edge features: cosine similarity, sequence distance
- Pooling: mean/max/attention for graph-level tasks

Key Constraints:
- NO label leakage (forbidden: mrr, recall_at_*, gold_rank, is_gold)
- Post-id disjoint 5-fold CV with nested tuning
- Dynamic-K: k_min=2, hard_cap=10, k_max_ratio=0.5
"""

from final_sc_review.gnn.config import (
    GNNConfig,
    GraphConstructionConfig,
    GNNModelConfig,
    GNNTrainingConfig,
    DynamicKConfig,
    GNNType,
    PoolingType,
    EdgeType,
    DynamicKPolicy,
)

__all__ = [
    # Config
    "GNNConfig",
    "GraphConstructionConfig",
    "GNNModelConfig",
    "GNNTrainingConfig",
    "DynamicKConfig",
    # Enums
    "GNNType",
    "PoolingType",
    "EdgeType",
    "DynamicKPolicy",
]
