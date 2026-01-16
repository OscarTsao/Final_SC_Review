"""GNN model implementations.

Models:
- P1: NEGateGNN - Graph-level classifier for has_evidence
- P2: DynamicKGNN - Node-level scoring for dynamic K selection
- P3: GraphRerankerGNN - Score refinement using graph structure
- P4: HeterogeneousCriterionGNN - Cross-criterion reasoning
"""

from final_sc_review.gnn.models.base import BaseGNNEncoder
from final_sc_review.gnn.models.pooling import (
    GraphPooling,
    MeanPooling,
    MaxPooling,
    AttentionPooling,
    get_pooling,
)
from final_sc_review.gnn.models.p1_ne_gate import NEGateGNN, NEGateLoss
from final_sc_review.gnn.models.p2_dynamic_k import DynamicKGNN, DynamicKLoss, JointNEDynamicKGNN
from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss
from final_sc_review.gnn.models.p4_hetero import HeterogeneousCriterionGNN, HeteroGraphLoss

__all__ = [
    # Base
    "BaseGNNEncoder",
    # Pooling
    "GraphPooling",
    "MeanPooling",
    "MaxPooling",
    "AttentionPooling",
    "get_pooling",
    # P1 NE Gate
    "NEGateGNN",
    "NEGateLoss",
    # P2 Dynamic-K
    "DynamicKGNN",
    "DynamicKLoss",
    "JointNEDynamicKGNN",
    # P3 Graph Reranker
    "GraphRerankerGNN",
    "GraphRerankerLoss",
    # P4 Heterogeneous
    "HeterogeneousCriterionGNN",
    "HeteroGraphLoss",
]
