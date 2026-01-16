"""Graph construction infrastructure for GNN models.

This module handles:
- Building graphs from embeddings and reranker scores
- Node feature extraction (inference-time only, NO gold labels)
- Edge construction (semantic kNN + adjacency)
- PyG data object creation
"""

from final_sc_review.gnn.graphs.builder import GraphBuilder
from final_sc_review.gnn.graphs.features import NodeFeatureExtractor, EdgeFeatureExtractor

__all__ = [
    "GraphBuilder",
    "NodeFeatureExtractor",
    "EdgeFeatureExtractor",
]
