"""Graph pooling methods for graph-level tasks.

Aggregates node representations into a single graph-level vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from final_sc_review.gnn.config import PoolingType


class GraphPooling(nn.Module, ABC):
    """Abstract base class for graph pooling."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool node features to graph-level representation.

        Args:
            x: Node features [n_nodes, hidden_dim]
            batch: Batch assignment [n_nodes]

        Returns:
            Graph representations [n_graphs, output_dim]
        """
        pass


class MeanPooling(GraphPooling):
    """Mean pooling over all nodes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            # Single graph
            return x.mean(dim=0, keepdim=True)
        return global_mean_pool(x, batch)


class MaxPooling(GraphPooling):
    """Max pooling over all nodes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            return x.max(dim=0, keepdim=True)[0]
        return global_max_pool(x, batch)


class AttentionPooling(GraphPooling):
    """Attention-based pooling with learnable query.

    Computes attention weights over nodes and aggregates.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention query (learnable)
        self.query = nn.Parameter(torch.randn(1, hidden_dim))

        # Attention projection
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temperature for attention
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention-weighted pooling.

        Args:
            x: Node features [n_nodes, hidden_dim]
            batch: Batch assignment [n_nodes]

        Returns:
            Graph representations [n_graphs, hidden_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Get number of graphs
        n_graphs = batch.max().item() + 1

        # Project to keys and values
        keys = self.key_proj(x)  # [n_nodes, hidden_dim]
        values = self.value_proj(x)  # [n_nodes, hidden_dim]

        # Compute attention scores per graph
        graph_outputs = []
        for g in range(n_graphs):
            mask = batch == g
            k = keys[mask]  # [n_nodes_g, hidden_dim]
            v = values[mask]  # [n_nodes_g, hidden_dim]

            # Attention scores: query dot keys
            scores = torch.matmul(self.query, k.T) * self.scale  # [1, n_nodes_g]
            attn = F.softmax(scores, dim=-1)  # [1, n_nodes_g]

            # Weighted sum of values
            out = torch.matmul(attn, v)  # [1, hidden_dim]
            graph_outputs.append(out)

        # Stack all graphs
        output = torch.cat(graph_outputs, dim=0)  # [n_graphs, hidden_dim]

        # Output projection
        output = self.out_proj(output)

        return output


class Set2SetPooling(GraphPooling):
    """Set2Set pooling for order-invariant aggregation.

    Uses LSTM to iteratively attend to nodes.
    """

    def __init__(self, hidden_dim: int, num_steps: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        try:
            from torch_geometric.nn import Set2Set
            self.pooling = Set2Set(hidden_dim, processing_steps=num_steps)
        except ImportError:
            raise ImportError("torch-geometric required for Set2Set pooling")

        # Project back to hidden_dim (Set2Set outputs 2*hidden_dim)
        self.out_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        out = self.pooling(x, batch)  # [n_graphs, 2*hidden_dim]
        out = self.out_proj(out)  # [n_graphs, hidden_dim]
        return out


def get_pooling(pooling_type: PoolingType, hidden_dim: int, **kwargs) -> GraphPooling:
    """Factory function to get pooling layer by type."""
    if pooling_type == PoolingType.MEAN:
        return MeanPooling(hidden_dim)
    elif pooling_type == PoolingType.MAX:
        return MaxPooling(hidden_dim)
    elif pooling_type == PoolingType.ATTENTION:
        return AttentionPooling(hidden_dim, **kwargs)
    elif pooling_type == PoolingType.SET2SET:
        return Set2SetPooling(hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
