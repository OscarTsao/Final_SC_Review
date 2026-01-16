"""Base GNN encoder with configurable layer types.

Supports:
- GCN (Graph Convolutional Network)
- GraphSAGE (Sample and Aggregate)
- GAT (Graph Attention Network)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, LayerNorm
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from final_sc_review.gnn.config import GNNType, GNNModelConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class GNNLayer(nn.Module):
    """Single GNN layer with normalization and dropout."""

    def __init__(
        self,
        gnn_type: GNNType,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.3,
        layer_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()

        self.gnn_type = gnn_type
        self.residual = residual and (in_dim == out_dim)
        self.dropout = dropout

        # GNN convolution
        if gnn_type == GNNType.GCN:
            self.conv = GCNConv(in_dim, out_dim)
        elif gnn_type == GNNType.SAGE:
            self.conv = SAGEConv(in_dim, out_dim)
        elif gnn_type == GNNType.GAT:
            # GAT output dim depends on concatenation
            if concat_heads:
                assert out_dim % num_heads == 0, f"out_dim {out_dim} must be divisible by num_heads {num_heads}"
                head_dim = out_dim // num_heads
            else:
                head_dim = out_dim
            self.conv = GATConv(in_dim, head_dim, heads=num_heads, concat=concat_heads, dropout=dropout)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Layer normalization
        self.norm = LayerNorm(out_dim) if layer_norm else None

        # Residual projection if dimensions don't match
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
            self.residual = True
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GNN layer.

        Args:
            x: Node features [n_nodes, in_dim]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge features (optional) [n_edges, edge_dim]

        Returns:
            Updated node features [n_nodes, out_dim]
        """
        # Store input for residual
        identity = x

        # GNN convolution
        if self.gnn_type == GNNType.GAT:
            x = self.conv(x, edge_index)
        else:
            x = self.conv(x, edge_index)

        # Activation
        x = F.elu(x)

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer norm
        if self.norm is not None:
            x = self.norm(x)

        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity

        return x


class BaseGNNEncoder(nn.Module):
    """Base GNN encoder with multiple layers.

    Encodes node features using stacked GNN layers.
    Output can be used for graph-level or node-level tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        num_layers: int = 3,
        gnn_type: GNNType = GNNType.GAT,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.3,
        layer_norm: bool = True,
        residual: bool = True,
    ):
        if not HAS_PYG:
            raise ImportError("torch-geometric required. Install with: pip install torch-geometric")

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = LayerNorm(hidden_dim) if layer_norm else None

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else self.output_dim

            self.layers.append(
                GNNLayer(
                    gnn_type=gnn_type,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    concat_heads=concat_heads,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    residual=residual,
                )
            )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge features (optional) [n_edges, edge_dim]
            batch: Batch assignment (optional) [n_nodes]

        Returns:
            Node embeddings [n_nodes, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return x

    @classmethod
    def from_config(cls, config: GNNModelConfig, input_dim: int) -> "BaseGNNEncoder":
        """Create encoder from config."""
        return cls(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_layers,
            gnn_type=config.gnn_type,
            num_heads=config.num_heads,
            concat_heads=config.concat_heads,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            residual=config.residual,
        )
