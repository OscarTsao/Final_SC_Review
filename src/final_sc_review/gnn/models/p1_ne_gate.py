"""P1: NE Gate GNN - Graph-level classifier for has_evidence.

Architecture:
- Input: Graph per query (nodes = candidates)
- Encoder: 3-layer GAT with attention pooling
- Output: P(has_evidence) logit
- Loss: Focal loss for class imbalance
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from final_sc_review.gnn.config import GNNModelConfig, PoolingType
from final_sc_review.gnn.models.base import BaseGNNEncoder
from final_sc_review.gnn.models.pooling import get_pooling
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class NEGateGNN(nn.Module):
    """Graph-level NE Gate classifier.

    Predicts P(has_evidence) for a query based on its candidate graph.

    Architecture:
    1. Node embedding via GNN encoder (GAT/GCN/SAGE)
    2. Graph pooling (mean/max/attention)
    3. MLP classifier head
    4. Sigmoid output

    Training:
    - Binary classification: has_evidence vs no_evidence
    - Focal loss for class imbalance (~45% positive rate)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling_type: PoolingType = PoolingType.ATTENTION,
        config: Optional[GNNModelConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Use config if provided
        if config is not None:
            self.encoder = BaseGNNEncoder.from_config(config, input_dim)
            hidden_dim = config.hidden_dim
            pooling_type = config.pooling_type
            dropout = config.dropout
        else:
            self.encoder = BaseGNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # Graph pooling
        self.pooling = get_pooling(pooling_type, hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Edge indices [2, n_edges]
            batch: Batch assignment [n_nodes]
            edge_attr: Edge features (optional) [n_edges, edge_dim]

        Returns:
            Logits [n_graphs, 1]
        """
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_attr, batch)

        # Pool to graph level
        graph_emb = self.pooling(node_emb, batch)

        # Classify
        logits = self.classifier(graph_emb)

        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict probabilities.

        Returns:
            Probabilities [n_graphs]
        """
        logits = self.forward(x, edge_index, batch, edge_attr)
        return torch.sigmoid(logits).squeeze(-1)

    def get_graph_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get graph-level embeddings before classification.

        Useful for analysis and visualization.

        Returns:
            Graph embeddings [n_graphs, hidden_dim]
        """
        node_emb = self.encoder(x, edge_index, edge_attr, batch)
        graph_emb = self.pooling(node_emb, batch)
        return graph_emb

    @classmethod
    def from_config(cls, config: GNNModelConfig, input_dim: int) -> "NEGateGNN":
        """Create model from config."""
        return cls(
            input_dim=input_dim,
            config=config,
        )


class NEGateGNNWithNodeFeatures(NEGateGNN):
    """NE Gate GNN that also outputs node-level features.

    Extends NEGateGNN to provide intermediate node representations
    that can be used for Dynamic-K selection or analysis.
    """

    def forward_with_node_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with node features.

        Returns:
            (logits, node_embeddings)
            - logits: [n_graphs, 1]
            - node_embeddings: [n_nodes, hidden_dim]
        """
        node_emb = self.encoder(x, edge_index, edge_attr, batch)
        graph_emb = self.pooling(node_emb, batch)
        logits = self.classifier(graph_emb)

        return logits, node_emb


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw model outputs [N, 1] or [N]
        targets: Binary labels [N]
        alpha: Weight for positive class
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    logits = logits.view(-1)
    targets = targets.view(-1)

    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    loss = focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class NEGateLoss(nn.Module):
    """Loss function for NE Gate training.

    Supports:
    - BCE loss
    - Focal loss
    - Weighted BCE
    """

    def __init__(
        self,
        loss_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: Model outputs [N, 1] or [N]
            targets: Binary labels [N]

        Returns:
            Loss value
        """
        if self.loss_type == "focal":
            return focal_loss(logits, targets, self.focal_alpha, self.focal_gamma)
        elif self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))
        elif self.loss_type == "weighted_bce":
            if self.pos_weight is None:
                # Auto-compute from batch
                n_pos = targets.sum().item()
                n_neg = len(targets) - n_pos
                pos_weight = n_neg / max(n_pos, 1)
            else:
                pos_weight = self.pos_weight
            return F.binary_cross_entropy_with_logits(
                logits.view(-1),
                targets.view(-1),
                pos_weight=torch.tensor(pos_weight, device=logits.device),
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
