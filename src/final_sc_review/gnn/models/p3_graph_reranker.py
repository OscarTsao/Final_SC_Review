"""P3: Graph Reranker GNN - Score refinement using graph structure.

Architecture:
- Input: reranker scores + candidate graph
- Output: refined_score = alpha * original + (1-alpha) * gnn_adjustment
- Placement: AFTER reranker, BEFORE NE gate

The model learns to adjust reranker scores based on graph context:
- High similarity neighbors with high scores -> boost
- Isolated high-scoring nodes -> potentially reduce
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from final_sc_review.gnn.config import GNNModelConfig
from final_sc_review.gnn.models.base import BaseGNNEncoder
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRerankerGNN(nn.Module):
    """Graph-based score refinement model.

    Refines reranker scores by incorporating graph structure information.
    The output is a convex combination of original and adjusted scores.

    Architecture:
    1. Node encoding via GNN
    2. Score adjustment prediction (per node)
    3. Learnable mixing coefficient (alpha)

    Output: refined_score_i = alpha * original_i + (1-alpha) * f(h_i)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        alpha_init: float = 0.7,
        learn_alpha: bool = True,
        config: Optional[GNNModelConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learn_alpha = learn_alpha

        # GNN encoder
        if config is not None:
            self.encoder = BaseGNNEncoder.from_config(config, input_dim)
            hidden_dim = config.hidden_dim
            dropout = config.dropout
        else:
            self.encoder = BaseGNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # Score adjustment head
        self.adjustment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Mixing coefficient
        if learn_alpha:
            self._alpha_logit = nn.Parameter(torch.tensor(self._logit(alpha_init)))
        else:
            self.register_buffer("_alpha", torch.tensor(alpha_init))

    @staticmethod
    def _logit(p: float) -> float:
        """Convert probability to logit."""
        import math
        p = max(min(p, 0.999), 0.001)
        return math.log(p / (1 - p))

    @property
    def alpha(self) -> torch.Tensor:
        """Get mixing coefficient (0-1)."""
        if self.learn_alpha:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        original_scores: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to get refined scores.

        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Edge indices [2, n_edges]
            original_scores: Original reranker scores [n_nodes]
            batch: Batch assignment [n_nodes]
            edge_attr: Edge features (optional)

        Returns:
            Refined scores [n_nodes]
        """
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_attr, batch)

        # Predict score adjustment
        adjustment = self.adjustment_head(node_emb).squeeze(-1)

        # Combine with original scores
        alpha = self.alpha
        refined = alpha * original_scores + (1 - alpha) * adjustment

        return refined

    def get_adjustments(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get raw adjustments without mixing."""
        node_emb = self.encoder(x, edge_index, edge_attr, batch)
        return self.adjustment_head(node_emb).squeeze(-1)


class GraphRerankerLoss(nn.Module):
    """Loss function for Graph Reranker training.

    Combines:
    - Ranking loss (correct order of refined scores)
    - Alignment loss (refined scores should preserve ordering of gold)
    - Regularization on adjustment magnitude
    """

    def __init__(
        self,
        alpha_rank: float = 1.0,
        alpha_align: float = 0.5,
        alpha_reg: float = 0.1,
        margin: float = 0.1,
    ):
        super().__init__()
        self.alpha_rank = alpha_rank
        self.alpha_align = alpha_align
        self.alpha_reg = alpha_reg
        self.margin = margin

    def forward(
        self,
        refined_scores: torch.Tensor,
        original_scores: torch.Tensor,
        node_labels: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss.

        Args:
            refined_scores: Refined scores from model [n_nodes]
            original_scores: Original reranker scores [n_nodes]
            node_labels: is_gold labels [n_nodes]
            batch: Batch assignment [n_nodes]

        Returns:
            (total_loss, components_dict)
        """
        components = {}

        # Ranking loss: gold nodes should have higher refined scores
        rank_losses = []
        n_graphs = batch.max().item() + 1

        for g in range(n_graphs):
            mask = batch == g
            scores_g = refined_scores[mask]
            labels_g = node_labels[mask]

            gold_mask = labels_g > 0.5
            non_gold_mask = ~gold_mask

            if not gold_mask.any() or not non_gold_mask.any():
                continue

            gold_scores = scores_g[gold_mask]
            non_gold_scores = scores_g[non_gold_mask]

            # Margin ranking: gold > non_gold + margin
            diffs = gold_scores[:, None] - non_gold_scores[None, :]
            loss = F.relu(self.margin - diffs).mean()
            rank_losses.append(loss)

        if rank_losses:
            rank_loss = torch.stack(rank_losses).mean()
        else:
            rank_loss = torch.tensor(0.0, device=refined_scores.device)
        components["rank"] = rank_loss.item()

        # Alignment loss: preserve relative ordering from original
        # (Only adjust, don't completely reorder)
        adjustment = refined_scores - original_scores
        align_loss = (adjustment ** 2).mean()
        components["align"] = align_loss.item()

        # Regularization: penalize large adjustments
        reg_loss = adjustment.abs().mean()
        components["reg"] = reg_loss.item()

        total = (
            self.alpha_rank * rank_loss +
            self.alpha_align * align_loss +
            self.alpha_reg * reg_loss
        )
        components["total"] = total.item()

        return total, components
