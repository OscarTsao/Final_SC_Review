"""Loss functions for GNN training.

Supports:
- Focal loss for class imbalance
- Ranking loss for pairwise ordering
- Combined losses for multi-task learning
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        logits: Raw predictions [N]
        targets: Binary labels [N]
        alpha: Weight for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    logits = logits.view(-1)
    targets = targets.view(-1).float()

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
    return loss


def ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 0.5,
    loss_type: str = "margin",
) -> torch.Tensor:
    """Pairwise ranking loss.

    Args:
        pos_scores: Scores for positive items [N_pos]
        neg_scores: Scores for negative items [N_neg]
        margin: Margin for hinge loss
        loss_type: 'margin' (hinge) or 'softplus' (smooth)

    Returns:
        Ranking loss
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=pos_scores.device)

    # All pairs: pos - neg
    diffs = pos_scores[:, None] - neg_scores[None, :]
    diffs = diffs.flatten()

    if loss_type == "margin":
        # Hinge: max(0, margin - diff)
        return F.relu(margin - diffs).mean()
    elif loss_type == "softplus":
        # Smooth: log(1 + exp(-diff))
        return F.softplus(-diffs).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def combined_ne_loss(
    graph_logits: torch.Tensor,
    graph_labels: torch.Tensor,
    node_logits: Optional[torch.Tensor] = None,
    node_labels: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    alpha_graph: float = 1.0,
    alpha_node: float = 0.5,
    alpha_rank: float = 0.3,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    rank_margin: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss for joint NE + Dynamic-K training.

    Components:
    1. Graph-level focal loss (NE detection)
    2. Node-level BCE loss (Dynamic-K)
    3. Ranking loss (gold > non-gold)

    Args:
        graph_logits: Graph predictions [N_graphs]
        graph_labels: Graph labels [N_graphs]
        node_logits: Node predictions [N_nodes] (optional)
        node_labels: Node labels [N_nodes] (optional)
        batch: Batch assignment [N_nodes] (optional)
        alpha_graph: Weight for graph loss
        alpha_node: Weight for node loss
        alpha_rank: Weight for ranking loss
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        rank_margin: Ranking loss margin

    Returns:
        (total_loss, components_dict)
    """
    components = {}

    # Graph-level focal loss
    graph_loss = focal_loss(
        graph_logits.view(-1),
        graph_labels.view(-1),
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    components["graph_focal"] = graph_loss.item()

    total = alpha_graph * graph_loss

    # Node-level losses (if provided)
    if node_logits is not None and node_labels is not None:
        # BCE loss
        node_bce = F.binary_cross_entropy_with_logits(
            node_logits.view(-1),
            node_labels.view(-1),
        )
        components["node_bce"] = node_bce.item()
        total = total + alpha_node * node_bce

        # Ranking loss per graph
        if batch is not None and alpha_rank > 0:
            rank_losses = []
            n_graphs = batch.max().item() + 1

            for g in range(n_graphs):
                mask = batch == g
                logits_g = node_logits[mask].view(-1)
                labels_g = node_labels[mask]

                pos_mask = labels_g > 0.5
                neg_mask = ~pos_mask

                if pos_mask.any() and neg_mask.any():
                    rank_loss = ranking_loss(
                        logits_g[pos_mask],
                        logits_g[neg_mask],
                        margin=rank_margin,
                    )
                    rank_losses.append(rank_loss)

            if rank_losses:
                rank_loss = torch.stack(rank_losses).mean()
                components["rank"] = rank_loss.item()
                total = total + alpha_rank * rank_loss

    components["total"] = total.item()

    return total, components


class WeightedBCELoss(nn.Module):
    """BCE with automatic pos_weight computation."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self._pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._pos_weight is not None:
            pos_weight = torch.tensor(self._pos_weight, device=logits.device)
        else:
            # Auto-compute from batch
            n_pos = targets.sum().item()
            n_neg = len(targets) - n_pos
            pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=logits.device)

        return F.binary_cross_entropy_with_logits(
            logits.view(-1), targets.view(-1), pos_weight=pos_weight
        )


class FocalLoss(nn.Module):
    """Focal loss module."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, targets, self.alpha, self.gamma)
