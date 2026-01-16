"""P2: Dynamic-K GNN - Node-level scoring for dynamic K selection.

Architecture:
- Input: Graph per query (nodes = candidates)
- Encoder: Shared GNN encoder with P1
- Output: Per-node P(select) logits
- K selection: threshold or mass-based policies

Dynamic-K Policies:
- DK-A: threshold (p_i >= tau)
- DK-B: mass (cumsum(p) >= gamma)
- Combined: both constraints

Constraints:
- k_min = 2 (always return at least 2)
- k_max = 10 (hard cap)
- k_max_ratio = 0.5 (at most 50% of candidates)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from final_sc_review.gnn.config import DynamicKConfig, DynamicKPolicy, GNNModelConfig
from final_sc_review.gnn.models.base import BaseGNNEncoder
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class DynamicKGNN(nn.Module):
    """Node-level scoring model for Dynamic-K selection.

    Predicts P(select) for each candidate node, then applies
    a selection policy to determine how many candidates to return.

    Architecture:
    1. Node embedding via shared GNN encoder
    2. Node-level MLP classifier
    3. Dynamic-K selection policy

    Training:
    - Node-level binary classification: is_gold
    - Ranking loss to ensure gold > non-gold
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        k_config: Optional[DynamicKConfig] = None,
        model_config: Optional[GNNModelConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_config = k_config or DynamicKConfig()

        # GNN encoder
        if model_config is not None:
            self.encoder = BaseGNNEncoder.from_config(model_config, input_dim)
            hidden_dim = model_config.hidden_dim
            dropout = model_config.dropout
        else:
            self.encoder = BaseGNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # Node-level classifier
        self.node_classifier = nn.Sequential(
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
        """Forward pass to get node-level logits.

        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Edge indices [2, n_edges]
            batch: Batch assignment [n_nodes]
            edge_attr: Edge features (optional)

        Returns:
            Node logits [n_nodes, 1]
        """
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_attr, batch)

        # Node-level classification
        logits = self.node_classifier(node_emb)

        return logits

    def predict_node_probs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict per-node probabilities.

        Returns:
            Node probabilities [n_nodes]
        """
        logits = self.forward(x, edge_index, batch, edge_attr)
        return torch.sigmoid(logits).squeeze(-1)

    def select_k(
        self,
        node_probs: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        reranker_scores: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Select K for each graph using configured policy.

        Args:
            node_probs: Node probabilities [n_nodes]
            batch: Batch assignment [n_nodes]
            reranker_scores: Original reranker scores for tiebreaking

        Returns:
            List of K values, one per graph
        """
        if batch is None:
            batch = torch.zeros(len(node_probs), dtype=torch.long, device=node_probs.device)

        n_graphs = batch.max().item() + 1
        k_values = []

        for g in range(n_graphs):
            mask = batch == g
            probs = node_probs[mask].cpu().numpy()
            n_candidates = len(probs)

            # Apply policy
            k = self._select_k_single(probs, n_candidates)
            k_values.append(k)

        return k_values

    def _select_k_single(self, probs: np.ndarray, n_candidates: int) -> int:
        """Select K for a single graph."""
        cfg = self.k_config

        # Compute k_max based on constraints
        k_max_from_ratio = int(n_candidates * cfg.k_max_ratio)
        k_max = min(cfg.k_max, k_max_from_ratio, n_candidates)
        k_max = max(k_max, cfg.k_min)  # Ensure at least k_min

        # Sort by probability (descending)
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]

        if cfg.policy == DynamicKPolicy.THRESHOLD:
            # DK-A: Count probs >= tau
            k = np.sum(sorted_probs >= cfg.threshold_tau)
        elif cfg.policy == DynamicKPolicy.MASS:
            # DK-B: Find smallest k such that cumsum(p) >= gamma
            cumsum = np.cumsum(sorted_probs)
            k = np.searchsorted(cumsum, cfg.mass_gamma) + 1
        elif cfg.policy == DynamicKPolicy.COMBINED:
            # Both: threshold AND mass
            k_thresh = np.sum(sorted_probs >= cfg.threshold_tau)
            cumsum = np.cumsum(sorted_probs)
            k_mass = np.searchsorted(cumsum, cfg.mass_gamma) + 1
            k = min(k_thresh, k_mass)
        else:
            raise ValueError(f"Unknown policy: {cfg.policy}")

        # Apply constraints
        k = max(cfg.k_min, min(k, k_max))

        return int(k)

    def get_selected_candidates(
        self,
        node_probs: torch.Tensor,
        batch: torch.Tensor,
        candidate_uids: List[List[str]],
    ) -> List[List[str]]:
        """Get selected candidate UIDs for each graph.

        Args:
            node_probs: Node probabilities [n_nodes]
            batch: Batch assignment [n_nodes]
            candidate_uids: List of candidate UID lists, one per graph

        Returns:
            List of selected UID lists
        """
        k_values = self.select_k(node_probs, batch)
        n_graphs = batch.max().item() + 1

        selected = []
        for g in range(n_graphs):
            mask = batch == g
            probs = node_probs[mask].cpu().numpy()
            uids = candidate_uids[g]
            k = k_values[g]

            # Sort by prob and take top-k
            sorted_idx = np.argsort(-probs)[:k]
            selected.append([uids[i] for i in sorted_idx])

        return selected


class DynamicKLoss(nn.Module):
    """Loss function for Dynamic-K training.

    Combines:
    - Node-level BCE for is_gold prediction
    - Ranking loss to ensure gold > non-gold scores
    """

    def __init__(
        self,
        alpha_bce: float = 1.0,
        alpha_rank: float = 0.5,
        margin: float = 0.5,
    ):
        super().__init__()
        self.alpha_bce = alpha_bce
        self.alpha_rank = alpha_rank
        self.margin = margin

    def forward(
        self,
        node_logits: torch.Tensor,
        node_labels: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            node_logits: Node predictions [n_nodes, 1]
            node_labels: Node labels (is_gold) [n_nodes]
            batch: Batch assignment [n_nodes]

        Returns:
            (total_loss, loss_components)
        """
        node_logits = node_logits.view(-1)

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(node_logits, node_labels)

        # Ranking loss (per graph)
        rank_losses = []
        n_graphs = batch.max().item() + 1

        for g in range(n_graphs):
            mask = batch == g
            logits_g = node_logits[mask]
            labels_g = node_labels[mask]

            # Get gold and non-gold indices
            gold_mask = labels_g > 0.5
            non_gold_mask = ~gold_mask

            if not gold_mask.any() or not non_gold_mask.any():
                continue

            gold_logits = logits_g[gold_mask]
            non_gold_logits = logits_g[non_gold_mask]

            # Margin ranking: gold should be > non_gold + margin
            # Loss = max(0, margin - (gold - non_gold))
            # Use all pairs or sampled pairs
            n_gold = gold_mask.sum().item()
            n_non_gold = non_gold_mask.sum().item()

            if n_gold * n_non_gold > 100:
                # Sample pairs
                idx_gold = torch.randint(n_gold, (100,), device=logits_g.device)
                idx_non_gold = torch.randint(n_non_gold, (100,), device=logits_g.device)
                diffs = gold_logits[idx_gold] - non_gold_logits[idx_non_gold]
            else:
                # All pairs
                diffs = gold_logits[:, None] - non_gold_logits[None, :]
                diffs = diffs.flatten()

            rank_loss = F.relu(self.margin - diffs).mean()
            rank_losses.append(rank_loss)

        if rank_losses:
            rank_loss = torch.stack(rank_losses).mean()
        else:
            rank_loss = torch.tensor(0.0, device=node_logits.device)

        # Total loss
        total = self.alpha_bce * bce_loss + self.alpha_rank * rank_loss

        components = {
            "bce": bce_loss.item(),
            "rank": rank_loss.item(),
            "total": total.item(),
        }

        return total, components


class JointNEDynamicKGNN(nn.Module):
    """Joint model for NE detection and Dynamic-K selection.

    Shares the GNN encoder between both tasks:
    - Graph-level output for has_evidence
    - Node-level output for dynamic K

    This enables end-to-end training and efficient inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        model_config: Optional[GNNModelConfig] = None,
    ):
        super().__init__()

        # Shared encoder
        if model_config is not None:
            self.encoder = BaseGNNEncoder.from_config(model_config, input_dim)
            hidden_dim = model_config.hidden_dim
            dropout = model_config.dropout
        else:
            self.encoder = BaseGNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # Graph-level head (NE gate)
        from final_sc_review.gnn.models.pooling import AttentionPooling
        self.graph_pooling = AttentionPooling(hidden_dim)
        self.ne_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Node-level head (Dynamic-K)
        self.node_classifier = nn.Sequential(
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both tasks.

        Returns:
            (graph_logits, node_logits)
            - graph_logits: [n_graphs, 1] for NE detection
            - node_logits: [n_nodes, 1] for Dynamic-K
        """
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_attr, batch)

        # Graph-level
        graph_emb = self.graph_pooling(node_emb, batch)
        graph_logits = self.ne_classifier(graph_emb)

        # Node-level
        node_logits = self.node_classifier(node_emb)

        return graph_logits, node_logits
