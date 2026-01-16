"""P4: Heterogeneous Graph GNN - Cross-criterion reasoning.

Architecture:
- Node types: criterion (10) + sentence (candidates)
- Edge types: criterion->sentence, sentence<->sentence
- Output: per-criterion P(has_evidence) + per-sentence P(select)

This model enables reasoning across multiple criteria:
- Shared evidence patterns across criteria
- Criterion-specific evidence thresholds
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, GATConv, Linear
    from torch_geometric.data import HeteroData
    HAS_HETERO = True
except ImportError:
    HAS_HETERO = False

from final_sc_review.gnn.config import GNNModelConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class HeteroGNNLayer(nn.Module):
    """Heterogeneous GNN layer with typed message passing."""

    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: int,
        edge_types: List[Tuple[str, str, str]],
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not HAS_HETERO:
            raise ImportError("torch-geometric heterogeneous support required")

        # Build typed convolutions
        convs = {}
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            src_dim = in_channels.get(src_type, out_channels)

            convs[edge_type] = GATConv(
                src_dim,
                out_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )

        self.conv = HeteroConv(convs, aggr="sum")
        self.dropout = dropout

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x_dict: Node features per type
            edge_index_dict: Edge indices per edge type

        Returns:
            Updated node features per type
        """
        out_dict = self.conv(x_dict, edge_index_dict)

        # Apply activation and dropout
        for node_type in out_dict:
            out_dict[node_type] = F.elu(out_dict[node_type])
            out_dict[node_type] = F.dropout(
                out_dict[node_type], p=self.dropout, training=self.training
            )

        return out_dict


class HeterogeneousCriterionGNN(nn.Module):
    """Heterogeneous GNN for cross-criterion evidence reasoning.

    Graph structure:
    - Node types:
        - 'criterion': DSM-5 criteria embeddings (10 nodes)
        - 'sentence': Candidate sentence embeddings

    - Edge types:
        - ('criterion', 'queries', 'sentence'): Criterion -> candidate query
        - ('sentence', 'similar', 'sentence'): Semantic similarity edges
        - ('sentence', 'adjacent', 'sentence'): Sequential adjacency edges

    Outputs:
    - Per-criterion has_evidence logits
    - Per-sentence selection probabilities
    """

    def __init__(
        self,
        criterion_dim: int,
        sentence_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not HAS_HETERO:
            raise ImportError("torch-geometric heterogeneous support required")

        self.criterion_dim = criterion_dim
        self.sentence_dim = sentence_dim
        self.hidden_dim = hidden_dim

        # Input projections
        self.criterion_proj = nn.Linear(criterion_dim, hidden_dim)
        self.sentence_proj = nn.Linear(sentence_dim, hidden_dim)

        # Define edge types
        self.edge_types = [
            ("criterion", "queries", "sentence"),
            ("sentence", "rev_queries", "criterion"),
            ("sentence", "similar", "sentence"),
        ]

        # Hetero GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = {"criterion": hidden_dim, "sentence": hidden_dim}
            self.layers.append(
                HeteroGNNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    edge_types=self.edge_types,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        # Output heads
        self.criterion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.sentence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_criterion: torch.Tensor,
        x_sentence: torch.Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_criterion: Criterion embeddings [n_criteria, criterion_dim]
            x_sentence: Sentence embeddings [n_sentences, sentence_dim]
            edge_index_dict: Edge indices per edge type

        Returns:
            (criterion_logits, sentence_logits)
            - criterion_logits: [n_criteria, 1]
            - sentence_logits: [n_sentences, 1]
        """
        # Project inputs
        x_dict = {
            "criterion": self.criterion_proj(x_criterion),
            "sentence": self.sentence_proj(x_sentence),
        }

        # Apply hetero GNN layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        # Output predictions
        criterion_logits = self.criterion_head(x_dict["criterion"])
        sentence_logits = self.sentence_head(x_dict["sentence"])

        return criterion_logits, sentence_logits

    def predict(
        self,
        x_criterion: torch.Tensor,
        x_sentence: torch.Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict probabilities."""
        criterion_logits, sentence_logits = self.forward(
            x_criterion, x_sentence, edge_index_dict
        )
        return torch.sigmoid(criterion_logits), torch.sigmoid(sentence_logits)


def build_hetero_graph(
    criterion_embeddings: torch.Tensor,  # [n_criteria, criterion_dim]
    sentence_embeddings: torch.Tensor,   # [n_sentences, sentence_dim]
    criterion_sentence_edges: torch.Tensor,  # [2, n_edges] criterion -> sentence
    sentence_similarity_edges: torch.Tensor,  # [2, n_edges] sentence <-> sentence
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
    """Build heterogeneous graph data structure.

    Args:
        criterion_embeddings: Criterion node features
        sentence_embeddings: Sentence node features
        criterion_sentence_edges: Edges from criteria to sentences they query
        sentence_similarity_edges: Semantic similarity edges between sentences

    Returns:
        (x_dict, edge_index_dict) suitable for HeterogeneousCriterionGNN
    """
    x_dict = {
        "criterion": criterion_embeddings,
        "sentence": sentence_embeddings,
    }

    edge_index_dict = {
        ("criterion", "queries", "sentence"): criterion_sentence_edges,
        ("sentence", "rev_queries", "criterion"): criterion_sentence_edges.flip(0),
        ("sentence", "similar", "sentence"): sentence_similarity_edges,
    }

    return x_dict, edge_index_dict


class HeteroGraphLoss(nn.Module):
    """Loss for heterogeneous graph model.

    Combines:
    - Criterion-level focal loss
    - Sentence-level BCE loss
    - Consistency regularization
    """

    def __init__(
        self,
        alpha_criterion: float = 1.0,
        alpha_sentence: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha_criterion = alpha_criterion
        self.alpha_sentence = alpha_sentence
        self.focal_gamma = focal_gamma

    def forward(
        self,
        criterion_logits: torch.Tensor,
        sentence_logits: torch.Tensor,
        criterion_labels: torch.Tensor,
        sentence_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss."""
        from final_sc_review.gnn.training.losses import focal_loss

        components = {}

        # Criterion-level loss
        crit_loss = focal_loss(
            criterion_logits.view(-1),
            criterion_labels.view(-1),
            gamma=self.focal_gamma,
        )
        components["criterion"] = crit_loss.item()

        # Sentence-level loss
        sent_loss = F.binary_cross_entropy_with_logits(
            sentence_logits.view(-1),
            sentence_labels.view(-1),
        )
        components["sentence"] = sent_loss.item()

        total = self.alpha_criterion * crit_loss + self.alpha_sentence * sent_loss
        components["total"] = total.item()

        return total, components
