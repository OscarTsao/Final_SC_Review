"""GNN training infrastructure.

Components:
- Trainer: Training loop with early stopping
- Losses: Focal loss, ranking loss, combined losses
"""

from final_sc_review.gnn.training.trainer import GNNTrainer
from final_sc_review.gnn.training.losses import focal_loss, ranking_loss, combined_ne_loss

__all__ = [
    "GNNTrainer",
    "focal_loss",
    "ranking_loss",
    "combined_ne_loss",
]
