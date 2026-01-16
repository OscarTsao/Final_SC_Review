"""GNN Trainer with early stopping and logging.

Handles:
- Training loop with gradient accumulation
- Validation and early stopping
- Learning rate scheduling
- Checkpoint saving/loading
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

try:
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from final_sc_review.gnn.config import GNNTrainingConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    epoch: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "train_losses": self.train_losses,
            "val_metrics": self.val_metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(
            epoch=d["epoch"],
            best_metric=d["best_metric"],
            best_epoch=d["best_epoch"],
            patience_counter=d["patience_counter"],
            train_losses=d["train_losses"],
            val_metrics=d["val_metrics"],
        )


class GNNTrainer:
    """Trainer for GNN models with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        config: GNNTrainingConfig,
        output_dir: Optional[Path] = None,
        metric_fn: Optional[Callable] = None,
        metric_name: str = "auroc",
        metric_mode: str = "max",
    ):
        """Initialize trainer.

        Args:
            model: GNN model to train
            loss_fn: Loss function (logits, labels) -> loss
            config: Training configuration
            output_dir: Directory for checkpoints and logs
            metric_fn: Validation metric function (preds, labels) -> float
            metric_name: Name of the metric for logging
            metric_mode: 'max' or 'min' for early stopping
        """
        if not HAS_PYG:
            raise ImportError("torch-geometric required")

        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = output_dir
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize scheduler
        if config.use_scheduler:
            if config.scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.max_epochs - config.warmup_epochs,
                )
            elif config.scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode=metric_mode,
                    patience=config.patience // 2,
                    factor=0.5,
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

        # Training state
        self.state = TrainingState()

        # Create output directory
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            # Compute loss
            if hasattr(batch, "y"):
                labels = batch.y
            else:
                labels = torch.zeros(logits.size(0), device=self.device)

            loss = self.loss_fn(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model.

        Returns:
            (primary_metric, all_metrics_dict)
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in val_loader:
            batch = batch.to(self.device)

            logits = self.model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            probs = torch.sigmoid(logits).view(-1)

            all_preds.append(probs.cpu())
            all_labels.append(batch.y.view(-1).cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        # Compute metrics
        metrics = self._compute_metrics(preds, labels)
        primary_metric = metrics.get(self.metric_name, 0.0)

        return primary_metric, metrics

    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute assessment metrics."""
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

        metrics = {}

        # AUROC
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = float(roc_auc_score(labels, preds))
            metrics["auprc"] = float(average_precision_score(labels, preds))

            # TPR at FPR levels
            fpr, tpr, thresholds = roc_curve(labels, preds)

            for target_fpr in [0.03, 0.05, 0.10]:
                idx = np.where(fpr <= target_fpr)[0]
                if len(idx) > 0:
                    metrics[f"tpr_at_fpr_{int(target_fpr*100)}pct"] = float(tpr[idx[-1]])
                else:
                    metrics[f"tpr_at_fpr_{int(target_fpr*100)}pct"] = 0.0
        else:
            metrics["auroc"] = 0.5
            metrics["auprc"] = labels.mean()
            metrics["tpr_at_fpr_3pct"] = 0.0
            metrics["tpr_at_fpr_5pct"] = 0.0
            metrics["tpr_at_fpr_10pct"] = 0.0

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping.

        Returns:
            Training history and best metrics
        """
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        best_model_state = None

        for epoch in range(self.config.max_epochs):
            self.state.epoch = epoch

            # Learning rate warmup
            if epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config.learning_rate * warmup_factor

            # Train
            train_loss = self.train_epoch(train_loader)
            self.state.train_losses.append(train_loss)

            # Validate
            val_metric, val_metrics = self.validate(val_loader)
            self.state.val_metrics.append(val_metrics)

            # Update scheduler
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metric)
                else:
                    self.scheduler.step()

            # Check for improvement
            improved = False
            if self.metric_mode == "max":
                if val_metric > self.state.best_metric + self.config.min_delta:
                    improved = True
            else:
                if val_metric < self.state.best_metric - self.config.min_delta:
                    improved = True

            if improved:
                self.state.best_metric = val_metric
                self.state.best_epoch = epoch
                self.state.patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                # Save checkpoint
                if self.output_dir:
                    self.save_checkpoint(self.output_dir / "best_model.pt")
            else:
                self.state.patience_counter += 1

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"{self.metric_name}: {val_metric:.4f} | "
                f"Best: {self.state.best_metric:.4f} @ {self.state.best_epoch+1} | "
                f"LR: {lr:.2e}"
            )

            # Early stopping
            if self.state.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {self.state.best_epoch+1}")

        # Final validation
        final_metric, final_metrics = self.validate(val_loader)

        return {
            "best_epoch": self.state.best_epoch,
            "best_metric": self.state.best_metric,
            "final_metrics": final_metrics,
            "train_losses": self.state.train_losses,
            "val_metrics": self.state.val_metrics,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_state": self.state.to_dict(),
            "config": self.config.to_dict(),
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.state = TrainingState.from_dict(checkpoint["training_state"])
