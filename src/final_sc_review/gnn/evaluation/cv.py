"""Cross-validation orchestration for GNN models.

Implements:
- 5-fold CV with post-id disjoint splits
- Nested tuning (inner split for threshold optimization)
- Model training and assessment per fold
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from final_sc_review.gnn.config import GNNConfig, GNNTrainingConfig
from final_sc_review.gnn.training.trainer import GNNTrainer
from final_sc_review.gnn.training.losses import FocalLoss
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """5-fold cross-validator for GNN models."""

    def __init__(
        self,
        model_class: Type[nn.Module],
        config: GNNConfig,
        output_dir: Path,
    ):
        """Initialize cross-validator.

        Args:
            model_class: GNN model class to instantiate
            config: GNN configuration
            output_dir: Directory for outputs
        """
        if not HAS_PYG:
            raise ImportError("torch-geometric required")

        self.model_class = model_class
        self.config = config
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def run_fold(
        self,
        fold_id: int,
        train_graphs: List[Data],
        val_graphs: List[Data],
        input_dim: int,
    ) -> Dict[str, Any]:
        """Run training and assessment for a single fold.

        Args:
            fold_id: Fold number (0-4)
            train_graphs: Training graphs
            val_graphs: Validation graphs
            input_dim: Input feature dimension

        Returns:
            Fold results dictionary
        """
        from final_sc_review.gnn.evaluation.metrics import NEGateMetrics

        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_id}")
        logger.info(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
        logger.info(f"{'='*50}")

        fold_dir = self.output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(exist_ok=True)

        # Split train into train/tune for nested threshold optimization
        n_train = len(train_graphs)
        n_tune = int(n_train * (1 - self.config.inner_train_ratio))

        np.random.seed(self.config.training.seed + fold_id)
        indices = np.random.permutation(n_train)

        tune_idx = indices[:n_tune]
        train_idx = indices[n_tune:]

        tune_graphs = [train_graphs[i] for i in tune_idx]
        inner_train_graphs = [train_graphs[i] for i in train_idx]

        logger.info(f"Inner split: train={len(inner_train_graphs)}, tune={len(tune_graphs)}")

        # Create data loaders
        train_loader = DataLoader(
            inner_train_graphs,
            batch_size=self.config.training.batch_size,
            shuffle=True,
        )
        tune_loader = DataLoader(
            tune_graphs,
            batch_size=self.config.training.batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_graphs,
            batch_size=self.config.training.batch_size,
            shuffle=False,
        )

        # Initialize model
        model = self.model_class(
            input_dim=input_dim,
            config=self.config.model,
        )

        # Initialize loss and trainer
        loss_fn = FocalLoss(
            alpha=self.config.training.focal_alpha,
            gamma=self.config.training.focal_gamma,
        )

        trainer = GNNTrainer(
            model=model,
            loss_fn=loss_fn,
            config=self.config.training,
            output_dir=fold_dir,
            metric_name="tpr_at_fpr_5pct",
            metric_mode="max",
        )

        # Train
        train_result = trainer.train(train_loader, tune_loader)

        # Get predictions on validation set
        model.eval()
        device = trainer.device

        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                probs = torch.sigmoid(logits).view(-1)
                val_preds.append(probs.cpu().numpy())
                val_labels.append(batch.y.view(-1).cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        # Compute metrics
        metrics = NEGateMetrics.compute(val_preds, val_labels)

        logger.info(f"Fold {fold_id} Results:")
        logger.info(f"  AUROC: {metrics.auroc:.4f}")
        logger.info(f"  TPR@5%FPR: {metrics.tpr_at_fpr_5pct:.4f}")
        logger.info(f"  AUPRC: {metrics.auprc:.4f}")

        # Save fold results
        fold_result = {
            "fold_id": fold_id,
            "metrics": metrics.to_dict(),
            "train_result": {
                "best_epoch": train_result["best_epoch"],
                "best_metric": train_result["best_metric"],
            },
            "n_train": len(inner_train_graphs),
            "n_tune": len(tune_graphs),
            "n_val": len(val_graphs),
        }

        with open(fold_dir / "results.json", "w") as f:
            json.dump(fold_result, f, indent=2)

        # Save predictions for analysis
        np.savez(
            fold_dir / "predictions.npz",
            preds=val_preds,
            labels=val_labels,
        )

        return fold_result

    def run_cv(
        self,
        all_graphs: List[Data],
        fold_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """Run full 5-fold cross-validation.

        Args:
            all_graphs: All graphs
            fold_ids: Fold assignment for each graph

        Returns:
            CV results with aggregated metrics
        """
        from final_sc_review.gnn.evaluation.metrics import NEGateMetrics, aggregate_fold_metrics

        # Get input dimension
        input_dim = all_graphs[0].x.shape[1]
        logger.info(f"Input dimension: {input_dim}")

        n_folds = len(np.unique(fold_ids))
        logger.info(f"Running {n_folds}-fold CV")

        fold_results = []
        fold_metrics = []

        for fold in range(n_folds):
            # Split by fold
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold

            train_graphs = [all_graphs[i] for i in np.where(train_mask)[0]]
            val_graphs = [all_graphs[i] for i in np.where(val_mask)[0]]

            # Run fold
            result = self.run_fold(fold, train_graphs, val_graphs, input_dim)
            fold_results.append(result)

            metric_dict = {k: v for k, v in result["metrics"].items()
                          if k not in ("has_evidence_rate", "n_samples")}
            fold_metrics.append(NEGateMetrics(
                n_samples=result["metrics"].get("n_samples", 0),
                has_evidence_rate=result["metrics"].get("has_evidence_rate", 0.0),
                **metric_dict
            ))

        # Aggregate results
        aggregated = aggregate_fold_metrics(fold_metrics)

        cv_result = {
            "fold_results": fold_results,
            "aggregated": {
                k: {"mean": v[0], "std": v[1]}
                for k, v in aggregated.items()
            },
            "n_folds": n_folds,
            "n_total_graphs": len(all_graphs),
            "config": self.config.to_dict(),
        }

        # Log summary
        logger.info("\n" + "="*50)
        logger.info("Cross-Validation Summary")
        logger.info("="*50)
        for key, (mean, std) in aggregated.items():
            logger.info(f"{key}: {mean:.4f} +/- {std:.4f}")

        # Save CV results
        with open(self.output_dir / "cv_results.json", "w") as f:
            json.dump(cv_result, f, indent=2)

        return cv_result
