#!/usr/bin/env python3
"""P4: Heterogeneous Graph GNN Script.

NOTE: P4 requires heterogeneous graphs with multiple node types (criterion, sentence)
and typed edges. This is more complex than P1-P3 which use homogeneous graphs.

Given that P1 NE Gate GNN did not outperform simpler baselines, P4 implementation
is provided as a simplified version that uses the existing graph structure with
a criterion-aware pooling mechanism.

If the simplified version shows promise, the full heterogeneous graph implementation
can be developed.

This script implements:
- Criterion-aware graph pooling using criterion ID as conditioning
- Per-criterion NE detection using shared GNN encoder
- Multi-label assessment across all criteria
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.gnn.config import GNNModelConfig
from final_sc_review.gnn.models.base import BaseGNNEncoder
from final_sc_review.gnn.models.pooling import AttentionPooling
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class CriterionAwareNEGNN(nn.Module):
    """Simplified P4: NE detection with criterion conditioning.

    Instead of full heterogeneous graph, this model:
    1. Encodes sentence graphs with GNN
    2. Uses criterion embedding to condition the prediction
    3. Produces per-criterion has_evidence predictions

    This is a stepping stone towards full P4 if results are promising.
    """

    def __init__(
        self,
        input_dim: int,
        criterion_dim: int = 64,
        num_criteria: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_criteria = num_criteria

        # Learnable criterion embeddings
        self.criterion_embeddings = nn.Embedding(num_criteria, criterion_dim)

        # GNN encoder for sentence graphs
        config = GNNModelConfig(
            gnn_type="gat",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.encoder = BaseGNNEncoder.from_config(config, input_dim)

        # Graph pooling
        self.pooling = AttentionPooling(hidden_dim)

        # Criterion-conditioned classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + criterion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        criterion_ids: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Edge indices [2, n_edges]
            batch: Batch assignment [n_nodes]
            criterion_ids: Criterion ID per graph [n_graphs]
            edge_attr: Edge features (optional)

        Returns:
            Graph-level logits [n_graphs, 1]
        """
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_attr, batch)

        # Pool to graph level
        graph_emb = self.pooling(node_emb, batch)

        # Get criterion embeddings
        crit_emb = self.criterion_embeddings(criterion_ids)

        # Combine and classify
        combined = torch.cat([graph_emb, crit_emb], dim=-1)
        logits = self.classifier(combined)

        return logits


def extract_criterion_id(graph: Data) -> int:
    """Extract criterion ID from graph metadata."""
    if hasattr(graph, 'criterion_id'):
        return graph.criterion_id
    if hasattr(graph, 'criterion'):
        # Try to parse criterion string like "A.1" -> 0, "A.2" -> 1, etc.
        crit = graph.criterion
        if isinstance(crit, str) and crit.startswith("A."):
            try:
                return int(crit.split(".")[1]) - 1
            except:
                pass
    return 0  # Default


def train_epoch(
    model: CriterionAwareNEGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    focal_gamma: float = 2.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Extract criterion IDs for each graph in batch
        n_graphs = batch.batch.max().item() + 1

        # Get criterion_id from batch if available
        if hasattr(batch, 'criterion_id') and batch.criterion_id is not None:
            crit_ids = batch.criterion_id
            if isinstance(crit_ids, torch.Tensor):
                criterion_ids = crit_ids.to(device)
            elif isinstance(crit_ids, list):
                # Parse criterion strings like "A.1" -> 0, "A.2" -> 1, etc.
                parsed = []
                for c in crit_ids:
                    if isinstance(c, str) and c.startswith("A."):
                        try:
                            parsed.append(int(c.split(".")[1]) - 1)
                        except:
                            parsed.append(0)
                    elif isinstance(c, (int, np.integer)):
                        parsed.append(int(c))
                    else:
                        parsed.append(0)
                criterion_ids = torch.tensor(parsed, dtype=torch.long, device=device)
            else:
                criterion_ids = torch.zeros(n_graphs, dtype=torch.long, device=device)
        else:
            # Default to criterion 0
            criterion_ids = torch.zeros(n_graphs, dtype=torch.long, device=device)

        # Forward
        logits = model(batch.x, batch.edge_index, batch.batch, criterion_ids, batch.edge_attr)

        # Get labels
        labels = batch.y.float().view(-1)

        # Focal loss
        probs = torch.sigmoid(logits.view(-1))
        pt = labels * probs + (1 - labels) * (1 - probs)
        focal_weight = (1 - pt) ** focal_gamma
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), labels, reduction='none')
        loss = (focal_weight * bce).mean()

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def assess_model(
    model: CriterionAwareNEGNN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Assess model performance."""
    model.train(False)

    all_probs = []
    all_labels = []
    per_criterion = {i: {"probs": [], "labels": []} for i in range(10)}

    for batch in tqdm(loader, desc="Assessing", leave=False):
        batch = batch.to(device)

        n_graphs = batch.batch.max().item() + 1
        if hasattr(batch, 'criterion_id') and batch.criterion_id is not None:
            crit_ids_raw = batch.criterion_id
            if isinstance(crit_ids_raw, torch.Tensor):
                criterion_ids = crit_ids_raw.to(device)
            elif isinstance(crit_ids_raw, list):
                # Parse criterion strings like "A.1" -> 0, "A.2" -> 1, etc.
                parsed = []
                for c in crit_ids_raw:
                    if isinstance(c, str) and c.startswith("A."):
                        try:
                            parsed.append(int(c.split(".")[1]) - 1)
                        except:
                            parsed.append(0)
                    elif isinstance(c, (int, np.integer)):
                        parsed.append(int(c))
                    else:
                        parsed.append(0)
                criterion_ids = torch.tensor(parsed, dtype=torch.long, device=device)
            else:
                criterion_ids = torch.zeros(n_graphs, dtype=torch.long, device=device)
        else:
            criterion_ids = torch.zeros(n_graphs, dtype=torch.long, device=device)

        logits = model(batch.x, batch.edge_index, batch.batch, criterion_ids, batch.edge_attr)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = batch.y.cpu().numpy().flatten()
        crit_ids = criterion_ids.cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels)

        # Per-criterion tracking
        for i, (p, l, c) in enumerate(zip(probs, labels, crit_ids)):
            per_criterion[c]["probs"].append(p)
            per_criterion[c]["labels"].append(l)

    # Compute overall metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        "n_samples": len(all_labels),
        "has_evidence_rate": float(all_labels.mean()),
    }

    if len(np.unique(all_labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(all_labels, all_probs))
        metrics["auprc"] = float(average_precision_score(all_labels, all_probs))

    # Per-criterion metrics
    criterion_metrics = {}
    for crit_id, data in per_criterion.items():
        if len(data["probs"]) > 0 and len(np.unique(data["labels"])) > 1:
            criterion_metrics[f"A.{crit_id+1}"] = {
                "n_samples": len(data["probs"]),
                "auroc": float(roc_auc_score(data["labels"], data["probs"])),
            }

    metrics["per_criterion"] = criterion_metrics

    return metrics


def run_fold(
    fold_id: int,
    train_graphs: List[Data],
    test_graphs: List[Data],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run training and testing for one fold."""
    logger.info(f"Fold {fold_id}: {len(train_graphs)} train, {len(test_graphs)} test")

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config["batch_size"], shuffle=False)

    # Get input dim
    input_dim = train_graphs[0].x.size(1)

    # Create model
    model = CriterionAwareNEGNN(
        input_dim=input_dim,
        criterion_dim=config.get("criterion_dim", 64),
        num_criteria=10,
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 3),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # Training loop
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = config.get("patience", 10)

    for epoch in range(config.get("max_epochs", 50)):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        test_metrics = assess_model(model, test_loader, device)

        auroc = test_metrics.get("auroc", 0.0)
        logger.info(
            f"Fold {fold_id} Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
            f"AUROC={auroc:.4f}"
        )

        if auroc > best_metric:
            best_metric = auroc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f"fold_{fold_id}_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best and final assessment
    model.load_state_dict(torch.load(output_dir / f"fold_{fold_id}_best.pt", weights_only=False))
    final_metrics = assess_model(model, test_loader, device)

    return {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_auroc": best_metric,
        "final_metrics": final_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="P4 Criterion-Aware GNN (Simplified)")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/gnn_research")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    setup_logging()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / "p4_hetero"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    graph_dir = Path(args.graph_dir)

    config = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "criterion_dim": 64,
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-5,
    }

    # Run 5-fold CV
    all_results = []

    for fold_id in range(5):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        if not fold_path.exists():
            logger.warning(f"Fold {fold_id} not found, skipping")
            continue

        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        train_graphs = []
        for other_fold in range(5):
            if other_fold == fold_id:
                continue
            other_path = graph_dir / f"fold_{other_fold}.pt"
            if other_path.exists():
                other_data = torch.load(other_path, weights_only=False)
                train_graphs.extend(other_data["graphs"])

        fold_result = run_fold(
            fold_id=fold_id,
            train_graphs=train_graphs,
            test_graphs=graphs,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(fold_result)

    # Aggregate
    logger.info("\n" + "=" * 60)
    logger.info("P4 Criterion-Aware GNN Results (Simplified)")
    logger.info("=" * 60)

    aurocs = [r["final_metrics"].get("auroc", 0) for r in all_results]
    auprcs = [r["final_metrics"].get("auprc", 0) for r in all_results]

    logger.info(f"AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    logger.info(f"AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")

    # Save results
    results = {
        "config": config,
        "fold_results": all_results,
        "aggregated": {
            "auroc": f"{np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}",
            "auprc": f"{np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}",
            "auroc_mean": float(np.mean(aurocs)),
            "auprc_mean": float(np.mean(auprcs)),
        },
        "timestamp": timestamp,
        "note": "Simplified P4 using criterion conditioning instead of full heterogeneous graph",
    }

    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
