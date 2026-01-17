#!/usr/bin/env python3
"""Train and assess P1 NE Gate GNN model with 5-fold CV.

This script:
1. Loads pre-built graph dataset
2. Runs 5-fold cross-validation
3. Reports AUROC and TPR@5%FPR
4. Saves best model per fold

Usage:
    python scripts/gnn/train_eval_ne_gnn.py \
        --graph_dir data/cache/gnn/<timestamp> \
        --output_dir outputs/gnn_research
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.gnn.config import GNNConfig, GNNModelConfig, GNNTrainingConfig
from final_sc_review.gnn.models.p1_ne_gate import NEGateGNN

logger = get_logger(__name__)


def load_graphs(graph_dir: Path, n_folds: int = 5) -> tuple:
    """Load all graphs and fold IDs."""
    all_graphs = []
    fold_ids = []

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        if not fold_path.exists():
            logger.warning(f"Fold file not found: {fold_path}")
            continue

        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        all_graphs.extend(graphs)
        fold_ids.extend([fold_id] * len(graphs))

        logger.info(f"Loaded fold {fold_id}: {len(graphs)} graphs")

    return all_graphs, np.array(fold_ids)


def main():
    parser = argparse.ArgumentParser(description="Train P1 NE Gate GNN")
    parser.add_argument("--graph_dir", type=str, required=True, help="Graph dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/gnn_research")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / "p1_ne_gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training P1 NE Gate GNN")
    logger.info(f"Graph dir: {args.graph_dir}")
    logger.info(f"Output: {output_dir}")

    # Create config
    model_config = GNNModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.3,
        num_heads=4,
    )

    training_config = GNNTrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    config = GNNConfig(
        model=model_config,
        training=training_config,
        output_dir=output_dir,
    )

    # Load graphs
    all_graphs, fold_ids = load_graphs(Path(args.graph_dir))
    logger.info(f"Total graphs: {len(all_graphs)}")

    if len(all_graphs) == 0:
        logger.error("No graphs loaded!")
        return

    # Get input dimension
    input_dim = all_graphs[0].x.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    # Run cross-validation
    from final_sc_review.gnn.evaluation.cv import CrossValidator

    cv = CrossValidator(
        model_class=NEGateGNN,
        config=config,
        output_dir=output_dir,
    )

    results = cv.run_cv(all_graphs, fold_ids)

    # Print summary
    print("\n" + "="*60)
    print("P1 NE Gate GNN Results")
    print("="*60)

    for key, stats in results["aggregated"].items():
        print(f"{key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
