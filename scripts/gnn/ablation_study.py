#!/usr/bin/env python3
"""Run ablation study for GNN architecture choices.

Ablation Matrix:
- GNN type: GCN vs SAGE vs GAT
- Edge types: semantic only vs adjacency only vs both
- Pooling: mean vs max vs attention
- Depth: 2 vs 3 vs 4 layers

Usage:
    python scripts/gnn/ablation_study.py \
        --graph_dir data/cache/gnn/<timestamp> \
        --output_dir outputs/gnn_research
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.gnn.config import (
    GNNConfig, GNNModelConfig, GNNTrainingConfig,
    GNNType, PoolingType, EdgeType
)

logger = get_logger(__name__)


ABLATION_CONFIGS = {
    "gnn_type": {
        "gcn": GNNType.GCN,
        "sage": GNNType.SAGE,
        "gat": GNNType.GAT,
    },
    "pooling": {
        "mean": PoolingType.MEAN,
        "max": PoolingType.MAX,
        "attention": PoolingType.ATTENTION,
    },
    "num_layers": [2, 3, 4],
    "edge_types": {
        "semantic_only": [EdgeType.SEMANTIC_KNN],
        "adjacency_only": [EdgeType.ADJACENCY],
        "both": [EdgeType.SEMANTIC_KNN, EdgeType.ADJACENCY],
    },
}


def load_graphs(graph_dir: Path, n_folds: int = 5) -> tuple:
    """Load all graphs and fold IDs."""
    all_graphs = []
    fold_ids = []

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        if not fold_path.exists():
            continue

        data = torch.load(fold_path)
        graphs = data["graphs"]
        all_graphs.extend(graphs)
        fold_ids.extend([fold_id] * len(graphs))

    return all_graphs, np.array(fold_ids)


def run_single_config(
    config_name: str,
    model_config: GNNModelConfig,
    training_config: GNNTrainingConfig,
    all_graphs: List,
    fold_ids: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single ablation configuration."""
    from final_sc_review.gnn.models.p1_ne_gate import NEGateGNN
    from final_sc_review.gnn.evaluation.cv import CrossValidator

    logger.info(f"\nRunning: {config_name}")

    config = GNNConfig(
        model=model_config,
        training=training_config,
        output_dir=output_dir / config_name,
    )

    cv = CrossValidator(
        model_class=NEGateGNN,
        config=config,
        output_dir=output_dir / config_name,
    )

    try:
        results = cv.run_cv(all_graphs, fold_ids)
        return {
            "config_name": config_name,
            "success": True,
            "auroc_mean": results["aggregated"]["auroc"]["mean"],
            "auroc_std": results["aggregated"]["auroc"]["std"],
            "tpr_at_5_mean": results["aggregated"]["tpr_at_fpr_5pct"]["mean"],
            "tpr_at_5_std": results["aggregated"]["tpr_at_fpr_5pct"]["std"],
        }
    except Exception as e:
        logger.error(f"Failed: {config_name} - {e}")
        return {
            "config_name": config_name,
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="GNN Ablation Study")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/gnn_research")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["gnn_type", "pooling", "depth", "edges", "all"])
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graphs
    all_graphs, fold_ids = load_graphs(Path(args.graph_dir))
    logger.info(f"Loaded {len(all_graphs)} graphs")

    if len(all_graphs) == 0:
        logger.error("No graphs loaded!")
        return

    input_dim = all_graphs[0].x.shape[1]

    # Base config
    base_training = GNNTrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
    )

    results = []

    # GNN Type ablation
    if args.ablation in ["gnn_type", "all"]:
        logger.info("\n=== GNN Type Ablation ===")
        for name, gnn_type in ABLATION_CONFIGS["gnn_type"].items():
            model_config = GNNModelConfig(
                gnn_type=gnn_type,
                hidden_dim=256,
                num_layers=3,
                pooling_type=PoolingType.ATTENTION,
            )
            result = run_single_config(
                f"gnn_type_{name}",
                model_config,
                base_training,
                all_graphs,
                fold_ids,
                output_dir,
            )
            results.append(result)

    # Pooling ablation
    if args.ablation in ["pooling", "all"]:
        logger.info("\n=== Pooling Ablation ===")
        for name, pooling_type in ABLATION_CONFIGS["pooling"].items():
            model_config = GNNModelConfig(
                gnn_type=GNNType.GAT,
                hidden_dim=256,
                num_layers=3,
                pooling_type=pooling_type,
            )
            result = run_single_config(
                f"pooling_{name}",
                model_config,
                base_training,
                all_graphs,
                fold_ids,
                output_dir,
            )
            results.append(result)

    # Depth ablation
    if args.ablation in ["depth", "all"]:
        logger.info("\n=== Depth Ablation ===")
        for num_layers in ABLATION_CONFIGS["num_layers"]:
            model_config = GNNModelConfig(
                gnn_type=GNNType.GAT,
                hidden_dim=256,
                num_layers=num_layers,
                pooling_type=PoolingType.ATTENTION,
            )
            result = run_single_config(
                f"depth_{num_layers}",
                model_config,
                base_training,
                all_graphs,
                fold_ids,
                output_dir,
            )
            results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    report_lines = [
        "# GNN Ablation Study Results",
        "",
        "## Summary",
        "",
        "| Config | AUROC | TPR@5%FPR |",
        "|--------|-------|-----------|",
    ]

    for r in results:
        if r.get("success", False):
            report_lines.append(
                f"| {r['config_name']} | "
                f"{r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f} | "
                f"{r['tpr_at_5_mean']:.4f} +/- {r['tpr_at_5_std']:.4f} |"
            )
        else:
            report_lines.append(f"| {r['config_name']} | FAILED | - |")

    report_lines.extend([
        "",
        "## Best Configuration",
        "",
    ])

    successful = [r for r in results if r.get("success", False)]
    if successful:
        best = max(successful, key=lambda x: x["tpr_at_5_mean"])
        report_lines.append(f"**{best['config_name']}**: TPR@5%FPR = {best['tpr_at_5_mean']:.4f}")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
