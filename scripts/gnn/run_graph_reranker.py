#!/usr/bin/env python3
"""P3: Graph Reranker GNN Script.

Trains GraphRerankerGNN for score refinement and measures
ranking improvement over original reranker scores.

Placement: AFTER reranker, BEFORE NE gate/dynamic-K

Metrics:
- nDCG@{1,3,5,10,20} improvement
- Recall@{1,3,5,10,20} improvement
- MRR improvement
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.gnn.config import GNNModelConfig
from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: List[int] = [1, 3, 5, 10, 20],
) -> Dict[str, float]:
    """Compute ranking metrics for a single query.

    Args:
        scores: Predicted scores per node
        labels: Ground truth labels (is_gold)
        ks: K values for metrics

    Returns:
        Dictionary with metrics
    """
    n_nodes = len(scores)
    n_gold = labels.sum()

    if n_gold == 0:
        return {"has_evidence": False}

    # Sort by score (descending)
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    metrics = {"has_evidence": True, "n_gold": int(n_gold), "n_candidates": n_nodes}

    # Recall@K
    for k in ks:
        if k <= n_nodes:
            recall_k = sorted_labels[:k].sum() / n_gold
            metrics[f"recall@{k}"] = float(recall_k)

    # nDCG@K
    for k in ks:
        if k <= n_nodes:
            relevance = labels[sorted_idx].reshape(1, -1)
            ideal = np.sort(labels)[::-1].reshape(1, -1)
            try:
                ndcg_k = ndcg_score(ideal, relevance, k=k)
                metrics[f"ndcg@{k}"] = float(ndcg_k)
            except:
                metrics[f"ndcg@{k}"] = 0.0

    # MRR
    gold_positions = np.where(sorted_labels > 0)[0]
    if len(gold_positions) > 0:
        mrr = 1.0 / (gold_positions[0] + 1)
    else:
        mrr = 0.0
    metrics["mrr"] = float(mrr)

    return metrics


def train_epoch(
    model: GraphRerankerGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: GraphRerankerLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_rank = 0.0
    total_align = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Get original scores from reranker_scores attribute
        if hasattr(batch, 'reranker_scores'):
            original_scores = batch.reranker_scores
        else:
            # Fallback: use last column of x
            original_scores = batch.x[:, -1]

        # Forward
        refined_scores = model(
            batch.x, batch.edge_index, original_scores, batch.batch, batch.edge_attr
        )

        # Get node labels
        if hasattr(batch, 'node_labels'):
            node_labels = batch.node_labels.float()
        else:
            node_labels = torch.zeros(batch.x.size(0), device=device)

        # Compute loss
        loss, components = loss_fn(refined_scores, original_scores, node_labels, batch.batch)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rank += components["rank"]
        total_align += components["align"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "rank": total_rank / n_batches,
        "align": total_align / n_batches,
    }


@torch.no_grad()
def run_ranking_assessment(
    model: GraphRerankerGNN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compare original vs refined rankings.

    Returns:
        (original_metrics, refined_metrics)
    """
    model.train(False)

    original_results = []
    refined_results = []

    for batch in tqdm(loader, desc="Assessing", leave=False):
        batch = batch.to(device)

        # Get original scores from reranker_scores attribute
        if hasattr(batch, 'reranker_scores'):
            original_scores = batch.reranker_scores
        else:
            original_scores = batch.x[:, -1]

        # Get refined scores
        refined_scores = model(
            batch.x, batch.edge_index, original_scores, batch.batch, batch.edge_attr
        )

        # Get node labels
        if hasattr(batch, 'node_labels'):
            labels = batch.node_labels.cpu().numpy()
        else:
            labels = np.zeros(batch.x.size(0))

        original_scores_np = original_scores.cpu().numpy()
        refined_scores_np = refined_scores.cpu().numpy()

        # Process each graph
        batch_idx = batch.batch.cpu().numpy()
        n_graphs = batch_idx.max() + 1

        for g in range(n_graphs):
            mask = batch_idx == g
            g_orig = original_scores_np[mask]
            g_ref = refined_scores_np[mask]
            g_labels = labels[mask]

            orig_metrics = compute_ranking_metrics(g_orig, g_labels)
            ref_metrics = compute_ranking_metrics(g_ref, g_labels)

            if orig_metrics.get("has_evidence", False):
                original_results.append(orig_metrics)
                refined_results.append(ref_metrics)

    # Aggregate
    def aggregate(results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {}

        agg = {"n_queries": len(results)}
        # Collect all possible keys from ALL results, not just the first one
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        metric_keys = [k for k in all_keys if k not in ["has_evidence", "n_gold", "n_candidates"]]

        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                agg[key] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))

        return agg

    return aggregate(original_results), aggregate(refined_results)


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

    # Get input dim from first graph
    input_dim = train_graphs[0].x.size(1)

    # Create model
    model_config = GNNModelConfig(
        gnn_type=config.get("gnn_type", "gat"),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.2),
    )

    model = GraphRerankerGNN(
        input_dim=input_dim,
        config=model_config,
        alpha_init=config.get("alpha_init", 0.7),
        learn_alpha=config.get("learn_alpha", True),
    ).to(device)

    # Loss and optimizer
    loss_fn = GraphRerankerLoss(
        alpha_rank=config.get("alpha_rank", 1.0),
        alpha_align=config.get("alpha_align", 0.5),
        alpha_reg=config.get("alpha_reg", 0.1),
        margin=config.get("margin", 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # Training loop
    best_metric = float('-inf')  # Start with -inf to ensure first epoch saves
    best_epoch = 0
    patience_counter = 0
    patience = config.get("patience", 10)

    for epoch in range(config.get("max_epochs", 30)):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Run assessment
        orig_metrics, ref_metrics = run_ranking_assessment(model, test_loader, device)

        # Track nDCG@5 improvement
        orig_ndcg = orig_metrics.get("ndcg@5", 0)
        ref_ndcg = ref_metrics.get("ndcg@5", 0)
        improvement = ref_ndcg - orig_ndcg

        logger.info(
            f"Fold {fold_id} Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
            f"nDCG@5: {orig_ndcg:.4f} -> {ref_ndcg:.4f} ({improvement:+.4f}), "
            f"alpha={model.alpha.item():.3f}"
        )

        # Early stopping on nDCG improvement
        if improvement > best_metric:
            best_metric = improvement
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f"fold_{fold_id}_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model and run final assessment
    best_model_path = output_dir / f"fold_{fold_id}_best.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
    else:
        logger.warning(f"No best model saved for fold {fold_id}, using final model state")
    final_orig, final_ref = run_ranking_assessment(model, test_loader, device)

    return {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_improvement": best_metric,
        "final_alpha": float(model.alpha.item()),
        "original_metrics": final_orig,
        "refined_metrics": final_ref,
    }


def main():
    parser = argparse.ArgumentParser(description="P3 Graph Reranker GNN")
    parser.add_argument("--graph_dir", type=str, required=True, help="Graph dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/gnn_research", help="Output directory")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    setup_logging()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / "p3_graph_reranker"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    graph_dir = Path(args.graph_dir)

    config = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "gnn_type": "gat",
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "alpha_init": 0.7,
        "learn_alpha": True,
        "alpha_rank": 1.0,
        "alpha_align": 0.5,
        "alpha_reg": 0.1,
        "margin": 0.1,
    }

    # Run 5-fold CV
    all_results = []

    for fold_id in range(5):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        if not fold_path.exists():
            logger.warning(f"Fold {fold_id} not found, skipping")
            continue

        # Load fold data
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        # Get other folds for training
        train_graphs = []
        for other_fold in range(5):
            if other_fold == fold_id:
                continue
            other_path = graph_dir / f"fold_{other_fold}.pt"
            if other_path.exists():
                other_data = torch.load(other_path, weights_only=False)
                train_graphs.extend(other_data["graphs"])

        # Run fold
        fold_result = run_fold(
            fold_id=fold_id,
            train_graphs=train_graphs,
            test_graphs=graphs,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(fold_result)

    # Aggregate results
    logger.info("\n" + "=" * 60)
    logger.info("P3 Graph Reranker Results")
    logger.info("=" * 60)

    # Compare original vs refined across folds
    metric_keys = ["mrr", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "recall@1", "recall@3", "recall@5", "recall@10"]

    aggregated = {"original": {}, "refined": {}, "improvement": {}}

    for key in metric_keys:
        orig_vals = [r["original_metrics"].get(key, 0) for r in all_results]
        ref_vals = [r["refined_metrics"].get(key, 0) for r in all_results]
        improvements = [r - o for o, r in zip(orig_vals, ref_vals)]

        aggregated["original"][key] = f"{np.mean(orig_vals):.4f} +/- {np.std(orig_vals):.4f}"
        aggregated["refined"][key] = f"{np.mean(ref_vals):.4f} +/- {np.std(ref_vals):.4f}"
        aggregated["improvement"][key] = f"{np.mean(improvements):+.4f} +/- {np.std(improvements):.4f}"

    logger.info("\nMetric Comparison:")
    logger.info(f"{'Metric':<12} {'Original':<20} {'Refined':<20} {'Improvement':<20}")
    logger.info("-" * 72)
    for key in metric_keys:
        logger.info(
            f"{key:<12} {aggregated['original'][key]:<20} "
            f"{aggregated['refined'][key]:<20} {aggregated['improvement'][key]:<20}"
        )

    # Save results
    results = {
        "config": config,
        "fold_results": all_results,
        "aggregated": aggregated,
        "timestamp": timestamp,
    }

    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
