#!/usr/bin/env python3
"""P2: Dynamic-K GNN Script.

Trains DynamicKGNN for node-level evidence scoring and runs
different K selection policies with proper constraints.

Dynamic-K Policies:
- DK-A: Threshold (p_i >= tau)
- DK-B: Mass/coverage (cumsum(p) >= gamma)
- DK-C: Fixed-K baseline (for reference only)

Constraints (Non-negotiable):
- k_min = 2 (always return at least 2)
- k_max = 10 (hard cap)
- k_max_ratio = 0.5 (at most 50% of candidates)
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

from final_sc_review.gnn.config import DynamicKConfig, DynamicKPolicy, GNNModelConfig
from final_sc_review.gnn.models.p2_dynamic_k import DynamicKGNN, DynamicKLoss
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_k_constraints(n_candidates: int, k_min: int = 2, k_max: int = 10, k_max_ratio: float = 0.5) -> Tuple[int, int]:
    """Compute k constraints for a given number of candidates."""
    k_max_from_ratio = int(np.ceil(n_candidates * k_max_ratio))
    actual_k_max = min(k_max, k_max_from_ratio, n_candidates)
    actual_k_max = max(actual_k_max, k_min)
    return k_min, actual_k_max


def select_k_threshold(probs: np.ndarray, tau: float, k_min: int, k_max: int) -> int:
    """DK-A: Select K based on probability threshold."""
    k = np.sum(probs >= tau)
    return max(k_min, min(k, k_max))


def select_k_mass(probs: np.ndarray, gamma: float, k_min: int, k_max: int) -> int:
    """DK-B: Select K based on cumulative mass."""
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    k = np.searchsorted(cumsum, gamma) + 1
    return max(k_min, min(k, k_max))


def select_k_fixed(n_candidates: int, fixed_k: int, k_min: int, k_max: int) -> int:
    """DK-C: Fixed K baseline."""
    return max(k_min, min(fixed_k, k_max))


def compute_evidence_metrics(
    node_probs: np.ndarray,
    node_labels: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Compute evidence selection metrics for a single graph.

    Args:
        node_probs: Predicted probabilities per node
        node_labels: Ground truth labels (is_gold)
        k: Number of candidates to select

    Returns:
        Dictionary with metrics
    """
    n_nodes = len(node_probs)
    n_gold = node_labels.sum()

    # Sort by probability (descending)
    sorted_idx = np.argsort(-node_probs)
    selected_idx = sorted_idx[:k]
    selected_labels = node_labels[selected_idx]

    # Evidence hit-rate: Did we select at least one gold?
    hit = 1.0 if selected_labels.sum() > 0 else 0.0

    # Evidence recall: Fraction of gold captured
    recall = selected_labels.sum() / n_gold if n_gold > 0 else 1.0

    # Precision: Fraction of selected that are gold
    precision = selected_labels.sum() / k if k > 0 else 0.0

    # nDCG@K (only if there are gold labels)
    if n_gold > 0:
        relevance = node_labels[sorted_idx].reshape(1, -1)
        ideal = np.sort(node_labels)[::-1].reshape(1, -1)
        try:
            ndcg = ndcg_score(ideal, relevance, k=k)
        except:
            ndcg = 0.0
    else:
        ndcg = 1.0  # Perfect for no-evidence queries

    return {
        "hit_rate": hit,
        "recall": recall,
        "precision": precision,
        "ndcg": ndcg,
        "k": k,
        "n_gold": n_gold,
        "n_candidates": n_nodes,
    }


def train_epoch(
    model: DynamicKGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DynamicKLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

        # Get node labels
        if hasattr(batch, 'node_labels'):
            node_labels = batch.node_labels.float()
        else:
            node_labels = torch.zeros(batch.x.size(0), device=device)

        # Compute loss
        loss, components = loss_fn(logits, node_labels, batch.batch)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_bce += components["bce"]
        total_rank += components["rank"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "bce": total_bce / n_batches,
        "rank": total_rank / n_batches,
    }


@torch.no_grad()
def run_policy_assessment(
    model: DynamicKGNN,
    loader: DataLoader,
    device: torch.device,
    policies: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Run policy assessment for K selection.

    Args:
        model: Trained model
        loader: Data loader
        device: Device
        policies: Dict of policy_name -> policy_config

    Returns:
        Dict of policy_name -> aggregated metrics
    """
    model.train(False)

    # Collect per-graph results for each policy
    policy_results = {name: [] for name in policies}

    for batch in tqdm(loader, desc="Assessing", leave=False):
        batch = batch.to(device)

        # Get predictions
        logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        # Get node labels
        if hasattr(batch, 'node_labels'):
            labels = batch.node_labels.cpu().numpy()
        else:
            labels = np.zeros(len(probs))

        # Process each graph in batch
        batch_idx = batch.batch.cpu().numpy()
        n_graphs = batch_idx.max() + 1

        for g in range(n_graphs):
            mask = batch_idx == g
            g_probs = probs[mask]
            g_labels = labels[mask]
            n_candidates = len(g_probs)

            k_min, k_max = compute_k_constraints(n_candidates)

            # Run each policy
            for policy_name, policy_config in policies.items():
                policy_type = policy_config["type"]

                if policy_type == "threshold":
                    k = select_k_threshold(g_probs, policy_config["tau"], k_min, k_max)
                elif policy_type == "mass":
                    k = select_k_mass(g_probs, policy_config["gamma"], k_min, k_max)
                elif policy_type == "fixed":
                    k = select_k_fixed(n_candidates, policy_config["k"], k_min, k_max)
                else:
                    raise ValueError(f"Unknown policy: {policy_type}")

                metrics = compute_evidence_metrics(g_probs, g_labels, k)
                policy_results[policy_name].append(metrics)

    # Aggregate results
    aggregated = {}
    for policy_name, results in policy_results.items():
        if not results:
            continue

        # Separate has-evidence and no-evidence queries
        has_evidence = [r for r in results if r["n_gold"] > 0]
        no_evidence = [r for r in results if r["n_gold"] == 0]

        agg = {
            "n_queries": len(results),
            "n_has_evidence": len(has_evidence),
            "n_no_evidence": len(no_evidence),
            "avg_k": np.mean([r["k"] for r in results]),
            "std_k": np.std([r["k"] for r in results]),
        }

        if has_evidence:
            agg["hit_rate"] = np.mean([r["hit_rate"] for r in has_evidence])
            agg["recall"] = np.mean([r["recall"] for r in has_evidence])
            agg["precision"] = np.mean([r["precision"] for r in has_evidence])
            agg["ndcg"] = np.mean([r["ndcg"] for r in has_evidence])

        # K distribution
        k_values = [r["k"] for r in results]
        agg["k_distribution"] = {
            "min": int(np.min(k_values)),
            "max": int(np.max(k_values)),
            "median": float(np.median(k_values)),
            "p25": float(np.percentile(k_values, 25)),
            "p75": float(np.percentile(k_values, 75)),
        }

        aggregated[policy_name] = agg

    return aggregated


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
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 3),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.3),
    )

    model = DynamicKGNN(
        input_dim=input_dim,
        model_config=model_config,
    ).to(device)

    # Loss and optimizer
    loss_fn = DynamicKLoss(
        alpha_bce=config.get("alpha_bce", 1.0),
        alpha_rank=config.get("alpha_rank", 0.5),
        margin=config.get("margin", 0.5),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # Training loop with early stopping
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = config.get("patience", 10)

    for epoch in range(config.get("max_epochs", 50)):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Run assessment with threshold policy for monitoring
        policies = {"threshold_0.5": {"type": "threshold", "tau": 0.5}}
        test_metrics = run_policy_assessment(model, test_loader, device, policies)

        hit_rate = test_metrics.get("threshold_0.5", {}).get("hit_rate", 0.0)

        logger.info(
            f"Fold {fold_id} Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
            f"hit_rate={hit_rate:.4f}"
        )

        # Early stopping on hit_rate
        if hit_rate > best_metric:
            best_metric = hit_rate
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / f"fold_{fold_id}_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model and run final assessment
    model.load_state_dict(torch.load(output_dir / f"fold_{fold_id}_best.pt", weights_only=False))

    policies = {
        # Threshold policies
        "threshold_0.3": {"type": "threshold", "tau": 0.3},
        "threshold_0.5": {"type": "threshold", "tau": 0.5},
        "threshold_0.7": {"type": "threshold", "tau": 0.7},
        # Mass policies
        "mass_0.8": {"type": "mass", "gamma": 0.8},
        "mass_0.9": {"type": "mass", "gamma": 0.9},
        "mass_0.95": {"type": "mass", "gamma": 0.95},
        # Fixed-K baselines
        "fixed_2": {"type": "fixed", "k": 2},
        "fixed_3": {"type": "fixed", "k": 3},
        "fixed_5": {"type": "fixed", "k": 5},
    }

    final_metrics = run_policy_assessment(model, test_loader, device, policies)

    return {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_hit_rate": best_metric,
        "policies": final_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="P2 Dynamic-K GNN")
    parser.add_argument("--graph_dir", type=str, required=True, help="Graph dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/gnn_research", help="Output directory")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    setup_logging()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / "p2_dynamic_k"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load graph dataset
    graph_dir = Path(args.graph_dir)

    config = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "gnn_type": "gat",
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "alpha_bce": 1.0,
        "alpha_rank": 0.5,
        "margin": 0.5,
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

    # Aggregate results across folds
    logger.info("\n" + "=" * 60)
    logger.info("P2 Dynamic-K GNN Results")
    logger.info("=" * 60)

    # Collect metrics per policy
    policy_names = list(all_results[0]["policies"].keys())
    aggregated = {}

    for policy in policy_names:
        metrics_list = [r["policies"].get(policy, {}) for r in all_results if policy in r["policies"]]
        if not metrics_list:
            continue

        hit_rates = [m.get("hit_rate", 0) for m in metrics_list]
        recalls = [m.get("recall", 0) for m in metrics_list]
        ndcgs = [m.get("ndcg", 0) for m in metrics_list]
        avg_ks = [m.get("avg_k", 0) for m in metrics_list]

        aggregated[policy] = {
            "hit_rate": f"{np.mean(hit_rates):.4f} +/- {np.std(hit_rates):.4f}",
            "recall": f"{np.mean(recalls):.4f} +/- {np.std(recalls):.4f}",
            "ndcg": f"{np.mean(ndcgs):.4f} +/- {np.std(ndcgs):.4f}",
            "avg_k": f"{np.mean(avg_ks):.2f} +/- {np.std(avg_ks):.2f}",
            "hit_rate_mean": float(np.mean(hit_rates)),
            "recall_mean": float(np.mean(recalls)),
            "ndcg_mean": float(np.mean(ndcgs)),
            "avg_k_mean": float(np.mean(avg_ks)),
        }

        logger.info(f"\n{policy}:")
        logger.info(f"  Hit Rate: {aggregated[policy]['hit_rate']}")
        logger.info(f"  Recall: {aggregated[policy]['recall']}")
        logger.info(f"  nDCG: {aggregated[policy]['ndcg']}")
        logger.info(f"  Avg K: {aggregated[policy]['avg_k']}")

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
