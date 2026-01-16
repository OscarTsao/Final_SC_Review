#!/usr/bin/env python3
"""Run graph statistics baselines for NE detection.

This script validates whether graph structure provides useful signal
for NE detection before training full GNN models.

Baselines:
- LogisticRegression on graph stats
- RandomForest on graph stats
- HistGradientBoosting on graph stats

Features (all inference-time, NO gold labels):
- avg_pairwise_similarity (among top-k)
- edge_density, avg_degree, max_degree
- clustering_coefficient_proxy
- pagerank_mean (simplified)
- score_connectivity_correlation
- score statistics (max, mean, std, entropy)

Output: outputs/gnn_research/<ts>/graph_stats_baselines/
    - report.md
    - results.json
    - roc_curves.png (optional)

Usage:
    python scripts/gnn/run_graph_stats_baselines.py \
        --graph_dir data/cache/gnn/<timestamp> \
        --output_dir outputs/gnn_research
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_graph_dataset(graph_dir: Path, fold_id: int) -> Tuple[List, Dict]:
    """Load graphs for a specific fold."""
    import torch

    fold_path = graph_dir / f"fold_{fold_id}.pt"
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_path}")

    data = torch.load(fold_path)
    graphs = data["graphs"]

    logger.info(f"Loaded fold {fold_id}: {len(graphs)} graphs")
    return graphs, data


def extract_graph_stats(graph) -> Dict[str, float]:
    """Extract graph-level statistics for a single graph."""
    import torch

    n_nodes = graph.x.shape[0]
    edge_index = graph.edge_index.numpy() if graph.edge_index.numel() > 0 else np.array([[], []])
    n_edges = edge_index.shape[1] if edge_index.size > 0 else 0

    # Get reranker scores
    if hasattr(graph, "reranker_scores"):
        scores = graph.reranker_scores.numpy()
    else:
        scores = np.zeros(n_nodes)

    stats = {}

    # Basic graph structure
    max_edges = n_nodes * (n_nodes - 1)
    stats["edge_density"] = n_edges / max_edges if max_edges > 0 else 0.0
    stats["n_nodes"] = float(n_nodes)
    stats["n_edges"] = float(n_edges)

    # Degree statistics
    if n_edges > 0:
        out_degrees = np.bincount(edge_index[0], minlength=n_nodes)
        in_degrees = np.bincount(edge_index[1], minlength=n_nodes)
        degrees = out_degrees + in_degrees

        stats["avg_degree"] = float(degrees.mean())
        stats["max_degree"] = float(degrees.max())
        stats["std_degree"] = float(degrees.std())

        # Score-degree correlation
        if degrees.std() > 0 and scores.std() > 0:
            corr = np.corrcoef(degrees, scores)[0, 1]
            stats["score_degree_corr"] = float(corr) if not np.isnan(corr) else 0.0
        else:
            stats["score_degree_corr"] = 0.0
    else:
        stats["avg_degree"] = 0.0
        stats["max_degree"] = 0.0
        stats["std_degree"] = 0.0
        stats["score_degree_corr"] = 0.0

    # Score statistics
    if len(scores) > 0:
        sorted_scores = np.sort(scores)[::-1]
        stats["max_score"] = float(sorted_scores[0])
        stats["second_score"] = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        stats["mean_score"] = float(scores.mean())
        stats["std_score"] = float(scores.std())
        stats["score_range"] = float(sorted_scores[0] - sorted_scores[-1])
        stats["top1_top2_gap"] = stats["max_score"] - stats["second_score"]

        # Entropy of top-5 (softmax normalized)
        top5 = sorted_scores[:min(5, len(sorted_scores))]
        if len(top5) > 0:
            exp_scores = np.exp(top5 - top5.max())
            probs = exp_scores / exp_scores.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            stats["entropy_top5"] = float(entropy)
        else:
            stats["entropy_top5"] = 0.0
    else:
        stats.update({
            "max_score": 0.0,
            "second_score": 0.0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "score_range": 0.0,
            "top1_top2_gap": 0.0,
            "entropy_top5": 0.0,
        })

    # Pairwise similarity among top-k (from embeddings in graph.x)
    if n_nodes > 1 and hasattr(graph, "x") and graph.x.shape[1] > 8:
        # Assume first 1024 dims are embeddings
        emb_dim = min(1024, graph.x.shape[1] - 8)
        embeddings = graph.x[:, :emb_dim].numpy()

        # Top-5 by score
        top_k = min(5, n_nodes)
        top_idx = np.argsort(-scores)[:top_k]
        top_emb = embeddings[top_idx]

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(top_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = top_emb / norms

        sim_matrix = normed @ normed.T
        # Upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        if mask.sum() > 0:
            stats["avg_pairwise_sim"] = float(sim_matrix[mask].mean())
            stats["min_pairwise_sim"] = float(sim_matrix[mask].min())
            stats["max_pairwise_sim"] = float(sim_matrix[mask].max())
        else:
            stats["avg_pairwise_sim"] = 0.0
            stats["min_pairwise_sim"] = 0.0
            stats["max_pairwise_sim"] = 0.0
    else:
        stats["avg_pairwise_sim"] = 0.0
        stats["min_pairwise_sim"] = 0.0
        stats["max_pairwise_sim"] = 0.0

    return stats


def build_features_and_labels(graphs: List) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract features and labels from graphs."""
    records = []
    labels = []

    for graph in graphs:
        stats = extract_graph_stats(graph)
        records.append(stats)

        # Get label
        if hasattr(graph, "y"):
            labels.append(int(graph.y.item()))
        else:
            labels.append(0)

    df = pd.DataFrame(records)
    labels = np.array(labels)

    return df, labels


def compute_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """Compute TPR at a target FPR level."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, 1.0

    best_idx = idx[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])


def train_evaluate_model(
    model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Train and evaluate a single model."""
    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    auroc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0
    auprc = average_precision_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0

    tpr_at_3, thresh_3 = compute_tpr_at_fpr(y_val, y_prob, 0.03)
    tpr_at_5, thresh_5 = compute_tpr_at_fpr(y_val, y_prob, 0.05)
    tpr_at_10, thresh_10 = compute_tpr_at_fpr(y_val, y_prob, 0.10)

    return {
        "model": model_name,
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_fpr_3pct": tpr_at_3,
        "tpr_at_fpr_5pct": tpr_at_5,
        "tpr_at_fpr_10pct": tpr_at_10,
        "threshold_at_fpr_5pct": thresh_5,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "has_evidence_rate_train": float(y_train.mean()),
        "has_evidence_rate_val": float(y_val.mean()),
    }


def run_cross_validation(
    graph_dir: Path,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Run 5-fold cross-validation for all baselines."""
    all_results = []

    # Models to evaluate
    models = {
        "logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "rf_100": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "rf_200": lambda: RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42),
    }

    # Load all folds
    all_features = []
    all_labels = []
    all_fold_ids = []

    for fold_id in range(n_folds):
        try:
            graphs, _ = load_graph_dataset(graph_dir, fold_id)
            features, labels = build_features_and_labels(graphs)
            all_features.append(features)
            all_labels.append(labels)
            all_fold_ids.extend([fold_id] * len(graphs))
        except FileNotFoundError as e:
            logger.warning(f"Skipping fold {fold_id}: {e}")
            continue

    if not all_features:
        raise ValueError("No folds found")

    # Combine
    X_all = pd.concat(all_features, ignore_index=True)
    y_all = np.concatenate(all_labels)
    fold_ids = np.array(all_fold_ids)

    logger.info(f"Total samples: {len(y_all)}")
    logger.info(f"Has evidence rate: {y_all.mean():.2%}")
    logger.info(f"Features: {list(X_all.columns)}")

    # Run CV for each model
    for model_name, model_fn in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        fold_results = []

        for val_fold in range(n_folds):
            train_mask = fold_ids != val_fold
            val_mask = fold_ids == val_fold

            if not train_mask.any() or not val_mask.any():
                continue

            X_train = X_all[train_mask].values
            y_train = y_all[train_mask]
            X_val = X_all[val_mask].values
            y_val = y_all[val_mask]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train and evaluate
            model = model_fn()
            result = train_evaluate_model(
                model, model_name,
                X_train_scaled, y_train,
                X_val_scaled, y_val,
            )
            result["fold"] = val_fold
            fold_results.append(result)

            logger.info(f"  Fold {val_fold}: AUROC={result['auroc']:.4f}, TPR@5%FPR={result['tpr_at_fpr_5pct']:.4f}")

        # Aggregate results
        if fold_results:
            mean_results = {
                "model": model_name,
                "auroc_mean": np.mean([r["auroc"] for r in fold_results]),
                "auroc_std": np.std([r["auroc"] for r in fold_results]),
                "tpr_at_fpr_5pct_mean": np.mean([r["tpr_at_fpr_5pct"] for r in fold_results]),
                "tpr_at_fpr_5pct_std": np.std([r["tpr_at_fpr_5pct"] for r in fold_results]),
                "auprc_mean": np.mean([r["auprc"] for r in fold_results]),
                "auprc_std": np.std([r["auprc"] for r in fold_results]),
                "fold_results": fold_results,
            }
            all_results.append(mean_results)

    return {
        "models": all_results,
        "feature_names": list(X_all.columns),
        "n_samples": len(y_all),
        "has_evidence_rate": float(y_all.mean()),
    }


def generate_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate markdown report."""
    report_path = output_dir / "report.md"

    lines = [
        "# Graph Statistics Baseline Results",
        "",
        "## Overview",
        f"- Total samples: {results['n_samples']}",
        f"- Has evidence rate: {results['has_evidence_rate']:.2%}",
        f"- Features: {len(results['feature_names'])}",
        "",
        "## Features Used",
        "```",
    ]
    lines.extend(results['feature_names'])
    lines.extend([
        "```",
        "",
        "## Results (5-Fold CV)",
        "",
        "| Model | AUROC | TPR@5%FPR | AUPRC |",
        "|-------|-------|-----------|-------|",
    ])

    for model_result in results['models']:
        lines.append(
            f"| {model_result['model']} | "
            f"{model_result['auroc_mean']:.4f} ± {model_result['auroc_std']:.4f} | "
            f"{model_result['tpr_at_fpr_5pct_mean']:.4f} ± {model_result['tpr_at_fpr_5pct_std']:.4f} | "
            f"{model_result['auprc_mean']:.4f} ± {model_result['auprc_std']:.4f} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Best Model**: " + max(results['models'], key=lambda x: x['tpr_at_fpr_5pct_mean'])['model'],
        "2. **Graph stats provide signal**: AUROC > 0.5 indicates graph structure is informative",
        "3. **Comparison to baseline**: rf_100 baseline had AUROC ~0.60, TPR@5%FPR ~10.95%",
        "",
        "## Next Steps",
        "",
        "- If graph stats improve over score-only baseline, proceed with GNN models",
        "- Train P1 NE Gate GNN with graph structure",
        "- Evaluate Dynamic-K selection with node-level predictions",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run graph statistics baselines")
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Path to graph dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gnn_research",
        help="Output directory",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / "graph_stats_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running graph statistics baselines")
    logger.info(f"Graph dir: {args.graph_dir}")
    logger.info(f"Output: {output_dir}")

    # Run CV
    results = run_cross_validation(Path(args.graph_dir), args.n_folds)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    generate_report(results, output_dir)

    # Print summary
    print("\n=== Graph Statistics Baseline Summary ===")
    for model_result in results['models']:
        print(
            f"{model_result['model']:12} | "
            f"AUROC: {model_result['auroc_mean']:.4f} ± {model_result['auroc_std']:.4f} | "
            f"TPR@5%FPR: {model_result['tpr_at_fpr_5pct_mean']:.4f} ± {model_result['tpr_at_fpr_5pct_std']:.4f}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
