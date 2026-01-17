#!/usr/bin/env python3
"""End-to-End GNN Pipeline Evaluation and Report Generation.

This script implements the gold-standard evaluation protocol:
1. Loads fold-wise raw predictions from P1/P2/P3/P4 components
2. Composes pipeline variants (ablations)
3. Evaluates with proper TRAIN/TUNE/EVAL splits
4. Generates comprehensive metrics and reports

Output: outputs/gnn_e2e_report/<timestamp>/
- report.md: Human-readable report
- summary.json: Machine-readable metrics
- leaderboard.csv: Variant comparison
- per_query_eval.parquet: Per-query predictions and labels
- raw_predictions/fold_{k}.parquet: Canonical raw predictions

Usage:
    python scripts/gnn/run_e2e_eval_and_report.py --graph_dir data/cache/gnn/20260117_003135
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.metrics.ranking import recall_at_k, mrr_at_k, ndcg_at_k, map_at_k

logger = get_logger(__name__)


# ============================================================================
# Data Classes for Structured Results
# ============================================================================

@dataclass
class NEMetrics:
    """NE Detection metrics at a specific operating point."""
    auroc: float = 0.0
    auprc: float = 0.0
    tpr_at_3pct_fpr: float = 0.0
    tpr_at_5pct_fpr: float = 0.0
    tpr_at_10pct_fpr: float = 0.0
    threshold_at_5pct_fpr: float = 0.5
    # At operating point
    tpr: float = 0.0
    fpr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mcc: float = 0.0
    balanced_acc: float = 0.0
    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    # Population stats
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingMetrics:
    """Ranking metrics for evidence queries."""
    mrr: float = 0.0
    map_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    n_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mrr": self.mrr,
            "map_at_k": self.map_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "n_queries": self.n_queries,
        }


@dataclass
class DynamicKMetrics:
    """Dynamic-K selection metrics."""
    avg_k_pred_pos: float = 0.0  # Among predicted positive
    avg_k_all: float = 0.0       # Including predicted negative (K=0)
    std_k: float = 0.0
    evidence_recall_conditional: float = 0.0   # Among pred_pos AND has_evidence
    evidence_recall_unconditional: float = 0.0 # All has_evidence queries
    k_distribution: Dict[str, float] = field(default_factory=dict)
    n_pred_pos: int = 0
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class E2EMetrics:
    """End-to-end deployment metrics."""
    ne_metrics: NEMetrics = field(default_factory=NEMetrics)
    ranking_metrics: RankingMetrics = field(default_factory=RankingMetrics)
    dynamic_k_metrics: DynamicKMetrics = field(default_factory=DynamicKMetrics)
    variant_name: str = ""
    fold_id: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "fold_id": self.fold_id,
            "ne_metrics": self.ne_metrics.to_dict(),
            "ranking_metrics": self.ranking_metrics.to_dict(),
            "dynamic_k_metrics": self.dynamic_k_metrics.to_dict(),
        }


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def compute_ne_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: Optional[float] = None,
    fpr_budget: float = 0.05,
) -> NEMetrics:
    """Compute NE detection metrics.

    Args:
        y_prob: Predicted probabilities
        y_true: Ground truth labels (0/1)
        threshold: If provided, use this threshold. Otherwise compute from fpr_budget.
        fpr_budget: Target FPR for threshold selection (used if threshold is None)

    Returns:
        NEMetrics object with all metrics
    """
    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_samples - n_positive

    # Handle edge cases
    if n_positive == 0 or n_negative == 0:
        return NEMetrics(
            auroc=0.5,
            auprc=n_positive / n_samples if n_samples > 0 else 0.0,
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
        )

    # Compute area metrics
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Compute ROC curve
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)

    # TPR at fixed FPR levels
    def get_tpr_at_fpr(target_fpr: float) -> Tuple[float, float]:
        idx = np.where(fpr_arr <= target_fpr)[0]
        if len(idx) == 0:
            return 0.0, 1.0
        best_idx = idx[-1]
        return float(tpr_arr[best_idx]), float(thresholds[best_idx])

    tpr_3, _ = get_tpr_at_fpr(0.03)
    tpr_5, thresh_5 = get_tpr_at_fpr(0.05)
    tpr_10, _ = get_tpr_at_fpr(0.10)

    # Determine operating threshold
    if threshold is None:
        threshold = thresh_5

    # Compute predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Derived metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return NEMetrics(
        auroc=auroc,
        auprc=auprc,
        tpr_at_3pct_fpr=tpr_3,
        tpr_at_5pct_fpr=tpr_5,
        tpr_at_10pct_fpr=tpr_10,
        threshold_at_5pct_fpr=thresh_5,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        balanced_acc=balanced_acc,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        n_samples=n_samples,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def select_threshold_on_tune(
    tune_probs: np.ndarray,
    tune_labels: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Select threshold on TUNE set that achieves target FPR.

    This is the GOLD STANDARD procedure:
    1. Compute ROC curve on TUNE set
    2. Find threshold where FPR <= target_fpr and TPR is maximized

    Args:
        tune_probs: Probabilities on tune set
        tune_labels: Labels on tune set
        target_fpr: Target FPR budget

    Returns:
        Selected threshold
    """
    if len(np.unique(tune_labels)) < 2:
        return 0.5  # Default if only one class

    fpr, tpr, thresholds = roc_curve(tune_labels, tune_probs)

    # Find indices where FPR <= target
    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
        # If can't achieve target FPR, use highest threshold
        return float(thresholds[0])

    # Among valid, pick the one with highest TPR (which is the last one since fpr is sorted)
    best_idx = valid_idx[-1]
    return float(thresholds[best_idx])


def compute_ranking_metrics(
    ranked_ids: List[List[str]],
    gold_ids: List[List[str]],
    k_values: List[int] = [1, 3, 5, 10, 20],
) -> RankingMetrics:
    """Compute ranking metrics for evidence queries.

    Args:
        ranked_ids: Per-query ranked candidate IDs
        gold_ids: Per-query gold evidence IDs
        k_values: K values for @K metrics

    Returns:
        RankingMetrics object
    """
    n_queries = len(ranked_ids)
    if n_queries == 0:
        return RankingMetrics()

    mrr_values = []
    recall_at_k_values = {k: [] for k in k_values}
    ndcg_at_k_values = {k: [] for k in k_values}
    map_at_k_values = {k: [] for k in k_values}
    hit_rate_at_k_values = {k: [] for k in k_values}

    for ranked, gold in zip(ranked_ids, gold_ids):
        if len(gold) == 0:
            continue  # Skip no-evidence queries

        # MRR
        mrr = mrr_at_k(gold, ranked, k=max(k_values))
        mrr_values.append(mrr)

        # Per-K metrics
        for k in k_values:
            # Recall@K
            recall = recall_at_k(gold, ranked, k=k)
            recall_at_k_values[k].append(recall)

            # nDCG@K
            ndcg = ndcg_at_k(gold, ranked, k=k)
            ndcg_at_k_values[k].append(ndcg)

            # MAP@K
            map_val = map_at_k(gold, ranked, k=k)
            map_at_k_values[k].append(map_val)

            # Hit Rate@K (at least one gold in top-K)
            gold_set = set(gold)
            top_k_set = set(ranked[:k])
            hit = 1.0 if len(gold_set & top_k_set) > 0 else 0.0
            hit_rate_at_k_values[k].append(hit)

    return RankingMetrics(
        mrr=np.mean(mrr_values) if mrr_values else 0.0,
        map_at_k={k: np.mean(v) if v else 0.0 for k, v in map_at_k_values.items()},
        ndcg_at_k={k: np.mean(v) if v else 0.0 for k, v in ndcg_at_k_values.items()},
        recall_at_k={k: np.mean(v) if v else 0.0 for k, v in recall_at_k_values.items()},
        hit_rate_at_k={k: np.mean(v) if v else 0.0 for k, v in hit_rate_at_k_values.items()},
        n_queries=len(mrr_values),
    )


def compute_dynamic_k_metrics(
    selected_k: np.ndarray,
    ne_pred: np.ndarray,
    has_evidence: np.ndarray,
    gold_counts: np.ndarray,
    retrieved_gold_counts: np.ndarray,
) -> DynamicKMetrics:
    """Compute Dynamic-K metrics.

    Args:
        selected_k: K selected for each query (0 if NE gate blocked)
        ne_pred: NE gate predictions (0/1)
        has_evidence: Ground truth has_evidence (0/1)
        gold_counts: Number of gold items per query
        retrieved_gold_counts: Number of gold items retrieved per query

    Returns:
        DynamicKMetrics object
    """
    n_total = len(selected_k)
    pred_pos_mask = ne_pred == 1
    n_pred_pos = int(pred_pos_mask.sum())

    # Average K among predicted positive
    if n_pred_pos > 0:
        avg_k_pred_pos = float(np.mean(selected_k[pred_pos_mask]))
        std_k = float(np.std(selected_k[pred_pos_mask]))
    else:
        avg_k_pred_pos = 0.0
        std_k = 0.0

    # Average K over all (including K=0 for predicted negative)
    avg_k_all = float(np.mean(selected_k))

    # Evidence Recall (Conditional): Among pred_pos AND has_evidence
    cond_mask = pred_pos_mask & (has_evidence == 1)
    if cond_mask.sum() > 0:
        cond_gold = gold_counts[cond_mask].sum()
        cond_retrieved = retrieved_gold_counts[cond_mask].sum()
        evidence_recall_conditional = cond_retrieved / cond_gold if cond_gold > 0 else 0.0
    else:
        evidence_recall_conditional = 0.0

    # Evidence Recall (Unconditional): All has_evidence queries
    has_ev_mask = has_evidence == 1
    if has_ev_mask.sum() > 0:
        total_gold = gold_counts[has_ev_mask].sum()
        total_retrieved = retrieved_gold_counts[has_ev_mask].sum()
        evidence_recall_unconditional = total_retrieved / total_gold if total_gold > 0 else 0.0
    else:
        evidence_recall_unconditional = 0.0

    # K distribution (among pred_pos)
    if n_pred_pos > 0:
        k_pred_pos = selected_k[pred_pos_mask]
        k_distribution = {
            "min": int(np.min(k_pred_pos)),
            "max": int(np.max(k_pred_pos)),
            "median": float(np.median(k_pred_pos)),
            "mean": float(np.mean(k_pred_pos)),
            "std": float(np.std(k_pred_pos)),
            "p25": float(np.percentile(k_pred_pos, 25)),
            "p75": float(np.percentile(k_pred_pos, 75)),
        }
    else:
        k_distribution = {}

    return DynamicKMetrics(
        avg_k_pred_pos=avg_k_pred_pos,
        avg_k_all=avg_k_all,
        std_k=std_k,
        evidence_recall_conditional=evidence_recall_conditional,
        evidence_recall_unconditional=evidence_recall_unconditional,
        k_distribution=k_distribution,
        n_pred_pos=n_pred_pos,
        n_total=n_total,
    )


# ============================================================================
# Dynamic-K Selection Functions (with bug fixes)
# ============================================================================

def compute_k_constraints(
    n_candidates: int,
    k_min: int = 2,
    k_max: int = 10,
    k_max_ratio: float = 0.5,
) -> Tuple[int, int]:
    """Compute K constraints for a query.

    Returns:
        (actual_k_min, actual_k_max)
    """
    k_max_from_ratio = int(np.ceil(n_candidates * k_max_ratio))
    actual_k_max = min(k_max, k_max_from_ratio, n_candidates)
    actual_k_max = max(actual_k_max, k_min)  # Ensure k_max >= k_min
    return k_min, actual_k_max


def select_k_threshold(
    probs: np.ndarray,
    tau: float,
    k_min: int,
    k_max: int,
) -> int:
    """DK-A: Select K based on probability threshold."""
    k = int(np.sum(probs >= tau))
    return max(k_min, min(k, k_max))


def select_k_mass(
    probs: np.ndarray,
    gamma: float,
    k_min: int,
    k_max: int,
) -> int:
    """DK-B: Select K based on cumulative probability mass.

    Select minimum K such that cumsum(sorted_probs) >= gamma.

    BUG FIX: The original implementation used raw probs.
    This version normalizes to valid probabilities first.
    """
    # Sort descending
    sorted_probs = np.sort(probs)[::-1]

    # Normalize to sum to 1 (like a probability distribution)
    prob_sum = sorted_probs.sum()
    if prob_sum > 0:
        normalized = sorted_probs / prob_sum
        cumsum = np.cumsum(normalized)
    else:
        # All zeros - default to k_min
        return k_min

    # Find first index where cumsum >= gamma
    idx = np.searchsorted(cumsum, gamma)
    k = idx + 1  # +1 because idx is 0-based

    return max(k_min, min(k, k_max))


def select_k_fixed(
    n_candidates: int,
    fixed_k: int,
    k_min: int,
    k_max: int,
) -> int:
    """DK-C: Fixed K (clamped to constraints)."""
    return max(k_min, min(fixed_k, k_max))


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_graph_dataset(graph_dir: Path) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache.

    Returns:
        (fold_graphs, metadata)
    """
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    fold_graphs = {}
    n_folds = metadata["n_folds"]

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        fold_graphs[fold_id] = data["graphs"]
        logger.info(f"Loaded fold {fold_id}: {len(data['graphs'])} graphs")

    return fold_graphs, metadata


def extract_raw_predictions(
    graphs: List,
    p4_preds: Optional[np.ndarray] = None,
    p3_refined_scores: Optional[List[np.ndarray]] = None,
    p2_node_probs: Optional[List[np.ndarray]] = None,
) -> pd.DataFrame:
    """Extract raw predictions into canonical format.

    Args:
        graphs: List of PyG Data objects
        p4_preds: P4 NE probabilities (optional)
        p3_refined_scores: P3 refined scores per query (optional)
        p2_node_probs: P2 node probabilities per query (optional)

    Returns:
        DataFrame with canonical raw prediction format
    """
    records = []

    for i, g in enumerate(graphs):
        # Extract graph attributes
        query_id = g.query_id
        post_id = g.post_id
        criterion_id = g.criterion_id

        # Ground truth
        has_evidence = int(g.y.item()) if hasattr(g, 'y') else 0
        gold_mask = g.node_labels.numpy() > 0 if hasattr(g, 'node_labels') else np.zeros(g.x.size(0), dtype=bool)

        # Candidate info
        candidate_ids = g.candidate_uids
        reranker_scores = g.reranker_scores.numpy()

        # Gold evidence IDs
        gold_ids = [candidate_ids[j] for j in np.where(gold_mask)[0]]

        # P4 NE probability
        ne_prob = p4_preds[i] if p4_preds is not None else 0.5

        # P3 refined scores
        if p3_refined_scores is not None and i < len(p3_refined_scores):
            refined_scores = p3_refined_scores[i]
        else:
            refined_scores = reranker_scores

        # P2 node probs
        if p2_node_probs is not None and i < len(p2_node_probs):
            node_probs = p2_node_probs[i]
        else:
            node_probs = np.ones(len(candidate_ids)) / len(candidate_ids)

        records.append({
            "query_id": query_id,
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gt_has_evidence": has_evidence,
            "n_gold": len(gold_ids),
            "gold_evidence_ids": gold_ids,
            "candidate_ids": candidate_ids,
            "reranker_scores": reranker_scores.tolist(),
            "refined_scores": refined_scores.tolist() if isinstance(refined_scores, np.ndarray) else refined_scores,
            "node_probs": node_probs.tolist() if isinstance(node_probs, np.ndarray) else node_probs,
            "ne_prob": float(ne_prob),
            "n_candidates": len(candidate_ids),
        })

    return pd.DataFrame(records)


# ============================================================================
# Pipeline Composition and Evaluation
# ============================================================================

def evaluate_variant(
    raw_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    use_p3: bool = False,
    use_p4: bool = True,
    use_p2: bool = True,
    fpr_budget: float = 0.05,
    dynamic_k_policy: str = "mass",
    dynamic_k_param: float = 0.8,
    k_min: int = 2,
    k_max: int = 10,
    k_max_ratio: float = 0.5,
    variant_name: str = "",
    fold_id: int = -1,
) -> Tuple[E2EMetrics, pd.DataFrame]:
    """Evaluate a pipeline variant.

    Args:
        raw_df: Eval set raw predictions
        tune_df: Tune set for threshold selection
        use_p3: Use P3 refined scores for ranking
        use_p4: Use P4 NE gate
        use_p2: Use P2 Dynamic-K (otherwise fixed K)
        fpr_budget: FPR budget for threshold selection
        dynamic_k_policy: "threshold", "mass", or "fixed"
        dynamic_k_param: tau for threshold, gamma for mass, K for fixed
        k_min, k_max, k_max_ratio: K constraints
        variant_name: Name for this variant
        fold_id: Fold identifier

    Returns:
        (E2EMetrics, per_query_df)
    """
    n_queries = len(raw_df)

    # ===== Step 1: NE Gate Threshold Selection (on TUNE set) =====
    if use_p4:
        tune_probs = tune_df["ne_prob"].values
        tune_labels = tune_df["gt_has_evidence"].values
        threshold = select_threshold_on_tune(tune_probs, tune_labels, fpr_budget)
    else:
        threshold = 0.0  # Always predict positive

    # ===== Step 2: Apply NE Gate =====
    ne_probs = raw_df["ne_prob"].values
    ne_preds = (ne_probs >= threshold).astype(int) if use_p4 else np.ones(n_queries, dtype=int)

    # ===== Step 3: Select K and retrieve candidates =====
    selected_ks = np.zeros(n_queries, dtype=int)
    retrieved_ids = []
    retrieved_gold_counts = np.zeros(n_queries, dtype=int)

    for i, row in raw_df.iterrows():
        if ne_preds[i] == 0:
            # NE gate blocked - return nothing
            selected_ks[i] = 0
            retrieved_ids.append([])
            continue

        n_candidates = row["n_candidates"]
        actual_k_min, actual_k_max = compute_k_constraints(n_candidates, k_min, k_max, k_max_ratio)

        # Get scores for ranking
        if use_p3:
            scores = np.array(row["refined_scores"])
        else:
            scores = np.array(row["reranker_scores"])

        # Select K
        if use_p2:
            node_probs = np.array(row["node_probs"])
            if dynamic_k_policy == "threshold":
                k = select_k_threshold(node_probs, dynamic_k_param, actual_k_min, actual_k_max)
            elif dynamic_k_policy == "mass":
                k = select_k_mass(node_probs, dynamic_k_param, actual_k_min, actual_k_max)
            elif dynamic_k_policy == "fixed":
                k = select_k_fixed(n_candidates, int(dynamic_k_param), actual_k_min, actual_k_max)
            else:
                raise ValueError(f"Unknown policy: {dynamic_k_policy}")
        else:
            # Default: use fixed K=5
            k = select_k_fixed(n_candidates, 5, actual_k_min, actual_k_max)

        selected_ks[i] = k

        # Retrieve top-K by score
        candidate_ids = row["candidate_ids"]
        sorted_idx = np.argsort(-scores)[:k]
        retrieved = [candidate_ids[j] for j in sorted_idx]
        retrieved_ids.append(retrieved)

        # Count retrieved gold
        gold_set = set(row["gold_evidence_ids"])
        retrieved_gold_counts[i] = len(set(retrieved) & gold_set)

    # ===== Step 4: Compute Metrics =====

    # NE Metrics
    ne_metrics = compute_ne_metrics(
        ne_probs,
        raw_df["gt_has_evidence"].values,
        threshold=threshold,
    )

    # Ranking Metrics (evidence queries only, that passed NE gate)
    evidence_mask = (raw_df["gt_has_evidence"].values == 1) & (ne_preds == 1)
    if evidence_mask.sum() > 0:
        evidence_df = raw_df[evidence_mask].reset_index(drop=True)
        evidence_retrieved = [retrieved_ids[i] for i, m in enumerate(evidence_mask) if m]
        evidence_gold = evidence_df["gold_evidence_ids"].tolist()
        ranking_metrics = compute_ranking_metrics(evidence_retrieved, evidence_gold)
    else:
        ranking_metrics = RankingMetrics()

    # Dynamic-K Metrics
    dynamic_k_metrics = compute_dynamic_k_metrics(
        selected_ks,
        ne_preds,
        raw_df["gt_has_evidence"].values,
        raw_df["n_gold"].values,
        retrieved_gold_counts,
    )

    # ===== Step 5: Build per-query output =====
    per_query_df = raw_df.copy()
    per_query_df["ne_pred"] = ne_preds
    per_query_df["ne_threshold"] = threshold
    per_query_df["selected_k"] = selected_ks
    per_query_df["retrieved_ids"] = retrieved_ids
    per_query_df["retrieved_gold_count"] = retrieved_gold_counts
    per_query_df["variant"] = variant_name
    per_query_df["fold_id"] = fold_id

    return E2EMetrics(
        ne_metrics=ne_metrics,
        ranking_metrics=ranking_metrics,
        dynamic_k_metrics=dynamic_k_metrics,
        variant_name=variant_name,
        fold_id=fold_id,
    ), per_query_df


def run_fold_evaluation(
    fold_id: int,
    train_graphs: List,
    tune_graphs: List,
    eval_graphs: List,
    output_dir: Path,
    variants: Dict[str, Dict],
) -> Dict[str, E2EMetrics]:
    """Run evaluation for all variants on one fold.

    Args:
        fold_id: Fold number
        train_graphs: Training graphs (not used for evaluation)
        tune_graphs: Tuning graphs (for threshold selection)
        eval_graphs: Evaluation graphs (for final metrics)
        output_dir: Output directory
        variants: Dict of variant_name -> variant_config

    Returns:
        Dict of variant_name -> E2EMetrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_id}: tune={len(tune_graphs)}, eval={len(eval_graphs)}")
    logger.info(f"{'='*60}")

    # Extract raw predictions (using baseline - no P3/P4/P2 inference in this script)
    # In a full implementation, you would load or infer P3/P4/P2 predictions here
    tune_df = extract_raw_predictions(tune_graphs)
    eval_df = extract_raw_predictions(eval_graphs)

    # Save raw predictions
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(exist_ok=True)
    eval_df.to_parquet(raw_dir / f"fold_{fold_id}_eval.parquet")
    tune_df.to_parquet(raw_dir / f"fold_{fold_id}_tune.parquet")

    results = {}
    all_per_query = []

    for variant_name, config in variants.items():
        logger.info(f"  Evaluating variant: {variant_name}")

        metrics, per_query_df = evaluate_variant(
            eval_df,
            tune_df,
            use_p3=config.get("use_p3", False),
            use_p4=config.get("use_p4", True),
            use_p2=config.get("use_p2", True),
            fpr_budget=config.get("fpr_budget", 0.05),
            dynamic_k_policy=config.get("dynamic_k_policy", "mass"),
            dynamic_k_param=config.get("dynamic_k_param", 0.8),
            variant_name=variant_name,
            fold_id=fold_id,
        )

        results[variant_name] = metrics
        all_per_query.append(per_query_df)

        # Log key metrics
        logger.info(f"    AUROC: {metrics.ne_metrics.auroc:.4f}")
        logger.info(f"    TPR@5%FPR: {metrics.ne_metrics.tpr_at_5pct_fpr:.4f}")
        logger.info(f"    Avg K (pred_pos): {metrics.dynamic_k_metrics.avg_k_pred_pos:.2f}")
        logger.info(f"    Evidence Recall: {metrics.dynamic_k_metrics.evidence_recall_unconditional:.4f}")

    # Save per-query results
    if all_per_query:
        combined_df = pd.concat(all_per_query, ignore_index=True)
        combined_df.to_parquet(output_dir / f"per_query_fold_{fold_id}.parquet")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def get_git_info() -> Dict[str, str]:
    """Get git commit hash and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except:
        return {"commit": "unknown", "branch": "unknown"}


def aggregate_fold_results(
    fold_results: List[Dict[str, E2EMetrics]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate results across folds."""
    aggregated = {}

    # Get all variant names
    variant_names = list(fold_results[0].keys())

    for variant_name in variant_names:
        variant_metrics = [fold[variant_name] for fold in fold_results]

        # Aggregate NE metrics
        aurocs = [m.ne_metrics.auroc for m in variant_metrics]
        auprcs = [m.ne_metrics.auprc for m in variant_metrics]
        tpr_5s = [m.ne_metrics.tpr_at_5pct_fpr for m in variant_metrics]
        tpr_3s = [m.ne_metrics.tpr_at_3pct_fpr for m in variant_metrics]
        tpr_10s = [m.ne_metrics.tpr_at_10pct_fpr for m in variant_metrics]

        # Aggregate Dynamic-K metrics
        avg_ks = [m.dynamic_k_metrics.avg_k_pred_pos for m in variant_metrics]
        ev_recalls = [m.dynamic_k_metrics.evidence_recall_unconditional for m in variant_metrics]

        # Aggregate ranking metrics
        mrrs = [m.ranking_metrics.mrr for m in variant_metrics]

        aggregated[variant_name] = {
            "ne": {
                "auroc": {"mean": np.mean(aurocs), "std": np.std(aurocs)},
                "auprc": {"mean": np.mean(auprcs), "std": np.std(auprcs)},
                "tpr_at_3pct_fpr": {"mean": np.mean(tpr_3s), "std": np.std(tpr_3s)},
                "tpr_at_5pct_fpr": {"mean": np.mean(tpr_5s), "std": np.std(tpr_5s)},
                "tpr_at_10pct_fpr": {"mean": np.mean(tpr_10s), "std": np.std(tpr_10s)},
            },
            "dynamic_k": {
                "avg_k_pred_pos": {"mean": np.mean(avg_ks), "std": np.std(avg_ks)},
                "evidence_recall": {"mean": np.mean(ev_recalls), "std": np.std(ev_recalls)},
            },
            "ranking": {
                "mrr": {"mean": np.mean(mrrs), "std": np.std(mrrs)},
            },
            "per_fold": [m.to_dict() for m in variant_metrics],
        }

    return aggregated


def generate_report(
    aggregated: Dict[str, Dict[str, Any]],
    output_dir: Path,
    metadata: Dict,
) -> None:
    """Generate markdown report."""
    report_lines = [
        "# GNN E2E Evaluation Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Dataset**: {metadata.get('total_queries', 0)} queries, {metadata.get('n_folds', 5)} folds",
        f"**Has Evidence Rate**: {metadata.get('has_evidence_rate', 0):.2%}",
        "",
        "---",
        "",
        "## NE Detection Leaderboard",
        "",
        "| Variant | AUROC | AUPRC | TPR@5%FPR |",
        "|---------|-------|-------|-----------|",
    ]

    for variant, metrics in aggregated.items():
        ne = metrics["ne"]
        auroc = f"{ne['auroc']['mean']:.4f} ± {ne['auroc']['std']:.4f}"
        auprc = f"{ne['auprc']['mean']:.4f} ± {ne['auprc']['std']:.4f}"
        tpr5 = f"{ne['tpr_at_5pct_fpr']['mean']:.4f} ± {ne['tpr_at_5pct_fpr']['std']:.4f}"
        report_lines.append(f"| {variant} | {auroc} | {auprc} | {tpr5} |")

    report_lines.extend([
        "",
        "## Dynamic-K Metrics",
        "",
        "| Variant | Avg K | Evidence Recall |",
        "|---------|-------|-----------------|",
    ])

    for variant, metrics in aggregated.items():
        dk = metrics["dynamic_k"]
        avg_k = f"{dk['avg_k_pred_pos']['mean']:.2f} ± {dk['avg_k_pred_pos']['std']:.2f}"
        ev_rec = f"{dk['evidence_recall']['mean']:.4f} ± {dk['evidence_recall']['std']:.4f}"
        report_lines.append(f"| {variant} | {avg_k} | {ev_rec} |")

    report_lines.extend([
        "",
        "---",
        "",
        "*Report generated by run_e2e_eval_and_report.py*",
    ])

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="GNN E2E Evaluation")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default="data/cache/gnn/20260117_003135",
        help="Graph dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/gnn_e2e_report/<timestamp>)",
    )
    parser.add_argument(
        "--inner_tune_ratio",
        type=float,
        default=0.3,
        help="Ratio of non-eval data to use for tuning",
    )
    parser.add_argument(
        "--fpr_budget",
        type=float,
        default=0.05,
        help="FPR budget for threshold selection",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/gnn_e2e_report") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Save environment info
    git_info = get_git_info()
    env_info = {
        "timestamp": timestamp,
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "args": vars(args),
        "command": " ".join(sys.argv),
    }
    with open(output_dir / "env.txt", "w") as f:
        json.dump(env_info, f, indent=2)

    # Load dataset
    graph_dir = Path(args.graph_dir)
    fold_graphs, metadata = load_graph_dataset(graph_dir)

    # Define variants to evaluate
    variants = {
        "V0_baseline": {
            "use_p3": False,
            "use_p4": False,  # No NE gate
            "use_p2": False,
            "dynamic_k_policy": "fixed",
            "dynamic_k_param": 5,
        },
        "V4_p4_only": {
            "use_p3": False,
            "use_p4": True,
            "use_p2": False,
            "dynamic_k_policy": "fixed",
            "dynamic_k_param": 5,
        },
        "V7_full_mass08": {
            "use_p3": False,  # Would be True if P3 preds available
            "use_p4": True,
            "use_p2": True,
            "dynamic_k_policy": "mass",
            "dynamic_k_param": 0.8,
        },
        "V7_full_mass09": {
            "use_p3": False,
            "use_p4": True,
            "use_p2": True,
            "dynamic_k_policy": "mass",
            "dynamic_k_param": 0.9,
        },
        "V7_full_thresh05": {
            "use_p3": False,
            "use_p4": True,
            "use_p2": True,
            "dynamic_k_policy": "threshold",
            "dynamic_k_param": 0.5,
        },
    }

    # Run evaluation for each fold
    n_folds = metadata["n_folds"]
    fold_results = []

    for eval_fold in range(n_folds):
        # Collect graphs
        eval_graphs = fold_graphs[eval_fold]

        # Combine other folds for train/tune
        other_graphs = []
        for f in range(n_folds):
            if f != eval_fold:
                other_graphs.extend(fold_graphs[f])

        # Split into train/tune
        n_other = len(other_graphs)
        n_tune = int(n_other * args.inner_tune_ratio)

        np.random.seed(42 + eval_fold)
        indices = np.random.permutation(n_other)
        tune_idx = indices[:n_tune]
        train_idx = indices[n_tune:]

        tune_graphs = [other_graphs[i] for i in tune_idx]
        train_graphs = [other_graphs[i] for i in train_idx]

        # Run evaluation
        results = run_fold_evaluation(
            eval_fold,
            train_graphs,
            tune_graphs,
            eval_graphs,
            output_dir,
            variants,
        )
        fold_results.append(results)

    # Aggregate results
    aggregated = aggregate_fold_results(fold_results)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Generate leaderboard CSV
    leaderboard_rows = []
    for variant, metrics in aggregated.items():
        leaderboard_rows.append({
            "variant": variant,
            "auroc_mean": metrics["ne"]["auroc"]["mean"],
            "auroc_std": metrics["ne"]["auroc"]["std"],
            "tpr_5pct_fpr_mean": metrics["ne"]["tpr_at_5pct_fpr"]["mean"],
            "tpr_5pct_fpr_std": metrics["ne"]["tpr_at_5pct_fpr"]["std"],
            "avg_k_mean": metrics["dynamic_k"]["avg_k_pred_pos"]["mean"],
            "avg_k_std": metrics["dynamic_k"]["avg_k_pred_pos"]["std"],
            "evidence_recall_mean": metrics["dynamic_k"]["evidence_recall"]["mean"],
            "evidence_recall_std": metrics["dynamic_k"]["evidence_recall"]["std"],
        })
    pd.DataFrame(leaderboard_rows).to_csv(output_dir / "leaderboard.csv", index=False)

    # Generate report
    generate_report(aggregated, output_dir, metadata)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output: {output_dir}")
    logger.info("\nLeaderboard:")
    for variant, metrics in aggregated.items():
        auroc = metrics["ne"]["auroc"]["mean"]
        tpr5 = metrics["ne"]["tpr_at_5pct_fpr"]["mean"]
        avg_k = metrics["dynamic_k"]["avg_k_pred_pos"]["mean"]
        logger.info(f"  {variant}: AUROC={auroc:.4f}, TPR@5%FPR={tpr5:.4f}, AvgK={avg_k:.2f}")


if __name__ == "__main__":
    main()
