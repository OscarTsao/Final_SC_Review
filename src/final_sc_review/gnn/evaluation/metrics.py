"""Metric computation for GNN models.

Metrics for NE Gate:
- AUROC, AUPRC
- TPR at FPR levels (3%, 5%, 10%)
- Threshold at target FPR

Metrics for Dynamic-K:
- Evidence recall at different K
- Average K
- F1 for evidence retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class NEGateMetrics:
    """Metrics for NE Gate assessment."""
    auroc: float
    auprc: float
    tpr_at_fpr_3pct: float
    tpr_at_fpr_5pct: float
    tpr_at_fpr_10pct: float
    threshold_at_fpr_5pct: float
    n_samples: int
    has_evidence_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "tpr_at_fpr_3pct": self.tpr_at_fpr_3pct,
            "tpr_at_fpr_5pct": self.tpr_at_fpr_5pct,
            "tpr_at_fpr_10pct": self.tpr_at_fpr_10pct,
            "threshold_at_fpr_5pct": self.threshold_at_fpr_5pct,
            "n_samples": self.n_samples,
            "has_evidence_rate": self.has_evidence_rate,
        }

    @classmethod
    def compute(
        cls,
        y_prob: np.ndarray,
        y_true: np.ndarray,
    ) -> "NEGateMetrics":
        """Compute all metrics from predictions and labels."""
        n_samples = len(y_true)
        has_evidence_rate = y_true.mean()

        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            return cls(
                auroc=0.5,
                auprc=has_evidence_rate,
                tpr_at_fpr_3pct=0.0,
                tpr_at_fpr_5pct=0.0,
                tpr_at_fpr_10pct=0.0,
                threshold_at_fpr_5pct=1.0,
                n_samples=n_samples,
                has_evidence_rate=has_evidence_rate,
            )

        # AUROC and AUPRC
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        # TPR at FPR levels
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        def get_tpr_at_fpr(target_fpr: float) -> Tuple[float, float]:
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) == 0:
                return 0.0, 1.0
            best_idx = idx[-1]
            return float(tpr[best_idx]), float(thresholds[best_idx])

        tpr_3, _ = get_tpr_at_fpr(0.03)
        tpr_5, thresh_5 = get_tpr_at_fpr(0.05)
        tpr_10, _ = get_tpr_at_fpr(0.10)

        return cls(
            auroc=auroc,
            auprc=auprc,
            tpr_at_fpr_3pct=tpr_3,
            tpr_at_fpr_5pct=tpr_5,
            tpr_at_fpr_10pct=tpr_10,
            threshold_at_fpr_5pct=thresh_5,
            n_samples=n_samples,
            has_evidence_rate=has_evidence_rate,
        )


@dataclass
class DynamicKMetrics:
    """Metrics for Dynamic-K assessment."""
    recall_mean: float  # Average recall across queries
    precision_mean: float  # Average precision
    f1_mean: float  # Average F1
    avg_k: float  # Average K selected
    recall_at_fixed_k: Dict[int, float]  # Recall at K=1,3,5,10
    n_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_mean": self.recall_mean,
            "precision_mean": self.precision_mean,
            "f1_mean": self.f1_mean,
            "avg_k": self.avg_k,
            "recall_at_fixed_k": self.recall_at_fixed_k,
            "n_queries": self.n_queries,
        }

    @classmethod
    def compute(
        cls,
        selected_ids: List[List[str]],  # Per-query selected UIDs
        gold_ids: List[List[str]],  # Per-query gold UIDs
        k_values: Optional[List[int]] = None,  # K per query
    ) -> "DynamicKMetrics":
        """Compute metrics from selected and gold IDs."""
        n_queries = len(selected_ids)
        if n_queries == 0:
            return cls(
                recall_mean=0.0,
                precision_mean=0.0,
                f1_mean=0.0,
                avg_k=0.0,
                recall_at_fixed_k={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
                n_queries=0,
            )

        recalls = []
        precisions = []
        k_list = []

        for i, (selected, gold) in enumerate(zip(selected_ids, gold_ids)):
            gold_set = set(gold)
            selected_set = set(selected)

            if len(gold_set) > 0:
                recall = len(selected_set & gold_set) / len(gold_set)
            else:
                recall = 1.0 if len(selected_set) == 0 else 0.0

            if len(selected_set) > 0:
                precision = len(selected_set & gold_set) / len(selected_set)
            else:
                precision = 1.0 if len(gold_set) == 0 else 0.0

            recalls.append(recall)
            precisions.append(precision)
            k_list.append(len(selected))

        recall_mean = np.mean(recalls)
        precision_mean = np.mean(precisions)
        f1_mean = 2 * recall_mean * precision_mean / (recall_mean + precision_mean + 1e-10)
        avg_k = np.mean(k_list)

        # Recall at fixed K
        recall_at_k = {}
        for k in [1, 3, 5, 10]:
            recalls_k = []
            for i, (selected, gold) in enumerate(zip(selected_ids, gold_ids)):
                gold_set = set(gold)
                selected_k = set(selected[:k])
                if len(gold_set) > 0:
                    recalls_k.append(len(selected_k & gold_set) / len(gold_set))
                else:
                    recalls_k.append(1.0)
            recall_at_k[k] = np.mean(recalls_k)

        return cls(
            recall_mean=recall_mean,
            precision_mean=precision_mean,
            f1_mean=f1_mean,
            avg_k=avg_k,
            recall_at_fixed_k=recall_at_k,
            n_queries=n_queries,
        )


def compute_threshold_at_fpr(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    target_fpr: float = 0.05,
) -> Tuple[float, float, float]:
    """Find threshold that achieves target FPR.

    Returns:
        (threshold, achieved_tpr, achieved_fpr)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 1.0, 0.0, 0.0

    best_idx = idx[-1]
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


def aggregate_fold_metrics(
    fold_metrics: List[NEGateMetrics],
) -> Dict[str, Tuple[float, float]]:
    """Aggregate metrics across folds with mean ± std."""
    result = {}

    for key in ["auroc", "auprc", "tpr_at_fpr_3pct", "tpr_at_fpr_5pct", "tpr_at_fpr_10pct"]:
        values = [getattr(m, key) for m in fold_metrics]
        result[key] = (np.mean(values), np.std(values))

    return result
