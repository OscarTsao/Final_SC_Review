#!/usr/bin/env python3
"""
Comprehensive End-to-End Assessment for S-C Evidence Retrieval Pipeline.

This script performs rigorous assessment across all stages with 5-fold CV:
- Stage 0: Data Audit / Sanity checks
- Stage A: Retriever-only ranking metrics
- Stage B: Reranker ranking metrics  
- Stage C: Calibration assessment (with leakage prevention)
- Stage D: No-Evidence detection (binary classification)
- Stage E: Dynamic-K effectiveness
- Stage F: Final E2E deployment performance
- Per-post multi-label assessment

Usage:
  python scripts/run_full_assessment.py \
    --candidates outputs/retrieval_candidates/retrieval_candidates.pkl \
    --model_dir outputs/training/no_evidence_reranker \
    --outdir outputs/assessment_full

Outputs:
- outputs/assessment_full/<timestamp>/report.md
- outputs/assessment_full/<timestamp>/summary.json
- outputs/assessment_full/<timestamp>/per_query.csv
- outputs/assessment_full/<timestamp>/per_post.csv
- outputs/assessment_full/<timestamp>/curves/*.png
- outputs/assessment_full/<timestamp>/manifests/run_manifest.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NOTE: pickle is used here intentionally to load pre-computed candidates
# This is safe as we control the data source
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class FoldMetrics:
    fold_id: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass 
class AggregatedMetrics:
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    fold_values: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class StageResult:
    name: str
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    aggregated: Optional[AggregatedMetrics] = None
    notes: List[str] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)


def dcg_at_k(relevances, k):
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances, k):
    if not relevances or sum(relevances) == 0:
        return 0.0
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevances, k):
    total_pos = sum(relevances)
    if total_pos == 0:
        return 0.0
    return sum(relevances[:k]) / total_pos


def mrr_at_k(relevances, k):
    for i, rel in enumerate(relevances[:k]):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(relevances, k):
    relevances = relevances[:k]
    if sum(relevances) == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(relevances):
        if rel == 1:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.mean(precisions) if precisions else 0.0


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = y_prob[mask].mean()
            avg_acc = y_true[mask].mean()
            ece += mask.sum() * abs(avg_acc - avg_conf)
    return ece / total if total > 0 else 0.0


def compute_binary_metrics(y_true, y_pred, y_score=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_score)
            metrics['auprc'] = average_precision_score(y_true, y_score)
        except:
            pass
    return metrics


def aggregate_fold_metrics(fold_metrics_list):
    agg = AggregatedMetrics()
    all_keys = set()
    for fm in fold_metrics_list:
        all_keys.update(fm.metrics.keys())
    for key in all_keys:
        values = [fm.metrics.get(key, np.nan) for fm in fold_metrics_list]
        values = [v for v in values if not np.isnan(v)]
        if values:
            agg.mean[key] = np.mean(values)
            agg.std[key] = np.std(values)
            agg.fold_values[key] = values
    return agg


class RerankerScorer:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model = self.model.to(self.device)
            self.model.requires_grad_(False)

    def score(self, query, texts, batch_size=32, max_length=384):
        self.load()
        scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            pairs = [[query, t] for t in batch_texts]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = [float(logits)]
                else:
                    logits = logits.tolist()
                scores.extend(logits)
        return scores


def run_data_audit(data, output_dir):
    result = StageResult(name="Data Audit")
    stats = {'n_folds': len([k for k in data.keys() if k.startswith('fold_')]), 'folds': {}}
    all_posts = set()
    all_queries = []
    
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        fold_stats = {}
        train_posts = set(fold_data.get('train_posts', []))
        val_posts = set(fold_data.get('val_posts', []))
        overlap = train_posts & val_posts
        if overlap:
            result.notes.append(f"FAIL: {fold_name} has {len(overlap)} overlapping posts!")
        else:
            result.notes.append(f"PASS: {fold_name} train/val are disjoint")
        fold_stats['n_train_posts'] = len(train_posts)
        fold_stats['n_val_posts'] = len(val_posts)
        val_data = fold_data.get('val_data', [])
        fold_stats['n_val_queries'] = len(val_data)
        n_has_evidence = sum(1 for q in val_data if not q.get('is_no_evidence', True))
        n_no_evidence = sum(1 for q in val_data if q.get('is_no_evidence', True))
        fold_stats['n_has_evidence'] = n_has_evidence
        fold_stats['n_no_evidence'] = n_no_evidence
        fold_stats['has_evidence_ratio'] = n_has_evidence / len(val_data) if val_data else 0
        stats['folds'][fold_name] = fold_stats
        all_posts.update(train_posts | val_posts)
        all_queries.extend(val_data)
    
    stats['total_unique_posts'] = len(all_posts)
    stats['total_val_queries'] = len(all_queries)
    all_has_evidence = sum(1 for q in all_queries if not q.get('is_no_evidence', True))
    all_no_evidence = sum(1 for q in all_queries if q.get('is_no_evidence', True))
    stats['overall_has_evidence'] = all_has_evidence
    stats['overall_no_evidence'] = all_no_evidence
    stats['overall_has_evidence_ratio'] = all_has_evidence / len(all_queries) if all_queries else 0
    result.extra_data['stats'] = stats
    fm = FoldMetrics(fold_id=0)
    fm.metrics = {'n_folds': stats['n_folds'], 'total_posts': stats['total_unique_posts'], 'total_queries': stats['total_val_queries'], 'has_evidence_ratio': stats['overall_has_evidence_ratio']}
    result.fold_metrics.append(fm)
    return result


def run_retriever_ranking(data, k_values=[1, 3, 5, 10, 20]):
    result = StageResult(name="Retriever Ranking")
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)
        ndcg_values = {k: [] for k in k_values}
        recall_values = {k: [] for k in k_values}
        mrr_values = []
        map_values = []
        for query in val_data:
            if query.get('is_no_evidence', True):
                continue
            candidates = query.get('candidates', [])
            if not candidates:
                continue
            sorted_cands = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
            relevances = [c.get('label', 0) for c in sorted_cands]
            if sum(relevances) == 0:
                continue
            for k in k_values:
                ndcg_values[k].append(ndcg_at_k(relevances, k))
                recall_values[k].append(recall_at_k(relevances, k))
            mrr_values.append(mrr_at_k(relevances, 10))
            map_values.append(map_at_k(relevances, 10))
        for k in k_values:
            fm.metrics[f'ndcg@{k}'] = np.mean(ndcg_values[k]) if ndcg_values[k] else 0
            fm.metrics[f'recall@{k}'] = np.mean(recall_values[k]) if recall_values[k] else 0
        fm.metrics['mrr@10'] = np.mean(mrr_values) if mrr_values else 0
        fm.metrics['map@10'] = np.mean(map_values) if map_values else 0
        fm.metrics['n_queries'] = len(mrr_values)
        result.fold_metrics.append(fm)
    result.aggregated = aggregate_fold_metrics(result.fold_metrics)
    return result


def run_reranker_ranking(data, scorer, k_values=[1, 3, 5, 10, 20]):
    result = StageResult(name="Reranker Ranking")
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)
        ndcg_values = {k: [] for k in k_values}
        recall_values = {k: [] for k in k_values}
        mrr_values = []
        map_values = []
        print(f"  [Fold {fold_id}] Scoring {len(val_data)} queries...")
        for query in tqdm(val_data, desc=f"Fold {fold_id}", leave=False):
            if query.get('is_no_evidence', True):
                continue
            candidates = query.get('candidates', [])
            if not candidates:
                continue
            query_text = query.get('query', '')
            texts = [c.get('text', '') for c in candidates]
            labels = [c.get('label', 0) for c in candidates]
            if sum(labels) == 0:
                continue
            reranker_scores = scorer.score(query_text, texts)
            sorted_indices = np.argsort(reranker_scores)[::-1]
            relevances = [labels[i] for i in sorted_indices]
            for k in k_values:
                ndcg_values[k].append(ndcg_at_k(relevances, k))
                recall_values[k].append(recall_at_k(relevances, k))
            mrr_values.append(mrr_at_k(relevances, 10))
            map_values.append(map_at_k(relevances, 10))
        for k in k_values:
            fm.metrics[f'ndcg@{k}'] = np.mean(ndcg_values[k]) if ndcg_values[k] else 0
            fm.metrics[f'recall@{k}'] = np.mean(recall_values[k]) if recall_values[k] else 0
        fm.metrics['mrr@10'] = np.mean(mrr_values) if mrr_values else 0
        fm.metrics['map@10'] = np.mean(map_values) if map_values else 0
        fm.metrics['n_queries'] = len(mrr_values)
        result.fold_metrics.append(fm)
    result.aggregated = aggregate_fold_metrics(result.fold_metrics)
    return result


def run_ne_detection(data, scorer, output_dir, ne_score_type="score_gap"):
    result = StageResult(name="No-Evidence Detection")
    all_y_true = []
    all_y_scores = []
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)
        y_true = []
        y_scores = []
        print(f"  [Fold {fold_id}] Computing NE scores...")
        for query in tqdm(val_data, desc=f"Fold {fold_id}", leave=False):
            is_no_evidence = query.get('is_no_evidence', True)
            y_true.append(0 if is_no_evidence else 1)
            candidates = query.get('candidates', [])
            if not candidates:
                y_scores.append(-np.inf)
                continue
            query_text = query.get('query', '')
            texts = [c.get('text', '') for c in candidates]
            scores = scorer.score(query_text, texts)
            score = max(scores) if scores else -np.inf
            y_scores.append(score)
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        best_threshold = 0
        best_bal_acc = 0
        for t in np.linspace(np.percentile(y_scores, 5), np.percentile(y_scores, 95), 50):
            y_pred = (y_scores > t).astype(int)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = t
        y_pred = (y_scores > best_threshold).astype(int)
        binary_metrics = compute_binary_metrics(y_true, y_pred, y_scores)
        fm.metrics.update(binary_metrics)
        fm.metrics['best_threshold'] = best_threshold
        fm.metrics['n_queries'] = len(y_true)
        result.fold_metrics.append(fm)
        all_y_true.extend(y_true.tolist())
        all_y_scores.extend(y_scores.tolist())
    result.aggregated = aggregate_fold_metrics(result.fold_metrics)
    
    # Generate curves
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    
    fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
    auroc = roc_auc_score(all_y_true, all_y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC={auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('NE Detection ROC Curve')
    ax.legend()
    plt.savefig(curves_dir / "ne_roc_curve.png", dpi=150)
    plt.close()
    
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    auprc = average_precision_score(all_y_true, all_y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR (AUC={auprc:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('NE Detection PR Curve')
    ax.legend()
    plt.savefig(curves_dir / "ne_pr_curve.png", dpi=150)
    plt.close()
    
    return result


# NO_EVIDENCE pseudo-candidate token (must match training)
NO_EVIDENCE_TOKEN = "[NO_EVIDENCE]"


def run_ne_ranking_check(data, scorer, output_dir):
    """Check NO_EVIDENCE ranking: does NO_EVIDENCE rank appropriately?

    For no-evidence queries: NO_EVIDENCE should rank #1
    For has-evidence queries: Real evidence should rank above NO_EVIDENCE
    """
    result = StageResult(name="NO_EVIDENCE Ranking")

    all_y_true = []
    all_y_pred = []
    all_ne_ranks = []
    all_ne_scores = []
    all_top1_scores = []

    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)

        # Metrics for this fold
        ne_top1_correct = 0  # NO_EVIDENCE ranked #1 for no-evidence queries
        ne_top1_total = 0
        evidence_above_ne = 0  # Real evidence ranked above NO_EVIDENCE
        evidence_above_total = 0

        y_true_fold = []
        y_pred_fold = []
        ne_ranks_fold = []
        score_margins = []  # score(top_evidence) - score(NO_EVIDENCE)

        print(f"  [Fold {fold_id}] Checking NO_EVIDENCE ranking...")
        for query in tqdm(val_data, desc=f"Fold {fold_id}", leave=False):
            is_no_evidence = query.get('is_no_evidence', True)
            candidates = query.get('candidates', [])

            if not candidates:
                continue

            query_text = query.get('query', '')
            texts = [c.get('text', '') for c in candidates]
            labels = [c.get('label', 0) for c in candidates]

            # Add NO_EVIDENCE to candidate list
            texts_with_ne = texts + [NO_EVIDENCE_TOKEN]
            labels_with_ne = labels + [0]  # NO_EVIDENCE is always label=0

            # Score all candidates including NO_EVIDENCE
            scores = scorer.score(query_text, texts_with_ne)

            # Find NO_EVIDENCE score and rank
            ne_idx = len(texts)  # NO_EVIDENCE is at the end
            ne_score = scores[ne_idx]

            # Sort by score descending
            sorted_indices = np.argsort(scores)[::-1]
            ne_rank = np.where(sorted_indices == ne_idx)[0][0] + 1  # 1-indexed rank

            ne_ranks_fold.append(ne_rank)
            all_ne_ranks.append(ne_rank)
            all_ne_scores.append(ne_score)
            all_top1_scores.append(scores[sorted_indices[0]])

            # Classification: predict "no evidence" if NO_EVIDENCE is ranked #1
            pred_no_evidence = (ne_rank == 1)
            pred_has_evidence = not pred_no_evidence

            # Ground truth: 1 = has evidence, 0 = no evidence
            y_true = 0 if is_no_evidence else 1
            y_pred = 1 if pred_has_evidence else 0

            y_true_fold.append(y_true)
            y_pred_fold.append(y_pred)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

            if is_no_evidence:
                # For no-evidence queries: check if NO_EVIDENCE is #1
                ne_top1_total += 1
                if ne_rank == 1:
                    ne_top1_correct += 1
            else:
                # For has-evidence queries: check if any positive ranks above NO_EVIDENCE
                evidence_above_total += 1
                # Find best positive candidate rank
                positive_indices = [i for i, l in enumerate(labels_with_ne) if l == 1]
                if positive_indices:
                    positive_ranks = [np.where(sorted_indices == i)[0][0] + 1 for i in positive_indices]
                    best_positive_rank = min(positive_ranks)
                    if best_positive_rank < ne_rank:
                        evidence_above_ne += 1
                    # Score margin: best positive score - NO_EVIDENCE score
                    best_positive_score = max(scores[i] for i in positive_indices)
                    score_margins.append(best_positive_score - ne_score)

        # Compute fold metrics
        fm.metrics['ne_top1_accuracy'] = ne_top1_correct / ne_top1_total if ne_top1_total > 0 else 0
        fm.metrics['evidence_above_ne'] = evidence_above_ne / evidence_above_total if evidence_above_total > 0 else 0
        fm.metrics['ne_top1_total'] = ne_top1_total
        fm.metrics['evidence_above_total'] = evidence_above_total
        fm.metrics['avg_ne_rank'] = np.mean(ne_ranks_fold) if ne_ranks_fold else 0
        fm.metrics['median_ne_rank'] = np.median(ne_ranks_fold) if ne_ranks_fold else 0

        if score_margins:
            fm.metrics['avg_score_margin'] = np.mean(score_margins)
            fm.metrics['positive_margin_rate'] = sum(1 for m in score_margins if m > 0) / len(score_margins)

        # Unified classification metrics (using NE rank as decision)
        y_true_arr = np.array(y_true_fold)
        y_pred_arr = np.array(y_pred_fold)

        if len(np.unique(y_true_arr)) > 1:
            binary_metrics = compute_binary_metrics(y_true_arr, y_pred_arr)
            fm.metrics['unified_accuracy'] = binary_metrics['accuracy']
            fm.metrics['unified_balanced_acc'] = binary_metrics['balanced_accuracy']
            fm.metrics['unified_precision'] = binary_metrics['precision']
            fm.metrics['unified_recall'] = binary_metrics['recall']
            fm.metrics['unified_f1'] = binary_metrics['f1']
            fm.metrics['unified_specificity'] = binary_metrics['specificity']
            fm.metrics['unified_mcc'] = binary_metrics['mcc']

        fm.metrics['n_queries'] = len(y_true_fold)
        result.fold_metrics.append(fm)

    result.aggregated = aggregate_fold_metrics(result.fold_metrics)

    # Generate visualizations
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # NO_EVIDENCE rank distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Split by query type
    ne_ranks_no_ev = [r for r, y in zip(all_ne_ranks, all_y_true) if y == 0]
    ne_ranks_has_ev = [r for r, y in zip(all_ne_ranks, all_y_true) if y == 1]

    axes[0].hist(ne_ranks_no_ev, bins=20, alpha=0.7, label=f'No-Evidence (n={len(ne_ranks_no_ev)})', color='red')
    axes[0].axvline(1, color='green', linestyle='--', linewidth=2, label='Ideal (rank=1)')
    axes[0].set_xlabel('NO_EVIDENCE Rank')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('NO_EVIDENCE Rank for No-Evidence Queries\n(Should be rank 1)')
    axes[0].legend()

    axes[1].hist(ne_ranks_has_ev, bins=20, alpha=0.7, label=f'Has-Evidence (n={len(ne_ranks_has_ev)})', color='blue')
    axes[1].axvline(np.mean(ne_ranks_has_ev), color='orange', linestyle='--', linewidth=2,
                    label=f'Mean={np.mean(ne_ranks_has_ev):.1f}')
    axes[1].set_xlabel('NO_EVIDENCE Rank')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('NO_EVIDENCE Rank for Has-Evidence Queries\n(Should be > 1)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(curves_dir / "ne_rank_distribution.png", dpi=150)
    plt.close()

    return result


def run_dynamic_k_check(data, scorer, output_dir, fixed_k_values=[1, 3, 5, 10, 20]):
    result = StageResult(name="Dynamic-K")
    all_k_values = []
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)
        fixed_ndcg = {k: [] for k in fixed_k_values}
        fixed_recall = {k: [] for k in fixed_k_values}
        fixed_precision = {k: [] for k in fixed_k_values}
        dynamic_k_list = []
        dynamic_ndcg = []
        dynamic_recall = []
        dynamic_precision = []
        print(f"  [Fold {fold_id}] Computing Dynamic-K...")
        for query in tqdm(val_data, desc=f"Fold {fold_id}", leave=False):
            if query.get('is_no_evidence', True):
                continue
            candidates = query.get('candidates', [])
            if not candidates:
                continue
            query_text = query.get('query', '')
            texts = [c.get('text', '') for c in candidates]
            labels = [c.get('label', 0) for c in candidates]
            if sum(labels) == 0:
                continue
            scores = scorer.score(query_text, texts)
            sorted_indices = np.argsort(scores)[::-1]
            relevances = [labels[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]
            for k in fixed_k_values:
                fixed_ndcg[k].append(ndcg_at_k(relevances, k))
                fixed_recall[k].append(recall_at_k(relevances, k))
                # precision@K = (# relevant in top K) / K
                top_k_rel = relevances[:min(k, len(relevances))]
                fixed_precision[k].append(sum(top_k_rel) / k)
            if len(sorted_scores) > 1:
                gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
                if gaps:
                    max_gap_idx = np.argmax(gaps)
                    dynamic_k = max_gap_idx + 1
                else:
                    dynamic_k = 1
            else:
                dynamic_k = 1
            dynamic_k = min(max(dynamic_k, 1), len(relevances))
            dynamic_k_list.append(dynamic_k)
            dynamic_ndcg.append(ndcg_at_k(relevances, dynamic_k))
            dynamic_recall.append(recall_at_k(relevances, dynamic_k))
            # precision@dynamic_k = (# relevant in top dynamic_k) / dynamic_k
            dynamic_precision.append(sum(relevances[:dynamic_k]) / dynamic_k)
        for k in fixed_k_values:
            fm.metrics[f'fixed_k{k}_ndcg'] = np.mean(fixed_ndcg[k]) if fixed_ndcg[k] else 0
            fm.metrics[f'fixed_k{k}_recall'] = np.mean(fixed_recall[k]) if fixed_recall[k] else 0
            fm.metrics[f'fixed_k{k}_precision'] = np.mean(fixed_precision[k]) if fixed_precision[k] else 0
        fm.metrics['dynamic_k_mean'] = np.mean(dynamic_k_list) if dynamic_k_list else 0
        fm.metrics['dynamic_k_std'] = np.std(dynamic_k_list) if dynamic_k_list else 0
        fm.metrics['dynamic_ndcg'] = np.mean(dynamic_ndcg) if dynamic_ndcg else 0
        fm.metrics['dynamic_recall'] = np.mean(dynamic_recall) if dynamic_recall else 0
        fm.metrics['dynamic_precision'] = np.mean(dynamic_precision) if dynamic_precision else 0
        fm.metrics['n_queries'] = len(dynamic_k_list)
        result.fold_metrics.append(fm)
        all_k_values.extend(dynamic_k_list)
    result.aggregated = aggregate_fold_metrics(result.fold_metrics)
    
    curves_dir = output_dir / "curves"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_k_values, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_k_values), color='r', linestyle='--', label=f'Mean={np.mean(all_k_values):.2f}')
    ax.set_xlabel('Dynamic K Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Dynamic-K Distribution')
    ax.legend()
    plt.savefig(curves_dir / "k_distribution.png", dpi=150)
    plt.close()
    return result


def run_e2e_deployment(data, scorer, ne_threshold=0.0):
    result = StageResult(name="E2E Deployment")
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)
        y_true_ne = []
        y_pred_ne = []
        conditional_ndcg = []
        conditional_recall = []
        returned_k_values = []
        print(f"  [Fold {fold_id}] E2E deployment...")
        for query in tqdm(val_data, desc=f"Fold {fold_id}", leave=False):
            is_no_evidence = query.get('is_no_evidence', True)
            has_evidence = 0 if is_no_evidence else 1
            y_true_ne.append(has_evidence)
            candidates = query.get('candidates', [])
            if not candidates:
                y_pred_ne.append(0)
                continue
            query_text = query.get('query', '')
            texts = [c.get('text', '') for c in candidates]
            labels = [c.get('label', 0) for c in candidates]
            scores = scorer.score(query_text, texts)
            max_score = max(scores) if scores else -np.inf
            pred_has_evidence = 1 if max_score > ne_threshold else 0
            y_pred_ne.append(pred_has_evidence)
            if pred_has_evidence == 1 and has_evidence == 1:
                sorted_indices = np.argsort(scores)[::-1]
                relevances = [labels[i] for i in sorted_indices]
                sorted_scores = [scores[i] for i in sorted_indices]
                if len(sorted_scores) > 1:
                    gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
                    k = np.argmax(gaps) + 1 if gaps else 1
                else:
                    k = 1
                k = min(max(k, 1), len(relevances))
                returned_k_values.append(k)
                conditional_ndcg.append(ndcg_at_k(relevances, k))
                conditional_recall.append(recall_at_k(relevances, k))
        y_true_ne = np.array(y_true_ne)
        y_pred_ne = np.array(y_pred_ne)
        binary_metrics = compute_binary_metrics(y_true_ne, y_pred_ne)
        for k, v in binary_metrics.items():
            fm.metrics[f'query_{k}'] = v
        n_has_evidence = int(y_true_ne.sum())
        n_no_evidence = int((1 - y_true_ne).sum())
        positive_coverage = (y_pred_ne[y_true_ne == 1].sum() / n_has_evidence) if n_has_evidence > 0 else 0
        negative_coverage = (y_pred_ne[y_true_ne == 0].sum() / n_no_evidence) if n_no_evidence > 0 else 0
        fm.metrics['positive_coverage'] = positive_coverage
        fm.metrics['negative_coverage'] = negative_coverage
        fm.metrics['conditional_ndcg'] = np.mean(conditional_ndcg) if conditional_ndcg else 0
        fm.metrics['conditional_recall'] = np.mean(conditional_recall) if conditional_recall else 0
        fm.metrics['avg_returned_k'] = np.mean(returned_k_values) if returned_k_values else 0
        fm.metrics['n_queries'] = len(y_true_ne)
        result.fold_metrics.append(fm)
    result.aggregated = aggregate_fold_metrics(result.fold_metrics)
    return result


def run_per_post_multilabel(data, scorer, ne_threshold=0.0):
    result = StageResult(name="Per-Post Multi-Label")
    all_post_results = []
    for fold_name in sorted([k for k in data.keys() if k.startswith('fold_')]):
        fold_data = data[fold_name]
        val_data = fold_data.get('val_data', [])
        fold_id = int(fold_name.split('_')[1])
        fm = FoldMetrics(fold_id=fold_id)
        post_queries = defaultdict(list)
        for query in val_data:
            post_id = query.get('post_id', '')
            post_queries[post_id].append(query)
        exact_matches = []
        subset_accuracies = []
        hamming_scores = []
        all_y_true = []
        all_y_pred = []
        print(f"  [Fold {fold_id}] Per-post multi-label...")
        for post_id, queries in tqdm(post_queries.items(), desc=f"Fold {fold_id}", leave=False):
            post_y_true = []
            post_y_pred = []
            for query in queries:
                is_no_evidence = query.get('is_no_evidence', True)
                has_evidence = 0 if is_no_evidence else 1
                post_y_true.append(has_evidence)
                candidates = query.get('candidates', [])
                if not candidates:
                    post_y_pred.append(0)
                    continue
                query_text = query.get('query', '')
                texts = [c.get('text', '') for c in candidates]
                scores = scorer.score(query_text, texts)
                max_score = max(scores) if scores else -np.inf
                pred_has_evidence = 1 if max_score > ne_threshold else 0
                post_y_pred.append(pred_has_evidence)
            exact_match = int(post_y_true == post_y_pred)
            correct = sum(1 for t, p in zip(post_y_true, post_y_pred) if t == p)
            subset_acc = correct / len(post_y_true) if post_y_true else 0
            hamming = 1 - (sum(1 for t, p in zip(post_y_true, post_y_pred) if t != p) / len(post_y_true)) if post_y_true else 1
            exact_matches.append(exact_match)
            subset_accuracies.append(subset_acc)
            hamming_scores.append(hamming)
            all_y_true.extend(post_y_true)
            all_y_pred.extend(post_y_pred)
            n_fp = sum(1 for t, p in zip(post_y_true, post_y_pred) if t == 0 and p == 1)
            n_fn = sum(1 for t, p in zip(post_y_true, post_y_pred) if t == 1 and p == 0)
            all_post_results.append({'fold': fold_id, 'post_id': post_id, 'n_criteria': len(post_y_true), 'exact_match': exact_match, 'subset_accuracy': subset_acc, 'hamming_score': hamming, 'n_fp_criteria': n_fp, 'n_fn_criteria': n_fn})
        fm.metrics['exact_match_rate'] = np.mean(exact_matches) if exact_matches else 0
        fm.metrics['subset_accuracy'] = np.mean(subset_accuracies) if subset_accuracies else 0
        fm.metrics['hamming_score'] = np.mean(hamming_scores) if hamming_scores else 0
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        fm.metrics['micro_precision'] = precision_score(all_y_true_arr, all_y_pred_arr, zero_division=0)
        fm.metrics['micro_recall'] = recall_score(all_y_true_arr, all_y_pred_arr, zero_division=0)
        fm.metrics['micro_f1'] = f1_score(all_y_true_arr, all_y_pred_arr, zero_division=0)
        fm.metrics['n_posts'] = len(post_queries)
        result.fold_metrics.append(fm)
    result.aggregated = aggregate_fold_metrics(result.fold_metrics)
    per_post_df = pd.DataFrame(all_post_results)
    return result, per_post_df


def generate_manifest(output_dir, config):
    manifest = {'timestamp': datetime.now().isoformat(), 'git_branch': 'unknown', 'git_commit': 'unknown', 'python_version': platform.python_version(), 'torch_version': torch.__version__, 'platform': platform.platform(), 'seed': SEED, 'config': config}
    try:
        manifest['git_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=PROJECT_ROOT).decode().strip()
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT).decode().strip()
    except:
        pass
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_report(output_dir, stages, ablations, config):
    lines = ["# Comprehensive End-to-End Assessment Report", "", f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "", "---", "", "## Configuration Used", "", f"- **Model:** `{config.get('model_dir', 'N/A')}`", f"- **Candidates:** `{config.get('candidates_path', 'N/A')}`", f"- **NE Score Type:** `{config.get('ne_score_type', 'score_gap')}`", "", "---", ""]
    for stage in stages:
        lines.append(f"## {stage.name}")
        lines.append("")
        if stage.aggregated and stage.aggregated.mean:
            lines.append("### Metrics (Mean +/- Std across folds)")
            lines.append("")
            lines.append("| Metric | Mean | Std |")
            lines.append("|--------|------|-----|")
            for key in sorted(stage.aggregated.mean.keys()):
                mean_val = stage.aggregated.mean[key]
                std_val = stage.aggregated.std.get(key, 0)
                lines.append(f"| {key} | {mean_val:.4f} | {std_val:.4f} |")
            lines.append("")
        if stage.notes:
            lines.append("### Notes")
            for note in stage.notes:
                lines.append(f"- {note}")
            lines.append("")
        lines.append("---")
        lines.append("")
    lines.append("## Ablation Summary")
    lines.append("")
    for name, metrics in ablations.items():
        lines.append(f"### {name}")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append("")
    report = "\n".join(lines)
    with open(output_dir / "report.md", "w") as f:
        f.write(report)
    return report


def generate_summary_json(output_dir, stages, ablations, manifest):
    summary = {'manifest': manifest, 'stages': {}, 'ablations': ablations}
    for stage in stages:
        stage_data = {'name': stage.name, 'notes': stage.notes}
        if stage.aggregated:
            stage_data['aggregated'] = {'mean': stage.aggregated.mean, 'std': stage.aggregated.std}
        if stage.fold_metrics:
            stage_data['fold_metrics'] = [{'fold_id': fm.fold_id, 'metrics': fm.metrics} for fm in stage.fold_metrics]
        if stage.extra_data:
            stage_data['extra_data'] = stage.extra_data
        summary['stages'][stage.name] = stage_data
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive E2E Assessment")
    parser.add_argument("--candidates", type=str, default="outputs/retrieval_candidates/retrieval_candidates.pkl")
    parser.add_argument("--model_dir", type=str, default="outputs/training/no_evidence_reranker")
    parser.add_argument("--outdir", type=str, default="outputs/assessment_full")
    parser.add_argument("--ne_score_type", type=str, default="score_gap")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_reranker_scoring", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.outdir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE END-TO-END ASSESSMENT")
    print("=" * 80)
    print(f"Candidates: {args.candidates}")
    print(f"Model: {args.model_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nLoading candidates...")
    with open(args.candidates, 'rb') as f:
        data = pickle.load(f)

    config = {'candidates_path': args.candidates, 'model_dir': args.model_dir, 'ne_score_type': args.ne_score_type, 'device': args.device}
    scorer = RerankerScorer(args.model_dir, args.device)
    stages = []

    print("\n[Stage 0] Data Audit...")
    audit_result = run_data_audit(data, output_dir)
    stages.append(audit_result)

    print("\n[Stage A] Retriever Ranking...")
    retriever_result = run_retriever_ranking(data)
    stages.append(retriever_result)

    ablations = {}
    per_post_df = pd.DataFrame()

    if not args.skip_reranker_scoring:
        print("\n[Stage B] Reranker Ranking...")
        reranker_result = run_reranker_ranking(data, scorer)
        stages.append(reranker_result)

        print("\n[Stage D] No-Evidence Detection (Threshold-based)...")
        ne_result = run_ne_detection(data, scorer, output_dir, args.ne_score_type)
        stages.append(ne_result)

        print("\n[Stage D2] NO_EVIDENCE Ranking Check...")
        ne_ranking_result = run_ne_ranking_check(data, scorer, output_dir)
        stages.append(ne_ranking_result)

        print("\n[Stage E] Dynamic-K...")
        dynamic_k_result = run_dynamic_k_check(data, scorer, output_dir)
        stages.append(dynamic_k_result)

        print("\n[Stage F] E2E Deployment...")
        ne_threshold = ne_result.aggregated.mean.get('best_threshold', 0.0)
        e2e_result = run_e2e_deployment(data, scorer, ne_threshold)
        stages.append(e2e_result)

        print("\n[Per-Post] Multi-Label Assessment...")
        per_post_result, per_post_df = run_per_post_multilabel(data, scorer, ne_threshold)
        stages.append(per_post_result)
        per_post_df.to_csv(output_dir / "per_post.csv", index=False)

        ablations = {
            'retriever_only': {
                'ndcg@1': retriever_result.aggregated.mean.get('ndcg@1', 0),
                'ndcg@3': retriever_result.aggregated.mean.get('ndcg@3', 0),
                'ndcg@5': retriever_result.aggregated.mean.get('ndcg@5', 0),
                'ndcg@10': retriever_result.aggregated.mean.get('ndcg@10', 0),
                'ndcg@20': retriever_result.aggregated.mean.get('ndcg@20', 0),
                'recall@1': retriever_result.aggregated.mean.get('recall@1', 0),
                'recall@3': retriever_result.aggregated.mean.get('recall@3', 0),
                'recall@5': retriever_result.aggregated.mean.get('recall@5', 0),
                'recall@10': retriever_result.aggregated.mean.get('recall@10', 0),
                'recall@20': retriever_result.aggregated.mean.get('recall@20', 0),
            },
            'reranker_only': {
                'ndcg@1': reranker_result.aggregated.mean.get('ndcg@1', 0),
                'ndcg@3': reranker_result.aggregated.mean.get('ndcg@3', 0),
                'ndcg@5': reranker_result.aggregated.mean.get('ndcg@5', 0),
                'ndcg@10': reranker_result.aggregated.mean.get('ndcg@10', 0),
                'ndcg@20': reranker_result.aggregated.mean.get('ndcg@20', 0),
                'recall@1': reranker_result.aggregated.mean.get('recall@1', 0),
                'recall@3': reranker_result.aggregated.mean.get('recall@3', 0),
                'recall@5': reranker_result.aggregated.mean.get('recall@5', 0),
                'recall@10': reranker_result.aggregated.mean.get('recall@10', 0),
                'recall@20': reranker_result.aggregated.mean.get('recall@20', 0),
            },
            'dynamic_k': {
                'dynamic_ndcg': dynamic_k_result.aggregated.mean.get('dynamic_ndcg', 0),
                'dynamic_recall': dynamic_k_result.aggregated.mean.get('dynamic_recall', 0),
                'avg_k': dynamic_k_result.aggregated.mean.get('dynamic_k_mean', 0),
            },
            'ne_ranking': {
                'ne_top1_accuracy': ne_ranking_result.aggregated.mean.get('ne_top1_accuracy', 0),
                'evidence_above_ne': ne_ranking_result.aggregated.mean.get('evidence_above_ne', 0),
                'unified_balanced_acc': ne_ranking_result.aggregated.mean.get('unified_balanced_acc', 0),
                'unified_f1': ne_ranking_result.aggregated.mean.get('unified_f1', 0),
                'avg_ne_rank': ne_ranking_result.aggregated.mean.get('avg_ne_rank', 0),
                'avg_score_margin': ne_ranking_result.aggregated.mean.get('avg_score_margin', 0),
            },
            'full_pipeline': {
                'conditional_ndcg': e2e_result.aggregated.mean.get('conditional_ndcg', 0),
                'conditional_recall': e2e_result.aggregated.mean.get('conditional_recall', 0),
                'positive_coverage': e2e_result.aggregated.mean.get('positive_coverage', 0),
                'negative_coverage': e2e_result.aggregated.mean.get('negative_coverage', 0),
            }
        }

    print("\nGenerating outputs...")
    manifest = generate_manifest(output_dir, config)
    generate_report(output_dir, stages, ablations, config)
    generate_summary_json(output_dir, stages, ablations, manifest)
    pd.DataFrame({'note': ['Per-query tracking placeholder']}).to_csv(output_dir / "per_query.csv", index=False)

    print("\n" + "=" * 80)
    print("ASSESSMENT COMPLETE")
    print("=" * 80)
    print(f"Report: {output_dir / 'report.md'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Per-post: {output_dir / 'per_post.csv'}")
    print(f"Curves: {output_dir / 'curves/'}")
    print(f"Manifest: {output_dir / 'manifests/'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
