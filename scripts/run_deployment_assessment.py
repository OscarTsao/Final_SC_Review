#!/usr/bin/env python3
"""
Comprehensive End-to-End Assessment for S-C Evidence Retrieval Pipeline.

Produces:
- outputs/final_assessment/report.md          - Human-readable report
- outputs/final_assessment/summary.json       - Machine-readable metrics
- outputs/final_assessment/per_query.csv      - Query-level results
- outputs/final_assessment/per_post.csv       - Post-level multi-label results
- outputs/final_assessment/curves/            - Calibration/PR/ROC curves

Stages:
0. Data Audit - sanity checks on splits and labels
A. Retriever Assessment - dense/sparse/ColBERT/fused metrics
B. Reranker Assessment - on gold-retrieval candidates
C. Calibration Assessment - 5-fold cross-val to prevent leakage
D. No-Evidence Detection - ranking vs threshold vs hybrid
E. Dynamic-K Assessment - adaptive cutoff selection
F. E2E Deployment Assessment - full pipeline metrics
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import NO_EVIDENCE_TOKEN, build_grouped_examples


# =============================================================================
# Metric Utilities
# =============================================================================

def ndcg_at_k(labels: List[int], k: int) -> float:
    """Compute nDCG@k from binary relevance labels."""
    if not labels or sum(labels) == 0:
        return 0.0
    labels = labels[:k]
    dcg = sum(l / np.log2(i + 2) for i, l in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum(l / np.log2(i + 2) for i, l in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(labels: List[int], k: int, total_pos: Optional[int] = None) -> float:
    """Compute Recall@k from binary relevance labels."""
    if total_pos is None:
        total_pos = sum(labels)
    if total_pos == 0:
        return 0.0
    return sum(labels[:k]) / total_pos


def mrr_at_k(labels: List[int], k: int) -> float:
    """Compute MRR@k from binary relevance labels."""
    for i, l in enumerate(labels[:k]):
        if l == 1:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(labels: List[int], k: int) -> float:
    """Compute MAP@k from binary relevance labels."""
    labels = labels[:k]
    if sum(labels) == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, l in enumerate(labels):
        if l == 1:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.mean(precisions) if precisions else 0.0


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() * abs(avg_accuracy - avg_confidence)
    return ece / len(y_true)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StageMetrics:
    """Container for stage-level metrics."""
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    per_query: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class AssessmentConfig:
    """Assessment configuration."""
    config_path: str
    model_dir: str
    output_dir: str
    device: str
    max_candidates: int
    k_values: List[int]
    calibration_folds: int
    ne_thresholds: List[float]


# =============================================================================
# Stage 0: Data Audit
# =============================================================================

def run_data_audit(groundtruth, criteria, splits: Dict) -> StageMetrics:
    """Audit data for quality and split integrity."""
    stage = StageMetrics(name="Data Audit")

    # Check post-ID disjointness
    train_posts = set(splits["train"])
    val_posts = set(splits["val"])
    test_posts = set(splits["test"])

    train_val_overlap = train_posts & val_posts
    train_test_overlap = train_posts & test_posts
    val_test_overlap = val_posts & test_posts

    stage.metrics["train_val_overlap"] = len(train_val_overlap)
    stage.metrics["train_test_overlap"] = len(train_test_overlap)
    stage.metrics["val_test_overlap"] = len(val_test_overlap)
    stage.metrics["splits_disjoint"] = int(len(train_val_overlap) == 0 and
                                           len(train_test_overlap) == 0 and
                                           len(val_test_overlap) == 0)

    # Count statistics
    stage.metrics["n_train_posts"] = len(train_posts)
    stage.metrics["n_val_posts"] = len(val_posts)
    stage.metrics["n_test_posts"] = len(test_posts)
    stage.metrics["n_criteria"] = len(criteria)

    # Count groundtruth distribution
    gt_df = pd.DataFrame([
        {"post_id": r.post_id, "criterion": r.criterion_id, "groundtruth": r.groundtruth}
        for r in groundtruth
    ])

    stage.metrics["total_annotations"] = len(gt_df)
    stage.metrics["positive_annotations"] = int(gt_df["groundtruth"].sum())
    stage.metrics["negative_annotations"] = int((~gt_df["groundtruth"].astype(bool)).sum())
    stage.metrics["pos_ratio"] = stage.metrics["positive_annotations"] / stage.metrics["total_annotations"]

    # Per-criterion stats
    for crit in criteria:
        crit_id = crit.criterion_id
        crit_gt = gt_df[gt_df["criterion"] == crit_id]
        stage.metrics[f"crit_{crit_id}_total"] = len(crit_gt)
        stage.metrics[f"crit_{crit_id}_pos"] = int(crit_gt["groundtruth"].sum())

    # Notes
    if stage.metrics["splits_disjoint"]:
        stage.notes.append("PASS: All splits are post-ID disjoint")
    else:
        stage.notes.append("FAIL: Split overlap detected - potential data leakage!")

    return stage


# =============================================================================
# Stage A: Retriever Component (Placeholder - requires retriever scores)
# =============================================================================

def run_retriever_check(config: AssessmentConfig, groundtruth, criteria, test_posts) -> StageMetrics:
    """Check retriever components if cached scores available."""
    stage = StageMetrics(name="Retriever Check")

    # Check for cached retriever scores
    cache_dir = PROJECT_ROOT / "data" / "cache" / "bge_m3"
    if not cache_dir.exists():
        stage.notes.append("SKIP: Retriever cache not found - run precompute first")
        return stage

    # This would load and check retriever scores
    # For now, mark as placeholder
    stage.notes.append("INFO: Retriever metrics require precomputed cache")
    stage.metrics["status"] = 0  # Not run

    return stage


# =============================================================================
# Stage B: Reranker Assessment (on gold retrieval)
# =============================================================================

def load_reranker_model(model_dir: str, device: str):
    """Load trained reranker model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.set_grad_enabled = False
    return model, tokenizer


def score_candidates(
    model, tokenizer, query: str, sentences: List[str],
    device: str = "cuda", max_length: int = 384, batch_size: int = 32
) -> List[float]:
    """Score all candidates for a query."""
    scores = []
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        pairs = [[query, sent] for sent in batch_sents]
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu().numpy()
            if logits.ndim == 0:
                logits = [float(logits)]
            else:
                logits = logits.tolist()
            scores.extend(logits)
    return scores


def run_reranker_check(
    config: AssessmentConfig, model, tokenizer, test_examples: List[Dict]
) -> StageMetrics:
    """Check reranker on test examples."""
    stage = StageMetrics(name="Reranker Check")

    # Containers for metrics
    all_ndcg = {k: [] for k in config.k_values}
    all_recall = {k: [] for k in config.k_values}
    all_mrr = {k: [] for k in config.k_values}
    all_map = {k: [] for k in config.k_values}

    print("\n[Stage B] Running reranker check...")
    for ex in tqdm(test_examples, desc="Reranker"):
        if ex["is_no_evidence"]:
            continue  # Skip no-evidence queries for ranking metrics

        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]

        # Filter out NO_EVIDENCE token for ranking metrics
        filtered_pairs = [(s, l) for s, l in zip(sentences, labels) if s != NO_EVIDENCE_TOKEN]
        if not filtered_pairs or sum(l for _, l in filtered_pairs) == 0:
            continue

        filtered_sents = [p[0] for p in filtered_pairs]
        filtered_labels = [p[1] for p in filtered_pairs]

        # Score candidates
        scores = score_candidates(model, tokenizer, query, filtered_sents, config.device)

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = [filtered_labels[i] for i in sorted_indices]

        # Compute metrics at each k
        for k in config.k_values:
            all_ndcg[k].append(ndcg_at_k(sorted_labels, k))
            all_recall[k].append(recall_at_k(sorted_labels, k))
            all_mrr[k].append(mrr_at_k(sorted_labels, k))
            all_map[k].append(map_at_k(sorted_labels, k))

        # Store per-query results
        stage.per_query.append({
            "query": query[:100],
            "n_candidates": len(filtered_sents),
            "n_positive": sum(filtered_labels),
            **{f"ndcg@{k}": ndcg_at_k(sorted_labels, k) for k in config.k_values},
            **{f"recall@{k}": recall_at_k(sorted_labels, k) for k in config.k_values},
        })

    # Aggregate metrics
    for k in config.k_values:
        stage.metrics[f"ndcg@{k}"] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0
        stage.metrics[f"recall@{k}"] = np.mean(all_recall[k]) if all_recall[k] else 0
        stage.metrics[f"mrr@{k}"] = np.mean(all_mrr[k]) if all_mrr[k] else 0
        stage.metrics[f"map@{k}"] = np.mean(all_map[k]) if all_map[k] else 0

    stage.metrics["n_queries_checked"] = len(stage.per_query)
    stage.notes.append(f"Checked {len(stage.per_query)} has-evidence queries")

    return stage


# =============================================================================
# Stage C: Calibration Assessment (5-fold CV)
# =============================================================================

def run_calibration_check(
    config: AssessmentConfig, model, tokenizer, all_examples: List[Dict], output_dir: Path
) -> StageMetrics:
    """Check calibration using 5-fold cross-validation."""
    stage = StageMetrics(name="Calibration Check")

    print("\n[Stage C] Running calibration check (5-fold CV)...")

    # Collect all scores and labels
    all_scores = []
    all_labels = []
    all_query_info = []

    for ex in tqdm(all_examples, desc="Scoring"):
        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]

        scores = score_candidates(model, tokenizer, query, sentences, config.device)

        for sent, score, label in zip(sentences, scores, labels):
            if sent != NO_EVIDENCE_TOKEN:
                all_scores.append(score)
                all_labels.append(label)
                all_query_info.append({"query": query[:50], "sentence": sent[:50]})

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Convert logits to probabilities
    all_probs = expit(all_scores)

    # Uncalibrated metrics
    stage.metrics["uncalibrated_ece"] = expected_calibration_error(all_labels, all_probs)

    # 5-fold cross-validation for calibration
    n_folds = config.calibration_folds
    fold_size = len(all_scores) // n_folds
    indices = np.arange(len(all_scores))
    np.random.seed(42)
    np.random.shuffle(indices)

    calibrated_probs_platt = np.zeros_like(all_probs)
    calibrated_probs_isotonic = np.zeros_like(all_probs)

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(indices)
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        # Platt scaling
        try:
            platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            platt.fit(all_scores[train_idx].reshape(-1, 1), all_labels[train_idx])
            calibrated_probs_platt[val_idx] = platt.predict_proba(
                all_scores[val_idx].reshape(-1, 1)
            )[:, 1]
        except Exception:
            calibrated_probs_platt[val_idx] = all_probs[val_idx]

        # Isotonic regression
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(all_scores[train_idx], all_labels[train_idx])
            calibrated_probs_isotonic[val_idx] = iso.predict(all_scores[val_idx])
        except Exception:
            calibrated_probs_isotonic[val_idx] = all_probs[val_idx]

    # Calibrated metrics
    stage.metrics["platt_ece"] = expected_calibration_error(all_labels, calibrated_probs_platt)
    stage.metrics["isotonic_ece"] = expected_calibration_error(all_labels, calibrated_probs_isotonic)

    # Reliability diagram
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (probs, name) in zip(axes, [
        (all_probs, "Uncalibrated"),
        (calibrated_probs_platt, "Platt Scaling"),
        (calibrated_probs_isotonic, "Isotonic Regression"),
    ]):
        fraction_pos, mean_pred = calibration_curve(all_labels, probs, n_bins=10)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.plot(mean_pred, fraction_pos, 's-', label=name)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig(curves_dir / "calibration_curves.png", dpi=150)
    plt.close()

    stage.notes.append(f"5-fold CV calibration complete")
    stage.notes.append(f"Uncalibrated ECE: {stage.metrics['uncalibrated_ece']:.4f}")
    stage.notes.append(f"Platt ECE: {stage.metrics['platt_ece']:.4f}")
    stage.notes.append(f"Isotonic ECE: {stage.metrics['isotonic_ece']:.4f}")

    return stage


# =============================================================================
# Stage D: No-Evidence Detection
# =============================================================================

def run_ne_detection_check(
    config: AssessmentConfig, model, tokenizer, test_examples: List[Dict], output_dir: Path
) -> StageMetrics:
    """Check no-evidence detection methods."""
    stage = StageMetrics(name="No-Evidence Detection")

    print("\n[Stage D] Running NE detection check...")

    # Collect predictions
    results = []
    for ex in tqdm(test_examples, desc="NE Detection"):
        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]
        is_no_evidence = ex["is_no_evidence"]

        scores = score_candidates(model, tokenizer, query, sentences, config.device)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_sentences = [sentences[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        # Find NO_EVIDENCE position and score
        try:
            ne_rank = sorted_sentences.index(NO_EVIDENCE_TOKEN) + 1
            ne_idx = sentences.index(NO_EVIDENCE_TOKEN)
            ne_score = scores[ne_idx]
        except ValueError:
            ne_rank = len(sentences) + 1
            ne_score = float('-inf')

        # Best evidence score (excluding NO_EVIDENCE)
        evidence_scores = [s for sent, s in zip(sentences, scores) if sent != NO_EVIDENCE_TOKEN]
        best_evidence_score = max(evidence_scores) if evidence_scores else float('-inf')
        score_gap = best_evidence_score - ne_score

        # Score std
        score_std = np.std(evidence_scores) if len(evidence_scores) > 1 else 0

        results.append({
            "query": query,
            "is_no_evidence": is_no_evidence,
            "ne_rank": ne_rank,
            "ne_score": ne_score,
            "best_evidence_score": best_evidence_score,
            "score_gap": score_gap,
            "score_std": score_std,
        })

    results_df = pd.DataFrame(results)

    # Method 1: Pure Ranking (NE rank == 1)
    y_true = results_df["is_no_evidence"].astype(int).values
    y_pred_rank = (results_df["ne_rank"] == 1).astype(int).values

    stage.metrics["ranking_accuracy"] = accuracy_score(y_true, y_pred_rank)
    stage.metrics["ranking_balanced_acc"] = balanced_accuracy_score(y_true, y_pred_rank)
    stage.metrics["ranking_precision"] = precision_score(y_true, y_pred_rank, zero_division=0)
    stage.metrics["ranking_recall"] = recall_score(y_true, y_pred_rank, zero_division=0)
    stage.metrics["ranking_f1"] = f1_score(y_true, y_pred_rank, zero_division=0)

    # Method 2: Score Gap threshold search
    best_gap_threshold = 0
    best_gap_bal_acc = 0
    for t in np.linspace(-2, 2, 41):
        y_pred_gap = (results_df["score_gap"] > t).astype(int).values
        y_pred_ne = 1 - y_pred_gap  # Low gap = no evidence
        bal_acc = balanced_accuracy_score(y_true, y_pred_ne)
        if bal_acc > best_gap_bal_acc:
            best_gap_bal_acc = bal_acc
            best_gap_threshold = t

    y_pred_gap_best = 1 - (results_df["score_gap"] > best_gap_threshold).astype(int).values
    stage.metrics["gap_best_threshold"] = best_gap_threshold
    stage.metrics["gap_balanced_acc"] = best_gap_bal_acc
    stage.metrics["gap_f1"] = f1_score(y_true, y_pred_gap_best, zero_division=0)

    # Method 3: Combined (Ranking AND/OR Gap)
    y_pred_combined_and = ((results_df["ne_rank"] == 1) & (results_df["score_gap"] <= best_gap_threshold)).astype(int).values
    y_pred_combined_or = ((results_df["ne_rank"] == 1) | (results_df["score_gap"] <= best_gap_threshold)).astype(int).values

    stage.metrics["combined_and_balanced_acc"] = balanced_accuracy_score(y_true, y_pred_combined_and)
    stage.metrics["combined_or_balanced_acc"] = balanced_accuracy_score(y_true, y_pred_combined_or)

    # AUROC/AUPRC for score gap
    if len(np.unique(y_true)) > 1:
        # Negative gap = more likely no evidence
        stage.metrics["gap_auroc"] = roc_auc_score(y_true, -results_df["score_gap"].values)
        stage.metrics["gap_auprc"] = average_precision_score(y_true, -results_df["score_gap"].values)

    # Plot ROC curve
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, -results_df["score_gap"].values)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Score Gap (AUROC={stage.metrics["gap_auroc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('NE Detection ROC Curve')
        plt.legend()
        plt.savefig(curves_dir / "ne_detection_roc.png", dpi=150)
        plt.close()

    stage.notes.append(f"Ranking method: Bal.Acc={stage.metrics['ranking_balanced_acc']:.4f}")
    stage.notes.append(f"Score gap method: Bal.Acc={stage.metrics['gap_balanced_acc']:.4f} (t={best_gap_threshold:.2f})")

    return stage


# =============================================================================
# Stage E: Dynamic-K Assessment
# =============================================================================

def run_dynamic_k_check(
    config: AssessmentConfig, model, tokenizer, test_examples: List[Dict]
) -> StageMetrics:
    """Check dynamic-K selection via NO_EVIDENCE position."""
    stage = StageMetrics(name="Dynamic-K Check")

    print("\n[Stage E] Running dynamic-K check...")

    k_values = []
    dynamic_ndcg = []
    dynamic_recall = []

    for ex in tqdm(test_examples, desc="Dynamic-K"):
        if ex["is_no_evidence"]:
            continue

        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]

        scores = score_candidates(model, tokenizer, query, sentences, config.device)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_sentences = [sentences[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]

        # Filter out NO_EVIDENCE
        filtered_labels = [l for s, l in zip(sorted_sentences, sorted_labels) if s != NO_EVIDENCE_TOKEN]

        if sum(filtered_labels) == 0:
            continue

        # Find dynamic K (position of NO_EVIDENCE)
        try:
            ne_pos = sorted_sentences.index(NO_EVIDENCE_TOKEN)
            k = ne_pos  # Evidence above NO_EVIDENCE
        except ValueError:
            k = len(filtered_labels)

        k_values.append(k)
        if k > 0:
            dynamic_ndcg.append(ndcg_at_k(filtered_labels, k))
            dynamic_recall.append(recall_at_k(filtered_labels, k))

    stage.metrics["avg_k"] = np.mean(k_values) if k_values else 0
    stage.metrics["median_k"] = np.median(k_values) if k_values else 0
    stage.metrics["std_k"] = np.std(k_values) if k_values else 0
    stage.metrics["dynamic_ndcg"] = np.mean(dynamic_ndcg) if dynamic_ndcg else 0
    stage.metrics["dynamic_recall"] = np.mean(dynamic_recall) if dynamic_recall else 0

    # Compare to fixed K
    for fixed_k in [1, 3, 5, 10]:
        fixed_ndcg = []
        fixed_recall = []
        for ex in test_examples:
            if ex["is_no_evidence"]:
                continue
            query = ex["query"]
            sentences = ex["sentences"]
            labels = ex["labels"]
            scores = score_candidates(model, tokenizer, query, sentences, config.device)
            sorted_indices = np.argsort(scores)[::-1]
            sorted_sentences = [sentences[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]
            filtered_labels = [l for s, l in zip(sorted_sentences, sorted_labels) if s != NO_EVIDENCE_TOKEN]
            if sum(filtered_labels) > 0:
                fixed_ndcg.append(ndcg_at_k(filtered_labels, fixed_k))
                fixed_recall.append(recall_at_k(filtered_labels, fixed_k))
        stage.metrics[f"fixed_k{fixed_k}_ndcg"] = np.mean(fixed_ndcg) if fixed_ndcg else 0
        stage.metrics[f"fixed_k{fixed_k}_recall"] = np.mean(fixed_recall) if fixed_recall else 0

    stage.notes.append(f"Dynamic K: avg={stage.metrics['avg_k']:.2f}, nDCG={stage.metrics['dynamic_ndcg']:.4f}")

    return stage


# =============================================================================
# Stage F: E2E Deployment Assessment
# =============================================================================

def run_e2e_check(
    config: AssessmentConfig, model, tokenizer, test_examples: List[Dict]
) -> StageMetrics:
    """End-to-end deployment check."""
    stage = StageMetrics(name="E2E Deployment")

    print("\n[Stage F] Running E2E deployment check...")

    # Simulate full pipeline behavior
    all_predictions = []
    all_ground_truth = []

    for ex in tqdm(test_examples, desc="E2E"):
        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]
        is_no_evidence = ex["is_no_evidence"]

        scores = score_candidates(model, tokenizer, query, sentences, config.device)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_sentences = [sentences[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]

        # Deployment behavior: return top-k evidence above NO_EVIDENCE
        try:
            ne_pos = sorted_sentences.index(NO_EVIDENCE_TOKEN)
            predicted_evidence = sorted_sentences[:ne_pos]
            predicted_labels = sorted_labels[:ne_pos]
        except ValueError:
            predicted_evidence = sorted_sentences
            predicted_labels = sorted_labels

        # Ground truth evidence
        gt_evidence = [s for s, l in zip(sentences, labels) if l == 1 and s != NO_EVIDENCE_TOKEN]

        # Store for aggregate metrics
        for sent, label in zip(sentences, labels):
            if sent != NO_EVIDENCE_TOKEN:
                pred = 1 if sent in predicted_evidence else 0
                all_predictions.append(pred)
                all_ground_truth.append(label)

        stage.per_query.append({
            "query": query[:100],
            "is_no_evidence": is_no_evidence,
            "n_predicted": len(predicted_evidence),
            "n_gt_positive": len(gt_evidence),
            "n_correct": sum(1 for s in predicted_evidence if s in gt_evidence) if gt_evidence else 0,
        })

    # Aggregate sentence-level metrics
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)

    stage.metrics["sentence_accuracy"] = accuracy_score(all_ground_truth, all_predictions)
    stage.metrics["sentence_precision"] = precision_score(all_ground_truth, all_predictions, zero_division=0)
    stage.metrics["sentence_recall"] = recall_score(all_ground_truth, all_predictions, zero_division=0)
    stage.metrics["sentence_f1"] = f1_score(all_ground_truth, all_predictions, zero_division=0)

    stage.notes.append(f"E2E Sentence F1: {stage.metrics['sentence_f1']:.4f}")

    return stage


# =============================================================================
# Per-Post Multi-Label Assessment
# =============================================================================

def run_per_post_check(
    config: AssessmentConfig, model, tokenizer, groundtruth, criteria, test_posts: List[str]
) -> Tuple[StageMetrics, pd.DataFrame]:
    """Per-post multi-label assessment."""
    stage = StageMetrics(name="Per-Post Multi-Label")

    print("\n[Per-Post] Running multi-label check...")

    # Group groundtruth by post
    post_gt = defaultdict(lambda: defaultdict(list))
    for row in groundtruth:
        if row.post_id in test_posts:
            post_gt[row.post_id][row.criterion_id].append({
                "sentence": row.sentence_text,
                "label": row.groundtruth
            })

    per_post_results = []

    for post_id in tqdm(test_posts, desc="Per-Post"):
        post_data = post_gt.get(post_id, {})
        if not post_data:
            continue

        post_metrics = {"post_id": post_id}
        criteria_predictions = []
        criteria_gt = []

        for crit in criteria:
            crit_id = crit.criterion_id
            crit_data = post_data.get(crit_id, [])
            if not crit_data:
                continue

            # Build query
            query = f"Criterion {crit_id}: {crit.text}"
            sentences = [d["sentence"] for d in crit_data]
            labels = [1 if d["label"] else 0 for d in crit_data]

            # Has evidence for this criterion?
            has_evidence = sum(labels) > 0
            criteria_gt.append(1 if has_evidence else 0)

            # Add NO_EVIDENCE token
            sentences.append(NO_EVIDENCE_TOKEN)
            labels.append(0 if has_evidence else 1)

            # Score
            scores = score_candidates(model, tokenizer, query, sentences, config.device)
            sorted_indices = np.argsort(scores)[::-1]
            sorted_sentences = [sentences[i] for i in sorted_indices]

            # Predict has-evidence if NO_EVIDENCE is not rank 1
            try:
                ne_rank = sorted_sentences.index(NO_EVIDENCE_TOKEN) + 1
                pred_has_evidence = ne_rank > 1
            except ValueError:
                pred_has_evidence = True

            criteria_predictions.append(1 if pred_has_evidence else 0)

        if criteria_predictions:
            # Per-post multi-label metrics
            post_metrics["n_criteria"] = len(criteria_predictions)
            post_metrics["gt_positive_criteria"] = sum(criteria_gt)
            post_metrics["pred_positive_criteria"] = sum(criteria_predictions)

            # Exact match
            post_metrics["exact_match"] = int(criteria_predictions == criteria_gt)

            # Subset accuracy
            correct = sum(1 for p, g in zip(criteria_predictions, criteria_gt) if p == g)
            post_metrics["subset_accuracy"] = correct / len(criteria_predictions)

            per_post_results.append(post_metrics)

    per_post_df = pd.DataFrame(per_post_results)

    if len(per_post_df) > 0:
        stage.metrics["n_posts"] = len(per_post_df)
        stage.metrics["exact_match_rate"] = per_post_df["exact_match"].mean()
        stage.metrics["avg_subset_accuracy"] = per_post_df["subset_accuracy"].mean()

    stage.notes.append(f"Checked {len(per_post_df)} posts")

    return stage, per_post_df


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    stages: List[StageMetrics],
    config: AssessmentConfig,
    output_dir: Path,
    per_query_df: pd.DataFrame,
    per_post_df: pd.DataFrame,
):
    """Generate final report and artifacts."""

    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": config.model_dir,
        "config_path": config.config_path,
        "stages": {},
    }

    for stage in stages:
        summary["stages"][stage.name] = {
            "metrics": stage.metrics,
            "notes": stage.notes,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-query CSV
    if len(per_query_df) > 0:
        per_query_df.to_csv(output_dir / "per_query.csv", index=False)

    # Per-post CSV
    if len(per_post_df) > 0:
        per_post_df.to_csv(output_dir / "per_post.csv", index=False)

    # Markdown Report
    report_lines = [
        "# End-to-End Assessment Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** `{config.model_dir}`",
        "",
        "---",
        "",
    ]

    for stage in stages:
        report_lines.append(f"## {stage.name}")
        report_lines.append("")

        if stage.metrics:
            report_lines.append("### Metrics")
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")
            for k, v in stage.metrics.items():
                if isinstance(v, float):
                    report_lines.append(f"| {k} | {v:.4f} |")
                else:
                    report_lines.append(f"| {k} | {v} |")
            report_lines.append("")

        if stage.notes:
            report_lines.append("### Notes")
            for note in stage.notes:
                report_lines.append(f"- {note}")
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # Ablation summary
    report_lines.append("## Ablation Summary")
    report_lines.append("")
    report_lines.append("| Method | Balanced Accuracy | F1 |")
    report_lines.append("|--------|-------------------|-----|")

    for stage in stages:
        if "NE Detection" in stage.name:
            report_lines.append(f"| Ranking | {stage.metrics.get('ranking_balanced_acc', 0):.4f} | {stage.metrics.get('ranking_f1', 0):.4f} |")
            report_lines.append(f"| Score Gap | {stage.metrics.get('gap_balanced_acc', 0):.4f} | {stage.metrics.get('gap_f1', 0):.4f} |")
            report_lines.append(f"| Combined (AND) | {stage.metrics.get('combined_and_balanced_acc', 0):.4f} | - |")
            report_lines.append(f"| Combined (OR) | {stage.metrics.get('combined_or_balanced_acc', 0):.4f} | - |")

    report_lines.append("")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to {output_dir / 'report.md'}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive E2E Assessment")
    parser.add_argument("--model_dir", type=str, default="outputs/training/no_evidence_reranker")
    parser.add_argument("--config", type=str, default="configs/reranker_hybrid.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/final_deployment_assessment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_candidates", type=int, default=32)
    parser.add_argument("--k_values", type=str, default="1,3,5,10")
    parser.add_argument("--calibration_folds", type=int, default=5)
    args = parser.parse_args()

    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]

    config = AssessmentConfig(
        config_path=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_candidates=args.max_candidates,
        k_values=k_values,
        calibration_folds=args.calibration_folds,
        ne_thresholds=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE END-TO-END ASSESSMENT")
    print("=" * 80)
    print(f"Model: {config.model_dir}")
    print(f"Output: {config.output_dir}")
    print("=" * 80)

    # Load config
    with open(config.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    gt_path = Path(cfg["paths"]["groundtruth"])
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

    # Load data
    groundtruth = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )

    # Build test examples
    test_examples = build_grouped_examples(
        groundtruth, criteria, splits["test"],
        max_candidates=config.max_candidates,
        seed=cfg["split"]["seed"],
        add_no_evidence=True,
        include_no_evidence_queries=True,
    )

    print(f"\nTest set: {len(test_examples)} examples")
    print(f"  Has-evidence: {sum(1 for ex in test_examples if not ex['is_no_evidence'])}")
    print(f"  No-evidence: {sum(1 for ex in test_examples if ex['is_no_evidence'])}")

    # Load model
    print(f"\nLoading model from {config.model_dir}...")
    model, tokenizer = load_reranker_model(config.model_dir, config.device)

    # Run all stages
    stages = []

    # Stage 0: Data Audit
    stages.append(run_data_audit(groundtruth, criteria, splits))

    # Stage A: Retriever (placeholder)
    stages.append(run_retriever_check(config, groundtruth, criteria, splits["test"]))

    # Stage B: Reranker
    stages.append(run_reranker_check(config, model, tokenizer, test_examples))

    # Stage C: Calibration
    stages.append(run_calibration_check(config, model, tokenizer, test_examples, output_dir))

    # Stage D: NE Detection
    stages.append(run_ne_detection_check(config, model, tokenizer, test_examples, output_dir))

    # Stage E: Dynamic-K
    stages.append(run_dynamic_k_check(config, model, tokenizer, test_examples))

    # Stage F: E2E
    e2e_stage = run_e2e_check(config, model, tokenizer, test_examples)
    stages.append(e2e_stage)

    # Per-post multi-label
    per_post_stage, per_post_df = run_per_post_check(
        config, model, tokenizer, groundtruth, criteria, splits["test"]
    )
    stages.append(per_post_stage)

    # Collect per-query results
    per_query_records = []
    for stage in stages:
        per_query_records.extend(stage.per_query)
    per_query_df = pd.DataFrame(per_query_records) if per_query_records else pd.DataFrame()

    # Generate report
    generate_report(stages, config, output_dir, per_query_df, per_post_df)

    print("\n" + "=" * 80)
    print("ASSESSMENT COMPLETE")
    print("=" * 80)
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Report: {output_dir / 'report.md'}")
    print(f"Per-query: {output_dir / 'per_query.csv'}")
    print(f"Per-post: {output_dir / 'per_post.csv'}")
    print(f"Curves: {output_dir / 'curves/'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
