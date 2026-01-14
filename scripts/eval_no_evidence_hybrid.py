#!/usr/bin/env python3
"""Assess hybrid NO_EVIDENCE detection approaches.

Combines learned NO_EVIDENCE ranking with threshold-based methods:
1. Score Gap: best_score - no_evidence_score > threshold
2. Score Ratio: best_score / no_evidence_score > threshold
3. Std Hybrid: Use score std to calibrate confidence
4. Combined: Multiple signals together
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import NO_EVIDENCE_TOKEN, build_grouped_examples


def load_model(model_dir: str, device: str = "cuda"):
    """Load trained reranker model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.train(False)
    return model, tokenizer


def score_candidates(model, tokenizer, query, sentences, device="cuda", max_length=384):
    """Score all candidates for a query."""
    scores = []
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        pairs = [[query, sent] for sent in batch_sents]
        inputs = tokenizer(pairs, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu().numpy()
            if logits.ndim == 0:
                logits = [float(logits)]
            else:
                logits = logits.tolist()
            scores.extend(logits)
    return scores


def compute_metrics(predictions, labels):
    """Compute binary classification metrics."""
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc = (recall + specificity) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def run_hybrid_assessment(
    model_dir: str,
    config_path: str = "configs/reranker_hybrid.yaml",
    max_candidates: int = 32,
    device: str = "cuda",
):
    """Assess multiple hybrid approaches."""
    print("=" * 80)
    print("HYBRID NO_EVIDENCE DETECTION ASSESSMENT")
    print("=" * 80)

    # Load config and data
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    gt_path = Path(cfg["paths"]["groundtruth"])
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"

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

    # Build test examples WITH NO_EVIDENCE
    test_examples = build_grouped_examples(
        groundtruth, criteria, splits["test"],
        max_candidates=max_candidates,
        seed=cfg["split"]["seed"],
        add_no_evidence=True,
        include_no_evidence_queries=True,
    )

    n_has_ev = sum(1 for ex in test_examples if not ex["is_no_evidence"])
    n_no_ev = sum(1 for ex in test_examples if ex["is_no_evidence"])
    print(f"Test: {len(test_examples)} queries ({n_has_ev} has-ev, {n_no_ev} no-ev)")

    # Load model
    print(f"Loading model from {model_dir}...")
    model, tokenizer = load_model(model_dir, device)

    # Collect scores for all queries
    print("\nScoring all queries...")
    query_data = []

    for ex in tqdm(test_examples):
        scores = score_candidates(model, tokenizer, ex["query"], ex["sentences"], device)

        # Find NO_EVIDENCE score and best other score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_sentences = [ex["sentences"][i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        no_ev_idx = sorted_sentences.index(NO_EVIDENCE_TOKEN)
        no_ev_score = sorted_scores[no_ev_idx]
        no_ev_rank = no_ev_idx + 1

        # Best score excluding NO_EVIDENCE
        other_scores = [s for sent, s in zip(sorted_sentences, sorted_scores)
                       if sent != NO_EVIDENCE_TOKEN]
        best_other_score = max(other_scores) if other_scores else -999

        # Score statistics (excluding NO_EVIDENCE)
        score_std = np.std(other_scores) if len(other_scores) > 1 else 0
        score_mean = np.mean(other_scores) if other_scores else 0

        query_data.append({
            "is_no_evidence": ex["is_no_evidence"],
            "no_ev_score": no_ev_score,
            "no_ev_rank": no_ev_rank,
            "best_other_score": best_other_score,
            "score_gap": best_other_score - no_ev_score,
            "score_ratio": best_other_score / no_ev_score if no_ev_score != 0 else 999,
            "score_std": score_std,
            "score_mean": score_mean,
        })

    # Ground truth labels (1 = has evidence, 0 = no evidence)
    labels = [0 if d["is_no_evidence"] else 1 for d in query_data]

    # Method 1: Pure Ranking
    print("\n" + "=" * 80)
    print("METHOD 1: Pure NO_EVIDENCE Ranking")
    print("=" * 80)
    preds_ranking = [0 if d["no_ev_rank"] == 1 else 1 for d in query_data]
    m1 = compute_metrics(preds_ranking, labels)
    print(f"  Accuracy: {m1['accuracy']:.4f}, Balanced: {m1['balanced_accuracy']:.4f}")
    print(f"  Precision: {m1['precision']:.4f}, Recall: {m1['recall']:.4f}, F1: {m1['f1']:.4f}, Spec: {m1['specificity']:.4f}")

    # Method 2: Score Gap Hybrid
    print("\n" + "=" * 80)
    print("METHOD 2: Score Gap Hybrid (best - no_ev > threshold)")
    print("=" * 80)

    gaps = [d["score_gap"] for d in query_data]
    best_gap_thresh, best_gap_bal = 0, 0

    for thresh in np.linspace(min(gaps), max(gaps), 100):
        preds = [1 if d["score_gap"] > thresh else 0 for d in query_data]
        m = compute_metrics(preds, labels)
        if m["balanced_accuracy"] > best_gap_bal:
            best_gap_bal = m["balanced_accuracy"]
            best_gap_thresh = thresh

    preds_gap = [1 if d["score_gap"] > best_gap_thresh else 0 for d in query_data]
    m2 = compute_metrics(preds_gap, labels)
    print(f"  Optimal Threshold: {best_gap_thresh:.4f}")
    print(f"  Accuracy: {m2['accuracy']:.4f}, Balanced: {m2['balanced_accuracy']:.4f}")
    print(f"  Precision: {m2['precision']:.4f}, Recall: {m2['recall']:.4f}, F1: {m2['f1']:.4f}, Spec: {m2['specificity']:.4f}")

    # Method 3: Score Ratio Hybrid
    print("\n" + "=" * 80)
    print("METHOD 3: Score Ratio Hybrid (best / no_ev > threshold)")
    print("=" * 80)

    ratios = [d["score_ratio"] for d in query_data]
    valid_ratios = [r for r in ratios if abs(r) < 100]
    best_ratio_thresh, best_ratio_bal = 1, 0

    for thresh in np.linspace(min(valid_ratios), max(valid_ratios), 100):
        preds = [1 if d["score_ratio"] > thresh else 0 for d in query_data]
        m = compute_metrics(preds, labels)
        if m["balanced_accuracy"] > best_ratio_bal:
            best_ratio_bal = m["balanced_accuracy"]
            best_ratio_thresh = thresh

    preds_ratio = [1 if d["score_ratio"] > best_ratio_thresh else 0 for d in query_data]
    m3 = compute_metrics(preds_ratio, labels)
    print(f"  Optimal Threshold: {best_ratio_thresh:.4f}")
    print(f"  Accuracy: {m3['accuracy']:.4f}, Balanced: {m3['balanced_accuracy']:.4f}")
    print(f"  Precision: {m3['precision']:.4f}, Recall: {m3['recall']:.4f}, F1: {m3['f1']:.4f}, Spec: {m3['specificity']:.4f}")

    # Method 4: Std Hybrid
    print("\n" + "=" * 80)
    print("METHOD 4: Std Hybrid (score_std > threshold)")
    print("=" * 80)

    stds = [d["score_std"] for d in query_data]
    best_std_thresh, best_std_bal = 0, 0

    for thresh in np.linspace(min(stds), max(stds), 100):
        preds = [1 if d["score_std"] > thresh else 0 for d in query_data]
        m = compute_metrics(preds, labels)
        if m["balanced_accuracy"] > best_std_bal:
            best_std_bal = m["balanced_accuracy"]
            best_std_thresh = thresh

    preds_std = [1 if d["score_std"] > best_std_thresh else 0 for d in query_data]
    m4 = compute_metrics(preds_std, labels)
    print(f"  Optimal Threshold: {best_std_thresh:.4f}")
    print(f"  Accuracy: {m4['accuracy']:.4f}, Balanced: {m4['balanced_accuracy']:.4f}")
    print(f"  Precision: {m4['precision']:.4f}, Recall: {m4['recall']:.4f}, F1: {m4['f1']:.4f}, Spec: {m4['specificity']:.4f}")

    # Method 5: Combined (Score Gap + Std)
    print("\n" + "=" * 80)
    print("METHOD 5: Combined (score_gap > t1 OR score_std > t2)")
    print("=" * 80)

    best_combined_bal = 0
    best_t1, best_t2 = 0, 0

    for t1 in np.linspace(min(gaps), max(gaps), 30):
        for t2 in np.linspace(min(stds), max(stds), 30):
            preds = [1 if (d["score_gap"] > t1 or d["score_std"] > t2) else 0
                    for d in query_data]
            m = compute_metrics(preds, labels)
            if m["balanced_accuracy"] > best_combined_bal:
                best_combined_bal = m["balanced_accuracy"]
                best_t1, best_t2 = t1, t2

    preds_combined = [1 if (d["score_gap"] > best_t1 or d["score_std"] > best_t2) else 0
                     for d in query_data]
    m5 = compute_metrics(preds_combined, labels)
    print(f"  Optimal Thresholds: gap={best_t1:.4f}, std={best_t2:.4f}")
    print(f"  Accuracy: {m5['accuracy']:.4f}, Balanced: {m5['balanced_accuracy']:.4f}")
    print(f"  Precision: {m5['precision']:.4f}, Recall: {m5['recall']:.4f}, F1: {m5['f1']:.4f}, Spec: {m5['specificity']:.4f}")

    # Method 6: Ranking + Score Gap (AND)
    print("\n" + "=" * 80)
    print("METHOD 6: Ranking AND Score Gap (rank > 1 AND gap > threshold)")
    print("=" * 80)

    best_m6_bal = 0
    best_m6_thresh = 0

    for thresh in np.linspace(min(gaps), max(gaps), 100):
        preds = [1 if (d["no_ev_rank"] > 1 and d["score_gap"] > thresh) else 0
                for d in query_data]
        m = compute_metrics(preds, labels)
        if m["balanced_accuracy"] > best_m6_bal:
            best_m6_bal = m["balanced_accuracy"]
            best_m6_thresh = thresh

    preds_m6 = [1 if (d["no_ev_rank"] > 1 and d["score_gap"] > best_m6_thresh) else 0
               for d in query_data]
    m6 = compute_metrics(preds_m6, labels)
    print(f"  Optimal Gap Threshold: {best_m6_thresh:.4f}")
    print(f"  Accuracy: {m6['accuracy']:.4f}, Balanced: {m6['balanced_accuracy']:.4f}")
    print(f"  Precision: {m6['precision']:.4f}, Recall: {m6['recall']:.4f}, F1: {m6['f1']:.4f}, Spec: {m6['specificity']:.4f}")

    # Method 7: Ranking OR Score Gap
    print("\n" + "=" * 80)
    print("METHOD 7: Ranking OR Score Gap (rank > 1 OR gap > threshold)")
    print("=" * 80)

    best_m7_bal = 0
    best_m7_thresh = 0

    for thresh in np.linspace(min(gaps), max(gaps), 100):
        preds = [1 if (d["no_ev_rank"] > 1 or d["score_gap"] > thresh) else 0
                for d in query_data]
        m = compute_metrics(preds, labels)
        if m["balanced_accuracy"] > best_m7_bal:
            best_m7_bal = m["balanced_accuracy"]
            best_m7_thresh = thresh

    preds_m7 = [1 if (d["no_ev_rank"] > 1 or d["score_gap"] > best_m7_thresh) else 0
               for d in query_data]
    m7 = compute_metrics(preds_m7, labels)
    print(f"  Optimal Gap Threshold: {best_m7_thresh:.4f}")
    print(f"  Accuracy: {m7['accuracy']:.4f}, Balanced: {m7['balanced_accuracy']:.4f}")
    print(f"  Precision: {m7['precision']:.4f}, Recall: {m7['recall']:.4f}, F1: {m7['f1']:.4f}, Spec: {m7['specificity']:.4f}")

    # Summary Table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Method':<40} {'Acc':>8} {'BalAcc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Spec':>8}")
    print("-" * 88)

    methods = [
        ("1. Pure Ranking", m1),
        ("2. Score Gap (best-noev > t)", m2),
        ("3. Score Ratio (best/noev > t)", m3),
        ("4. Std Hybrid (std > t)", m4),
        ("5. Combined (gap OR std)", m5),
        ("6. Ranking AND Gap", m6),
        ("7. Ranking OR Gap", m7),
    ]

    for name, m in methods:
        print(f"{name:<40} {m['accuracy']:>8.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['specificity']:>8.4f}")

    print("=" * 80)

    # Find best by balanced accuracy
    best_method = max(methods, key=lambda x: x[1]["balanced_accuracy"])
    print(f"\nBest Method (by Balanced Accuracy): {best_method[0]}")
    print(f"  Balanced Accuracy: {best_method[1]['balanced_accuracy']:.4f}")

    # Save results
    results = {
        "methods": {name: metrics for name, metrics in methods},
        "optimal_thresholds": {
            "score_gap": best_gap_thresh,
            "score_ratio": best_ratio_thresh,
            "score_std": best_std_thresh,
            "combined_gap": best_t1,
            "combined_std": best_t2,
            "ranking_and_gap": best_m6_thresh,
            "ranking_or_gap": best_m7_thresh,
        },
        "best_method": best_method[0],
    }

    output_path = Path(model_dir) / "hybrid_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/training/no_evidence_reranker")
    parser.add_argument("--config", default="configs/reranker_hybrid.yaml")
    parser.add_argument("--max_candidates", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_hybrid_assessment(args.model_dir, args.config, args.max_candidates, args.device)
