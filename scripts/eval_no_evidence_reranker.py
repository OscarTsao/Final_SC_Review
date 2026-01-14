#!/usr/bin/env python3
"""Evaluate NO_EVIDENCE reranker on test set.

Measures:
1. No-evidence detection accuracy (when NO_EVIDENCE ranks first)
2. Ranking metrics for has-evidence queries
3. Dynamic-k via NO_EVIDENCE position
"""

import json
import sys
from collections import defaultdict
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


def compute_ndcg_at_k(labels: list, k: int) -> float:
    """Compute nDCG@k from binary relevance labels."""
    if not labels or sum(labels) == 0:
        return 0.0
    labels = labels[:k]
    dcg = sum(l / np.log2(i + 2) for i, l in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum(l / np.log2(i + 2) for i, l in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_recall_at_k(labels: list, k: int) -> float:
    """Compute Recall@k from binary relevance labels."""
    total_pos = sum(labels)
    if total_pos == 0:
        return 0.0
    hits = sum(labels[:k])
    return hits / total_pos


def load_model(model_dir: str, device: str = "cuda"):
    """Load trained reranker model."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def score_candidates(
    model,
    tokenizer,
    query: str,
    sentences: list,
    device: str = "cuda",
    max_length: int = 384,
    batch_size: int = 32,
):
    """Score all candidates for a query."""
    scores = []

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        pairs = [[query, sent] for sent in batch_sents]

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
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


def run_evaluation(
    model_dir: str,
    config_path: str = "configs/reranker_hybrid.yaml",
    max_candidates: int = 32,
    device: str = "cuda",
):
    """Run full test set assessment."""
    print("=" * 80)
    print("NO_EVIDENCE RERANKER ASSESSMENT")
    print("=" * 80)

    # Load config
    with open(config_path, "r") as f:
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

    # Build test examples WITH NO_EVIDENCE
    test_examples = build_grouped_examples(
        groundtruth,
        criteria,
        splits["test"],
        max_candidates=max_candidates,
        seed=cfg["split"]["seed"],
        add_no_evidence=True,
        include_no_evidence_queries=True,
    )

    print(f"Test examples: {len(test_examples)}")
    n_has_evidence = sum(1 for ex in test_examples if not ex["is_no_evidence"])
    n_no_evidence = sum(1 for ex in test_examples if ex["is_no_evidence"])
    print(f"  Has-evidence: {n_has_evidence}")
    print(f"  No-evidence: {n_no_evidence}")

    # Load model
    print(f"\nLoading model from {model_dir}...")
    model, tokenizer = load_model(model_dir, device)

    # Assessment containers
    results = {
        "has_evidence": {
            "correct_detection": 0,
            "total": 0,
            "ndcg_at_1": [],
            "ndcg_at_5": [],
            "ndcg_at_10": [],
            "recall_at_1": [],
            "recall_at_5": [],
            "recall_at_10": [],
            "no_evidence_ranks": [],
        },
        "no_evidence": {
            "correct_detection": 0,
            "total": 0,
            "no_evidence_ranks": [],
        },
        "dynamic_k": {
            "k_values": [],
            "ndcg": [],
            "recall": [],
        },
    }

    print("\nRunning assessment...")
    for ex in tqdm(test_examples):
        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]
        is_no_evidence = ex["is_no_evidence"]

        # Score all candidates
        scores = score_candidates(model, tokenizer, query, sentences, device)

        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        sorted_sentences = [sentences[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        # Find NO_EVIDENCE position
        try:
            no_ev_rank = sorted_sentences.index(NO_EVIDENCE_TOKEN) + 1
        except ValueError:
            no_ev_rank = len(sorted_sentences) + 1

        if is_no_evidence:
            results["no_evidence"]["total"] += 1
            results["no_evidence"]["no_evidence_ranks"].append(no_ev_rank)
            if no_ev_rank == 1:
                results["no_evidence"]["correct_detection"] += 1
        else:
            results["has_evidence"]["total"] += 1
            results["has_evidence"]["no_evidence_ranks"].append(no_ev_rank)
            if no_ev_rank > 1:
                results["has_evidence"]["correct_detection"] += 1

            # Filter out NO_EVIDENCE for ranking metrics
            filtered_labels = [l for s, l in zip(sorted_sentences, sorted_labels) if s != NO_EVIDENCE_TOKEN]

            if sum(filtered_labels) > 0:
                results["has_evidence"]["ndcg_at_1"].append(compute_ndcg_at_k(filtered_labels, 1))
                results["has_evidence"]["ndcg_at_5"].append(compute_ndcg_at_k(filtered_labels, 5))
                results["has_evidence"]["ndcg_at_10"].append(compute_ndcg_at_k(filtered_labels, 10))
                results["has_evidence"]["recall_at_1"].append(compute_recall_at_k(filtered_labels, 1))
                results["has_evidence"]["recall_at_5"].append(compute_recall_at_k(filtered_labels, 5))
                results["has_evidence"]["recall_at_10"].append(compute_recall_at_k(filtered_labels, 10))

            k = no_ev_rank - 1
            results["dynamic_k"]["k_values"].append(k)
            if k > 0 and sum(filtered_labels) > 0:
                results["dynamic_k"]["ndcg"].append(compute_ndcg_at_k(filtered_labels, k))
                results["dynamic_k"]["recall"].append(compute_recall_at_k(filtered_labels, k))

    # Compute summary
    summary = {}

    ne_total = results["no_evidence"]["total"]
    ne_correct = results["no_evidence"]["correct_detection"]
    summary["no_evidence_detection_accuracy"] = ne_correct / ne_total if ne_total > 0 else 0
    summary["no_evidence_avg_rank"] = np.mean(results["no_evidence"]["no_evidence_ranks"]) if results["no_evidence"]["no_evidence_ranks"] else 0

    he_total = results["has_evidence"]["total"]
    he_correct = results["has_evidence"]["correct_detection"]
    summary["has_evidence_detection_accuracy"] = he_correct / he_total if he_total > 0 else 0
    summary["has_evidence_no_ev_avg_rank"] = np.mean(results["has_evidence"]["no_evidence_ranks"]) if results["has_evidence"]["no_evidence_ranks"] else 0

    total = ne_total + he_total
    correct = ne_correct + he_correct
    summary["overall_detection_accuracy"] = correct / total if total > 0 else 0

    tp = ne_correct
    tn = he_correct
    fp = he_total - he_correct
    fn = ne_total - ne_correct

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    summary["ne_precision"] = precision
    summary["ne_recall"] = recall
    summary["ne_f1"] = f1
    summary["ne_specificity"] = specificity

    for metric in ["ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "recall_at_1", "recall_at_5", "recall_at_10"]:
        values = results["has_evidence"][metric]
        summary[f"has_evidence_{metric}"] = np.mean(values) if values else 0

    summary["dynamic_k_avg"] = np.mean(results["dynamic_k"]["k_values"]) if results["dynamic_k"]["k_values"] else 0
    summary["dynamic_k_ndcg"] = np.mean(results["dynamic_k"]["ndcg"]) if results["dynamic_k"]["ndcg"] else 0
    summary["dynamic_k_recall"] = np.mean(results["dynamic_k"]["recall"]) if results["dynamic_k"]["recall"] else 0

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n## No-Evidence Detection (Binary Classification)")
    print(f"  Accuracy: {summary['overall_detection_accuracy']:.4f}")
    print(f"  Precision: {summary['ne_precision']:.4f}")
    print(f"  Recall: {summary['ne_recall']:.4f}")
    print(f"  F1: {summary['ne_f1']:.4f}")
    print(f"  Specificity: {summary['ne_specificity']:.4f}")

    print("\n## Detection by Query Type")
    print(f"  No-evidence queries: {ne_correct}/{ne_total} ({summary['no_evidence_detection_accuracy']:.2%}) correctly detected")
    print(f"    Avg NO_EVIDENCE rank: {summary['no_evidence_avg_rank']:.2f}")
    print(f"  Has-evidence queries: {he_correct}/{he_total} ({summary['has_evidence_detection_accuracy']:.2%}) correctly detected")
    print(f"    Avg NO_EVIDENCE rank: {summary['has_evidence_no_ev_avg_rank']:.2f}")

    print("\n## Ranking Metrics (Has-Evidence Queries)")
    print(f"  nDCG@1: {summary['has_evidence_ndcg_at_1']:.4f}")
    print(f"  nDCG@5: {summary['has_evidence_ndcg_at_5']:.4f}")
    print(f"  nDCG@10: {summary['has_evidence_ndcg_at_10']:.4f}")
    print(f"  Recall@1: {summary['has_evidence_recall_at_1']:.4f}")
    print(f"  Recall@5: {summary['has_evidence_recall_at_5']:.4f}")
    print(f"  Recall@10: {summary['has_evidence_recall_at_10']:.4f}")

    print("\n## Learned Dynamic-K (Evidence Above NO_EVIDENCE)")
    print(f"  Average K: {summary['dynamic_k_avg']:.2f}")
    print(f"  Dynamic-K nDCG: {summary['dynamic_k_ndcg']:.4f}")
    print(f"  Dynamic-K Recall: {summary['dynamic_k_recall']:.4f}")

    print("=" * 80)

    output_dir = Path(model_dir)
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'test_results.json'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/training/no_evidence_reranker")
    parser.add_argument("--config", type=str, default="configs/reranker_hybrid.yaml")
    parser.add_argument("--max_candidates", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_evaluation(
        args.model_dir,
        args.config,
        args.max_candidates,
        args.device,
    )
