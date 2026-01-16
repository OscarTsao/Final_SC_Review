#!/usr/bin/env python3
"""
Assess reranker trained with NO_EVIDENCE pseudo-candidate.

This script assesses the model's ability to:
1. Rank real evidence above NO_EVIDENCE for has-evidence queries
2. Rank NO_EVIDENCE above all candidates for no-evidence queries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import build_grouped_examples, NO_EVIDENCE_TOKEN
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def score_group(
    model,
    tokenizer,
    query: str,
    sentences: List[str],
    device: str,
    max_length: int = 256,
    batch_size: int = 32
) -> List[float]:
    """Score all sentences in a group against the query."""
    scores = []

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        pairs = [[query, s] for s in batch_sents]

        enc = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        with torch.inference_mode():
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())

    return scores


def run_assessment(
    model_path: str,
    examples: List[Dict],
    max_length: int = 256,
    batch_size: int = 32
) -> Dict:
    """
    Run assessment on examples.

    Returns metrics for:
    - NE detection (binary classification)
    - Ranking quality for has-evidence queries
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True
    )
    model.to(device)

    # Collect predictions
    results = []

    for ex in tqdm(examples, desc="Assessing"):
        query = ex["query"]
        sentences = ex["sentences"]
        labels = ex["labels"]
        is_no_evidence = ex.get("is_no_evidence", False)

        # Score all candidates
        scores = score_group(model, tokenizer, query, sentences, device, max_length, batch_size)

        # Find NO_EVIDENCE position and score
        ne_idx = None
        ne_score = None
        for i, s in enumerate(sentences):
            if s == NO_EVIDENCE_TOKEN:
                ne_idx = i
                ne_score = scores[i]
                break

        # Find best real candidate score
        real_scores = [s for i, s in enumerate(scores) if sentences[i] != NO_EVIDENCE_TOKEN]
        best_real_score = max(real_scores) if real_scores else None

        # Compute margin
        margin = best_real_score - ne_score if (best_real_score is not None and ne_score is not None) else None

        # Count candidates above NO_EVIDENCE
        n_above_ne = sum(1 for i, s in enumerate(scores)
                        if sentences[i] != NO_EVIDENCE_TOKEN and s > ne_score) if ne_score is not None else 0

        results.append({
            "post_id": ex["post_id"],
            "criterion_id": ex["criterion_id"],
            "is_no_evidence": is_no_evidence,
            "gt_has_evidence": not is_no_evidence,
            "ne_score": ne_score,
            "best_real_score": best_real_score,
            "margin": margin,
            "n_above_ne": n_above_ne,
            "n_candidates": len(sentences),
        })

    df = pd.DataFrame(results)

    # Binary classification: predict has_evidence based on margin
    # has_evidence if margin > 0 (best real candidate > NO_EVIDENCE)
    df["pred_has_evidence"] = (df["margin"] > 0).astype(int)

    y_true = df["gt_has_evidence"].astype(int).values
    y_pred = df["pred_has_evidence"].values
    y_score = df["margin"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.5

    metrics = {
        "n_total": len(df),
        "n_has_evidence": int(df["gt_has_evidence"].sum()),
        "n_no_evidence": int((~df["gt_has_evidence"]).sum()),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "precision": float(precision),
        "f1": float(f1),
        "mcc": float(mcc),
        "auroc": float(auroc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # Distribution analysis
    has_ev_df = df[df["gt_has_evidence"]]
    no_ev_df = df[~df["gt_has_evidence"]]

    metrics["has_evidence_margin_mean"] = float(has_ev_df["margin"].mean()) if len(has_ev_df) > 0 else 0
    metrics["has_evidence_margin_std"] = float(has_ev_df["margin"].std()) if len(has_ev_df) > 0 else 0
    metrics["no_evidence_margin_mean"] = float(no_ev_df["margin"].mean()) if len(no_ev_df) > 0 else 0
    metrics["no_evidence_margin_std"] = float(no_ev_df["margin"].std()) if len(no_ev_df) > 0 else 0

    # n_above_ne distribution
    metrics["has_evidence_n_above_ne_mean"] = float(has_ev_df["n_above_ne"].mean()) if len(has_ev_df) > 0 else 0
    metrics["no_evidence_n_above_ne_mean"] = float(no_ev_df["n_above_ne"].mean()) if len(no_ev_df) > 0 else 0

    return metrics, df


def main():
    parser = argparse.ArgumentParser(description="Assess NO_EVIDENCE reranker")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/reranker_with_no_evidence.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--outdir", type=str, default="outputs/assessment_no_evidence_model")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load data
    gt = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")

    post_ids = sorted({row.post_id for row in gt})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )

    # Build assessment examples with NO_EVIDENCE
    assessment_examples = build_grouped_examples(
        gt, criteria, splits[args.split],
        max_candidates=cfg["data"]["max_candidates"],
        seed=cfg["split"]["seed"],
        add_no_evidence=True,
        include_no_evidence_queries=True,
    )

    logger.info(f"Assessing on {len(assessment_examples)} {args.split} examples")

    # Run assessment
    metrics, results_df = run_assessment(
        args.model,
        assessment_examples,
        max_length=cfg["models"]["max_length"],
    )

    # Print results
    print("\n" + "="*60)
    print("ASSESSMENT RESULTS")
    print("="*60)
    print(f"\nDataset: {args.split} ({metrics['n_total']} queries)")
    print(f"  Has-evidence: {metrics['n_has_evidence']}")
    print(f"  No-evidence: {metrics['n_no_evidence']}")

    print("\n--- Binary Classification (margin > 0) ---")
    print(f"TPR (recall): {metrics['tpr']:.2%}")
    print(f"FPR: {metrics['fpr']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"MCC: {metrics['mcc']:.3f}")
    print(f"AUROC: {metrics['auroc']:.3f}")

    print("\n--- Confusion Matrix ---")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}, TN: {metrics['tn']}")

    print("\n--- Margin Distribution ---")
    print(f"Has-evidence: {metrics['has_evidence_margin_mean']:.3f} ± {metrics['has_evidence_margin_std']:.3f}")
    print(f"No-evidence: {metrics['no_evidence_margin_mean']:.3f} ± {metrics['no_evidence_margin_std']:.3f}")

    print("\n--- n_above_ne Distribution ---")
    print(f"Has-evidence: {metrics['has_evidence_n_above_ne_mean']:.2f}")
    print(f"No-evidence: {metrics['no_evidence_n_above_ne_mean']:.2f}")

    # Compare with targets
    print("\n--- vs Targets (high_recall_low_hallucination) ---")
    target_fpr = 0.05
    target_tpr = 0.90
    print(f"FPR: {metrics['fpr']:.2%} (target: ≤{target_fpr:.0%}) {'✅' if metrics['fpr'] <= target_fpr else '❌'}")
    print(f"TPR: {metrics['tpr']:.2%} (target: ≥{target_tpr:.0%}) {'✅' if metrics['tpr'] >= target_tpr else '❌'}")

    # Save results
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    results_df.to_parquet(outdir / "results.parquet")

    print(f"\nResults saved to {outdir}")


if __name__ == "__main__":
    main()
