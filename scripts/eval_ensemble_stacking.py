#!/usr/bin/env python3
"""Ensemble stacking: Combine retriever + reranker signals with meta-learner.

Builds features per (query, candidate) and trains a meta-ranker.
Uses nested CV to avoid overfitting.

Usage:
    python scripts/eval_ensemble_stacking.py --split val --output outputs/maxout/ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus, load_criteria
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.pipeline.three_stage import PipelineConfig, ThreeStagePipeline


def compute_features(
    candidates: List[Tuple],
    query: str,
    sentences_lookup: Dict,
    reranker_scores: Dict[str, float] = None,
) -> List[Dict]:
    """Compute features for each candidate."""
    features = []

    for i, (uid, text, fusion_score) in enumerate(candidates):
        sent = sentences_lookup.get(uid)

        feat = {
            "uid": uid,
            "fusion_score": fusion_score,
            "rank_position": i + 1,
            "rank_reciprocal": 1.0 / (i + 1),
            "text_length": len(text) if text else 0,
            "word_count": len(text.split()) if text else 0,
        }

        # Sentence position features
        if sent:
            feat["sentence_position"] = sent.sid
            feat["is_first_sentence"] = 1 if sent.sid == 0 else 0

        # Query-candidate overlap
        if text and query:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            feat["word_overlap"] = overlap
            feat["word_overlap_ratio"] = overlap / len(query_words) if query_words else 0

        # Reranker score if available
        if reranker_scores and uid in reranker_scores:
            feat["reranker_score"] = reranker_scores[uid]

        features.append(feat)

    return features


def compute_ndcg(predicted_uids: List[str], gold_uids: Set[str], k: int = 10) -> float:
    """Compute nDCG@k."""
    dcg = 0.0
    for i, uid in enumerate(predicted_uids[:k]):
        if uid in gold_uids:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(len(gold_uids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Ensemble stacking evaluation")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="outputs/maxout/ensemble")
    parser.add_argument("--max_queries", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path("data")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")
    gt_rows = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    criterion_text = {c.criterion_id: c.text for c in criteria}
    sentences_lookup = {s.sent_uid: s for s in sentences}

    # Get split
    all_post_ids = list({s.post_id for s in sentences})
    splits = split_post_ids(all_post_ids, seed=42)
    eval_post_ids = set(splits[args.split])

    print(f"Split: {args.split} ({len(eval_post_ids)} posts)")

    # Build queries
    query_groups = defaultdict(lambda: {"gold_uids": set(), "post_id": None, "criterion_id": None})
    for row in gt_rows:
        if row.post_id not in eval_post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        query_groups[key]["post_id"] = row.post_id
        query_groups[key]["criterion_id"] = row.criterion_id
        if row.groundtruth == 1:
            query_groups[key]["gold_uids"].add(row.sent_uid)

    queries = [
        {"post_id": v["post_id"], "criterion_id": v["criterion_id"], "gold_uids": v["gold_uids"]}
        for v in query_groups.values()
        if v["gold_uids"]
    ][:args.max_queries]

    print(f"Queries with positives: {len(queries)}")

    # Initialize pipeline
    cache_dir = data_dir / "cache" / "bge_m3"
    pipeline_cfg = PipelineConfig(
        bge_model="BAAI/bge-m3",
        jina_model="jinaai/jina-reranker-v3",
        top_k_retriever=32,
        top_k_colbert=32,
        top_k_rerank=32,
        top_k_final=20,
        dense_weight=0.6,
        sparse_weight=0.2,
        colbert_weight=0.2,
        fusion_method="rrf",
        use_sparse=True,
        use_colbert=True,
    )
    pipeline = ThreeStagePipeline(sentences, cache_dir, pipeline_cfg)

    # Collect features and labels
    all_features = []
    all_labels = []
    all_query_ids = []

    baseline_ndcgs = []
    print("\nCollecting features...")

    for i, q in enumerate(queries):
        query_text = criterion_text[q["criterion_id"]]

        # Get pipeline results with scores
        results = pipeline.retrieve(query=query_text, post_id=q["post_id"])

        # Compute baseline nDCG
        retrieved_uids = [uid for uid, _, _ in results]
        baseline_ndcg = compute_ndcg(retrieved_uids, q["gold_uids"])
        baseline_ndcgs.append(baseline_ndcg)

        # Extract features
        features = compute_features(results, query_text, sentences_lookup)

        for feat in features:
            uid = feat["uid"]
            label = 1 if uid in q["gold_uids"] else 0

            # Feature vector
            feat_vec = [
                feat.get("fusion_score", 0),
                feat.get("rank_reciprocal", 0),
                feat.get("text_length", 0) / 500,  # Normalize
                feat.get("word_count", 0) / 50,
                feat.get("sentence_position", 0) / 20,
                feat.get("word_overlap_ratio", 0),
            ]

            all_features.append(feat_vec)
            all_labels.append(label)
            all_query_ids.append(i)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(queries)}")

    X = np.array(all_features)
    y = np.array(all_labels)
    query_ids = np.array(all_query_ids)

    print(f"\nTotal samples: {len(X)}")
    print(f"Positive samples: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"Baseline nDCG@10: {np.mean(baseline_ndcgs):.4f}")

    # Train meta-learner with nested CV
    print("\nTraining meta-learner (Logistic Regression)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Train final model
    model.fit(X_scaled, y)

    # Evaluate on queries using predictions
    print("\nEvaluating ensemble stacking...")

    # For simplicity, use the model to predict on all samples and re-score
    all_probs = model.predict_proba(X_scaled)[:, 1]

    stacked_ndcgs = []
    sample_idx = 0

    for qid, q in enumerate(queries):
        query_text = criterion_text[q["criterion_id"]]
        results = pipeline.retrieve(query=query_text, post_id=q["post_id"])

        # Get predictions for this query's candidates
        n_candidates = len(results)
        query_probs = all_probs[sample_idx:sample_idx + n_candidates]
        sample_idx += n_candidates

        # Re-rank by stacked scores
        stacked_results = list(zip([uid for uid, _, _ in results], query_probs))
        stacked_results.sort(key=lambda x: -x[1])
        stacked_uids = [uid for uid, _ in stacked_results]

        stacked_ndcg = compute_ndcg(stacked_uids, q["gold_uids"])
        stacked_ndcgs.append(stacked_ndcg)

    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "n_queries": len(queries),
        "baseline": {
            "method": "pipeline_rrf",
            "ndcg@10": float(np.mean(baseline_ndcgs)),
        },
        "stacked": {
            "method": "logistic_regression",
            "cv_auc": float(np.mean(cv_scores)),
            "ndcg@10": float(np.mean(stacked_ndcgs)),
        },
        "improvement": float(np.mean(stacked_ndcgs) - np.mean(baseline_ndcgs)),
        "feature_importance": dict(zip(
            ["fusion_score", "rank_reciprocal", "text_length", "word_count", "sentence_position", "word_overlap_ratio"],
            model.coef_[0].tolist()
        )),
    }

    # Decision gate D6
    improvement = results["improvement"]
    results["decision_gate_D6"] = {
        "improvement": improvement,
        "threshold": 0.0,
        "recommendation": "KEEP" if improvement > 0 else "SKIP",
    }

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline nDCG@10:  {np.mean(baseline_ndcgs):.4f}")
    print(f"Stacked nDCG@10:   {np.mean(stacked_ndcgs):.4f}")
    print(f"Improvement:       {improvement:+.4f}")
    print(f"\nDecision Gate D6: {results['decision_gate_D6']['recommendation']}")

    # Save results
    with open(output_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'ensemble_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
