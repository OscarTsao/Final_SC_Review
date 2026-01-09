#!/usr/bin/env python3
"""Phase 10: Final Test Evaluation - ONE-TIME RUN ON TEST SET.

WARNING: This script should only be run ONCE with the locked configuration.
Running multiple times on the test set violates experimental protocol.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from create_splits import load_split
from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.metrics.ranking import recall_at_k, ndcg_at_k, mrr_at_k
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def run_final_test(
    retriever_model_id: str,
    retriever_name: str,
    reranker_model_id: str,
    groundtruth,
    criteria,
    sentences,
    post_ids: Set[str],
    ks: List[int] = [3, 5, 10],
    top_k_retriever: int = 20,
    device: str = "cuda",
) -> Dict:
    """Run final test evaluation."""

    criterion_text = {c.criterion_id: c.text for c in criteria}
    post_to_sentences = defaultdict(list)
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)

    groups = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth:
        if row.post_id not in post_ids:
            continue
        key = (row.post_id, row.criterion_id)
        groups[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            groups[key]["gold_uids"].add(row.sent_uid)

    # Load models
    logger.info(f"Loading retriever: {retriever_name}")
    retriever = SentenceTransformer(retriever_model_id, trust_remote_code=True)

    logger.info(f"Loading reranker: {reranker_model_id}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_id)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        reranker_model_id,
        num_labels=1,
        trust_remote_code=True,
    ).to(device)

    if reranker_tokenizer.pad_token is None:
        reranker_tokenizer.pad_token = reranker_tokenizer.eos_token
    if reranker_model.config.pad_token_id is None:
        reranker_model.config.pad_token_id = reranker_tokenizer.pad_token_id

    reranker_model.eval()

    # Metrics storage
    metrics = {f"oracle@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"ndcg@{k}": [] for k in ks})
    metrics.update({f"mrr@{k}": [] for k in ks})

    retriever_metrics = {f"retriever_oracle@{k}": [] for k in ks}

    n_positive = 0
    n_empty = 0
    per_query_rows = []

    total = len(groups)
    for idx, ((post_id, criterion_id), data) in enumerate(groups.items()):
        if idx % 100 == 0:
            logger.info(f"Processing {idx}/{total}...")

        query_text = criterion_text.get(criterion_id)
        if not query_text:
            continue

        post_sents = post_to_sentences.get(post_id, [])
        if not post_sents:
            continue

        sent_texts = [s.text for s in post_sents]
        sent_uids = [s.sent_uid for s in post_sents]
        gold_uids = data["gold_uids"]

        # Encode
        if "bge" in retriever_name.lower():
            query_emb = retriever.encode([query_text], normalize_embeddings=True)
            sent_embs = retriever.encode(sent_texts, normalize_embeddings=True)
        else:
            query_emb = retriever.encode([query_text], normalize_embeddings=True)
            sent_embs = retriever.encode(sent_texts, normalize_embeddings=True)

        retriever_scores = np.dot(query_emb, sent_embs.T)[0]
        ranked_indices = np.argsort(retriever_scores)[::-1][:top_k_retriever]
        retriever_ranked_uids = [sent_uids[i] for i in ranked_indices]

        # Rerank
        candidates = [(sent_uids[i], sent_texts[i]) for i in ranked_indices]
        pairs = [(query_text, text) for uid, text in candidates]

        with torch.no_grad():
            inputs = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            reranker_scores = reranker_model(**inputs).logits.squeeze(-1).cpu().numpy()

        reranked_indices = np.argsort(reranker_scores)[::-1]
        reranked_uids = [candidates[i][0] for i in reranked_indices]
        reranked_scores = [float(reranker_scores[i]) for i in reranked_indices]

        # Save per-query data
        per_query_rows.append({
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gold_ids": "|".join(gold_uids) if gold_uids else "",
            "retriever_topk": "|".join(retriever_ranked_uids),
            "reranked_topk": "|".join(reranked_uids),
            "scores": "|".join([f"{s:.6f}" for s in reranked_scores]),
            "has_positive": 1 if gold_uids else 0,
        })

        # Compute metrics
        if not gold_uids:
            n_empty += 1
            continue

        n_positive += 1
        n_candidates = len(reranked_uids)

        for k in ks:
            k_eff = min(k, n_candidates)

            # Retriever-only metrics
            ret_top_k = set(retriever_ranked_uids[:k_eff])
            retriever_metrics[f"retriever_oracle@{k}"].append(1.0 if ret_top_k & gold_uids else 0.0)

            # Reranked metrics
            top_k_uids = set(reranked_uids[:k_eff])
            oracle = 1.0 if top_k_uids & gold_uids else 0.0
            metrics[f"oracle@{k}"].append(oracle)
            metrics[f"recall@{k}"].append(recall_at_k(gold_uids, reranked_uids, k_eff))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(gold_uids, reranked_uids, k_eff))
            metrics[f"mrr@{k}"].append(mrr_at_k(gold_uids, reranked_uids, k_eff))

    # Aggregate results
    result = {
        "n_queries": len(groups),
        "n_positive": n_positive,
        "n_empty": n_empty,
        "empty_prevalence": n_empty / len(groups) if groups else 0,
    }

    # Retriever-only metrics
    for key, values in retriever_metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    # Reranked metrics
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    return result, pd.DataFrame(per_query_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locked_best_config.yaml")
    parser.add_argument("--output_dir", default="outputs/final_test_results")
    parser.add_argument("--confirm", action="store_true", help="Confirm running on test set")
    args = parser.parse_args()

    if not args.confirm:
        print("=" * 70)
        print("WARNING: This will run the FINAL TEST EVALUATION.")
        print("This should only be done ONCE with the locked configuration.")
        print("=" * 70)
        print("\nTo proceed, run with --confirm flag")
        return 1

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Retriever: {config['models']['retriever']['name']}")
    logger.info(f"Reranker: {config['models']['reranker']['name']}")

    # Load data
    data_dir = Path(config["paths"]["data_dir"])
    groundtruth = load_groundtruth(Path(config["paths"]["groundtruth"]))
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    sentences = load_sentence_corpus(Path(config["paths"]["sentence_corpus"]))

    # Load TEST split
    test_ids = set(load_split("test"))
    logger.info(f"Test posts: {len(test_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run evaluation
    results, per_query_df = run_final_test(
        config["models"]["retriever"]["model_id"],
        config["models"]["retriever"]["name"],
        config["models"]["reranker"]["model_id"],
        groundtruth,
        criteria,
        sentences,
        test_ids,
        ks=config["k_policy"]["k_primary"],
        top_k_retriever=config["k_policy"]["top_k_retriever"],
        device=device,
    )

    # Add metadata
    results["config"] = {
        "retriever": config["models"]["retriever"]["name"],
        "reranker": config["models"]["reranker"]["name"],
        "retriever_model_id": config["models"]["retriever"]["model_id"],
        "reranker_model_id": config["models"]["reranker"]["model_id"],
    }
    results["timestamp"] = datetime.now().isoformat()
    results["split"] = "test"

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "final_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    per_query_df.to_csv(output_dir / "per_query.csv", index=False)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total queries: {results['n_queries']}")
    logger.info(f"With positives: {results['n_positive']}")
    logger.info(f"Empty: {results['n_empty']} ({results['empty_prevalence']:.1%})")
    logger.info("")
    logger.info("Retriever-only:")
    for k in config["k_policy"]["k_primary"]:
        logger.info(f"  Oracle@{k}: {results[f'retriever_oracle@{k}']:.4f}")
    logger.info("")
    logger.info("Reranked (FINAL):")
    for k in config["k_policy"]["k_primary"]:
        logger.info(f"  Oracle@{k}: {results[f'oracle@{k}']:.4f}")
        logger.info(f"  Recall@{k}: {results[f'recall@{k}']:.4f}")
        logger.info(f"  nDCG@{k}: {results[f'ndcg@{k}']:.4f}")
        logger.info(f"  MRR@{k}: {results[f'mrr@{k}']:.4f}")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
