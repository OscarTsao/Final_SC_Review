#!/usr/bin/env python3
"""Phase 6: Interaction Check - Test top retrievers with best reranker."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
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


@dataclass
class InteractionConfig:
    """Configuration for interaction check."""
    top_retrievers: List[Dict] = None
    best_reranker: Dict = None
    k_primary: List[int] = None
    top_k_retriever: int = 20
    output_dir: str = "outputs/interaction_check"

    def __post_init__(self):
        if self.k_primary is None:
            self.k_primary = [3, 5, 10]
        if self.top_retrievers is None:
            self.top_retrievers = [
                {"name": "e5-large-v2", "model_id": "intfloat/e5-large-v2"},
                {"name": "bge-large-en-v1.5", "model_id": "BAAI/bge-large-en-v1.5"},
                {"name": "gte-large-en-v1.5", "model_id": "Alibaba-NLP/gte-large-en-v1.5"},
            ]
        if self.best_reranker is None:
            self.best_reranker = {
                "name": "bge-reranker-v2-m3",
                "model_id": "BAAI/bge-reranker-v2-m3",
            }


def compute_retriever_rankings(
    retriever: SentenceTransformer,
    retriever_name: str,
    groundtruth,
    criteria,
    sentences,
    post_ids: Set[str],
    top_k: int = 20,
) -> Dict[tuple, List[str]]:
    """Compute retriever rankings for each (post_id, criterion_id) pair."""
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

    rankings = {}

    for (post_id, criterion_id), data in groups.items():
        query_text = criterion_text.get(criterion_id)
        if not query_text:
            continue

        post_sents = post_to_sentences.get(post_id, [])
        if not post_sents:
            continue

        sent_texts = [s.text for s in post_sents]
        sent_uids = [s.sent_uid for s in post_sents]

        # Encode based on model type
        if "e5" in retriever_name.lower():
            query_emb = retriever.encode(["query: " + query_text], normalize_embeddings=True)
            sent_embs = retriever.encode(["passage: " + t for t in sent_texts], normalize_embeddings=True)
        elif "bge" in retriever_name.lower():
            query_emb = retriever.encode([query_text], normalize_embeddings=True)
            sent_embs = retriever.encode(sent_texts, normalize_embeddings=True)
        else:
            query_emb = retriever.encode([query_text], normalize_embeddings=True)
            sent_embs = retriever.encode(sent_texts, normalize_embeddings=True)

        scores = np.dot(query_emb, sent_embs.T)[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_uids = [sent_uids[i] for i in ranked_indices]

        rankings[(post_id, criterion_id)] = {
            "ranked_uids": ranked_uids,
            "gold_uids": data["gold_uids"],
            "candidates": [(sent_uids[i], sent_texts[i]) for i in ranked_indices],
        }

    return rankings


def rerank_with_model(
    reranker_model,
    reranker_tokenizer,
    rankings: Dict,
    criteria,
    device: str = "cuda",
) -> Dict[tuple, List[str]]:
    """Rerank candidates using the reranker model."""
    criterion_text = {c.criterion_id: c.text for c in criteria}
    reranked = {}

    reranker_model.eval()

    for (post_id, criterion_id), data in rankings.items():
        query = criterion_text.get(criterion_id)
        if not query:
            continue

        candidates = data["candidates"]
        if not candidates:
            continue

        pairs = [(query, text) for uid, text in candidates]

        with torch.no_grad():
            inputs = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            scores = reranker_model(**inputs).logits.squeeze(-1).cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1]
        reranked_uids = [candidates[i][0] for i in ranked_indices]

        reranked[(post_id, criterion_id)] = {
            "ranked_uids": reranked_uids,
            "gold_uids": data["gold_uids"],
        }

    return reranked


def compute_metrics(rankings: Dict, ks: List[int]) -> Dict:
    """Compute metrics from rankings."""
    metrics = {f"oracle@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"ndcg@{k}": [] for k in ks})
    metrics.update({f"mrr@{k}": [] for k in ks})

    n_positive = 0
    n_empty = 0

    for key, data in rankings.items():
        gold_uids = data["gold_uids"]
        ranked_uids = data["ranked_uids"]

        if not gold_uids:
            n_empty += 1
            continue

        n_positive += 1
        n_candidates = len(ranked_uids)

        for k in ks:
            k_eff = min(k, n_candidates)
            top_k_uids = set(ranked_uids[:k_eff])

            oracle = 1.0 if top_k_uids & gold_uids else 0.0
            metrics[f"oracle@{k}"].append(oracle)
            metrics[f"recall@{k}"].append(recall_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(gold_uids, ranked_uids, k_eff))
            metrics[f"mrr@{k}"].append(mrr_at_k(gold_uids, ranked_uids, k_eff))

    result = {
        "n_positive": n_positive,
        "n_empty": n_empty,
    }
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if values else 0.0

    return result


def run_interaction_check(config: InteractionConfig):
    """Run full interaction check."""
    logger.info("=" * 60)
    logger.info("PHASE 6: INTERACTION CHECK")
    logger.info("=" * 60)

    data_dir = Path("data")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")

    dev_select_ids = set(load_split("dev_select"))
    logger.info(f"dev_select posts: {len(dev_select_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load reranker
    logger.info(f"\nLoading reranker: {config.best_reranker['name']}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(config.best_reranker["model_id"])
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        config.best_reranker["model_id"],
        num_labels=1,
        trust_remote_code=True,
    ).to(device)

    if reranker_tokenizer.pad_token is None:
        reranker_tokenizer.pad_token = reranker_tokenizer.eos_token
    if reranker_model.config.pad_token_id is None:
        reranker_model.config.pad_token_id = reranker_tokenizer.pad_token_id

    results = []

    for retriever_cfg in config.top_retrievers:
        logger.info(f"\n--- Testing: {retriever_cfg['name']} + {config.best_reranker['name']} ---")

        # Load retriever
        retriever = SentenceTransformer(retriever_cfg["model_id"], trust_remote_code=True)

        # Get retriever rankings
        logger.info("Computing retriever rankings...")
        retriever_rankings = compute_retriever_rankings(
            retriever,
            retriever_cfg["name"],
            groundtruth,
            criteria,
            sentences,
            dev_select_ids,
            top_k=config.top_k_retriever,
        )

        # Compute retriever-only metrics
        retriever_metrics = compute_metrics(retriever_rankings, config.k_primary)
        logger.info(f"Retriever-only Oracle@10: {retriever_metrics['oracle@10']:.4f}")

        # Rerank
        logger.info("Reranking...")
        reranked_rankings = rerank_with_model(
            reranker_model,
            reranker_tokenizer,
            retriever_rankings,
            criteria,
            device,
        )

        # Compute reranked metrics
        reranked_metrics = compute_metrics(reranked_rankings, config.k_primary)
        logger.info(f"Reranked Oracle@10: {reranked_metrics['oracle@10']:.4f}")

        results.append({
            "retriever": retriever_cfg["name"],
            "reranker": config.best_reranker["name"],
            "retriever_metrics": retriever_metrics,
            "reranked_metrics": reranked_metrics,
        })

        # Clean up
        del retriever
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Find best combination
    best_combo = max(results, key=lambda x: x["reranked_metrics"]["oracle@10"])
    logger.info("\n" + "=" * 60)
    logger.info("BEST COMBINATION:")
    logger.info(f"  Retriever: {best_combo['retriever']}")
    logger.info(f"  Reranker: {best_combo['reranker']}")
    logger.info(f"  Oracle@10: {best_combo['reranked_metrics']['oracle@10']:.4f}")
    logger.info("=" * 60)

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "interaction_results.json", "w") as f:
        json.dump({
            "results": results,
            "best_combination": {
                "retriever": best_combo["retriever"],
                "reranker": best_combo["reranker"],
                "oracle@10": best_combo["reranked_metrics"]["oracle@10"],
            },
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'interaction_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/interaction_check")
    args = parser.parse_args()

    config = InteractionConfig(output_dir=args.output_dir)
    results = run_interaction_check(config)

    # Print summary table
    print("\n" + "=" * 70)
    print("INTERACTION CHECK RESULTS")
    print("=" * 70)
    print(f"{'Retriever':<25} {'Oracle@3':<10} {'Oracle@5':<10} {'Oracle@10':<10}")
    print("-" * 70)
    for r in results:
        m = r["reranked_metrics"]
        print(f"{r['retriever']:<25} {m['oracle@3']:.4f}     {m['oracle@5']:.4f}     {m['oracle@10']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
