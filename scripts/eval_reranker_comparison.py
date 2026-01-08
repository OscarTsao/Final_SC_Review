#!/usr/bin/env python3
"""Compare multiple rerankers on the validation split.

Quick evaluation to determine which rerankers are worth training HPO.

Usage:
    python scripts/eval_reranker_comparison.py --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus, load_criteria
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.bge_m3 import BgeM3Retriever


class JinaReranker:
    """Wrapper for Jina reranker that doesn't use CrossEncoder."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Set to inference mode
        self.model.training = False
        for param in self.model.parameters():
            param.requires_grad = False

    def predict(self, pairs):
        import torch
        scores = []
        batch_size = 16

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            inputs = self.tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Handle different output shapes
                if logits.dim() == 1:
                    batch_scores = logits.cpu().numpy().tolist()
                elif logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().numpy().tolist()
                else:
                    # Binary classification - take positive class score
                    batch_scores = logits[:, -1].cpu().numpy().tolist()
                if isinstance(batch_scores, (int, float)):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

        return scores


def load_reranker(model_id: str, device: str = "cuda"):
    """Load a cross-encoder reranker model."""
    try:
        if "jina" in model_id.lower():
            print(f"Loading Jina reranker: {model_id}")
            return JinaReranker(model_id, device=device)
        else:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker: {model_id}")
            model = CrossEncoder(model_id, device=device, max_length=512)
            return model
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None


def compute_metrics(
    retrieved_uids: List[str],
    gold_uids: Set[str],
    ks: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Compute retrieval metrics."""
    metrics = {}

    for k in ks:
        top_k = set(retrieved_uids[:k])

        # Recall
        hits = len(gold_uids & top_k)
        recall = hits / len(gold_uids) if gold_uids else 0.0
        metrics[f"recall@{k}"] = recall

        # MRR
        mrr = 0.0
        for i, uid in enumerate(retrieved_uids[:k]):
            if uid in gold_uids:
                mrr = 1.0 / (i + 1)
                break
        metrics[f"mrr@{k}"] = mrr

        # nDCG
        dcg = 0.0
        for i, uid in enumerate(retrieved_uids[:k]):
            if uid in gold_uids:
                dcg += 1.0 / np.log2(i + 2)
        ideal_hits = min(len(gold_uids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare rerankers")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="outputs/maxout/reranker/comparison_results.json")
    parser.add_argument("--max_queries", type=int, default=50, help="Max queries for quick eval")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rerankers to compare
    rerankers_config = [
        {"name": "jina-reranker-v3", "model_id": "jinaai/jina-reranker-v3"},
        {"name": "bge-reranker-v2-m3", "model_id": "BAAI/bge-reranker-v2-m3"},
    ]

    # Load data
    data_dir = Path("data")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")
    gt_rows = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Build sentence lookup
    sent_lookup = {s.sent_uid: s for s in sentences}

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

    print(f"Evaluating {len(queries)} queries")

    # Initialize retriever for getting candidates
    cache_dir = data_dir / "cache" / "bge_m3"
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        model_name="BAAI/bge-m3",
        rebuild_cache=False,
    )

    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "n_queries": len(queries),
        "rerankers": {},
    }

    # First, get retriever baseline
    print(f"\n{'='*60}")
    print("Evaluating: retriever_baseline")
    print(f"{'='*60}")

    baseline_metrics = defaultdict(list)
    all_candidates = {}  # Store for reranking

    for i, q in enumerate(queries):
        candidates = retriever.retrieve_within_post(
            query=criterion_text[q["criterion_id"]],
            post_id=q["post_id"],
            top_k_retriever=32,
            top_k_colbert=32,
            use_sparse=True,
            use_colbert=True,
            fusion_method="rrf",
        )
        retrieved_uids = [uid for uid, _, _ in candidates]
        all_candidates[(q["post_id"], q["criterion_id"])] = candidates

        metrics = compute_metrics(retrieved_uids, q["gold_uids"])
        for k, v in metrics.items():
            baseline_metrics[k].append(v)

    agg_baseline = {k: float(np.mean(v)) for k, v in baseline_metrics.items()}
    results["rerankers"]["retriever_baseline"] = {"metrics": agg_baseline}
    print(f"  nDCG@10: {agg_baseline['ndcg@10']:.4f}")
    print(f"  MRR@10: {agg_baseline['mrr@10']:.4f}")

    # Evaluate each reranker
    for rr_config in rerankers_config:
        name = rr_config["name"]
        model_id = rr_config["model_id"]

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        reranker = load_reranker(model_id)
        if reranker is None:
            results["rerankers"][name] = {"error": "Failed to load"}
            continue

        rr_metrics = defaultdict(list)

        for i, q in enumerate(queries):
            key = (q["post_id"], q["criterion_id"])
            candidates = all_candidates[key]
            query_text = criterion_text[q["criterion_id"]]

            # Prepare pairs for reranking
            pairs = [(query_text, sent_lookup[uid].text) for uid, _, _ in candidates if uid in sent_lookup]
            uids = [uid for uid, _, _ in candidates if uid in sent_lookup]

            if not pairs:
                continue

            # Rerank
            scores = reranker.predict(pairs)

            # Sort by reranker score
            reranked = sorted(zip(uids, scores), key=lambda x: -x[1])
            reranked_uids = [uid for uid, _ in reranked]

            metrics = compute_metrics(reranked_uids, q["gold_uids"])
            for k, v in metrics.items():
                rr_metrics[k].append(v)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(queries)}")

        agg_rr = {k: float(np.mean(v)) for k, v in rr_metrics.items()}
        results["rerankers"][name] = {"metrics": agg_rr, "model_id": model_id}
        print(f"  nDCG@10: {agg_rr['ndcg@10']:.4f}")
        print(f"  MRR@10: {agg_rr['mrr@10']:.4f}")

        # Clean up GPU memory
        del reranker
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Reranker':<25} {'nDCG@10':<10} {'MRR@10':<10} {'Recall@20':<12}")
    print("-" * 60)

    for name, data in results["rerankers"].items():
        if "error" in data:
            print(f"{name:<25} ERROR")
        else:
            m = data["metrics"]
            print(f"{name:<25} {m['ndcg@10']:<10.4f} {m['mrr@10']:<10.4f} {m['recall@20']:<12.4f}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
