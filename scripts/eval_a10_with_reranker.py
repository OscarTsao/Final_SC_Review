#!/usr/bin/env python3
"""
A.10 ablation with both retriever and reranker.

Compares performance with and without A.10 criterion.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(ranked_uids: List[str], gold_uids: Set[str], k: int) -> Dict:
    """Compute ranking metrics."""
    ranked_k = ranked_uids[:k]
    gold_set = gold_uids

    hits = sum(1 for uid in ranked_k if uid in gold_set)
    recall = hits / len(gold_uids) if gold_uids else 0.0

    mrr = 0.0
    for i, uid in enumerate(ranked_k):
        if uid in gold_set:
            mrr = 1.0 / (i + 1)
            break

    dcg = sum(1.0 / np.log2(i + 2) for i, uid in enumerate(ranked_k) if uid in gold_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_uids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def rerank_candidates(
    query: str,
    candidates: List[tuple],
    reranker_model,
    reranker_tokenizer,
    device: str = "cuda",
    batch_size: int = 16,
) -> List[tuple]:
    """Rerank candidates using reranker model."""
    if not candidates:
        return []

    texts = [c[1] for c in candidates]
    pairs = [[query, text] for text in texts]

    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = reranker_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
            if scores.ndim == 0:
                scores = np.array([scores])
            all_scores.extend(scores.tolist())

    scored = [(candidates[i][0], candidates[i][1], all_scores[i]) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def run_evaluation(
    retriever,
    reranker_model,
    reranker_tokenizer,
    queries: List[tuple],
    criterion_text: Dict,
    k_values: List[int],
    device: str,
    top_k_retriever: int = 50,
) -> Dict:
    """Run evaluation returning both retriever and reranked metrics."""
    ret_metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
    rer_metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    n_done = 0

    for post_id, criterion_id, gold_uids in queries:
        if not gold_uids:
            continue

        n_done += 1
        query_text = criterion_text.get(criterion_id, criterion_id)

        results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k_retriever=top_k_retriever,
        )
        retriever_uids = [r[0] for r in results]

        for k in k_values:
            m = compute_metrics(retriever_uids, gold_uids, k)
            ret_metrics[k]["recall"].append(m["recall"])
            ret_metrics[k]["mrr"].append(m["mrr"])
            ret_metrics[k]["ndcg"].append(m["ndcg"])

        if reranker_model is not None:
            reranked = rerank_candidates(
                query_text, results[:20], reranker_model, reranker_tokenizer, device
            )
            reranked_uids = [r[0] for r in reranked]
        else:
            reranked_uids = retriever_uids

        for k in k_values:
            m = compute_metrics(reranked_uids, gold_uids, k)
            rer_metrics[k]["recall"].append(m["recall"])
            rer_metrics[k]["mrr"].append(m["mrr"])
            rer_metrics[k]["ndcg"].append(m["ndcg"])

    ret_summary = {}
    rer_summary = {}
    for k in k_values:
        for metric in ["recall", "mrr", "ndcg"]:
            ret_summary[f"{metric}@{k}"] = np.mean(ret_metrics[k][metric]) if ret_metrics[k][metric] else 0.0
            rer_summary[f"{metric}@{k}"] = np.mean(rer_metrics[k][metric]) if rer_metrics[k][metric] else 0.0

    return {"retriever": ret_summary, "reranked": rer_summary, "n_queries": n_done}


def main():
    repo_root = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("A.10 ABLATION WITH RETRIEVER AND RERANKER")
    print("Timestamp:", datetime.now().isoformat())
    print("Device:", device)
    print("=" * 80)

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("\n[1/5] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    print("\n[2/5] Building queries...")
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}

    criterion_positive = defaultdict(int)
    for row in gt_rows:
        if row.groundtruth == 1:
            criterion_positive[row.criterion_id] += 1

    print("\nPositive labels by criterion:")
    for cid in sorted(criterion_positive.keys()):
        print("  %s: %d" % (cid, criterion_positive[cid]))

    print("\n[3/5] Splitting posts...")
    splits = split_post_ids(list(all_posts), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    print("\n[4/5] Initializing models...")
    retriever = BgeM3Retriever(sentences=sentences, cache_dir=cache_dir, rebuild_cache=False)

    reranker_path = repo_root / "outputs" / "reranker_hybrid"
    reranker_model = None
    reranker_tokenizer = None

    if (reranker_path / "model.safetensors").exists():
        print("  Loading reranker from:", reranker_path)
        try:
            reranker_tokenizer = AutoTokenizer.from_pretrained(str(reranker_path))
            reranker_model = AutoModelForSequenceClassification.from_pretrained(
                str(reranker_path), num_labels=1, trust_remote_code=True
            ).to(device)
            if reranker_tokenizer.pad_token is None:
                reranker_tokenizer.pad_token = reranker_tokenizer.eos_token
            if reranker_model.config.pad_token_id is None:
                reranker_model.config.pad_token_id = reranker_tokenizer.pad_token_id
            reranker_model.eval()
            print("  Reranker loaded successfully")
        except Exception as e:
            print("  [WARN] Could not load reranker:", e)
            reranker_model = None
    else:
        print("  [INFO] No reranker model found")

    print("\n[5/5] Running evaluations...")
    k_values = [1, 5, 10]
    all_results = {}

    for split_name, split_posts in [("dev_select", splits["val"]), ("test", splits["test"])]:
        print("\n  === %s SPLIT (%d posts) ===" % (split_name.upper(), len(split_posts)))
        eval_posts = set(split_posts)

        queries_all = []
        queries_no_a10 = []

        for (post_id, crit_id), gold_uids in query_gold.items():
            if post_id not in eval_posts:
                continue
            queries_all.append((post_id, crit_id, gold_uids))
            if crit_id != "A.10":
                queries_no_a10.append((post_id, crit_id, gold_uids))

        print("    All criteria: %d queries" % len(queries_all))
        print("    Excluding A.10: %d queries" % len(queries_no_a10))

        print("    Running ALL criteria...")
        res_all = run_evaluation(
            retriever, reranker_model, reranker_tokenizer,
            queries_all, criterion_text, k_values, device
        )

        print("    Running EXCLUDING A.10...")
        res_no_a10 = run_evaluation(
            retriever, reranker_model, reranker_tokenizer,
            queries_no_a10, criterion_text, k_values, device
        )

        all_results[split_name] = {"all_criteria": res_all, "excluding_a10": res_no_a10}

    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    for split_name in ["dev_select", "test"]:
        res = all_results[split_name]
        r_all = res["all_criteria"]
        r_no = res["excluding_a10"]

        print("\n### %s Split" % split_name.upper())
        print("Queries: %d (all), %d (excl. A.10)" % (r_all["n_queries"], r_no["n_queries"]))

        print("\n| Model | Criteria | nDCG@1 | nDCG@5 | nDCG@10 | Recall@10 | MRR@10 |")
        print("|-------|----------|--------|--------|---------|-----------|--------|")

        ret_all = r_all["retriever"]
        ret_no = r_no["retriever"]
        print("| Retriever (BGE-M3) | All | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            ret_all["ndcg@1"], ret_all["ndcg@5"], ret_all["ndcg@10"],
            ret_all["recall@10"], ret_all["mrr@10"]))
        print("| Retriever (BGE-M3) | Excl. A.10 | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            ret_no["ndcg@1"], ret_no["ndcg@5"], ret_no["ndcg@10"],
            ret_no["recall@10"], ret_no["mrr@10"]))

        rer_all = r_all["reranked"]
        rer_no = r_no["reranked"]
        print("| + Reranker (Jina-v3) | All | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            rer_all["ndcg@1"], rer_all["ndcg@5"], rer_all["ndcg@10"],
            rer_all["recall@10"], rer_all["mrr@10"]))
        print("| + Reranker (Jina-v3) | Excl. A.10 | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            rer_no["ndcg@1"], rer_no["ndcg@5"], rer_no["ndcg@10"],
            rer_no["recall@10"], rer_no["mrr@10"]))

        delta_ret = ret_no["ndcg@10"] - ret_all["ndcg@10"]
        delta_rer = rer_no["ndcg@10"] - rer_all["ndcg@10"]
        print("| **Delta (Excl-All)** | Retriever | - | - | %+.3f | - | - |" % delta_ret)
        print("| **Delta (Excl-All)** | Reranked | - | - | %+.3f | - | - |" % delta_rer)

    output_dir = repo_root / "outputs" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "criterion_positive_counts": dict(criterion_positive),
        "results": all_results,
    }

    results_path = output_dir / "a10_ablation_with_reranker.json"
    results_path.write_text(json.dumps(results_data, indent=2, default=float))
    print("\n\nResults saved to:", results_path)

    print("\n" + "=" * 80)
    print("[SUCCESS] A.10 ablation with reranker complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
