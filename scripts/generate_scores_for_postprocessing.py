#!/usr/bin/env python3
"""Generate per_query.csv with scores for postprocessing evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from create_splits import load_split
from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def generate_per_query_with_scores(
    retriever_model_id: str,
    retriever_name: str,
    reranker_model_id: str,
    groundtruth,
    criteria,
    sentences,
    post_ids: Set[str],
    top_k_retriever: int = 20,
    device: str = "cuda",
) -> pd.DataFrame:
    """Generate per_query results with scores."""

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

    # Load retriever
    logger.info(f"Loading retriever: {retriever_name}")
    retriever = SentenceTransformer(retriever_model_id, trust_remote_code=True)

    # Load reranker
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

    rows = []
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

        rows.append({
            "post_id": post_id,
            "criterion_id": criterion_id,
            "gold_ids": "|".join(gold_uids) if gold_uids else "",
            "retriever_topk": "|".join(retriever_ranked_uids),
            "reranked_topk": "|".join(reranked_uids),
            "scores": "|".join([f"{s:.6f}" for s in reranked_scores]),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--retriever_name", default="bge-large-en-v1.5")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--split", default="dev_select")
    parser.add_argument("--output", default="outputs/postprocessing/per_query_with_scores.csv")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path("data")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")

    post_ids = set(load_split(args.split))
    logger.info(f"{args.split} posts: {len(post_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = generate_per_query_with_scores(
        args.retriever,
        args.retriever_name,
        args.reranker,
        groundtruth,
        criteria,
        sentences,
        post_ids,
        top_k_retriever=args.top_k,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} queries to {output_path}")

    # Print summary
    n_with_gold = (df["gold_ids"] != "").sum()
    n_empty = (df["gold_ids"] == "").sum()
    print(f"\nSummary:")
    print(f"  Total queries: {len(df)}")
    print(f"  With positives: {n_with_gold}")
    print(f"  Empty: {n_empty}")
    print(f"  Empty prevalence: {n_empty/len(df):.2%}")


if __name__ == "__main__":
    main()
