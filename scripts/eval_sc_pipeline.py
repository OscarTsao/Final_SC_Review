#!/usr/bin/env python3
"""Evaluate retriever-only and reranked performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.retrieval_eval import evaluate_rankings
from final_sc_review.pipeline.three_stage import PipelineConfig, ThreeStagePipeline
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/eval_summary.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    eval_split = cfg["evaluation"].get("split", "test")
    eval_posts = set(splits[eval_split])

    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    cache_dir = Path(cfg["paths"]["cache_dir"])

    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        model_name=cfg["models"]["bge_m3"],
        device=cfg.get("device"),
        query_max_length=cfg["models"].get("bge_query_max_length", 128),
        passage_max_length=cfg["models"].get("bge_passage_max_length", 256),
        use_fp16=cfg["models"].get("bge_use_fp16", True),
        batch_size=cfg["models"].get("bge_batch_size", 64),
        rebuild_cache=cfg["retriever"].get("rebuild_cache", False),
    )
    pipeline_cfg = PipelineConfig(
        bge_model=cfg["models"]["bge_m3"],
        jina_model=cfg["models"]["jina_v3"],
        bge_query_max_length=cfg["models"].get("bge_query_max_length", 128),
        bge_passage_max_length=cfg["models"].get("bge_passage_max_length", 256),
        bge_use_fp16=cfg["models"].get("bge_use_fp16", True),
        bge_batch_size=cfg["models"].get("bge_batch_size", 64),
        dense_weight=cfg["retriever"].get("dense_weight", 0.7),
        sparse_weight=cfg["retriever"].get("sparse_weight", 0.3),
        colbert_weight=cfg["retriever"].get("colbert_weight", 0.0),
        fusion_method=cfg["retriever"].get("fusion_method", "weighted_sum"),
        score_normalization=cfg["retriever"].get("score_normalization", "none"),
        rrf_k=cfg["retriever"].get("rrf_k", 60),
        use_sparse=cfg["retriever"].get("use_sparse", True),
        use_colbert=cfg["retriever"].get("use_colbert", True),
        top_k_retriever=cfg["retriever"]["top_k_retriever"],
        top_k_colbert=cfg["retriever"].get("top_k_colbert", cfg["retriever"]["top_k_retriever"]),
        top_k_final=cfg["retriever"]["top_k_final"],
        reranker_max_length=cfg["models"].get("reranker_max_length", cfg["models"].get("max_length", 512)),
        reranker_chunk_size=cfg["models"].get("reranker_chunk_size", 64),
        reranker_dtype=cfg["models"].get("reranker_dtype", "auto"),
        reranker_use_listwise=cfg["models"].get("reranker_use_listwise", True),
        device=cfg.get("device"),
    )
    pipeline = ThreeStagePipeline(sentences, cache_dir, pipeline_cfg, rebuild_cache=False)

    # Group groundtruth by (post_id, criterion)
    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in eval_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    ks = cfg["evaluation"]["ks"]
    retriever_results = []
    reranked_results = []
    per_query_rows = []

    for (post_id, criterion_id), rows in sorted(grouped.items()):
        query = criteria_map.get(criterion_id)
        if query is None:
            continue
        gold_ids = [r.sent_uid for r in rows if r.groundtruth == 1]

        retriever_results_stage = retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=cfg["retriever"]["top_k_retriever"],
            top_k_colbert=cfg["retriever"].get("top_k_colbert", cfg["retriever"]["top_k_retriever"]),
            dense_weight=cfg["retriever"].get("dense_weight", 0.7),
            sparse_weight=cfg["retriever"].get("sparse_weight", 0.3),
            colbert_weight=cfg["retriever"].get("colbert_weight", 0.0),
            use_sparse=cfg["retriever"].get("use_sparse", True),
            use_colbert=cfg["retriever"].get("use_colbert", True),
            fusion_method=cfg["retriever"].get("fusion_method", "weighted_sum"),
            score_normalization=cfg["retriever"].get("score_normalization", "none"),
            rrf_k=cfg["retriever"].get("rrf_k", 60),
        )
        retriever_ranked_ids = [sid for sid, _, _ in retriever_results_stage][
            : cfg["retriever"].get("top_k_colbert", cfg["retriever"]["top_k_retriever"])
        ]

        reranked = pipeline.retrieve(query=query, post_id=post_id)
        reranked_ids = [sid for sid, _, _ in reranked]

        retriever_results.append(
            {
                "post_id": post_id,
                "criterion_id": criterion_id,
                "gold_ids": gold_ids,
                "ranked_ids": retriever_ranked_ids,
            }
        )
        reranked_results.append(
            {
                "post_id": post_id,
                "criterion_id": criterion_id,
                "gold_ids": gold_ids,
                "ranked_ids": reranked_ids,
            }
        )

        if cfg["evaluation"].get("save_query_csv", False):
            per_query_rows.append(
                {
                    "post_id": post_id,
                    "criterion_id": criterion_id,
                    "gold_ids": "|".join(gold_ids),
                    "retriever_topk": "|".join(retriever_ranked_ids),
                    "reranked_topk": "|".join(reranked_ids),
                }
            )

    summary = {
        "retriever": evaluate_rankings(
            retriever_results, ks=ks, skip_no_positives=cfg["evaluation"].get("skip_no_positives", True)
        ),
        "reranked": evaluate_rankings(
            reranked_results, ks=ks, skip_no_positives=cfg["evaluation"].get("skip_no_positives", True)
        ),
        "eval_split": eval_split,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", output_path)

    if per_query_rows:
        csv_path = output_path.with_suffix(".queries.csv")
        pd.DataFrame(per_query_rows).to_csv(csv_path, index=False)
        logger.info("Saved per-query results to %s", csv_path)


if __name__ == "__main__":
    main()
