#!/usr/bin/env python3
"""Run a single final evaluation on the test split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.hpo.reporting import build_run_manifest, write_json
from final_sc_review.metrics.retrieval_eval import dual_evaluate, evaluate_rankings, format_dual_metrics
from final_sc_review.pipeline.three_stage import PipelineConfig, ThreeStagePipeline
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--dual_eval",
        action="store_true",
        help="Run dual evaluation (both positives-only and all-queries modes)",
    )
    args = parser.parse_args()

    with open(args.best_config, "r", encoding="utf-8") as f:
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
    test_posts = set(splits["test"])

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
        rebuild_cache=False,
    )

    # Support both old (top_k_colbert) and new (top_k_rerank) config options
    top_k_rerank = cfg["retriever"].get("top_k_rerank")
    if top_k_rerank is None:
        top_k_rerank = cfg["retriever"].get("top_k_colbert", cfg["retriever"]["top_k_retriever"])

    pipeline_cfg = PipelineConfig(
        bge_model=cfg["models"]["bge_m3"],
        jina_model=cfg["models"]["jina_v3"],
        bge_query_max_length=cfg["models"].get("bge_query_max_length", 128),
        bge_passage_max_length=cfg["models"].get("bge_passage_max_length", 256),
        bge_use_fp16=cfg["models"].get("bge_use_fp16", True),
        bge_batch_size=cfg["models"].get("bge_batch_size", 64),
        dense_weight=cfg["retriever"].get("dense_weight", 0.6),
        sparse_weight=cfg["retriever"].get("sparse_weight", 0.2),
        colbert_weight=cfg["retriever"].get("colbert_weight", 0.2),
        fusion_method=cfg["retriever"].get("fusion_method", "weighted_sum"),
        score_normalization=cfg["retriever"].get("score_normalization", "none"),
        rrf_k=cfg["retriever"].get("rrf_k", 60),
        use_sparse=cfg["retriever"].get("use_sparse", True),
        use_colbert=cfg["retriever"].get("use_colbert", True),
        top_k_retriever=cfg["retriever"]["top_k_retriever"],
        top_k_colbert=top_k_rerank,  # Deprecated, use top_k_rerank
        top_k_rerank=top_k_rerank,
        top_k_final=cfg["retriever"]["top_k_final"],
        reranker_max_length=cfg["models"].get("reranker_max_length", 512),
        reranker_chunk_size=cfg["models"].get("reranker_chunk_size", 64),
        reranker_dtype=cfg["models"].get("reranker_dtype", "auto"),
        reranker_use_listwise=cfg["models"].get("reranker_use_listwise", True),
        device=cfg.get("device"),
    )
    pipeline = ThreeStagePipeline(sentences, cache_dir, pipeline_cfg, rebuild_cache=False)

    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in test_posts:
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
            dense_weight=cfg["retriever"].get("dense_weight", 0.6),
            sparse_weight=cfg["retriever"].get("sparse_weight", 0.2),
            colbert_weight=cfg["retriever"].get("colbert_weight", 0.2),
            use_sparse=cfg["retriever"].get("use_sparse", True),
            use_colbert=cfg["retriever"].get("use_colbert", True),
            fusion_method=cfg["retriever"].get("fusion_method", "weighted_sum"),
            score_normalization=cfg["retriever"].get("score_normalization", "none"),
            rrf_k=cfg["retriever"].get("rrf_k", 60),
        )
        retriever_ranked_ids = [sid for sid, _, _ in retriever_results_stage]

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
        per_query_rows.append(
            {
                "post_id": post_id,
                "criterion_id": criterion_id,
                "gold_ids": "|".join(gold_ids),
                "retriever_topk": "|".join(retriever_ranked_ids),
                "reranked_topk": "|".join(reranked_ids),
            }
        )

    if args.dual_eval:
        # Dual evaluation: both positives-only and all-queries modes
        retriever_dual = dual_evaluate(retriever_results, ks=ks)
        reranked_dual = dual_evaluate(reranked_results, ks=ks)
        summary = {
            "retriever": retriever_dual,
            "reranked": reranked_dual,
            "eval_split": "test",
            "eval_mode": "dual",
        }
        logger.info("\n=== Retriever Results ===")
        logger.info("\n%s", format_dual_metrics(retriever_dual, k=10))
        logger.info("\n=== Reranked Results ===")
        logger.info("\n%s", format_dual_metrics(reranked_dual, k=10))
    else:
        # Standard evaluation with configured skip_no_positives
        summary = {
            "retriever": evaluate_rankings(
                retriever_results, ks=ks, skip_no_positives=cfg["evaluation"]["skip_no_positives"]
            ),
            "reranked": evaluate_rankings(
                reranked_results, ks=ks, skip_no_positives=cfg["evaluation"]["skip_no_positives"]
            ),
            "eval_split": "test",
            "eval_mode": "standard",
        }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/final_eval") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(per_query_rows).to_csv(out_dir / "per_query.csv", index=False)

    manifest = build_run_manifest(
        repo_dir=Path("."),
        config_snapshot=cfg,
        dataset_checksums={
            "groundtruth": _checksum(cfg["paths"]["groundtruth"]),
            "sentence_corpus": _checksum(cfg["paths"]["sentence_corpus"]),
            "criteria": _checksum(Path(cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json"),
        },
    )
    write_json(out_dir / "manifest.json", manifest)

    logger.info("Saved final evaluation to %s", out_dir)


def _checksum(path: str | Path) -> str:
    from final_sc_review.hpo.reporting import compute_sha256

    return compute_sha256(Path(path))


if __name__ == "__main__":
    main()
