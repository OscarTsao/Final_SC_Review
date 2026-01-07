#!/usr/bin/env python3
"""Run ablation studies comparing different configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.retrieval_eval import dual_evaluate
from final_sc_review.pipeline.three_stage import PipelineConfig, ThreeStagePipeline
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    base_cfg = _load_yaml(Path("configs/default.yaml"))

    groundtruth = load_groundtruth(Path(base_cfg["paths"]["groundtruth"]))
    criteria = load_criteria(Path(base_cfg["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}
    sentences = load_sentence_corpus(Path(base_cfg["paths"]["sentence_corpus"]))
    cache_dir = Path(base_cfg["paths"]["cache_dir"])

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=base_cfg["split"]["seed"],
        train_ratio=base_cfg["split"]["train_ratio"],
        val_ratio=base_cfg["split"]["val_ratio"],
        test_ratio=base_cfg["split"]["test_ratio"],
    )
    test_posts = set(splits["test"])

    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in test_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    # Ablation configurations
    ablations = {
        "baseline_default": {
            "top_k_retriever": 64,
            "top_k_rerank": 64,
            "top_k_final": 10,
            "use_sparse": True,
            "use_colbert": True,
            "fusion_method": "weighted_sum",
        },
        "best_v2_rrf": {
            "top_k_retriever": 32,
            "top_k_rerank": 32,
            "top_k_final": 20,
            "use_sparse": True,
            "use_colbert": True,
            "fusion_method": "rrf",
        },
        "large_pool_100_50": {
            "top_k_retriever": 100,
            "top_k_rerank": 50,
            "top_k_final": 20,
            "use_sparse": True,
            "use_colbert": True,
            "fusion_method": "rrf",
        },
        "dense_only": {
            "top_k_retriever": 32,
            "top_k_rerank": 32,
            "top_k_final": 20,
            "use_sparse": False,
            "use_colbert": False,
            "fusion_method": "weighted_sum",
        },
        "no_reranker": {
            "top_k_retriever": 20,
            "top_k_rerank": 20,
            "top_k_final": 20,
            "use_sparse": True,
            "use_colbert": True,
            "fusion_method": "rrf",
        },
    }

    results = {}

    for name, cfg_overrides in ablations.items():
        logger.info("Running ablation: %s", name)

        # Merge config
        ret_cfg = {**base_cfg["retriever"], **cfg_overrides}

        retriever = BgeM3Retriever(
            sentences=sentences,
            cache_dir=cache_dir,
            model_name=base_cfg["models"]["bge_m3"],
            device=base_cfg.get("device"),
            query_max_length=base_cfg["models"].get("bge_query_max_length", 128),
            passage_max_length=base_cfg["models"].get("bge_passage_max_length", 256),
            use_fp16=base_cfg["models"].get("bge_use_fp16", True),
            batch_size=base_cfg["models"].get("bge_batch_size", 64),
            rebuild_cache=False,
        )

        pipeline_cfg = PipelineConfig(
            bge_model=base_cfg["models"]["bge_m3"],
            jina_model=base_cfg["models"]["jina_v3"],
            bge_query_max_length=base_cfg["models"].get("bge_query_max_length", 128),
            bge_passage_max_length=base_cfg["models"].get("bge_passage_max_length", 256),
            bge_use_fp16=base_cfg["models"].get("bge_use_fp16", True),
            bge_batch_size=base_cfg["models"].get("bge_batch_size", 64),
            dense_weight=ret_cfg.get("dense_weight", 0.6),
            sparse_weight=ret_cfg.get("sparse_weight", 0.2),
            colbert_weight=ret_cfg.get("colbert_weight", 0.2),
            fusion_method=ret_cfg["fusion_method"],
            score_normalization=ret_cfg.get("score_normalization", "none"),
            rrf_k=ret_cfg.get("rrf_k", 60),
            use_sparse=ret_cfg["use_sparse"],
            use_colbert=ret_cfg["use_colbert"],
            top_k_retriever=ret_cfg["top_k_retriever"],
            top_k_colbert=ret_cfg["top_k_rerank"],
            top_k_rerank=ret_cfg["top_k_rerank"],
            top_k_final=ret_cfg["top_k_final"],
            reranker_max_length=base_cfg["models"].get("reranker_max_length", 512),
            reranker_chunk_size=base_cfg["models"].get("reranker_chunk_size", 64),
            reranker_dtype=base_cfg["models"].get("reranker_dtype", "auto"),
            reranker_use_listwise=base_cfg["models"].get("reranker_use_listwise", True),
            device=base_cfg.get("device"),
        )
        pipeline = ThreeStagePipeline(sentences, cache_dir, pipeline_cfg, rebuild_cache=False)

        retriever_results = []
        reranked_results = []

        for (post_id, criterion_id), rows in sorted(grouped.items()):
            query = criteria_map.get(criterion_id)
            if query is None:
                continue
            gold_ids = [r.sent_uid for r in rows if r.groundtruth == 1]

            retriever_stage = retriever.retrieve_within_post(
                query=query,
                post_id=post_id,
                top_k_retriever=ret_cfg["top_k_retriever"],
                top_k_colbert=ret_cfg["top_k_rerank"],
                dense_weight=ret_cfg.get("dense_weight", 0.6),
                sparse_weight=ret_cfg.get("sparse_weight", 0.2),
                colbert_weight=ret_cfg.get("colbert_weight", 0.2),
                use_sparse=ret_cfg["use_sparse"],
                use_colbert=ret_cfg["use_colbert"],
                fusion_method=ret_cfg["fusion_method"],
                score_normalization=ret_cfg.get("score_normalization", "none"),
                rrf_k=ret_cfg.get("rrf_k", 60),
            )
            retriever_ranked_ids = [sid for sid, _, _ in retriever_stage]

            reranked = pipeline.retrieve(query=query, post_id=post_id)
            reranked_ids = [sid for sid, _, _ in reranked]

            retriever_results.append({
                "post_id": post_id,
                "criterion_id": criterion_id,
                "gold_ids": gold_ids,
                "ranked_ids": retriever_ranked_ids,
            })
            reranked_results.append({
                "post_id": post_id,
                "criterion_id": criterion_id,
                "gold_ids": gold_ids,
                "ranked_ids": reranked_ids,
            })

        ks = [1, 5, 10, 20]
        retriever_dual = dual_evaluate(retriever_results, ks=ks)
        reranked_dual = dual_evaluate(reranked_results, ks=ks)

        results[name] = {
            "config": cfg_overrides,
            "retriever": {
                "ndcg@10": retriever_dual["positives_only"]["ndcg@10"],
                "recall@10": retriever_dual["positives_only"]["recall@10"],
                "mrr@10": retriever_dual["positives_only"]["mrr@10"],
            },
            "reranked": {
                "ndcg@10": reranked_dual["positives_only"]["ndcg@10"],
                "recall@10": reranked_dual["positives_only"]["recall@10"],
                "mrr@10": reranked_dual["positives_only"]["mrr@10"],
            },
        }

        logger.info(
            "  Retriever: nDCG@10=%.4f, Recall@10=%.4f",
            results[name]["retriever"]["ndcg@10"],
            results[name]["retriever"]["recall@10"],
        )
        logger.info(
            "  Reranked:  nDCG@10=%.4f, Recall@10=%.4f",
            results[name]["reranked"]["ndcg@10"],
            results[name]["reranked"]["recall@10"],
        )

    # Save results
    output_dir = Path("outputs/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n=== Ablation Results (TEST split, positives-only) ===")
    print(f"{'Config':<25} {'Retriever nDCG@10':>18} {'Reranked nDCG@10':>18} {'Delta':>10}")
    print("-" * 75)
    for name, res in results.items():
        ret_ndcg = res["retriever"]["ndcg@10"]
        rer_ndcg = res["reranked"]["ndcg@10"]
        delta = rer_ndcg - ret_ndcg
        print(f"{name:<25} {ret_ndcg:>18.4f} {rer_ndcg:>18.4f} {delta:>+10.4f}")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
