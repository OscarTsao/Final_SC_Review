"""Pipeline runner API."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import yaml

from final_sc_review.data.io import load_sentence_corpus
from final_sc_review.pipeline.three_stage import PipelineConfig, ThreeStagePipeline


def load_pipeline_from_config(config_path: Path, rebuild_cache: bool = False) -> ThreeStagePipeline:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    cache_dir = Path(cfg["paths"]["cache_dir"])
    pipe_cfg = PipelineConfig(
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
        top_k_retriever=cfg["retriever"].get("top_k_retriever", 50),
        top_k_colbert=cfg["retriever"].get("top_k_colbert", 50),
        top_k_final=cfg["retriever"].get("top_k_final", 20),
        reranker_max_length=cfg["models"].get("reranker_max_length", cfg["models"].get("max_length", 512)),
        reranker_chunk_size=cfg["models"].get("reranker_chunk_size", 64),
        reranker_dtype=cfg["models"].get("reranker_dtype", "auto"),
        reranker_use_listwise=cfg["models"].get("reranker_use_listwise", True),
        device=cfg.get("device"),
    )
    return ThreeStagePipeline(sentences=sentences, cache_dir=cache_dir, config=pipe_cfg, rebuild_cache=rebuild_cache)


def run_single(
    config_path: Path,
    post_id: str,
    criterion_text: str,
    rebuild_cache: bool = False,
) -> List[Tuple[str, str, float]]:
    pipeline = load_pipeline_from_config(config_path, rebuild_cache=rebuild_cache)
    return pipeline.retrieve(query=criterion_text, post_id=post_id)
