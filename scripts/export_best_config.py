#!/usr/bin/env python3
"""Export best HPO parameters to a pipeline config."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import yaml

from final_sc_review.hpo.storage import get_or_create_study
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo_config", type=str, default="configs/hpo_inference.yaml")
    parser.add_argument("--base_config", type=str, default="configs/default.yaml")
    parser.add_argument("--study_name", type=str, default="sc_inference")
    parser.add_argument("--storage", type=str, default="sqlite:///outputs/hpo/optuna.db")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    hpo_cfg = _load_yaml(Path(args.hpo_config))
    base_cfg = _load_yaml(Path(args.base_config))

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    if study.best_trial is None:
        raise ValueError("No best trial found")
    params = study.best_trial.params

    _apply_hpo_params(base_cfg, hpo_cfg, params)

    output_path = Path(args.output) if args.output else Path(hpo_cfg["paths"]["hpo_output_dir"]) / args.study_name / "best_config.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)

    logger.info("Wrote best config to %s", output_path)


def _apply_hpo_params(base_cfg: dict, hpo_cfg: dict, params: dict) -> None:
    # Models
    base_cfg["models"]["bge_m3"] = hpo_cfg["models"]["bge_m3"]
    base_cfg["models"]["bge_query_max_length"] = hpo_cfg["models"]["bge_query_max_length"]
    base_cfg["models"]["bge_passage_max_length"] = hpo_cfg["models"]["bge_passage_max_length"]
    base_cfg["models"]["bge_use_fp16"] = hpo_cfg["models"]["bge_use_fp16"]
    base_cfg["models"]["bge_batch_size"] = hpo_cfg["models"]["bge_batch_size"]
    base_cfg["models"]["jina_v3"] = hpo_cfg["reranker"]["model_name"]
    base_cfg["models"]["reranker_chunk_size"] = hpo_cfg["reranker"]["chunk_size"]
    base_cfg["models"]["reranker_dtype"] = hpo_cfg["reranker"].get("dtype", "auto")
    base_cfg["models"]["reranker_use_listwise"] = True

    # Retriever
    retriever_cfg = base_cfg["retriever"]
    retriever_cfg["top_k_retriever"] = int(params["top_k_retriever"])

    # V2: Support decoupled rerank pool size
    top_k_rerank = int(params.get("top_k_rerank", params["top_k_retriever"]))
    retriever_cfg["top_k_rerank"] = top_k_rerank
    retriever_cfg["top_k_colbert"] = top_k_rerank  # Backward compat

    retriever_cfg["top_k_final"] = int(params["top_k_final"])
    retriever_cfg["use_sparse"] = bool(params["use_sparse"])
    retriever_cfg["use_colbert"] = bool(params["use_multiv"])
    retriever_cfg["fusion_method"] = params["fusion_method"]
    retriever_cfg["score_normalization"] = params["score_normalization"]
    retriever_cfg["rrf_k"] = int(hpo_cfg["search_space"].get("rrf_k", 60))
    retriever_cfg["dense_weight"] = float(params.get("w_dense", retriever_cfg.get("dense_weight", 0.6)))
    retriever_cfg["sparse_weight"] = float(params.get("w_sparse", retriever_cfg.get("sparse_weight", 0.2)))
    retriever_cfg["colbert_weight"] = float(params.get("w_multiv", retriever_cfg.get("colbert_weight", 0.2)))


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
