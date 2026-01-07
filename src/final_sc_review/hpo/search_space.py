"""Search space definitions for HPO."""

from __future__ import annotations

from typing import Dict, List

import optuna


def sample_inference_params(trial: optuna.Trial, cfg: Dict) -> Dict:
    """Sample inference-stage parameters for Optuna.

    Supports both v1 (top_k_retriever only) and v2 (top_k_retriever + top_k_rerank) configs.
    """
    space = cfg["search_space"]
    params: Dict[str, object] = {}

    params["top_k_retriever"] = trial.suggest_categorical(
        "top_k_retriever", space["top_k_retriever"]
    )

    # V2: Support decoupled rerank pool size
    if "top_k_rerank" in space:
        # Filter valid rerank values (<= retriever)
        valid_rerank = [k for k in space["top_k_rerank"] if k <= params["top_k_retriever"]]
        if not valid_rerank:
            valid_rerank = [params["top_k_retriever"]]  # Fallback
        params["top_k_rerank"] = trial.suggest_categorical(
            "top_k_rerank", valid_rerank
        )
    else:
        # V1 backward compat: rerank pool = retriever pool
        params["top_k_rerank"] = params["top_k_retriever"]

    params["top_k_final"] = trial.suggest_categorical(
        "top_k_final", space["top_k_final"]
    )

    params["use_sparse"] = trial.suggest_categorical("use_sparse", space["use_sparse"])
    params["use_multiv"] = trial.suggest_categorical("use_multiv", space["use_multiv"])
    params["fusion_method"] = trial.suggest_categorical(
        "fusion_method", space["fusion_method"]
    )
    params["score_normalization"] = trial.suggest_categorical(
        "score_normalization", space["score_normalization"]
    )

    if params["fusion_method"] == "weighted_sum":
        w_dense = trial.suggest_float("w_dense", 0.0, 1.0)
        w_sparse = trial.suggest_float("w_sparse", 0.0, 1.0)
        w_multiv = trial.suggest_float("w_multiv", 0.0, 1.0)
        weights = _normalize_weights(w_dense, w_sparse, w_multiv)
        params.update(weights)
    else:
        params["w_dense"] = 1.0
        params["w_sparse"] = 0.0
        params["w_multiv"] = 0.0

    params["rrf_k"] = space.get("rrf_k", 60)
    return params


def _normalize_weights(w_dense: float, w_sparse: float, w_multiv: float) -> Dict[str, float]:
    total = w_dense + w_sparse + w_multiv
    if total <= 0:
        return {"w_dense": 1.0, "w_sparse": 0.0, "w_multiv": 0.0}
    return {
        "w_dense": w_dense / total,
        "w_sparse": w_sparse / total,
        "w_multiv": w_multiv / total,
    }
