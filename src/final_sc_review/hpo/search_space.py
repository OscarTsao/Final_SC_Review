"""Search space definitions for HPO."""

from __future__ import annotations

from typing import Dict

import optuna


def sample_inference_params(trial: optuna.Trial, cfg: Dict) -> Dict:
    """Sample inference-stage parameters for Optuna."""
    space = cfg["search_space"]
    params: Dict[str, object] = {}

    params["top_k_retriever"] = trial.suggest_categorical(
        "top_k_retriever", space["top_k_retriever"]
    )
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
