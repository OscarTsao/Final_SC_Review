"""Inference-only HPO objective using cached scores."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import optuna

from final_sc_review.hpo.constraints import validate_inference_params
from final_sc_review.hpo.reporting import TrialLogWriter
from final_sc_review.metrics.retrieval_eval import evaluate_rankings
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceCache:
    query_id: List[str]
    post_id: List[str]
    criterion_id: List[str]
    criterion_text: List[str]
    candidate_ids: List[List[str]]
    dense_scores: List[List[float]]
    sparse_scores: List[Optional[List[float]]]
    multiv_scores: List[Optional[List[float]]]
    jina_scores: List[List[float]]
    gold_ids: List[List[str]]


@dataclass
class ObjectiveConfig:
    ks: List[int]
    skip_no_positives: bool
    objective_metric: str
    prune_chunk_frac: float
    prune_min_queries: int


class InferenceObjective:
    def __init__(self, cache: InferenceCache, cfg: ObjectiveConfig, trial_log: TrialLogWriter, seed: int):
        self.cache = cache
        self.cfg = cfg
        self.trial_log = trial_log
        self.seed = seed

    def __call__(self, trial: optuna.Trial, params: Dict) -> float:
        start = time.time()
        try:
            metric = self._evaluate_trial(trial, params)
            self.trial_log.append(trial.number, metric, "complete", time.time() - start, params)
            return metric
        except optuna.TrialPruned:
            self.trial_log.append(trial.number, float("nan"), "pruned", time.time() - start, params)
            raise
        except Exception as exc:
            self.trial_log.append(trial.number, float("nan"), "fail", time.time() - start, params)
            raise exc

    def _evaluate_trial(self, trial: optuna.Trial, params: Dict) -> float:
        max_candidates = max(len(c) for c in self.cache.candidate_ids)
        try:
            validate_inference_params(params, max_candidates)
        except ValueError as exc:
            raise optuna.TrialPruned(str(exc)) from exc

        order = _shuffled_indices(len(self.cache.query_id), self.seed + trial.number)
        chunk_size = max(1, int(len(order) * self.cfg.prune_chunk_frac))
        chunk_size = min(chunk_size, len(order))

        retriever_results = []
        reranked_results = []

        for chunk_end in range(chunk_size, len(order) + 1, chunk_size):
            subset = order[:chunk_end]
            retriever_results.clear()
            reranked_results.clear()
            for idx in subset:
                retriever_ranked, reranked = _rank_query(self.cache, idx, params)
                retriever_results.append(
                    {
                        "post_id": self.cache.post_id[idx],
                        "criterion_id": self.cache.criterion_id[idx],
                        "gold_ids": self.cache.gold_ids[idx],
                        "ranked_ids": retriever_ranked,
                    }
                )
                reranked_results.append(
                    {
                        "post_id": self.cache.post_id[idx],
                        "criterion_id": self.cache.criterion_id[idx],
                        "gold_ids": self.cache.gold_ids[idx],
                        "ranked_ids": reranked,
                    }
                )

            retriever_metrics = evaluate_rankings(
                retriever_results,
                ks=self.cfg.ks,
                skip_no_positives=self.cfg.skip_no_positives,
            )
            reranked_metrics = evaluate_rankings(
                reranked_results,
                ks=self.cfg.ks,
                skip_no_positives=self.cfg.skip_no_positives,
            )
            value = _select_metric_dual(retriever_metrics, reranked_metrics, self.cfg.objective_metric)
            trial.report(value, step=chunk_end)
            if chunk_end >= self.cfg.prune_min_queries and trial.should_prune():
                raise optuna.TrialPruned()

        return _select_metric_dual(retriever_metrics, reranked_metrics, self.cfg.objective_metric)


def load_cache(cache_path: Path) -> InferenceCache:
    data = np.load(cache_path, allow_pickle=True)
    cache = InferenceCache(
        query_id=list(data["query_id"]),
        post_id=list(data["post_id"]),
        criterion_id=list(data["criterion_id"]),
        criterion_text=list(data["criterion_text"]),
        candidate_ids=list(data["candidate_ids"]),
        dense_scores=list(data["dense_scores"]),
        sparse_scores=list(data["sparse_scores"]),
        multiv_scores=list(data["multiv_scores"]),
        jina_scores=list(data["jina_scores"]),
        gold_ids=list(data["gold_ids"]),
    )
    _validate_cache(cache)
    return cache


def _validate_cache(cache: InferenceCache) -> None:
    for idx, gold in enumerate(cache.gold_ids):
        if gold is None:
            raise ValueError("Cache missing gold_ids; groundtruth is required")
        cand = cache.candidate_ids[idx]
        if len(cand) != len(cache.dense_scores[idx]) or len(cand) != len(cache.jina_scores[idx]):
            raise ValueError("Cache candidate_ids and score lists are misaligned")
        if cache.sparse_scores[idx] is not None and len(cand) != len(cache.sparse_scores[idx]):
            raise ValueError("Cache sparse scores misaligned")
        if cache.multiv_scores[idx] is not None and len(cand) != len(cache.multiv_scores[idx]):
            raise ValueError("Cache multiv scores misaligned")


def _rank_query(cache: InferenceCache, idx: int, params: Dict) -> (List[str], List[str]):
    candidate_ids = cache.candidate_ids[idx]
    dense_scores = np.array(cache.dense_scores[idx], dtype=float)
    sparse_scores = cache.sparse_scores[idx]
    multiv_scores = cache.multiv_scores[idx]
    jina_scores = np.array(cache.jina_scores[idx], dtype=float)

    sparse_arr = np.array(sparse_scores, dtype=float) if sparse_scores is not None else None
    multiv_arr = np.array(multiv_scores, dtype=float) if multiv_scores is not None else None

    components = [dense_scores]
    if params["use_sparse"] and sparse_arr is not None:
        components.append(sparse_arr)
    if params["use_multiv"] and multiv_arr is not None:
        components.append(multiv_arr)

    if params["fusion_method"] == "rrf":
        fused = _rrf_fusion(candidate_ids, components, params["rrf_k"])
    else:
        fused = _weighted_sum_fusion(dense_scores, sparse_arr, multiv_arr, params)

    ranked_indices = _sorted_indices(candidate_ids, fused)
    top_k_retriever = int(params["top_k_retriever"])
    retriever_ranked = [candidate_ids[i] for i in ranked_indices[:top_k_retriever]]

    # V2: Use decoupled rerank pool size (defaults to retriever pool for backward compat)
    top_k_rerank = int(params.get("top_k_rerank", top_k_retriever))
    top_k_final = int(params["top_k_final"])

    # Rerank with cached Jina scores (only on top_k_rerank candidates)
    rerank_candidates = ranked_indices[:top_k_rerank]
    rerank_scored = []
    for i in rerank_candidates:
        rerank_scored.append((candidate_ids[i], float(jina_scores[i])))
    rerank_scored.sort(key=lambda x: (-x[1], x[0]))
    reranked = [sid for sid, _ in rerank_scored[:top_k_final]]
    return retriever_ranked, reranked


def _weighted_sum_fusion(
    dense: np.ndarray,
    sparse: Optional[np.ndarray],
    multiv: Optional[np.ndarray],
    params: Dict,
) -> np.ndarray:
    w_dense = params["w_dense"]
    w_sparse = params["w_sparse"] if params["use_sparse"] and sparse is not None else 0.0
    w_multiv = params["w_multiv"] if params["use_multiv"] and multiv is not None else 0.0
    total = w_dense + w_sparse + w_multiv
    if total <= 0:
        w_dense, w_sparse, w_multiv = 1.0, 0.0, 0.0
    else:
        w_dense, w_sparse, w_multiv = w_dense / total, w_sparse / total, w_multiv / total

    scores = _normalize(dense, params["score_normalization"]) * w_dense
    if w_sparse > 0.0:
        scores = scores + _normalize(sparse, params["score_normalization"]) * w_sparse
    if w_multiv > 0.0:
        scores = scores + _normalize(multiv, params["score_normalization"]) * w_multiv
    return scores


def _rrf_fusion(candidate_ids: List[str], components: List[np.ndarray], rrf_k: int) -> np.ndarray:
    scores = np.zeros(len(candidate_ids), dtype=float)
    for comp in components:
        ranks = _rank_indices(candidate_ids, comp)
        for idx, rank in enumerate(ranks):
            scores[idx] += 1.0 / (rrf_k + rank)
    return scores


def _rank_indices(candidate_ids: List[str], scores: np.ndarray) -> List[int]:
    indexed = list(range(len(candidate_ids)))
    indexed.sort(key=lambda i: (-scores[i], candidate_ids[i]))
    ranks = [0] * len(candidate_ids)
    for rank, idx in enumerate(indexed, start=1):
        ranks[idx] = rank
    return ranks


def _sorted_indices(candidate_ids: List[str], scores: np.ndarray) -> List[int]:
    indexed = list(range(len(candidate_ids)))
    indexed.sort(key=lambda i: (-scores[i], candidate_ids[i]))
    return indexed


def _normalize(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores
    if mode == "minmax_per_query":
        min_v = float(scores.min())
        max_v = float(scores.max())
        if max_v > min_v:
            return (scores - min_v) / (max_v - min_v)
        return np.zeros_like(scores)
    if mode == "zscore_per_query":
        mean = float(scores.mean())
        std = float(scores.std())
        if std > 0:
            return (scores - mean) / std
        return np.zeros_like(scores)
    raise ValueError(f"Unknown score_normalization: {mode}")


def _select_metric(metrics: Dict[str, float], key: str) -> float:
    cleaned = key.replace("_reranked", "")
    if cleaned not in metrics:
        raise KeyError(f"Objective metric not found: {cleaned}")
    return float(metrics[cleaned])


def _select_metric_dual(retriever_metrics: Dict[str, float], reranked_metrics: Dict[str, float], key: str) -> float:
    """Select metric from retriever or reranked results based on suffix."""
    if "_retriever" in key:
        cleaned = key.replace("_retriever", "")
        if cleaned not in retriever_metrics:
            raise KeyError(f"Objective metric not found in retriever metrics: {cleaned}")
        return float(retriever_metrics[cleaned])
    else:
        cleaned = key.replace("_reranked", "")
        if cleaned not in reranked_metrics:
            raise KeyError(f"Objective metric not found in reranked metrics: {cleaned}")
        return float(reranked_metrics[cleaned])


def _shuffled_indices(n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return indices.tolist()
