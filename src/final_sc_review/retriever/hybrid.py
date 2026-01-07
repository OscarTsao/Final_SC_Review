"""Hybrid fusion of dense and sparse scores."""

from __future__ import annotations

from typing import Dict, List, Tuple


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return scores
    values = list(scores.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return scores
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def fuse_scores(
    dense_results: List[Tuple[str, str, float]],
    sparse_results: List[Tuple[str, str, float]],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """Fuse dense and sparse scores with min-max normalization."""
    dense_map = {sid: score for sid, _, score in dense_results}
    sparse_map = {sid: score for sid, _, score in sparse_results}
    texts = {sid: text for sid, text, _ in dense_results}
    for sid, text, _ in sparse_results:
        texts.setdefault(sid, text)

    norm_dense = _normalize(dense_map)
    norm_sparse = _normalize(sparse_map)

    all_ids = sorted(set(norm_dense) | set(norm_sparse))
    fused = []
    for sid in all_ids:
        d = norm_dense.get(sid, 0.0)
        s = norm_sparse.get(sid, 0.0)
        fused.append((sid, texts.get(sid, ""), dense_weight * d + sparse_weight * s))
    fused.sort(key=lambda x: (-x[2], x[0]))
    return fused
