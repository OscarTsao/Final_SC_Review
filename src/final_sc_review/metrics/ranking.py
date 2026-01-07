"""Ranking metrics for retrieval evaluation."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def recall_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """Recall@K: fraction of gold items appearing in top-K."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = set(ranked_ids[:k]) & gold
    return len(hits) / len(gold)


def mrr_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """MRR@K: reciprocal rank of first relevant item in top-K."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            return 1.0 / idx
    return 0.0


def map_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """MAP@K: mean average precision with binary relevance."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(len(gold), k)


def ndcg_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """nDCG@K with binary relevance."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if sent_id in gold else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
