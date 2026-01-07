"""Hard negative mining for reranker training.

Rationale: A reranker trained on random negatives underperforms on "near-miss"
candidates - sentences that are retrieved by the first-stage retriever but are
not actually relevant. Mining hard negatives improves reranker discrimination.

Methods:
- top_k_hard: Use top-k retriever results (excluding gold) as hard negatives
- semi_hard: Use candidates ranked between k1 and k2 (not too easy, not too hard)
- in_batch: Sample hard negatives from other queries in the same batch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from final_sc_review.data.io import Sentence
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HardNegativeConfig:
    """Configuration for hard negative mining."""

    enabled: bool = True
    method: str = "top_k_hard"  # top_k_hard, semi_hard, in_batch
    k_neg: int = 3  # Number of hard negatives per query
    # For semi_hard method
    rank_low: int = 5  # Start rank for semi-hard (inclusive)
    rank_high: int = 50  # End rank for semi-hard (exclusive)
    # For retrieval
    retriever_top_k: int = 100  # How many candidates to retrieve for mining


class HardNegativeMiner:
    """Mine hard negatives from retriever results."""

    def __init__(
        self,
        retriever: BgeM3Retriever,
        config: Optional[HardNegativeConfig] = None,
    ):
        """Initialize hard negative miner.

        Args:
            retriever: BGE-M3 retriever for candidate retrieval.
            config: Mining configuration.
        """
        self.retriever = retriever
        self.config = config or HardNegativeConfig()

    def mine_negatives(
        self,
        query: str,
        post_id: str,
        gold_ids: Set[str],
        k_neg: Optional[int] = None,
    ) -> List[str]:
        """Mine hard negatives for a query.

        Args:
            query: Query text (criterion text).
            post_id: Post ID to constrain search.
            gold_ids: Set of gold sentence UIDs to exclude.
            k_neg: Number of negatives to mine (overrides config).

        Returns:
            List of hard negative sentence UIDs.
        """
        k_neg = k_neg or self.config.k_neg
        method = self.config.method

        if method == "top_k_hard":
            return self._mine_top_k_hard(query, post_id, gold_ids, k_neg)
        elif method == "semi_hard":
            return self._mine_semi_hard(query, post_id, gold_ids, k_neg)
        else:
            raise ValueError(f"Unknown mining method: {method}")

    def _mine_top_k_hard(
        self,
        query: str,
        post_id: str,
        gold_ids: Set[str],
        k_neg: int,
    ) -> List[str]:
        """Mine top-k hard negatives (highest scoring non-gold candidates).

        These are the "near-miss" candidates that the retriever thinks are
        relevant but are actually not.
        """
        results = self.retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=self.config.retriever_top_k,
        )

        hard_negatives = []
        for sent_uid, _, _ in results:
            if sent_uid not in gold_ids:
                hard_negatives.append(sent_uid)
                if len(hard_negatives) >= k_neg:
                    break

        return hard_negatives

    def _mine_semi_hard(
        self,
        query: str,
        post_id: str,
        gold_ids: Set[str],
        k_neg: int,
    ) -> List[str]:
        """Mine semi-hard negatives (ranked between rank_low and rank_high).

        These are candidates that are not at the very top (too hard to learn from)
        but also not too easy (random negatives). Good for curriculum learning.
        """
        results = self.retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=self.config.retriever_top_k,
        )

        # Filter out gold and take candidates in the semi-hard range
        candidates = []
        rank = 0
        for sent_uid, _, _ in results:
            if sent_uid not in gold_ids:
                rank += 1
                if self.config.rank_low <= rank < self.config.rank_high:
                    candidates.append(sent_uid)

        # Sample k_neg from the semi-hard candidates
        if len(candidates) <= k_neg:
            return candidates
        rng = np.random.default_rng()
        indices = rng.choice(len(candidates), size=k_neg, replace=False)
        return [candidates[i] for i in indices]

    def mine_batch_negatives(
        self,
        batch: List[Dict],
        k_neg: Optional[int] = None,
    ) -> List[Dict]:
        """Mine hard negatives for a batch of queries.

        Args:
            batch: List of dicts with keys: query, post_id, gold_ids.
            k_neg: Number of negatives per query.

        Returns:
            List of dicts with additional key: hard_negative_ids.
        """
        results = []
        for item in batch:
            query = item["query"]
            post_id = item["post_id"]
            gold_ids = set(item["gold_ids"])

            hard_negatives = self.mine_negatives(query, post_id, gold_ids, k_neg)

            results.append(
                {
                    **item,
                    "hard_negative_ids": hard_negatives,
                }
            )

        return results


class InBatchNegativeMiner:
    """Mine in-batch negatives by sampling from other queries in the batch."""

    def __init__(self, sentences: List[Sentence], k_neg: int = 3):
        """Initialize in-batch negative miner.

        Args:
            sentences: List of all sentences.
            k_neg: Number of negatives per query.
        """
        self.sentences = sentences
        self.sent_uid_to_sentence = {s.sent_uid: s for s in sentences}
        self.k_neg = k_neg

    def mine_in_batch(self, batch: List[Dict], seed: int = 42) -> List[Dict]:
        """Mine in-batch negatives.

        For each query, sample negatives from sentences in other queries' posts.
        This creates "easy" negatives that are clearly from different contexts.

        Args:
            batch: List of dicts with keys: query, post_id, gold_ids.
            seed: Random seed for reproducibility.

        Returns:
            List of dicts with additional key: in_batch_negative_ids.
        """
        rng = np.random.default_rng(seed)
        results = []

        # Collect all sentences from each post
        post_sentences: Dict[str, List[str]] = {}
        for item in batch:
            post_id = item["post_id"]
            if post_id not in post_sentences:
                post_sentences[post_id] = [
                    s.sent_uid for s in self.sentences if s.post_id == post_id
                ]

        for item in batch:
            current_post = item["post_id"]
            gold_ids = set(item["gold_ids"])

            # Collect candidates from other posts
            other_candidates = []
            for post_id, sent_uids in post_sentences.items():
                if post_id != current_post:
                    other_candidates.extend(sent_uids)

            # Sample negatives
            if len(other_candidates) <= self.k_neg:
                negatives = other_candidates
            else:
                indices = rng.choice(len(other_candidates), size=self.k_neg, replace=False)
                negatives = [other_candidates[i] for i in indices]

            results.append(
                {
                    **item,
                    "in_batch_negative_ids": negatives,
                }
            )

        return results


def create_training_examples(
    batch_with_negatives: List[Dict],
    sentences: List[Sentence],
    include_in_batch: bool = False,
) -> List[Dict]:
    """Create training examples from mined negatives.

    Args:
        batch_with_negatives: Output from HardNegativeMiner.mine_batch_negatives().
        sentences: List of all sentences.
        include_in_batch: Whether to include in-batch negatives.

    Returns:
        List of training examples with query, positive, and negative texts.
    """
    sent_map = {s.sent_uid: s.text for s in sentences}
    examples = []

    for item in batch_with_negatives:
        query = item["query"]
        gold_ids = item["gold_ids"]
        hard_neg_ids = item.get("hard_negative_ids", [])
        in_batch_neg_ids = item.get("in_batch_negative_ids", []) if include_in_batch else []

        for gold_id in gold_ids:
            if gold_id not in sent_map:
                continue

            positive_text = sent_map[gold_id]

            # Create example with hard negatives
            negative_texts = [sent_map[nid] for nid in hard_neg_ids if nid in sent_map]
            if include_in_batch:
                negative_texts.extend([sent_map[nid] for nid in in_batch_neg_ids if nid in sent_map])

            if negative_texts:
                examples.append(
                    {
                        "query": query,
                        "positive": positive_text,
                        "negatives": negative_texts,
                        "post_id": item["post_id"],
                        "gold_id": gold_id,
                    }
                )

    return examples
