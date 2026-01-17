"""LLM-based reranker for evidence candidates.

This module implements listwise reranking using Gemini 1.5 Flash.
Operates on top-M candidates post-P3 graph reranking.
"""

import logging
import random
from typing import Dict, List, Tuple

import numpy as np

from final_sc_review.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class LLMReranker:
    """LLM-based listwise reranker for evidence candidates."""

    def __init__(
        self,
        client: GeminiClient,
        top_m: int = 10,
        use_position_randomization: bool = True,
    ):
        """Initialize LLM reranker.

        Args:
            client: Gemini API client
            top_m: Number of top candidates to rerank
            use_position_randomization: Randomize candidate order to reduce position bias
        """
        self.client = client
        self.top_m = top_m
        self.use_position_randomization = use_position_randomization

        logger.info(
            f"Initialized LLMReranker with top_m={top_m}, "
            f"position_randomization={use_position_randomization}"
        )

    def rerank(
        self,
        query_text: str,
        criterion_text: str,
        candidates: List[str],
        candidate_ids: List[str],
        current_scores: List[float],
    ) -> Tuple[List[int], List[float]]:
        """Rerank candidates using LLM.

        Args:
            query_text: Full query text (post content)
            criterion_text: Criterion description
            candidates: List of candidate sentence texts
            candidate_ids: List of candidate UIDs
            current_scores: Current scores from P3 (used for fallback)

        Returns:
            (reranked_indices, llm_scores)
            - reranked_indices: Permutation indices (0-indexed)
            - llm_scores: Normalized LLM scores [0, 1]
        """
        # Take top-M by current scores
        top_m = min(self.top_m, len(candidates))
        sorted_indices = np.argsort(current_scores)[::-1]
        top_indices = sorted_indices[:top_m]

        # Extract top-M candidates
        top_candidates = [candidates[i] for i in top_indices]
        top_ids = [candidate_ids[i] for i in top_indices]

        # Create position randomization mapping if enabled
        if self.use_position_randomization:
            shuffled_order = list(range(len(top_candidates)))
            random.shuffle(shuffled_order)
            # Track original -> shuffled position
            reverse_mapping = {shuffled: orig for orig, shuffled in enumerate(shuffled_order)}
        else:
            shuffled_order = list(range(len(top_candidates)))
            reverse_mapping = {i: i for i in range(len(top_candidates))}

        # Build prompt
        prompt = self._build_reranking_prompt(
            query_text=query_text,
            criterion_text=criterion_text,
            candidates=[top_candidates[i] for i in shuffled_order],
            candidate_labels=[f"[{i}]" for i in range(len(top_candidates))],
        )

        # Get LLM response
        try:
            result = self.client.generate_json(
                prompt=prompt,
                schema={
                    "type": "object",
                    "required": ["rankings"],
                    "properties": {
                        "rankings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["candidate_id", "relevance_score"],
                                "properties": {
                                    "candidate_id": {"type": "integer"},
                                    "relevance_score": {"type": "number"},
                                },
                            },
                        }
                    },
                },
            )

            # Parse rankings
            llm_rankings = result["rankings"]

            # Map back to original order
            llm_scores_dict = {}
            for item in llm_rankings:
                shuffled_idx = item["candidate_id"]
                original_idx = reverse_mapping[shuffled_idx]
                llm_scores_dict[original_idx] = item["relevance_score"]

            # Create score array
            llm_scores_top_m = np.array([llm_scores_dict.get(i, 0.0) for i in range(len(top_candidates))])

            # Normalize scores to [0, 1]
            if llm_scores_top_m.max() > llm_scores_top_m.min():
                llm_scores_top_m = (llm_scores_top_m - llm_scores_top_m.min()) / (
                    llm_scores_top_m.max() - llm_scores_top_m.min()
                )
            else:
                llm_scores_top_m = np.ones_like(llm_scores_top_m) * 0.5

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}. Falling back to current scores.")
            # Fallback: use current scores
            llm_scores_top_m = np.array([current_scores[i] for i in top_indices])
            llm_scores_top_m = (llm_scores_top_m - llm_scores_top_m.min()) / (
                llm_scores_top_m.max() - llm_scores_top_m.min() + 1e-8
            )

        # Sort top-M by LLM scores
        reranked_top_m_order = np.argsort(llm_scores_top_m)[::-1]

        # Map back to full candidate list indices
        reranked_top_m_indices = [top_indices[i] for i in reranked_top_m_order]

        # Keep bottom (N - M) candidates in original order
        bottom_indices = [i for i in sorted_indices[top_m:]]

        # Combine
        final_reranked_indices = reranked_top_m_indices + bottom_indices

        # Create full score array (top-M get LLM scores, rest get 0)
        full_llm_scores = np.zeros(len(candidates))
        for i, orig_idx in enumerate(top_indices):
            full_llm_scores[orig_idx] = llm_scores_top_m[i]

        return final_reranked_indices, full_llm_scores.tolist()

    def _build_reranking_prompt(
        self,
        query_text: str,
        criterion_text: str,
        candidates: List[str],
        candidate_labels: List[str],
    ) -> str:
        """Build prompt for listwise reranking."""

        # Format candidates
        candidates_str = "\n".join(
            [f"{label} {text}" for label, text in zip(candidate_labels, candidates)]
        )

        prompt = f"""You are an expert clinical psychologist evaluating evidence for mental health diagnoses.

**Task**: Rank the following candidate sentences by their relevance as evidence for the given diagnostic criterion.

**Diagnostic Criterion**:
{criterion_text}

**Patient Post** (context):
{query_text[:500]}...

**Candidate Evidence Sentences**:
{candidates_str}

**Instructions**:
1. For each candidate, assess how well it supports the diagnostic criterion
2. Assign a relevance score from 0 (not relevant) to 10 (highly relevant)
3. Consider:
   - Does the sentence describe symptoms/behaviors mentioned in the criterion?
   - Is the evidence specific and concrete?
   - Does it clearly support the criterion (not just tangentially related)?

**Output Format** (JSON):
{{
  "rankings": [
    {{"candidate_id": 0, "relevance_score": 8.5}},
    {{"candidate_id": 1, "relevance_score": 6.0}},
    ...
  ]
}}

Return ONLY the JSON object, no additional text."""

        return prompt


if __name__ == "__main__":
    # Test reranker
    logging.basicConfig(level=logging.INFO)

    client = GeminiClient()
    reranker = LLMReranker(client)

    # Test case
    query = "I've been feeling really down lately. Can't sleep, lost appetite."
    criterion = "Depressed mood most of the day, nearly every day"
    candidates = [
        "I've been feeling really down lately.",
        "Can't sleep at night.",
        "I love pizza.",
        "Lost my appetite completely.",
    ]
    candidate_ids = ["s1", "s2", "s3", "s4"]
    current_scores = [0.9, 0.7, 0.1, 0.8]

    indices, scores = reranker.rerank(query, criterion, candidates, candidate_ids, current_scores)

    print("Reranked order:")
    for i in indices:
        print(f"  {i}: {candidates[i]} (score={scores[i]:.3f})")
