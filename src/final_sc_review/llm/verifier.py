"""LLM-based evidence verifier.

This module implements LLM-as-judge for binary verification:
Does this sentence support the diagnostic criterion?
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from final_sc_review.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class LLMVerifier:
    """LLM-based evidence verifier (LLM-as-judge)."""

    def __init__(
        self,
        client: GeminiClient,
        verification_mode: str = "positives_only",
        confidence_threshold: float = 0.5,
    ):
        """Initialize LLM verifier.

        Args:
            client: Gemini API client
            verification_mode: 'all' (verify all), 'positives_only' (only predicted positives),
                             or 'uncertain' (only uncertain predictions)
            confidence_threshold: Minimum confidence for 'supports' label
        """
        self.client = client
        self.verification_mode = verification_mode
        self.confidence_threshold = confidence_threshold

        logger.info(
            f"Initialized LLMVerifier with mode={verification_mode}, "
            f"confidence_threshold={confidence_threshold}"
        )

    def verify_batch(
        self,
        query_text: str,
        criterion_text: str,
        candidates: List[str],
        candidate_ids: List[str],
        ne_prob: float = 1.0,
    ) -> Tuple[List[bool], List[float]]:
        """Verify a batch of candidates for a single query.

        Args:
            query_text: Full query text (post content)
            criterion_text: Criterion description
            candidates: List of candidate sentence texts
            candidate_ids: List of candidate UIDs
            ne_prob: NE gate probability (for filtering in 'uncertain' mode)

        Returns:
            (supports_labels, confidence_scores)
            - supports_labels: Boolean list indicating if each candidate supports criterion
            - confidence_scores: Confidence scores [0, 1]
        """
        # Decide which candidates to verify
        if self.verification_mode == "positives_only":
            # Only verify if NE gate predicts positive
            if ne_prob < 0.5:
                # Skip verification, return all False
                return [False] * len(candidates), [0.0] * len(candidates)
        elif self.verification_mode == "uncertain":
            # Only verify if NE gate is uncertain (e.g., 0.3 < prob < 0.7)
            if ne_prob < 0.3 or ne_prob > 0.7:
                # High confidence prediction, skip verification
                if ne_prob >= 0.7:
                    # Assume all support
                    return [True] * len(candidates), [1.0] * len(candidates)
                else:
                    # Assume none support
                    return [False] * len(candidates), [0.0] * len(candidates)

        # Batch verification (send multiple candidates in one call to reduce cost)
        prompt = self._build_verification_prompt(
            query_text=query_text,
            criterion_text=criterion_text,
            candidates=candidates,
        )

        try:
            result = self.client.generate_json(
                prompt=prompt,
                schema={
                    "type": "object",
                    "required": ["verifications"],
                    "properties": {
                        "verifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["candidate_id", "supports", "confidence"],
                                "properties": {
                                    "candidate_id": {"type": "integer"},
                                    "supports": {"type": "boolean"},
                                    "confidence": {"type": "number"},
                                },
                            },
                        }
                    },
                },
            )

            # Parse verifications
            verifications = result["verifications"]

            # Create output arrays
            supports_labels = []
            confidence_scores = []

            for i in range(len(candidates)):
                # Find verification for this candidate
                verification = next(
                    (v for v in verifications if v["candidate_id"] == i), None
                )

                if verification:
                    supports = verification["supports"]
                    confidence = verification["confidence"]

                    # Apply confidence threshold
                    if supports and confidence >= self.confidence_threshold:
                        supports_labels.append(True)
                        confidence_scores.append(confidence)
                    else:
                        supports_labels.append(False)
                        confidence_scores.append(1.0 - confidence if not supports else confidence)
                else:
                    # Missing verification, default to False
                    logger.warning(f"Missing verification for candidate {i}")
                    supports_labels.append(False)
                    confidence_scores.append(0.0)

            return supports_labels, confidence_scores

        except Exception as e:
            logger.error(f"LLM verification failed: {e}. Returning all False.")
            # Fallback: assume no support
            return [False] * len(candidates), [0.0] * len(candidates)

    def _build_verification_prompt(
        self,
        query_text: str,
        criterion_text: str,
        candidates: List[str],
    ) -> str:
        """Build prompt for batch verification."""

        # Format candidates
        candidates_str = "\n".join(
            [f"[{i}] {text}" for i, text in enumerate(candidates)]
        )

        prompt = f"""You are an expert clinical psychologist evaluating evidence for mental health diagnoses.

**Task**: For each candidate sentence, determine if it SUPPORTS the given diagnostic criterion.

**Diagnostic Criterion**:
{criterion_text}

**Patient Post** (context):
{query_text[:800]}...

**Candidate Evidence Sentences**:
{candidates_str}

**Instructions**:
1. For each candidate, answer: Does this sentence provide evidence that supports the criterion?
2. A sentence SUPPORTS the criterion if it:
   - Describes a symptom, behavior, or experience mentioned in the criterion
   - Provides concrete, specific evidence (not vague or tangential)
   - Directly relates to the criterion (not just general distress)

3. Assign:
   - supports: true/false
   - confidence: 0.0 (very uncertain) to 1.0 (very certain)

**Examples**:
- Criterion: "Depressed mood most of the day"
  - Sentence: "I feel sad all day every day" → supports=true, confidence=0.95
  - Sentence: "I had a bad day yesterday" → supports=false, confidence=0.8
  - Sentence: "I like ice cream" → supports=false, confidence=1.0

**Output Format** (JSON):
{{
  "verifications": [
    {{"candidate_id": 0, "supports": true, "confidence": 0.9}},
    {{"candidate_id": 1, "supports": false, "confidence": 0.85}},
    ...
  ]
}}

Return ONLY the JSON object, no additional text."""

        return prompt


def verify_and_filter(
    verifier: LLMVerifier,
    query_text: str,
    criterion_text: str,
    candidates: List[str],
    candidate_ids: List[str],
    ne_prob: float,
    min_supported: int = 1,
) -> Tuple[bool, List[int]]:
    """Verify candidates and return final prediction.

    Args:
        verifier: LLM verifier instance
        query_text: Query text
        criterion_text: Criterion text
        candidates: Candidate sentences
        candidate_ids: Candidate UIDs
        ne_prob: NE gate probability
        min_supported: Minimum number of supported candidates to predict positive

    Returns:
        (has_evidence, supported_indices)
        - has_evidence: Final binary prediction
        - supported_indices: Indices of candidates that LLM verified as supporting
    """
    supports_labels, confidence_scores = verifier.verify_batch(
        query_text=query_text,
        criterion_text=criterion_text,
        candidates=candidates,
        candidate_ids=candidate_ids,
        ne_prob=ne_prob,
    )

    # Get indices of supported candidates
    supported_indices = [i for i, supports in enumerate(supports_labels) if supports]

    # Final decision
    has_evidence = len(supported_indices) >= min_supported

    return has_evidence, supported_indices


if __name__ == "__main__":
    # Test verifier
    logging.basicConfig(level=logging.INFO)

    client = GeminiClient()
    verifier = LLMVerifier(client, verification_mode="all")

    # Test case
    query = "I've been feeling really down lately. Can't sleep, lost appetite. I love pizza though."
    criterion = "Depressed mood most of the day, nearly every day"
    candidates = [
        "I've been feeling really down lately.",
        "Can't sleep at night.",
        "I love pizza.",
        "Lost my appetite completely.",
    ]
    candidate_ids = ["s1", "s2", "s3", "s4"]

    supports, confidence = verifier.verify_batch(query, criterion, candidates, candidate_ids)

    print("Verification results:")
    for i, (cand, sup, conf) in enumerate(zip(candidates, supports, confidence)):
        print(f"  [{i}] {cand}")
        print(f"      → supports={sup}, confidence={conf:.3f}")
