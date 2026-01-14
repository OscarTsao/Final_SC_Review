"""Dataset utilities for reranker training."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence

from final_sc_review.data.schemas import Criterion, GroundTruthRow

# Special token for NO_EVIDENCE pseudo-candidate
NO_EVIDENCE_TOKEN = "[NO_EVIDENCE]"


def build_grouped_examples(
    groundtruth: Sequence[GroundTruthRow],
    criteria: Sequence[Criterion],
    post_ids: Sequence[str],
    max_candidates: int = 32,
    seed: int = 42,
    add_no_evidence: bool = False,
    include_no_evidence_queries: bool = False,
) -> List[Dict]:
    """Build grouped examples by (post_id, criterion_id).

    Each group contains all positives and sampled negatives from the same post.

    Args:
        groundtruth: List of groundtruth rows.
        criteria: List of criteria with text.
        post_ids: Post IDs to include.
        max_candidates: Maximum candidates per query.
        seed: Random seed for reproducibility.
        add_no_evidence: If True, add a NO_EVIDENCE pseudo-candidate to each query.
            For has-evidence queries: NO_EVIDENCE has label=0 (negative).
            For no-evidence queries: NO_EVIDENCE has label=1 (the target).
        include_no_evidence_queries: If True, include queries with no positive
            evidence. Only meaningful when add_no_evidence=True, as these queries
            need the NO_EVIDENCE candidate as their positive target.

    Returns:
        List of grouped examples with query, sentences, and labels.
    """
    criteria_map = {c.criterion_id: c.text for c in criteria}
    allowed_posts = set(post_ids)
    groups: Dict[tuple, List[GroundTruthRow]] = {}
    for row in groundtruth:
        if row.post_id not in allowed_posts:
            continue
        key = (row.post_id, row.criterion_id)
        groups.setdefault(key, []).append(row)

    rng = random.Random(seed)
    examples: List[Dict] = []
    for (post_id, criterion_id), rows in sorted(groups.items()):
        query = criteria_map.get(criterion_id)
        if query is None:
            continue
        positives = [r for r in rows if r.groundtruth == 1]
        negatives = [r for r in rows if r.groundtruth == 0]

        # Determine if this is a no-evidence query
        is_no_evidence_query = len(positives) == 0

        # Skip no-evidence queries unless explicitly included
        if is_no_evidence_query and not include_no_evidence_queries:
            continue

        # For no-evidence queries, we need add_no_evidence to be meaningful
        if is_no_evidence_query and not add_no_evidence:
            continue

        # Build candidate list
        sentences: List[str] = []
        labels: List[float] = []

        if is_no_evidence_query:
            # No-evidence query: sample negatives, NO_EVIDENCE is the positive
            if max_candidates is not None:
                # Reserve 1 slot for NO_EVIDENCE
                n_neg = min(max_candidates - 1, len(negatives))
            else:
                n_neg = len(negatives)
            if n_neg > 0 and negatives:
                sampled_negs = rng.sample(negatives, n_neg)
                sentences.extend([r.sentence_text for r in sampled_negs])
                labels.extend([0.0] * len(sampled_negs))
            # Add NO_EVIDENCE as the positive (target)
            sentences.append(NO_EVIDENCE_TOKEN)
            labels.append(1.0)
        else:
            # Has-evidence query: positives + sampled negatives + optional NO_EVIDENCE
            candidates = positives[:]
            if max_candidates is not None:
                # Reserve 1 slot for NO_EVIDENCE if enabled
                reserved = 1 if add_no_evidence else 0
                remaining = max(0, max_candidates - len(positives) - reserved)
            else:
                remaining = len(negatives)
            if remaining > 0 and negatives:
                sampled_negs = rng.sample(negatives, min(remaining, len(negatives)))
                candidates.extend(sampled_negs)

            sentences = [c.sentence_text for c in candidates]
            labels = [float(c.groundtruth) for c in candidates]

            # Add NO_EVIDENCE as a negative
            if add_no_evidence:
                sentences.append(NO_EVIDENCE_TOKEN)
                labels.append(0.0)

        # Shuffle candidates
        combined = list(zip(sentences, labels))
        rng.shuffle(combined)
        sentences, labels = zip(*combined) if combined else ([], [])

        examples.append(
            {
                "post_id": post_id,
                "criterion_id": criterion_id,
                "query": query,
                "sentences": list(sentences),
                "labels": list(labels),
                "is_no_evidence": is_no_evidence_query,
            }
        )
    return examples


class GroupedRerankerDataset:
    """Simple dataset wrapper for grouped examples."""

    def __init__(self, examples: Sequence[Dict]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]
