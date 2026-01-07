"""Dataset utilities for reranker training."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence

from final_sc_review.data.schemas import Criterion, GroundTruthRow


def build_grouped_examples(
    groundtruth: Sequence[GroundTruthRow],
    criteria: Sequence[Criterion],
    post_ids: Sequence[str],
    max_candidates: int = 32,
    seed: int = 42,
) -> List[Dict]:
    """Build grouped examples by (post_id, criterion_id).

    Each group contains all positives and sampled negatives from the same post.
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
        if not positives:
            continue
        candidates = positives[:]
        if max_candidates is not None:
            remaining = max(0, max_candidates - len(positives))
        else:
            remaining = len(negatives)
        if remaining > 0 and negatives:
            sampled_negs = rng.sample(negatives, min(remaining, len(negatives)))
            candidates.extend(sampled_negs)
        rng.shuffle(candidates)
        examples.append(
            {
                "post_id": post_id,
                "criterion_id": criterion_id,
                "query": query,
                "sentences": [c.sentence_text for c in candidates],
                "labels": [float(c.groundtruth) for c in candidates],
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
