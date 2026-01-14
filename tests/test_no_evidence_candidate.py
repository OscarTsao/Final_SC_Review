"""Tests for NO_EVIDENCE pseudo-candidate in reranker training."""

import pytest

from final_sc_review.data.schemas import Criterion, GroundTruthRow
from final_sc_review.reranker.dataset import (
    NO_EVIDENCE_TOKEN,
    build_grouped_examples,
)


def make_groundtruth_rows(post_id: str, criterion_id: str, n_pos: int, n_neg: int):
    """Helper to create test groundtruth rows."""
    rows = []
    for i in range(n_pos):
        rows.append(
            GroundTruthRow(
                post_id=post_id,
                criterion_id=criterion_id,
                sid=i,
                sent_uid=f"{post_id}_{i}",
                sentence_text=f"positive sentence {i}",
                groundtruth=1,
            )
        )
    for i in range(n_neg):
        rows.append(
            GroundTruthRow(
                post_id=post_id,
                criterion_id=criterion_id,
                sid=n_pos + i,
                sent_uid=f"{post_id}_{n_pos + i}",
                sentence_text=f"negative sentence {i}",
                groundtruth=0,
            )
        )
    return rows


class TestNoEvidenceCandidate:
    """Test NO_EVIDENCE pseudo-candidate functionality."""

    def test_no_evidence_disabled_by_default(self):
        """By default, NO_EVIDENCE should not be added."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=2, n_neg=5)
        criteria = [Criterion("c1", "test criterion")]

        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=10, seed=42
        )

        assert len(examples) == 1
        assert NO_EVIDENCE_TOKEN not in examples[0]["sentences"]

    def test_no_evidence_added_to_has_evidence_query(self):
        """NO_EVIDENCE should be added as negative for has-evidence queries."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=2, n_neg=5)
        criteria = [Criterion("c1", "test criterion")]

        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=10, seed=42,
            add_no_evidence=True,
        )

        assert len(examples) == 1
        ex = examples[0]

        # NO_EVIDENCE should be present
        assert NO_EVIDENCE_TOKEN in ex["sentences"]

        # Find NO_EVIDENCE index and check its label
        idx = ex["sentences"].index(NO_EVIDENCE_TOKEN)
        assert ex["labels"][idx] == 0.0, "NO_EVIDENCE should have label=0 for has-evidence query"

        # is_no_evidence flag should be False
        assert ex["is_no_evidence"] is False

    def test_no_evidence_query_skipped_by_default(self):
        """No-evidence queries should be skipped when add_no_evidence=False."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=0, n_neg=5)  # No positives
        criteria = [Criterion("c1", "test criterion")]

        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=10, seed=42,
            add_no_evidence=False,
        )

        assert len(examples) == 0, "No-evidence query should be skipped"

    def test_no_evidence_query_included_when_enabled(self):
        """No-evidence queries should be included with NO_EVIDENCE as positive."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=0, n_neg=5)  # No positives
        criteria = [Criterion("c1", "test criterion")]

        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=10, seed=42,
            add_no_evidence=True,
            include_no_evidence_queries=True,
        )

        assert len(examples) == 1
        ex = examples[0]

        # NO_EVIDENCE should be present
        assert NO_EVIDENCE_TOKEN in ex["sentences"]

        # Find NO_EVIDENCE index and check its label
        idx = ex["sentences"].index(NO_EVIDENCE_TOKEN)
        assert ex["labels"][idx] == 1.0, "NO_EVIDENCE should have label=1 for no-evidence query"

        # All other labels should be 0
        for i, label in enumerate(ex["labels"]):
            if i != idx:
                assert label == 0.0, f"Sentence at index {i} should have label=0"

        # is_no_evidence flag should be True
        assert ex["is_no_evidence"] is True

    def test_no_evidence_respects_max_candidates(self):
        """NO_EVIDENCE should be included within max_candidates limit."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=2, n_neg=10)
        criteria = [Criterion("c1", "test criterion")]

        max_candidates = 5
        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=max_candidates, seed=42,
            add_no_evidence=True,
        )

        assert len(examples) == 1
        ex = examples[0]

        # Should have exactly max_candidates
        assert len(ex["sentences"]) == max_candidates

        # NO_EVIDENCE should be present
        assert NO_EVIDENCE_TOKEN in ex["sentences"]

        # Should have 2 positives + 2 negatives + 1 NO_EVIDENCE = 5
        n_positives = sum(1 for l in ex["labels"] if l == 1.0)
        n_negatives = sum(1 for l in ex["labels"] if l == 0.0)
        assert n_positives == 2
        assert n_negatives == 3  # 2 sampled negatives + NO_EVIDENCE

    def test_mixed_has_and_no_evidence_queries(self):
        """Test dataset with both has-evidence and no-evidence queries."""
        groundtruth = (
            make_groundtruth_rows("p1", "c1", n_pos=2, n_neg=5) +  # Has evidence
            make_groundtruth_rows("p1", "c2", n_pos=0, n_neg=5) +  # No evidence
            make_groundtruth_rows("p2", "c1", n_pos=1, n_neg=3)    # Has evidence
        )
        criteria = [Criterion("c1", "criterion 1"), Criterion("c2", "criterion 2")]

        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1", "p2"], max_candidates=10, seed=42,
            add_no_evidence=True,
            include_no_evidence_queries=True,
        )

        # Should have 3 examples
        assert len(examples) == 3

        # Check each example
        has_evidence_count = 0
        no_evidence_count = 0
        for ex in examples:
            assert NO_EVIDENCE_TOKEN in ex["sentences"]
            idx = ex["sentences"].index(NO_EVIDENCE_TOKEN)

            if ex["is_no_evidence"]:
                assert ex["labels"][idx] == 1.0
                no_evidence_count += 1
            else:
                assert ex["labels"][idx] == 0.0
                has_evidence_count += 1

        assert has_evidence_count == 2
        assert no_evidence_count == 1

    def test_no_evidence_query_all_negatives_sampled(self):
        """For no-evidence queries, all negatives should be sampled up to limit."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=0, n_neg=3)
        criteria = [Criterion("c1", "test criterion")]

        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=10, seed=42,
            add_no_evidence=True,
            include_no_evidence_queries=True,
        )

        assert len(examples) == 1
        ex = examples[0]

        # Should have 3 negatives + 1 NO_EVIDENCE = 4 candidates
        assert len(ex["sentences"]) == 4
        assert NO_EVIDENCE_TOKEN in ex["sentences"]

    def test_backward_compatibility(self):
        """Existing code without new parameters should work unchanged."""
        groundtruth = make_groundtruth_rows("p1", "c1", n_pos=2, n_neg=5)
        criteria = [Criterion("c1", "test criterion")]

        # Call without new parameters (backward compatible)
        examples = build_grouped_examples(
            groundtruth, criteria, post_ids=["p1"], max_candidates=10, seed=42,
        )

        assert len(examples) == 1
        ex = examples[0]

        # Should work as before - no NO_EVIDENCE, no is_no_evidence field expected
        assert NO_EVIDENCE_TOKEN not in ex["sentences"]
        # But is_no_evidence field is now added
        assert ex["is_no_evidence"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
