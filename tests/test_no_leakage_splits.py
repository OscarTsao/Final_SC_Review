from final_sc_review.data.schemas import Criterion, GroundTruthRow
from final_sc_review.data.splits import split_post_ids
from final_sc_review.reranker.dataset import build_grouped_examples


def test_split_post_ids_disjoint():
    post_ids = ["p1", "p2", "p3", "p4", "p5"]
    splits = split_post_ids(post_ids, seed=1, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_negatives_within_post_only():
    groundtruth = [
        GroundTruthRow("p1", "c1", 0, "p1_0", "p1_pos", 1),
        GroundTruthRow("p1", "c1", 1, "p1_1", "p1_neg", 0),
        GroundTruthRow("p2", "c1", 0, "p2_0", "p2_pos", 1),
        GroundTruthRow("p2", "c1", 1, "p2_1", "p2_neg", 0),
    ]
    criteria = [Criterion("c1", "criterion")]
    examples = build_grouped_examples(groundtruth, criteria, post_ids=["p1"], max_candidates=2, seed=1)
    assert len(examples) == 1
    ex = examples[0]
    assert ex["post_id"] == "p1"
    assert all("p1_" in sent for sent in ex["sentences"])
