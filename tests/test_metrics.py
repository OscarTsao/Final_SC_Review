import math

from final_sc_review.metrics.ranking import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_ranking_metrics_basic():
    gold = ["a"]
    ranked = ["b", "a", "c"]

    assert recall_at_k(gold, ranked, k=2) == 1.0
    assert mrr_at_k(gold, ranked, k=2) == 0.5
    assert map_at_k(gold, ranked, k=2) == 0.5

    ndcg = ndcg_at_k(gold, ranked, k=2)
    expected = 1.0 / math.log2(3)
    assert abs(ndcg - expected) < 1e-6
