import pytest

from final_sc_review.hpo.cache_builder import _validate_candidate_pool


def test_candidate_pool_within_post_only():
    _validate_candidate_pool("p1", ["p1_0", "p1_1"])
    with pytest.raises(ValueError):
        _validate_candidate_pool("p1", ["p1_0", "p2_0"])
