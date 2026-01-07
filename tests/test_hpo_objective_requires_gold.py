from pathlib import Path

import numpy as np
import pytest

from final_sc_review.hpo.objective_inference import load_cache


def test_objective_requires_gold(tmp_path: Path):
    cache_path = tmp_path / "dev_cache.npz"
    np.savez_compressed(
        cache_path,
        query_id=np.array(["p1::c1"], dtype=object),
        post_id=np.array(["p1"], dtype=object),
        criterion_id=np.array(["c1"], dtype=object),
        criterion_text=np.array(["crit"], dtype=object),
        candidate_ids=np.array([["p1_0", "p1_1"]], dtype=object),
        dense_scores=np.array([[0.1, 0.2]], dtype=object),
        sparse_scores=np.array([[0.0, 0.0]], dtype=object),
        multiv_scores=np.array([[0.0, 0.0]], dtype=object),
        jina_scores=np.array([[0.3, 0.1]], dtype=object),
        gold_ids=np.array([None], dtype=object),
    )

    with pytest.raises(ValueError):
        load_cache(cache_path)
