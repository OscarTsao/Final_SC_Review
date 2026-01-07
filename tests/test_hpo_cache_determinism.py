from final_sc_review.hpo.cache_builder import _cache_fingerprint


def test_hpo_cache_fingerprint_deterministic():
    cfg = {
        "models": {"bge_m3": "BAAI/bge-m3"},
        "cache": {"dense_topk_max": 32, "sparse_topk_max": 32, "superset_max": 64},
        "split": {"seed": 42, "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "reranker": {"model_name": "jinaai/jina-reranker-v3"},
    }
    checksums = {"groundtruth": "a", "sentence_corpus": "b", "criteria": "c"}
    f1 = _cache_fingerprint(cfg, checksums)
    f2 = _cache_fingerprint(cfg, checksums)
    assert f1 == f2
