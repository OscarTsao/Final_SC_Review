from final_sc_review.hpo.cache_builder import _cache_fingerprint


def test_cache_key_changes_with_config():
    cfg_a = {
        "models": {"bge_m3": "BAAI/bge-m3", "bge_query_max_length": 128},
        "cache": {"dense_topk_max": 32, "sparse_topk_max": 32, "superset_max": 64},
        "split": {"seed": 42, "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "reranker": {"model_name": "jinaai/jina-reranker-v3"},
    }
    cfg_b = {
        "models": {"bge_m3": "BAAI/bge-m3", "bge_query_max_length": 256},
        "cache": {"dense_topk_max": 32, "sparse_topk_max": 32, "superset_max": 64},
        "split": {"seed": 42, "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "reranker": {"model_name": "jinaai/jina-reranker-v3"},
    }
    checksums = {"groundtruth": "a", "sentence_corpus": "b", "criteria": "c"}
    assert _cache_fingerprint(cfg_a, checksums) != _cache_fingerprint(cfg_b, checksums)
