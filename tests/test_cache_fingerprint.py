from pathlib import Path

from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.utils.hashing import corpus_fingerprint


def test_corpus_fingerprint_changes():
    records_a = [("p1", 0, "a"), ("p1", 1, "b")]
    records_b = [("p1", 0, "a"), ("p1", 1, "c")]
    assert corpus_fingerprint(records_a) != corpus_fingerprint(records_b)


def test_cache_paths_extensions(tmp_path: Path):
    retriever = BgeM3Retriever.__new__(BgeM3Retriever)
    retriever.cache_dir = tmp_path
    emb_path, fp_path = retriever._cache_paths()
    assert emb_path.name.endswith(".npy")
    assert fp_path.name == "fingerprint.json"
