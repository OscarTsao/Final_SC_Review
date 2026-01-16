"""Cache builder for inference-stage HPO."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.hpo.reporting import compute_sha256, write_json
from final_sc_review.retriever.bge_m3 import BgeM3Retriever, maxsim_score
from final_sc_review.reranker.jina_v3 import JinaV3Reranker
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def build_cache(config_path: Path, force_rebuild: bool = False) -> Path:
    cfg = _load_config(config_path)
    cache_dir = Path(cfg["paths"]["hpo_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_checksums = _dataset_checksums(cfg)
    cache_fingerprint = _cache_fingerprint(cfg, dataset_checksums)
    out_dir = cache_dir / cache_fingerprint
    manifest_path = out_dir / "manifest.json"

    if out_dir.exists() and manifest_path.exists() and not force_rebuild:
        logger.info("Cache exists at %s", out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".lock"
    _acquire_lock(lock_path, cfg["cache"].get("lock_timeout_sec", 3600))
    try:
        _build_cache(cfg, out_dir, dataset_checksums)
    finally:
        _release_lock(lock_path)
    return out_dir


def load_manifest(cache_dir: Path) -> Dict:
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_cache_split(cache_dir: Path, expected_split: str) -> None:
    manifest = load_manifest(cache_dir)
    actual = manifest.get("split")
    if actual != expected_split:
        raise ValueError(f"Cache split mismatch: expected {expected_split}, got {actual}")


def _build_cache(cfg: Dict, out_dir: Path, dataset_checksums: Dict[str, str]) -> None:
    groundtruth = load_groundtruth(Path(cfg["paths"]["groundtruth"]))
    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    criteria = load_criteria(Path(cfg["paths"]["criteria"]))
    criteria_map = {c.criterion_id: c.text for c in criteria}

    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg["split"]["seed"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )
    dev_split = cfg["split"].get("dev_split", "val")
    dev_posts = set(splits[dev_split])

    sentences_sorted = sorted(sentences, key=lambda s: (s.post_id, s.sid))
    post_to_indices: Dict[str, List[int]] = {}
    for idx, sent in enumerate(sentences_sorted):
        post_to_indices.setdefault(sent.post_id, []).append(idx)

    retriever_cache_dir = out_dir / "bge_cache"
    bge_cfg = cfg["models"]
    retriever = BgeM3Retriever(
        sentences=sentences_sorted,
        cache_dir=retriever_cache_dir,
        model_name=bge_cfg["bge_m3"],
        device=cfg.get("device"),
        query_max_length=bge_cfg.get("bge_query_max_length", 128),
        passage_max_length=bge_cfg.get("bge_passage_max_length", 256),
        use_fp16=bge_cfg.get("bge_use_fp16", True),
        batch_size=bge_cfg.get("bge_batch_size", 64),
        rebuild_cache=cfg["cache"].get("rebuild_embeddings", False),
    )

    reranker_cfg = cfg["reranker"]
    reranker = JinaV3Reranker(
        model_name=reranker_cfg["model_name"],
        device=cfg.get("device"),
        max_length=reranker_cfg.get("max_length", 512),
        listwise_chunk_size=reranker_cfg["chunk_size"],
        dtype=reranker_cfg.get("dtype", "auto"),
        use_listwise=True,  # Will fallback to pairwise if model lacks rerank() API
    )
    # Note: reranker.score_pairs() works for both listwise and pairwise modes

    dense_topk = cfg["cache"]["dense_topk_max"]
    sparse_topk = cfg["cache"]["sparse_topk_max"]
    superset_max = cfg["cache"]["superset_max"]
    use_sparse = cfg["cache"].get("use_sparse", True)
    use_multiv = cfg["cache"].get("use_multiv", True)
    trim_method = cfg["cache"].get("trim_method", "dense")

    # Group groundtruth by (post_id, criterion)
    grouped: Dict[Tuple[str, str], List] = {}
    for row in groundtruth:
        if row.post_id not in dev_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    query_ids = []
    post_ids_out = []
    criterion_ids = []
    criterion_texts = []
    candidate_ids = []
    dense_scores_out = []
    sparse_scores_out = []
    multiv_scores_out = []
    jina_scores_out = []
    gold_ids_out = []

    for (post_id, criterion_id), rows in sorted(grouped.items()):
        query_text = criteria_map.get(criterion_id)
        if query_text is None:
            continue
        indices = post_to_indices.get(post_id, [])
        if not indices:
            continue

        gold_ids = sorted({r.sent_uid for r in rows if r.groundtruth == 1})
        for gid in gold_ids:
            if gid not in retriever.sent_uid_to_index:
                raise ValueError(f"gold_id not in corpus: {gid}")

        query_out = retriever.encoder.encode(
            [query_text],
            return_dense=True,
            return_sparse=use_sparse,
            return_colbert=use_multiv,
            max_length=retriever.query_max_length,
        )
        query_dense = query_out["dense_vecs"][0]
        query_sparse = {}
        if use_sparse:
            query_sparse = (query_out.get("lexical_weights") or [{}])[0]
        query_colbert = None
        if use_multiv:
            query_colbert = (query_out.get("colbert_vecs") or [None])[0]

        candidate_dense = retriever.dense_vecs[indices]
        dense_scores = candidate_dense @ query_dense
        dense_ranked = _topk_indices(indices, dense_scores, dense_topk, retriever)

        sparse_ranked = []
        if use_sparse:
            sparse_scores_all = []
            for idx in indices:
                doc_sparse = retriever.sparse_weights[idx] if idx < len(retriever.sparse_weights) else {}
                score = sum(query_sparse.get(tok, 0.0) * weight for tok, weight in doc_sparse.items())
                sparse_scores_all.append(score)
            sparse_ranked = _topk_indices(indices, np.array(sparse_scores_all), sparse_topk, retriever)
        union_indices = sorted(set(dense_ranked) | set(sparse_ranked))

        if not union_indices:
            continue

        limit = min(len(union_indices), superset_max)
        # Apply selected trim method
        if trim_method == "rrf_dense_sparse" and use_sparse:
            union_indices = _trim_by_rrf(union_indices, retriever, query_dense, query_sparse, limit)
        elif trim_method == "max_normalized" and use_sparse:
            union_indices = _trim_by_max_normalized(union_indices, retriever, query_dense, query_sparse, limit)
        else:
            # Default: dense-only trim (legacy behavior)
            union_indices = _trim_by_dense(union_indices, retriever, query_dense, limit)

        candidate_list = []
        dense_list = []
        sparse_list = []
        multiv_list = []

        for idx in union_indices:
            sent = sentences_sorted[idx]
            candidate_list.append(sent.sent_uid)
            dense_list.append(float(retriever.dense_vecs[idx] @ query_dense))
            if use_sparse:
                doc_sparse = retriever.sparse_weights[idx] if idx < len(retriever.sparse_weights) else {}
                sparse_list.append(
                    float(sum(query_sparse.get(tok, 0.0) * weight for tok, weight in doc_sparse.items()))
                )
            if use_multiv:
                if query_colbert is None or not retriever.colbert_vecs:
                    multiv_list.append(0.0)
                else:
                    multiv_list.append(float(maxsim_score(query_colbert, retriever.colbert_vecs[idx])))

        candidate_texts = [sentences_sorted[idx].text for idx in union_indices]
        jina_scores = reranker.score_pairs(query_text, candidate_texts)

        _validate_candidate_pool(post_id, candidate_list)

        query_id = f"{post_id}::{criterion_id}"
        query_ids.append(query_id)
        post_ids_out.append(post_id)
        criterion_ids.append(criterion_id)
        criterion_texts.append(query_text)
        candidate_ids.append(candidate_list)
        dense_scores_out.append(dense_list)
        sparse_scores_out.append(sparse_list if use_sparse else None)
        multiv_scores_out.append(multiv_list if use_multiv else None)
        jina_scores_out.append(jina_scores)
        gold_ids_out.append(gold_ids)

    cache_path = out_dir / "dev_cache.npz"
    np.savez_compressed(
        cache_path,
        query_id=np.array(query_ids, dtype=object),
        post_id=np.array(post_ids_out, dtype=object),
        criterion_id=np.array(criterion_ids, dtype=object),
        criterion_text=np.array(criterion_texts, dtype=object),
        candidate_ids=np.array(candidate_ids, dtype=object),
        dense_scores=np.array(dense_scores_out, dtype=object),
        sparse_scores=np.array(sparse_scores_out, dtype=object),
        multiv_scores=np.array(multiv_scores_out, dtype=object),
        jina_scores=np.array(jina_scores_out, dtype=object),
        gold_ids=np.array(gold_ids_out, dtype=object),
    )

    manifest = {
        "cache_path": str(cache_path),
        "split": dev_split,
        "num_queries": len(query_ids),
        "dataset_checksums": dataset_checksums,
        "config": cfg,
    }
    write_json(out_dir / "manifest.json", manifest)
    logger.info("Saved cache to %s", cache_path)


def _validate_candidate_pool(post_id: str, candidate_ids: List[str]) -> None:
    for sent_uid in candidate_ids:
        if not sent_uid.startswith(f"{post_id}_"):
            raise ValueError("candidate_id not within post_id pool")


def _topk_indices(indices: List[int], scores: np.ndarray, k: int, retriever: BgeM3Retriever) -> List[int]:
    sorted_pairs = []
    for idx, score in zip(indices, scores.tolist()):
        sent_uid = retriever.sentences[idx].sent_uid
        sorted_pairs.append((idx, score, sent_uid))
    sorted_pairs.sort(key=lambda x: (-x[1], x[2]))
    return [idx for idx, _, _ in sorted_pairs[:k]]


def _trim_by_dense(indices: List[int], retriever: BgeM3Retriever, query_dense: np.ndarray, k: int) -> List[int]:
    """Trim candidates by dense score (legacy method)."""
    scored = []
    for idx in indices:
        score = float(retriever.dense_vecs[idx] @ query_dense)
        sent_uid = retriever.sentences[idx].sent_uid
        scored.append((idx, score, sent_uid))
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [idx for idx, _, _ in scored[:k]]


def _trim_by_rrf(
    indices: List[int],
    retriever: BgeM3Retriever,
    query_dense: np.ndarray,
    query_sparse: dict,
    k: int,
    rrf_k: int = 60,
) -> List[int]:
    """Trim candidates by RRF(dense_rank, sparse_rank) for fairer selection."""
    # Compute dense scores
    dense_scored = []
    for idx in indices:
        score = float(retriever.dense_vecs[idx] @ query_dense)
        dense_scored.append((idx, score))
    dense_scored.sort(key=lambda x: -x[1])
    dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_scored)}

    # Compute sparse scores
    sparse_scored = []
    for idx in indices:
        doc_sparse = retriever.sparse_weights[idx] if idx < len(retriever.sparse_weights) else {}
        score = sum(query_sparse.get(tok, 0.0) * weight for tok, weight in doc_sparse.items())
        sparse_scored.append((idx, score))
    sparse_scored.sort(key=lambda x: -x[1])
    sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_scored)}

    # Compute RRF scores
    rrf_scored = []
    for idx in indices:
        rrf_score = 1.0 / (rrf_k + dense_ranks[idx]) + 1.0 / (rrf_k + sparse_ranks[idx])
        sent_uid = retriever.sentences[idx].sent_uid
        rrf_scored.append((idx, rrf_score, sent_uid))
    rrf_scored.sort(key=lambda x: (-x[1], x[2]))
    return [idx for idx, _, _ in rrf_scored[:k]]


def _trim_by_max_normalized(
    indices: List[int],
    retriever: BgeM3Retriever,
    query_dense: np.ndarray,
    query_sparse: dict,
    k: int,
) -> List[int]:
    """Trim candidates by max(normalized_dense, normalized_sparse) for fairer selection."""
    # Compute dense scores
    dense_scores = []
    for idx in indices:
        score = float(retriever.dense_vecs[idx] @ query_dense)
        dense_scores.append(score)

    # Compute sparse scores
    sparse_scores = []
    for idx in indices:
        doc_sparse = retriever.sparse_weights[idx] if idx < len(retriever.sparse_weights) else {}
        score = sum(query_sparse.get(tok, 0.0) * weight for tok, weight in doc_sparse.items())
        sparse_scores.append(score)

    # Normalize to [0, 1]
    def minmax_normalize(scores):
        min_v, max_v = min(scores), max(scores)
        if max_v > min_v:
            return [(s - min_v) / (max_v - min_v) for s in scores]
        return [0.0] * len(scores)

    norm_dense = minmax_normalize(dense_scores)
    norm_sparse = minmax_normalize(sparse_scores)

    # Max-fused score
    fused_scored = []
    for i, idx in enumerate(indices):
        fused = max(norm_dense[i], norm_sparse[i])
        sent_uid = retriever.sentences[idx].sent_uid
        fused_scored.append((idx, fused, sent_uid))
    fused_scored.sort(key=lambda x: (-x[1], x[2]))
    return [idx for idx, _, _ in fused_scored[:k]]


def _select_trim_method(method: str):
    """Return the appropriate trim function based on config."""
    if method == "rrf_dense_sparse":
        return _trim_by_rrf
    elif method == "max_normalized":
        return _trim_by_max_normalized
    else:
        return None  # Default to dense-only


def _load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _dataset_checksums(cfg: Dict) -> Dict[str, str]:
    checksums = {
        "groundtruth": compute_sha256(Path(cfg["paths"]["groundtruth"])),
        "sentence_corpus": compute_sha256(Path(cfg["paths"]["sentence_corpus"])),
        "criteria": compute_sha256(Path(cfg["paths"]["criteria"])),
    }
    return checksums


def _cache_fingerprint(cfg: Dict, dataset_checksums: Dict[str, str]) -> str:
    payload = {
        "dataset": dataset_checksums,
        "models": cfg["models"],
        "cache": cfg["cache"],
        "split": cfg["split"],
        "reranker": cfg["reranker"],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _acquire_lock(lock_path: Path, timeout_sec: int) -> None:
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Cache lock timed out: {lock_path}")
            time.sleep(1)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
