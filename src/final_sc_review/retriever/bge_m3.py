"""BGE-M3 hybrid retriever with dense, sparse, and ColBERT scoring."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from final_sc_review.data.schemas import Sentence
from final_sc_review.utils.hashing import corpus_fingerprint
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def maxsim_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Compute MaxSim score between query and doc token vectors."""
    if len(query_vecs) == 0 or len(doc_vecs) == 0:
        return 0.0
    similarities = np.dot(query_vecs, doc_vecs.T)
    max_sims = similarities.max(axis=1)
    return float(max_sims.mean())


class BGEM3HybridEncoder:
    """BGE-M3 encoder supporting dense, sparse, and multi-vector outputs."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "FlagEmbedding is required for BGE-M3. Install with: pip install FlagEmbedding"
            ) from exc

        logger.info("Loading BGE-M3 model %s on %s", model_name, device)
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 and device == "cuda",
            device=device,
        )

    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict:
        if not texts:
            return {
                "dense_vecs": np.array([]) if return_dense else None,
                "lexical_weights": [] if return_sparse else None,
                "colbert_vecs": [] if return_colbert else None,
            }

        batch_size = batch_size or self.batch_size
        max_length = max_length or self.max_length

        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

        result: Dict[str, object] = {}
        if return_dense:
            dense = output.get("dense_vecs")
            if isinstance(dense, torch.Tensor):
                dense = dense.cpu().numpy()
            result["dense_vecs"] = dense
        if return_sparse:
            result["lexical_weights"] = output.get("lexical_weights") or []
        if return_colbert:
            colbert = output.get("colbert_vecs") or []
            colbert_list = []
            for vec in colbert:
                if isinstance(vec, torch.Tensor):
                    colbert_list.append(vec.cpu().numpy())
                else:
                    colbert_list.append(vec)
            result["colbert_vecs"] = colbert_list
        return result

    def encode_corpus(
        self,
        documents: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict:
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.max_length

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    range(0, len(documents), batch_size),
                    desc="Encoding corpus",
                    total=(len(documents) + batch_size - 1) // batch_size,
                )
            except Exception:  # pragma: no cover - fallback
                iterator = range(0, len(documents), batch_size)
        else:
            iterator = range(0, len(documents), batch_size)

        all_dense = [] if return_dense else None
        all_sparse = [] if return_sparse else None
        all_colbert = [] if return_colbert else None

        for start in iterator:
            batch = documents[start : start + batch_size]
            out = self.encode(
                batch,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
                batch_size=batch_size,
                max_length=max_length,
            )
            if return_dense and out.get("dense_vecs") is not None:
                all_dense.append(out["dense_vecs"])
            if return_sparse and out.get("lexical_weights"):
                all_sparse.extend(out["lexical_weights"])
            if return_colbert and out.get("colbert_vecs"):
                all_colbert.extend(out["colbert_vecs"])

        result = {}
        result["dense_vecs"] = np.vstack(all_dense) if return_dense and all_dense else None
        result["lexical_weights"] = all_sparse if return_sparse else []
        result["colbert_vecs"] = all_colbert if return_colbert else []
        return result


class BgeM3Retriever:
    """Retriever using BGE-M3 dense, sparse, and ColBERT signals."""

    def __init__(
        self,
        sentences: List[Sentence],
        cache_dir: Path,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        query_max_length: int = 128,
        passage_max_length: int = 256,
        use_fp16: bool = True,
        batch_size: int = 64,
        rebuild_cache: bool = False,
    ):
        self.sentences = sentences
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = BGEM3HybridEncoder(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size,
            max_length=passage_max_length,
        )
        self.model_name = model_name
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.post_to_indices: Dict[str, List[int]] = {}
        self.sent_uid_to_index: Dict[str, int] = {}
        for idx, sent in enumerate(sentences):
            self.post_to_indices.setdefault(sent.post_id, []).append(idx)
            self.sent_uid_to_index[sent.sent_uid] = idx
        self.dense_vecs: np.ndarray
        self.sparse_weights: List[Dict[str, float]]
        self.colbert_vecs: List[np.ndarray]
        self._load_or_build_indexes(rebuild_cache)

    def _cache_paths(self) -> Tuple[Path, Path]:
        return self.cache_dir / "dense.npy", self.cache_dir / "fingerprint.json"

    def _all_cache_paths(self) -> Dict[str, Path]:
        return {
            "dense": self.cache_dir / "dense.npy",
            "sparse": self.cache_dir / "sparse.pkl",
            "colbert": self.cache_dir / "colbert.pkl",
            "fingerprint": self.cache_dir / "fingerprint.json",
        }

    def _load_or_build_indexes(self, rebuild_cache: bool) -> None:
        paths = self._all_cache_paths()
        if not rebuild_cache and all(p.exists() for p in paths.values()):
            with open(paths["fingerprint"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected = meta.get("fingerprint")
            current = corpus_fingerprint((s.post_id, s.sid, s.text) for s in self.sentences)
            config_ok = meta.get("model_name") == self.model_name
            config_ok = config_ok and meta.get("query_max_length") == self.query_max_length
            config_ok = config_ok and meta.get("passage_max_length") == self.passage_max_length
            config_ok = config_ok and meta.get("use_fp16") == self.use_fp16
            config_ok = config_ok and meta.get("batch_size") == self.batch_size
            if expected == current and config_ok:
                self._load_indexes(paths)
                return
            logger.info("Cache fingerprint mismatch; rebuilding indexes")
        self._build_and_cache_indexes(paths)

    def _build_and_cache_indexes(self, paths: Dict[str, Path]) -> None:
        texts = [s.text for s in self.sentences]
        output = self.encoder.encode_corpus(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            batch_size=self.batch_size,
            max_length=self.passage_max_length,
            show_progress=True,
        )
        dense = output.get("dense_vecs")
        self.dense_vecs = dense if dense is not None else np.zeros((0, 768), dtype=np.float32)
        self.sparse_weights = output.get("lexical_weights") or []
        self.colbert_vecs = output.get("colbert_vecs") or []

        np.save(paths["dense"], self.dense_vecs)
        with open(paths["sparse"], "wb") as f:
            pickle.dump(self.sparse_weights, f)
        with open(paths["colbert"], "wb") as f:
            pickle.dump(self.colbert_vecs, f)

        fingerprint = corpus_fingerprint((s.post_id, s.sid, s.text) for s in self.sentences)
        with open(paths["fingerprint"], "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "num_sentences": len(self.sentences),
                    "model_name": self.model_name,
                    "query_max_length": self.query_max_length,
                    "passage_max_length": self.passage_max_length,
                    "use_fp16": self.use_fp16,
                    "batch_size": self.batch_size,
                },
                f,
                indent=2,
            )

    def _load_indexes(self, paths: Dict[str, Path]) -> None:
        self.dense_vecs = np.load(paths["dense"])
        with open(paths["sparse"], "rb") as f:
            self.sparse_weights = pickle.load(f)
        with open(paths["colbert"], "rb") as f:
            self.colbert_vecs = pickle.load(f)

    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k_retriever: int = 50,
        top_k_colbert: int = 50,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.0,
        use_sparse: bool = True,
        use_colbert: bool = True,
        fusion_method: str = "weighted_sum",
        score_normalization: str = "none",
        rrf_k: int = 60,
    ) -> List[Tuple[str, str, float]]:
        indices = self.post_to_indices.get(post_id, [])
        if not indices:
            return []

        query_out = self.encoder.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            max_length=self.query_max_length,
        )
        if query_out.get("dense_vecs") is None or len(query_out["dense_vecs"]) == 0:
            return []
        query_dense = query_out["dense_vecs"][0]
        query_sparse = query_out["lexical_weights"][0] if query_out["lexical_weights"] else {}
        query_colbert = query_out["colbert_vecs"][0] if query_out["colbert_vecs"] else None

        candidate_dense = self.dense_vecs[indices]
        dense_scores = candidate_dense @ query_dense

        sparse_scores = None
        if use_sparse:
            sparse_scores = []
            for idx in indices:
                doc_sparse = self.sparse_weights[idx] if idx < len(self.sparse_weights) else {}
                score = sum(query_sparse.get(tok, 0.0) * weight for tok, weight in doc_sparse.items())
                sparse_scores.append(score)
            sparse_scores = np.array(sparse_scores, dtype=float)

        colbert_scores = None
        compute_colbert = use_colbert and query_colbert is not None and self.colbert_vecs
        if compute_colbert and (fusion_method == "rrf" or colbert_weight > 0.0):
            colbert_scores = []
            for idx in indices:
                doc_colbert = self.colbert_vecs[idx]
                colbert_scores.append(maxsim_score(query_colbert, doc_colbert))
            colbert_scores = np.array(colbert_scores, dtype=float)

        fused_scores = _fuse_components(
            dense_scores,
            sparse_scores,
            colbert_scores,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            colbert_weight=colbert_weight,
            fusion_method=fusion_method,
            score_normalization=score_normalization,
            rrf_k=rrf_k,
        )

        ranked = sorted(
            zip(indices, fused_scores.tolist()),
            key=lambda x: (-x[1], self.sentences[x[0]].sent_uid),
        )
        pool_k = min(top_k_retriever, top_k_colbert)
        results = []
        for idx, score in ranked[:pool_k]:
            sent = self.sentences[idx]
            results.append((sent.sent_uid, sent.text, float(score)))
        return results


def _fuse_components(
    dense_scores: np.ndarray,
    sparse_scores: Optional[np.ndarray],
    colbert_scores: Optional[np.ndarray],
    dense_weight: float,
    sparse_weight: float,
    colbert_weight: float,
    fusion_method: str,
    score_normalization: str,
    rrf_k: int,
) -> np.ndarray:
    if fusion_method == "rrf":
        return _rrf_fusion(dense_scores, sparse_scores, colbert_scores, rrf_k)
    weights = _normalize_weights(dense_weight, sparse_weight, colbert_weight)
    fused = _normalize_scores(dense_scores, score_normalization) * weights[0]
    if sparse_scores is not None:
        fused = fused + _normalize_scores(sparse_scores, score_normalization) * weights[1]
    if colbert_scores is not None:
        fused = fused + _normalize_scores(colbert_scores, score_normalization) * weights[2]
    return fused


def _normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores
    if mode == "minmax_per_query":
        min_v = float(scores.min())
        max_v = float(scores.max())
        if max_v > min_v:
            return (scores - min_v) / (max_v - min_v)
        return np.zeros_like(scores)
    if mode == "zscore_per_query":
        mean = float(scores.mean())
        std = float(scores.std())
        if std > 0:
            return (scores - mean) / std
        return np.zeros_like(scores)
    raise ValueError(f"Unknown score_normalization: {mode}")


def _normalize_weights(w_dense: float, w_sparse: float, w_colbert: float) -> Tuple[float, float, float]:
    total = w_dense + w_sparse + w_colbert
    if total <= 0:
        return (1.0, 0.0, 0.0)
    return (w_dense / total, w_sparse / total, w_colbert / total)


def _rrf_fusion(
    dense_scores: np.ndarray,
    sparse_scores: Optional[np.ndarray],
    colbert_scores: Optional[np.ndarray],
    rrf_k: int,
) -> np.ndarray:
    scores = np.zeros(len(dense_scores), dtype=float)
    for comp in (dense_scores, sparse_scores, colbert_scores):
        if comp is None:
            continue
        ranks = _rank_indices(comp)
        for idx, rank in enumerate(ranks):
            scores[idx] += 1.0 / (rrf_k + rank)
    return scores


def _rank_indices(scores: np.ndarray) -> List[int]:
    idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(idxs, start=1):
        ranks[idx] = rank
    return ranks
