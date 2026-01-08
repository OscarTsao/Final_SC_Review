"""Retriever Zoo: Unified interface for multiple retriever models.

Supports:
- BGE-M3 (existing hybrid: dense + sparse + ColBERT)
- Dense embedders from HuggingFace (configurable model IDs)
- BM25 (lexical baseline)
- SPLADE (if installable)
- ColBERTv2 (if installable)

Each retriever outputs identical per-query candidate lists with component scores.

Note: Uses pickle for internal caching of BM25 objects (same pattern as bge_m3.py).
This is safe as caches are only written/read by this system.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from final_sc_review.data.schemas import Sentence
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Standard result format for all retrievers."""
    sent_uid: str
    text: str
    score: float
    component_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for a retriever in the zoo."""
    name: str
    model_id: str
    retriever_type: str  # "hybrid", "dense", "lexical", "sparse"
    max_length: int = 512
    batch_size: int = 64
    use_fp16: bool = True
    pooling: str = "cls"  # "cls", "mean", "last"
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    min_vram_gb: float = 0.0


class BaseRetriever(ABC):
    """Abstract base class for all retrievers in the zoo."""

    def __init__(self, config: RetrieverConfig, sentences: List[Sentence], cache_dir: Path):
        self.config = config
        self.sentences = sentences
        self.cache_dir = cache_dir / config.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.post_to_indices: Dict[str, List[int]] = {}
        self.sent_uid_to_index: Dict[str, int] = {}
        for idx, sent in enumerate(sentences):
            self.post_to_indices.setdefault(sent.post_id, []).append(idx)
            self.sent_uid_to_index[sent.sent_uid] = idx

    @abstractmethod
    def encode_corpus(self, rebuild: bool = False) -> None:
        """Encode and cache corpus embeddings."""
        pass

    @abstractmethod
    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k: int = 100,
    ) -> List[RetrievalResult]:
        """Retrieve candidates from within a specific post."""
        pass

    def _corpus_fingerprint(self) -> str:
        """Compute fingerprint of corpus for cache validation."""
        h = hashlib.sha256()
        for sent in self.sentences:
            h.update(f"{sent.post_id}|{sent.sid}|{sent.text}".encode())
        return h.hexdigest()[:16]

    def _config_fingerprint(self) -> str:
        """Compute fingerprint of config for cache validation."""
        config_str = f"{self.config.model_id}|{self.config.max_length}|{self.config.pooling}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]


class DenseRetriever(BaseRetriever):
    """Dense embedding retriever using HuggingFace sentence-transformers."""

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        device: Optional[str] = None,
    ):
        super().__init__(config, sentences, cache_dir)
        self.device = device
        self.embeddings: Optional[np.ndarray] = None
        self.model = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading dense model: {self.config.model_id}")
        self.model = SentenceTransformer(self.config.model_id, device=self.device)

    def encode_corpus(self, rebuild: bool = False) -> None:
        """Encode and cache corpus embeddings."""
        cache_path = self.cache_dir / "embeddings.npy"
        fingerprint_path = self.cache_dir / "fingerprint.json"

        if not rebuild and cache_path.exists() and fingerprint_path.exists():
            with open(fingerprint_path) as f:
                meta = json.load(f)
            if (meta.get("corpus") == self._corpus_fingerprint() and
                meta.get("config") == self._config_fingerprint()):
                logger.info(f"Loading cached embeddings from {cache_path}")
                self.embeddings = np.load(cache_path)
                return

        self._load_model()
        texts = [sent.text for sent in self.sentences]

        if self.config.passage_prefix:
            texts = [self.config.passage_prefix + t for t in texts]

        logger.info(f"Encoding {len(texts)} sentences with {self.config.name}")
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )

        np.save(cache_path, self.embeddings)
        with open(fingerprint_path, "w") as f:
            json.dump({
                "corpus": self._corpus_fingerprint(),
                "config": self._config_fingerprint(),
                "model_id": self.config.model_id,
                "num_sentences": len(self.sentences),
            }, f, indent=2)

    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k: int = 100,
    ) -> List[RetrievalResult]:
        """Retrieve candidates from within a specific post."""
        if self.embeddings is None:
            self.encode_corpus()

        indices = self.post_to_indices.get(post_id, [])
        if not indices:
            return []

        self._load_model()

        query_text = self.config.query_prefix + query if self.config.query_prefix else query
        query_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )[0]

        candidate_embs = self.embeddings[indices]
        scores = candidate_embs @ query_emb

        ranked_indices = np.argsort(-scores)[:top_k]
        results = []
        for rank_idx in ranked_indices:
            idx = indices[rank_idx]
            sent = self.sentences[idx]
            results.append(RetrievalResult(
                sent_uid=sent.sent_uid,
                text=sent.text,
                score=float(scores[rank_idx]),
                component_scores={"dense": float(scores[rank_idx])},
            ))
        return results


class BM25Retriever(BaseRetriever):
    """BM25 lexical retriever baseline."""

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        super().__init__(config, sentences, cache_dir)
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus: Optional[List[List[str]]] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def encode_corpus(self, rebuild: bool = False) -> None:
        """Build BM25 index from corpus."""
        cache_path = self.cache_dir / "bm25_cache.pkl"
        fingerprint_path = self.cache_dir / "fingerprint.json"

        if not rebuild and cache_path.exists() and fingerprint_path.exists():
            with open(fingerprint_path) as f:
                meta = json.load(f)
            if meta.get("corpus") == self._corpus_fingerprint():
                logger.info(f"Loading cached BM25 index from {cache_path}")
                # Using pickle for internal BM25 caching (same pattern as bge_m3.py)
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.tokenized_corpus = data["tokenized_corpus"]
                return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required. Install with: pip install rank-bm25"
            )

        logger.info("Building BM25 index...")
        self.tokenized_corpus = [self._tokenize(sent.text) for sent in self.sentences]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        # Using pickle for internal BM25 caching (same pattern as bge_m3.py)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)
        with open(fingerprint_path, "w") as f:
            json.dump({
                "corpus": self._corpus_fingerprint(),
                "k1": self.k1,
                "b": self.b,
                "num_sentences": len(self.sentences),
            }, f, indent=2)

    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k: int = 100,
    ) -> List[RetrievalResult]:
        """Retrieve candidates from within a specific post."""
        if self.bm25 is None:
            self.encode_corpus()

        indices = self.post_to_indices.get(post_id, [])
        if not indices:
            return []

        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
        all_scores = self.bm25.get_scores(tokenized_query)

        # Filter to within-post candidates
        candidate_scores = [(idx, all_scores[idx]) for idx in indices]
        candidate_scores.sort(key=lambda x: -x[1])

        results = []
        for idx, score in candidate_scores[:top_k]:
            sent = self.sentences[idx]
            results.append(RetrievalResult(
                sent_uid=sent.sent_uid,
                text=sent.text,
                score=float(score),
                component_scores={"bm25": float(score)},
            ))
        return results


class BGEM3ZooRetriever(BaseRetriever):
    """BGE-M3 wrapper for zoo interface."""

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        device: Optional[str] = None,
    ):
        super().__init__(config, sentences, cache_dir)
        self.device = device
        self._bge_retriever = None

    def _load_retriever(self):
        """Lazy load the BGE-M3 retriever."""
        if self._bge_retriever is not None:
            return

        from final_sc_review.retriever.bge_m3 import BgeM3Retriever

        self._bge_retriever = BgeM3Retriever(
            sentences=self.sentences,
            cache_dir=self.cache_dir,
            model_name=self.config.model_id,
            device=self.device,
            query_max_length=128,
            passage_max_length=self.config.max_length,
            use_fp16=self.config.use_fp16,
            batch_size=self.config.batch_size,
            rebuild_cache=False,
        )

    def encode_corpus(self, rebuild: bool = False) -> None:
        """BGE-M3 builds cache on init."""
        self._load_retriever()

    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k: int = 100,
    ) -> List[RetrievalResult]:
        """Retrieve using BGE-M3 hybrid retrieval."""
        self._load_retriever()

        candidates = self._bge_retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=top_k,
            top_k_colbert=top_k,
            use_sparse=True,
            use_colbert=True,
            fusion_method="rrf",
        )

        results = []
        for sent_uid, text, score in candidates:
            results.append(RetrievalResult(
                sent_uid=sent_uid,
                text=text,
                score=score,
                component_scores={"hybrid_rrf": score},
            ))
        return results


class RetrieverZoo:
    """Factory and manager for multiple retrievers."""

    # Default retriever configurations
    DEFAULT_RETRIEVERS = [
        RetrieverConfig(
            name="bge-m3",
            model_id="BAAI/bge-m3",
            retriever_type="hybrid",
            max_length=256,
            batch_size=64,
        ),
        RetrieverConfig(
            name="bge-large-en-v1.5",
            model_id="BAAI/bge-large-en-v1.5",
            retriever_type="dense",
            max_length=512,
            batch_size=64,
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        RetrieverConfig(
            name="e5-large-v2",
            model_id="intfloat/e5-large-v2",
            retriever_type="dense",
            max_length=512,
            batch_size=64,
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        RetrieverConfig(
            name="gte-large-en-v1.5",
            model_id="Alibaba-NLP/gte-large-en-v1.5",
            retriever_type="dense",
            max_length=8192,
            batch_size=32,
        ),
        RetrieverConfig(
            name="bm25",
            model_id="bm25",
            retriever_type="lexical",
        ),
    ]

    def __init__(
        self,
        sentences: List[Sentence],
        cache_dir: Path,
        configs: Optional[List[RetrieverConfig]] = None,
        device: Optional[str] = None,
    ):
        self.sentences = sentences
        self.cache_dir = cache_dir / "retriever_zoo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.configs = configs or self.DEFAULT_RETRIEVERS
        self._retrievers: Dict[str, BaseRetriever] = {}

    def get_retriever(self, name: str) -> BaseRetriever:
        """Get or create a retriever by name."""
        if name in self._retrievers:
            return self._retrievers[name]

        config = None
        for c in self.configs:
            if c.name == name:
                config = c
                break

        if config is None:
            raise ValueError(f"Unknown retriever: {name}")

        retriever = self._create_retriever(config)
        self._retrievers[name] = retriever
        return retriever

    def _create_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create a retriever from config."""
        if config.retriever_type == "hybrid":
            return BGEM3ZooRetriever(
                config=config,
                sentences=self.sentences,
                cache_dir=self.cache_dir,
                device=self.device,
            )
        elif config.retriever_type == "dense":
            return DenseRetriever(
                config=config,
                sentences=self.sentences,
                cache_dir=self.cache_dir,
                device=self.device,
            )
        elif config.retriever_type == "lexical":
            return BM25Retriever(
                config=config,
                sentences=self.sentences,
                cache_dir=self.cache_dir,
            )
        else:
            raise ValueError(f"Unknown retriever type: {config.retriever_type}")

    def list_retrievers(self) -> List[str]:
        """List available retriever names."""
        return [c.name for c in self.configs]

    def encode_all(self, rebuild: bool = False) -> None:
        """Encode corpus for all retrievers."""
        for config in self.configs:
            logger.info(f"Encoding corpus for {config.name}")
            try:
                retriever = self.get_retriever(config.name)
                retriever.encode_corpus(rebuild=rebuild)
            except Exception as e:
                logger.warning(f"Failed to encode {config.name}: {e}")

    def compute_oracle_recall(
        self,
        queries: List[Dict],  # [{"query": str, "post_id": str, "gold_uids": set}]
        k_values: List[int] = [50, 100, 200, 400, 800],
        retriever_name: Optional[str] = None,
    ) -> Dict[str, Dict[int, float]]:
        """Compute oracle recall at various K for each retriever."""
        retriever_names = [retriever_name] if retriever_name else self.list_retrievers()
        results = {}

        for name in retriever_names:
            try:
                retriever = self.get_retriever(name)
                retriever.encode_corpus()
            except Exception as e:
                logger.warning(f"Skipping {name}: {e}")
                continue

            recalls = {k: [] for k in k_values}

            for q in queries:
                if not q.get("gold_uids"):
                    continue

                candidates = retriever.retrieve_within_post(
                    query=q["query"],
                    post_id=q["post_id"],
                    top_k=max(k_values),
                )

                retrieved_uids = [r.sent_uid for r in candidates]
                gold_uids = set(q["gold_uids"])

                for k in k_values:
                    hits = len(gold_uids & set(retrieved_uids[:k]))
                    recall = hits / len(gold_uids) if gold_uids else 0.0
                    recalls[k].append(recall)

            results[name] = {k: np.mean(v) if v else 0.0 for k, v in recalls.items()}
            logger.info(f"{name} oracle recalls: {results[name]}")

        return results
