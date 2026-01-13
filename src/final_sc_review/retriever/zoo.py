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
    retriever_type: str  # "hybrid", "dense", "lexical", "sparse", "late_interaction"
    max_length: int = 512
    batch_size: int = 64
    use_fp16: bool = True
    pooling: str = "cls"  # "cls", "mean", "last"
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    min_vram_gb: float = 0.0
    trust_remote_code: bool = False  # For models with custom code (GTE, Qwen, etc.)
    use_4bit: bool = False  # Enable 4-bit quantization for large models
    use_bf16: bool = True  # Use BF16 with FP16 fallback (AMP-style)


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


def _patch_dynamic_cache_compat():
    """Patch DynamicCache for backward compatibility with older model code.

    Some models like NV-Embed-v2 use the deprecated `get_usable_length` method
    which was removed in newer transformers versions. This adds it back.
    Also fixes `from_legacy_cache(None)` returning None instead of empty cache.
    """
    try:
        from transformers.cache_utils import DynamicCache

        # Patch 1: Add missing get_usable_length method
        if not hasattr(DynamicCache, 'get_usable_length'):
            def get_usable_length(self, new_seq_length: int) -> int:
                """Return the sequence length of the cached states."""
                # In older versions, this returned min(self.get_seq_length(), new_seq_length)
                # but for most use cases, get_seq_length() is what we need
                return self.get_seq_length()

            DynamicCache.get_usable_length = get_usable_length
            logger.debug("Patched DynamicCache.get_usable_length for backward compatibility")

        # Patch 2: Fix from_legacy_cache(None) returning None
        # Older model code expects an empty DynamicCache, not None
        _original_from_legacy_cache = DynamicCache.from_legacy_cache

        @classmethod
        def _patched_from_legacy_cache(cls, past_key_values):
            if past_key_values is None:
                return cls()  # Return empty DynamicCache
            return _original_from_legacy_cache.__func__(cls, past_key_values)

        if not getattr(DynamicCache, '_patched_from_legacy', False):
            DynamicCache.from_legacy_cache = _patched_from_legacy_cache
            DynamicCache._patched_from_legacy = True
            logger.debug("Patched DynamicCache.from_legacy_cache for None handling")

    except ImportError:
        pass  # transformers not installed


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

        # Apply DynamicCache compatibility patch before loading (for NV-Embed-v2, etc.)
        _patch_dynamic_cache_compat()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        import torch

        logger.info(f"Loading dense model: {self.config.model_id}")

        # Determine dtype: prefer BF16, fallback to FP16
        model_kwargs = {}
        if self.config.use_bf16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("  Using BF16 precision")
            elif self.config.use_fp16:
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("  BF16 not supported, using FP16 fallback")

        # Handle 4-bit quantization for large models (only if BF16/FP16 not enough)
        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("  Using 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        if self.config.trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        self.model = SentenceTransformer(
            self.config.model_id,
            device=self.device,
            model_kwargs=model_kwargs if model_kwargs else None,
            trust_remote_code=self.config.trust_remote_code,
        )

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


class ColBERTv2Retriever(BaseRetriever):
    """ColBERTv2 late-interaction retriever.

    Uses token-level embeddings with MaxSim scoring for fine-grained matching.
    Higher quality but more expensive than single-vector methods.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        device: Optional[str] = None,
    ):
        super().__init__(config, sentences, cache_dir)
        self.device = device or "cuda"
        self.model = None
        self.doc_embeddings: Optional[List[np.ndarray]] = None

    def _load_model(self):
        """Lazy load ColBERTv2 model."""
        if self.model is not None:
            return

        try:
            from colbert.infra import ColBERTConfig
            from colbert.modeling.checkpoint import Checkpoint
        except ImportError:
            raise ImportError(
                "ColBERT is required. Install with: pip install colbert-ai"
            )

        logger.info(f"Loading ColBERTv2 model: {self.config.model_id}")
        colbert_config = ColBERTConfig(
            doc_maxlen=self.config.max_length,
            query_maxlen=128,
        )
        self.model = Checkpoint(self.config.model_id, colbert_config)

    def encode_corpus(self, rebuild: bool = False) -> None:
        """Encode and cache corpus embeddings (per-document token embeddings)."""
        cache_path = self.cache_dir / "colbert_doc_embs.pkl"
        fingerprint_path = self.cache_dir / "fingerprint.json"

        if not rebuild and cache_path.exists() and fingerprint_path.exists():
            with open(fingerprint_path) as f:
                meta = json.load(f)
            if (meta.get("corpus") == self._corpus_fingerprint() and
                meta.get("config") == self._config_fingerprint()):
                logger.info(f"Loading cached ColBERTv2 embeddings from {cache_path}")
                with open(cache_path, "rb") as f:
                    self.doc_embeddings = pickle.load(f)
                return

        self._load_model()
        texts = [sent.text for sent in self.sentences]

        logger.info(f"Encoding {len(texts)} documents with ColBERTv2")
        self.doc_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # ColBERTv2 returns per-token embeddings
            embs = self.model.docFromText(batch)
            for emb in embs:
                self.doc_embeddings.append(emb.cpu().numpy())

            if (i + batch_size) % 1000 == 0:
                logger.info(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

        with open(cache_path, "wb") as f:
            pickle.dump(self.doc_embeddings, f)
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
        """Retrieve using ColBERTv2 MaxSim scoring."""
        if self.doc_embeddings is None:
            self.encode_corpus()

        indices = self.post_to_indices.get(post_id, [])
        if not indices:
            return []

        self._load_model()

        # Encode query
        query_emb = self.model.queryFromText([query])[0].cpu().numpy()  # [num_query_tokens, dim]

        # Compute MaxSim scores
        scores = []
        for idx in indices:
            doc_emb = self.doc_embeddings[idx]  # [num_doc_tokens, dim]
            # MaxSim: for each query token, find max similarity with any doc token
            sim_matrix = query_emb @ doc_emb.T  # [num_query_tokens, num_doc_tokens]
            maxsim = sim_matrix.max(axis=1).sum()  # Sum of max similarities
            scores.append(maxsim)

        # Rank and return
        ranked = sorted(zip(indices, scores), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ranked:
            sent = self.sentences[idx]
            results.append(RetrievalResult(
                sent_uid=sent.sent_uid,
                text=sent.text,
                score=float(score),
                component_scores={"colbertv2_maxsim": float(score)},
            ))
        return results


class SPLADERetriever(BaseRetriever):
    """SPLADE sparse neural retriever.

    Produces sparse lexical-semantic embeddings with learned term expansion.
    Combines benefits of lexical (BM25) and neural methods.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        device: Optional[str] = None,
    ):
        super().__init__(config, sentences, cache_dir)
        self.device = device or "cuda"
        self.model = None
        self.tokenizer = None
        self.doc_embeddings: Optional[List[Dict[int, float]]] = None

    def _load_model(self):
        """Lazy load SPLADE model."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for SPLADE"
            )

        logger.info(f"Loading SPLADE model: {self.config.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.config.model_id)
        self.model.to(self.device)
        self.model.eval()

    def _encode_splade(self, texts: List[str]) -> List[Dict[int, float]]:
        """Encode texts to SPLADE sparse vectors."""
        import torch

        sparse_vecs = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # SPLADE: log(1 + ReLU(logits)) * attention_mask, then max over sequence
                logits = outputs.logits
                # Apply SPLADE transformation
                splade_vecs = torch.log1p(torch.relu(logits))
                # Max pooling over sequence dimension, masked
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                splade_vecs = splade_vecs * attention_mask
                splade_vecs = splade_vecs.max(dim=1).values  # [batch, vocab_size]

            # Convert to sparse dicts
            for vec in splade_vecs.cpu().numpy():
                nonzero = np.nonzero(vec)[0]
                sparse_dict = {int(idx): float(vec[idx]) for idx in nonzero if vec[idx] > 0}
                sparse_vecs.append(sparse_dict)

        return sparse_vecs

    def encode_corpus(self, rebuild: bool = False) -> None:
        """Encode and cache corpus sparse embeddings."""
        cache_path = self.cache_dir / "splade_sparse.pkl"
        fingerprint_path = self.cache_dir / "fingerprint.json"

        if not rebuild and cache_path.exists() and fingerprint_path.exists():
            with open(fingerprint_path) as f:
                meta = json.load(f)
            if (meta.get("corpus") == self._corpus_fingerprint() and
                meta.get("config") == self._config_fingerprint()):
                logger.info(f"Loading cached SPLADE embeddings from {cache_path}")
                with open(cache_path, "rb") as f:
                    self.doc_embeddings = pickle.load(f)
                return

        self._load_model()
        texts = [sent.text for sent in self.sentences]

        logger.info(f"Encoding {len(texts)} documents with SPLADE")
        self.doc_embeddings = self._encode_splade(texts)

        with open(cache_path, "wb") as f:
            pickle.dump(self.doc_embeddings, f)
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
        """Retrieve using SPLADE sparse dot product."""
        if self.doc_embeddings is None:
            self.encode_corpus()

        indices = self.post_to_indices.get(post_id, [])
        if not indices:
            return []

        self._load_model()

        # Encode query
        query_sparse = self._encode_splade([query])[0]

        # Compute sparse dot product scores
        scores = []
        for idx in indices:
            doc_sparse = self.doc_embeddings[idx]
            # Sparse dot product
            score = sum(query_sparse.get(k, 0) * v for k, v in doc_sparse.items())
            scores.append(score)

        # Rank and return
        ranked = sorted(zip(indices, scores), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ranked:
            sent = self.sentences[idx]
            results.append(RetrievalResult(
                sent_uid=sent.sent_uid,
                text=sent.text,
                score=float(score),
                component_scores={"splade": float(score)},
            ))
        return results


class NVEmbedRetriever(BaseRetriever):
    """NV-Embed-v2 retriever using custom encoding methods.

    NV-Embed-v2 is a Mistral-7B based model that requires special handling:
    - Uses `model._do_encode()` for corpus encoding (no instruction)
    - Uses `model.encode()` for query encoding (with instruction)
    """

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        device: Optional[str] = None,
    ):
        super().__init__(config, sentences, cache_dir)
        self.device = device or "cuda"
        self.embeddings: Optional[np.ndarray] = None
        self.model = None

    def _load_model(self):
        """Lazy load the NV-Embed-v2 model."""
        if self.model is not None:
            return

        # Apply DynamicCache compatibility patch
        _patch_dynamic_cache_compat()

        import torch
        from transformers import AutoModel

        logger.info(f"Loading NV-Embed-v2 model: {self.config.model_id}")

        model_kwargs = {
            "trust_remote_code": True,
        }

        # Determine dtype and quantization
        if self.config.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("  Using 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, using BF16")
                model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("  Using BF16 precision")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("  Using FP16 precision")

        self.model = AutoModel.from_pretrained(self.config.model_id, **model_kwargs)

        # Only move to device if not quantized (quantized models handle this)
        if not self.config.use_4bit:
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("  NV-Embed-v2 loaded successfully!")

    def encode_corpus(self, rebuild: bool = False) -> None:
        """Encode and cache corpus embeddings using _do_encode()."""
        cache_path = self.cache_dir / "embeddings.npy"
        fingerprint_path = self.cache_dir / "fingerprint.json"

        if not rebuild and cache_path.exists() and fingerprint_path.exists():
            with open(fingerprint_path) as f:
                meta = json.load(f)
            if (meta.get("corpus") == self._corpus_fingerprint() and
                meta.get("config") == self._config_fingerprint()):
                logger.info(f"Loading cached NV-Embed-v2 embeddings from {cache_path}")
                self.embeddings = np.load(cache_path)
                return

        self._load_model()
        texts = [sent.text for sent in self.sentences]

        logger.info(f"Encoding {len(texts)} sentences with NV-Embed-v2")

        # NV-Embed-v2 uses _do_encode for batch encoding
        # For passages, no instruction prefix is needed
        corpus_embeddings = self.model._do_encode(
            texts,
            batch_size=self.config.batch_size,
            instruction="",  # No instruction for passages
            max_length=self.config.max_length,
            num_workers=0,
        )

        # Convert to numpy and normalize
        corpus_embeddings = corpus_embeddings.cpu().numpy()
        norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        self.embeddings = corpus_embeddings / (norms + 1e-8)

        np.save(cache_path, self.embeddings)
        with open(fingerprint_path, "w") as f:
            json.dump({
                "corpus": self._corpus_fingerprint(),
                "config": self._config_fingerprint(),
                "model_id": self.config.model_id,
                "num_sentences": len(self.sentences),
            }, f, indent=2)

        logger.info(f"Saved NV-Embed-v2 embeddings to {cache_path}")

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

        # Encode query with instruction
        query_instruction = self.config.query_prefix or "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

        query_emb = self.model.encode(
            [query],
            instruction=query_instruction,
            max_length=self.config.max_length,
        )
        query_emb = query_emb.cpu().numpy()[0]
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Get within-post candidates
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


class RetrieverZoo:
    """Factory and manager for multiple retrievers."""

    # Default retriever configurations - optimized for RTX 5090 32GB
    DEFAULT_RETRIEVERS = [
        RetrieverConfig(
            name="bge-m3",
            model_id="BAAI/bge-m3",
            retriever_type="hybrid",
            max_length=256,
            batch_size=256,  # 4x for RTX 5090 32GB
        ),
        RetrieverConfig(
            name="bge-large-en-v1.5",
            model_id="BAAI/bge-large-en-v1.5",
            retriever_type="dense",
            max_length=512,
            batch_size=256,  # 4x for RTX 5090 32GB
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        RetrieverConfig(
            name="e5-large-v2",
            model_id="intfloat/e5-large-v2",
            retriever_type="dense",
            max_length=512,
            batch_size=256,  # 4x for RTX 5090 32GB
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        RetrieverConfig(
            name="gte-large-en-v1.5",
            model_id="Alibaba-NLP/gte-large-en-v1.5",
            retriever_type="dense",
            max_length=8192,
            batch_size=128,  # 4x for RTX 5090 32GB
            trust_remote_code=True,
        ),
        RetrieverConfig(
            name="bm25",
            model_id="bm25",
            retriever_type="lexical",
        ),
        # SPLADE sparse neural retriever (Tier-0)
        RetrieverConfig(
            name="splade-cocondenser",
            model_id="naver/splade-cocondenser-ensembledistil",
            retriever_type="sparse",
            max_length=256,
            batch_size=128,  # 4x for RTX 5090 32GB
        ),
        # ColBERTv2 late interaction (Tier-2)
        RetrieverConfig(
            name="colbertv2",
            model_id="colbert-ir/colbertv2.0",
            retriever_type="late_interaction",
            max_length=256,
            batch_size=128,  # 4x for RTX 5090 32GB
        ),
        # E5-Mistral instruction-tuned (Tier-1) - BF16
        RetrieverConfig(
            name="e5-mistral-7b",
            model_id="intfloat/e5-mistral-7b-instruct",
            retriever_type="dense",
            max_length=4096,
            batch_size=8,  # 4x for RTX 5090 32GB
            query_prefix="Instruct: Retrieve evidence sentences supporting the criterion.\nQuery: ",
        ),
        # GTE-Qwen2 instruction-tuned (Tier-1) - BF16
        RetrieverConfig(
            name="gte-qwen2-7b",
            model_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
            retriever_type="dense",
            max_length=8192,
            batch_size=8,  # 4x for RTX 5090 32GB
            trust_remote_code=True,
        ),
        # === Models from retriever_lists.md ===
        # Qwen3-Embedding family (instruction-aware, MRL support)
        RetrieverConfig(
            name="qwen3-embed-0.6b",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            retriever_type="dense",
            max_length=8192,
            batch_size=128,  # 4x for RTX 5090 32GB
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            trust_remote_code=True,
        ),
        RetrieverConfig(
            name="qwen3-embed-4b",
            model_id="Qwen/Qwen3-Embedding-4B",
            retriever_type="dense",
            max_length=8192,
            batch_size=32,  # 4x for RTX 5090 32GB
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            trust_remote_code=True,
        ),
        RetrieverConfig(
            name="qwen3-embed-8b",
            model_id="Qwen/Qwen3-Embedding-8B",
            retriever_type="dense",
            max_length=8192,
            batch_size=16,  # 4x for RTX 5090 32GB
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            trust_remote_code=True,
        ),
        # Stella (English-focused mid-size)
        RetrieverConfig(
            name="stella-1.5b",
            model_id="NovaSearch/stella_en_1.5B_v5",
            retriever_type="dense",
            max_length=512,
            batch_size=64,  # 4x for RTX 5090 32GB
            query_prefix="Instruct: Retrieve evidence sentences.\nQuery: ",
        ),
        # Mixedbread (strong permissive baseline)
        RetrieverConfig(
            name="mxbai-embed-large",
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            retriever_type="dense",
            max_length=512,
            batch_size=128,  # 4x for RTX 5090 32GB
        ),
        # UAE-Large (MIT, universal embedder)
        RetrieverConfig(
            name="uae-large",
            model_id="WhereIsAI/UAE-Large-V1",
            retriever_type="dense",
            max_length=512,
            batch_size=128,  # 4x for RTX 5090 32GB
        ),
        # Snowflake Arctic (open, lightweight)
        RetrieverConfig(
            name="arctic-embed-l",
            model_id="Snowflake/snowflake-arctic-embed-l",
            retriever_type="dense",
            max_length=512,
            batch_size=128,  # 4x for RTX 5090 32GB
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        # NVIDIA Llama-Embed (research only, SOTA)
        RetrieverConfig(
            name="llama-embed-8b",
            model_id="nvidia/llama-embed-nemotron-8b",
            retriever_type="dense",
            max_length=512,  # Reduced from 32768 for memory
            batch_size=2,
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            trust_remote_code=True,
            use_4bit=True,  # Enable 4-bit quantization for 8B model
        ),
        # NVIDIA NV-Embed-v2 (Mistral-7B based, research only)
        # Note: Must use BF16 (no 4-bit) - 4-bit causes position embeddings issue
        RetrieverConfig(
            name="nv-embed-v2",
            model_id="nvidia/NV-Embed-v2",
            retriever_type="dense",
            max_length=512,
            batch_size=8,  # BF16 fits in 32GB with batch=8
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            trust_remote_code=True,
            use_4bit=False,  # BF16 required - 4-bit causes position embeddings issue
        ),
        # Salesforce SFR-Embedding-Mistral (research only)
        RetrieverConfig(
            name="sfr-embedding-mistral",
            model_id="Salesforce/SFR-Embedding-Mistral",
            retriever_type="dense",
            max_length=512,
            batch_size=2,
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            use_4bit=True,  # Enable 4-bit quantization for 7B model
        ),
        # Qwen3-8B with 4-bit quantization (retry OOM)
        RetrieverConfig(
            name="qwen3-embed-8b-4bit",
            model_id="Qwen/Qwen3-Embedding-8B",
            retriever_type="dense",
            max_length=512,  # Reduced for memory
            batch_size=2,
            query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
            trust_remote_code=True,
            use_4bit=True,  # Enable 4-bit quantization
        ),
        # === Additional models from comprehensive list ===
        # BGE-en-ICL (in-context learning retriever)
        RetrieverConfig(
            name="bge-en-icl",
            model_id="BAAI/bge-en-icl",
            retriever_type="dense",
            max_length=512,
            batch_size=64,
            trust_remote_code=True,
        ),
        # GTE-Qwen2-1.5B (smaller instruction-tuned)
        RetrieverConfig(
            name="gte-qwen2-1.5b",
            model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            retriever_type="dense",
            max_length=8192,
            batch_size=64,
            trust_remote_code=True,
        ),
        # Stella 400M (smaller variant)
        RetrieverConfig(
            name="stella-400m",
            model_id="NovaSearch/stella_en_400M_v5",
            retriever_type="dense",
            max_length=512,
            batch_size=128,
            query_prefix="Instruct: Retrieve evidence sentences.\nQuery: ",
        ),
        # Snowflake Arctic Embed v1.5 medium
        RetrieverConfig(
            name="arctic-embed-m-v1.5",
            model_id="Snowflake/snowflake-arctic-embed-m-v1.5",
            retriever_type="dense",
            max_length=512,
            batch_size=128,
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        # Snowflake Arctic Embed v2.0 medium
        RetrieverConfig(
            name="arctic-embed-m-v2",
            model_id="Snowflake/snowflake-arctic-embed-m-v2.0",
            retriever_type="dense",
            max_length=512,
            batch_size=128,
            trust_remote_code=True,
        ),
        # Snowflake Arctic Embed v2.0 large
        RetrieverConfig(
            name="arctic-embed-l-v2",
            model_id="Snowflake/snowflake-arctic-embed-l-v2.0",
            retriever_type="dense",
            max_length=512,
            batch_size=64,
            trust_remote_code=True,
        ),
        # Jina Embeddings v3 (multilingual, MRL)
        RetrieverConfig(
            name="jina-embed-v3",
            model_id="jinaai/jina-embeddings-v3",
            retriever_type="dense",
            max_length=8192,
            batch_size=64,
            trust_remote_code=True,
        ),
        # Nomic Embed Text v1.5 (open, dynamic context)
        RetrieverConfig(
            name="nomic-embed-v1.5",
            model_id="nomic-ai/nomic-embed-text-v1.5",
            retriever_type="dense",
            max_length=8192,
            batch_size=128,
            trust_remote_code=True,
            query_prefix="search_query: ",
            passage_prefix="search_document: ",
        ),
        # SPLADE v2 distil (lighter sparse)
        RetrieverConfig(
            name="splade-v2-distil",
            model_id="naver/splade_v2_distil",
            retriever_type="sparse",
            max_length=256,
            batch_size=128,
        ),
        # mxbai ColBERT large (late interaction)
        RetrieverConfig(
            name="mxbai-colbert-large",
            model_id="mixedbread-ai/mxbai-colbert-large-v1",
            retriever_type="late_interaction",
            max_length=512,
            batch_size=64,
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
            # Use NVEmbedRetriever for NV-Embed-v2 (requires special encoding methods)
            if config.name == "nv-embed-v2" or "NV-Embed" in config.model_id:
                return NVEmbedRetriever(
                    config=config,
                    sentences=self.sentences,
                    cache_dir=self.cache_dir,
                    device=self.device,
                )
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
        elif config.retriever_type == "sparse":
            return SPLADERetriever(
                config=config,
                sentences=self.sentences,
                cache_dir=self.cache_dir,
                device=self.device,
            )
        elif config.retriever_type == "late_interaction":
            return ColBERTv2Retriever(
                config=config,
                sentences=self.sentences,
                cache_dir=self.cache_dir,
                device=self.device,
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
