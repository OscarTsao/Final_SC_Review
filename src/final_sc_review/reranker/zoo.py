"""Reranker Zoo: Unified interface for multiple reranker models.

Implements reranker models from the research plan:
- SOTA-class: jina-reranker-v3, mxbai-rerank-v2, Qwen3-Reranker
- Baselines: BGE-reranker-v2-m3, bge-reranker-v2.5-gemma2-lightweight

Each reranker scores (query, document) pairs for ranking.

Hardware optimizations:
- BF16 with FP16 fallback (AMP-style)
- Flash-Attention 2 if available
- TF32 for tensor cores
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def setup_hardware_optimizations():
    """Configure hardware optimizations for NVIDIA GPUs."""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True

        logger.debug("Enabled TF32 and cuDNN benchmark mode")


def get_optimal_dtype():
    """Get optimal dtype: BF16 if supported, else FP16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def check_flash_attention_available():
    """Check if Flash Attention 2 is available."""
    try:
        from transformers.utils import is_flash_attn_2_available
        return is_flash_attn_2_available()
    except ImportError:
        return False


# Apply hardware optimizations on module load
setup_hardware_optimizations()


@dataclass
class RerankerConfig:
    """Configuration for a reranker in the zoo."""
    name: str
    model_id: str
    reranker_type: str  # "cross-encoder", "listwise", "lightweight"
    max_length: int = 512
    batch_size: int = 32
    use_fp16: bool = True
    trust_remote_code: bool = False
    # Instruction templates for instruction-aware rerankers
    query_instruction: str = ""
    doc_instruction: str = ""
    # Listwise-specific
    listwise_max_docs: int = 32  # Max docs per forward for listwise models
    # Lightweight model params (BGE)
    cutoff_layers: Optional[int] = None
    compress_ratio: Optional[int] = None


@dataclass
class RerankerResult:
    """Result from reranking a single query."""
    sent_uid: str
    text: str
    score: float
    rank: int


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    def __init__(self, config: RerankerConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the reranker model."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # [(sent_uid, text), ...]
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        """Rerank candidates for a query.

        Args:
            query: Query text
            candidates: List of (sent_uid, text) tuples
            top_k: Return top-k results (None = all)

        Returns:
            Sorted list of RerankerResult
        """
        pass

    def score_pairs(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score query-document pairs. Override for efficiency."""
        results = self.rerank(query, [(f"doc_{i}", d) for i, d in enumerate(documents)])
        # Return scores in original order
        score_map = {r.sent_uid: r.score for r in results}
        return [score_map.get(f"doc_{i}", 0.0) for i in range(len(documents))]

    def rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Tuple[str, str]]]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankerResult]]:
        """Batch rerank multiple queries efficiently.

        Args:
            queries_and_candidates: List of (query, candidates) pairs
                where candidates is List[(sent_uid, text)]
            top_k: Return top-k results per query (None = all)

        Returns:
            List of result lists, one per query
        """
        # Default implementation: process sequentially
        # Subclasses can override for batched processing
        return [
            self.rerank(query, candidates, top_k)
            for query, candidates in queries_and_candidates
        ]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, config: RerankerConfig, device: Optional[str] = None):
        super().__init__(config, device)

    def load_model(self) -> None:
        if self.model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading cross-encoder: {self.config.model_id}")

        # Use optimal dtype (BF16 if supported, else FP16)
        dtype = get_optimal_dtype() if self.config.use_fp16 else torch.float32

        # Build model kwargs for automodel - don't force flash attention
        # as many models don't support it; let the model choose automatically
        model_kwargs = {"torch_dtype": dtype}

        self.model = CrossEncoder(
            self.config.model_id,
            max_length=self.config.max_length,
            device=self.device,
            trust_remote_code=self.config.trust_remote_code,
            model_kwargs=model_kwargs,
        )

        # Set model to inference mode (disables dropout)
        self.model.model.train(False)

        # Disabled torch.compile - causes overhead with dynamic shapes
        # For consistent batch inference, enable manually if needed

        logger.info(f"  Loaded {self.config.name} with dtype={dtype}")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        if self.model is None:
            self.load_model()

        if not candidates:
            return []

        # Apply instruction if configured
        if self.config.query_instruction:
            query = self.config.query_instruction + query

        # Build pairs
        pairs = []
        for sent_uid, text in candidates:
            doc = self.config.doc_instruction + text if self.config.doc_instruction else text
            pairs.append((query, doc))

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Build results
        results = []
        for i, (sent_uid, text) in enumerate(candidates):
            results.append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(scores[i]),
                rank=0,  # Will be set after sorting
            ))

        # Sort by score descending
        results.sort(key=lambda x: -x.score)

        # Set ranks
        for i, r in enumerate(results):
            r.rank = i + 1

        if top_k:
            results = results[:top_k]

        return results

    def rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Tuple[str, str]]]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankerResult]]:
        """Batch rerank multiple queries efficiently.

        Combines all pairs across queries into one batch for GPU efficiency.
        """
        if self.model is None:
            self.load_model()

        if not queries_and_candidates:
            return []

        # Build all pairs and track query boundaries
        all_pairs = []
        query_boundaries = []  # (start_idx, end_idx, query_idx)
        current_idx = 0
        candidate_info = []  # Store (sent_uid, text) for rebuilding results

        for query_idx, (query, candidates) in enumerate(queries_and_candidates):
            if not candidates:
                query_boundaries.append((current_idx, current_idx, query_idx))
                continue

            # Apply instruction if configured
            q = self.config.query_instruction + query if self.config.query_instruction else query

            start_idx = current_idx
            for sent_uid, text in candidates:
                doc = self.config.doc_instruction + text if self.config.doc_instruction else text
                all_pairs.append((q, doc))
                candidate_info.append((sent_uid, text, query_idx))
                current_idx += 1

            query_boundaries.append((start_idx, current_idx, query_idx))

        if not all_pairs:
            return [[] for _ in queries_and_candidates]

        # Score all pairs in one batch
        all_scores = self.model.predict(
            all_pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Rebuild results per query
        all_results = [[] for _ in queries_and_candidates]

        for i, (sent_uid, text, query_idx) in enumerate(candidate_info):
            all_results[query_idx].append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(all_scores[i]),
                rank=0,
            ))

        # Sort and rank each query's results
        for results in all_results:
            results.sort(key=lambda x: -x.score)
            for i, r in enumerate(results):
                r.rank = i + 1
            if top_k:
                results[:] = results[:top_k]

        return all_results


class ListwiseReranker(BaseReranker):
    """Listwise reranker (e.g., jina-reranker-v3) that scores multiple docs at once."""

    def __init__(self, config: RerankerConfig, device: Optional[str] = None):
        super().__init__(config, device)
        self.tokenizer = None

    def load_model(self) -> None:
        if self.model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading listwise reranker: {self.config.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        # Ensure pad_token is set (required for batched inference)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use optimal dtype (BF16 if supported, else FP16)
        dtype = get_optimal_dtype() if self.config.use_fp16 else torch.float32

        # Let the model choose optimal attention implementation automatically
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )
        # Set pad_token_id on model config (required for batched inference)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        # Set to inference mode (disables dropout)
        self.model.train(False)

        logger.info(f"  Loaded {self.config.name} with dtype={dtype}")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        if self.model is None:
            self.load_model()

        if not candidates:
            return []

        # Apply instruction
        if self.config.query_instruction:
            query = self.config.query_instruction + query

        # Score in batches respecting listwise_max_docs
        all_scores = []
        batch_size = min(self.config.listwise_max_docs, len(candidates))

        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            batch_texts = [text for _, text in batch_candidates]

            # Tokenize
            inputs = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.float().cpu().numpy()
                # Handle different output shapes
                if logits.ndim == 1:
                    scores = logits.tolist()
                elif logits.ndim == 2 and logits.shape[1] == 1:
                    scores = logits.squeeze(-1).tolist()
                else:
                    # Multi-class or unexpected shape - take first column
                    scores = logits[:, 0].tolist()

            # Ensure scores is a flat list
            if not isinstance(scores, list):
                scores = [float(scores)]

            all_scores.extend(scores)

        # Build results
        results = []
        for i, (sent_uid, text) in enumerate(candidates):
            results.append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(all_scores[i]),
                rank=0,
            ))

        # Sort and rank
        results.sort(key=lambda x: -x.score)
        for i, r in enumerate(results):
            r.rank = i + 1

        if top_k:
            results = results[:top_k]

        return results

    def rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Tuple[str, str]]]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankerResult]]:
        """Batch rerank multiple queries efficiently.

        Combines pairs across queries for better GPU utilization.
        """
        if self.model is None:
            self.load_model()

        if not queries_and_candidates:
            return []

        # Build all pairs
        all_queries = []
        all_texts = []
        candidate_info = []  # (sent_uid, text, query_idx)

        for query_idx, (query, candidates) in enumerate(queries_and_candidates):
            if not candidates:
                continue

            q = self.config.query_instruction + query if self.config.query_instruction else query

            for sent_uid, text in candidates:
                all_queries.append(q)
                all_texts.append(text)
                candidate_info.append((sent_uid, text, query_idx))

        if not all_queries:
            return [[] for _ in queries_and_candidates]

        # Score all pairs in batches
        all_scores = []
        batch_size = self.config.batch_size

        for i in range(0, len(all_queries), batch_size):
            batch_queries = all_queries[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.float().cpu().numpy()
                if logits.ndim == 1:
                    scores = logits.tolist()
                elif logits.ndim == 2 and logits.shape[1] == 1:
                    scores = logits.squeeze(-1).tolist()
                else:
                    scores = logits[:, 0].tolist()

                if not isinstance(scores, list):
                    scores = [float(scores)]

                all_scores.extend(scores)

        # Rebuild results per query
        all_results = [[] for _ in queries_and_candidates]

        for i, (sent_uid, text, query_idx) in enumerate(candidate_info):
            all_results[query_idx].append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(all_scores[i]),
                rank=0,
            ))

        # Sort and rank each query's results
        for results in all_results:
            results.sort(key=lambda x: -x.score)
            for i, r in enumerate(results):
                r.rank = i + 1
            if top_k:
                results[:] = results[:top_k]

        return all_results


class BGELightweightReranker(BaseReranker):
    """BGE lightweight reranker with layer cutoff and token compression."""

    def __init__(self, config: RerankerConfig, device: Optional[str] = None):
        super().__init__(config, device)

    def load_model(self) -> None:
        if self.model is not None:
            return

        try:
            from FlagEmbedding import FlagLLMReranker, LayerWiseFlagLLMReranker
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required. Install with: pip install FlagEmbedding"
            )

        logger.info(f"Loading BGE reranker: {self.config.model_id}")

        # BGE rerankers use FP16 by default; BF16 not directly supported by FlagEmbedding
        dtype_str = "fp16" if self.config.use_fp16 else "fp32"

        if self.config.cutoff_layers:
            self.model = LayerWiseFlagLLMReranker(
                self.config.model_id,
                use_fp16=self.config.use_fp16,
                device=self.device,
            )
        else:
            self.model = FlagLLMReranker(
                self.config.model_id,
                use_fp16=self.config.use_fp16,
                device=self.device,
            )

        logger.info(f"  Loaded {self.config.name} with dtype={dtype_str}")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        if self.model is None:
            self.load_model()

        if not candidates:
            return []

        # Build pairs for FlagEmbedding
        pairs = [[query, text] for _, text in candidates]

        # Compute scores
        if self.config.cutoff_layers:
            scores = self.model.compute_score(
                pairs,
                cutoff_layers=[self.config.cutoff_layers],
                batch_size=self.config.batch_size,
            )
            # LayerWise returns nested list
            if isinstance(scores[0], list):
                scores = [s[0] for s in scores]
        else:
            scores = self.model.compute_score(
                pairs,
                batch_size=self.config.batch_size,
            )

        # Build results
        results = []
        for i, (sent_uid, text) in enumerate(candidates):
            results.append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(scores[i]),
                rank=0,
            ))

        # Sort and rank
        results.sort(key=lambda x: -x.score)
        for i, r in enumerate(results):
            r.rank = i + 1

        if top_k:
            results = results[:top_k]

        return results


class RerankerZoo:
    """Factory and manager for multiple rerankers."""

    # Default reranker configurations from research plan
    DEFAULT_RERANKERS = [
        # === SOTA-class open rerankers ===
        # jina-reranker-v3 (strong BEIR, Qwen3-based)
        RerankerConfig(
            name="jina-reranker-v3",
            model_id="jinaai/jina-reranker-v3",
            reranker_type="listwise",  # Uses custom tokenizer handling
            max_length=1024,
            batch_size=128,  # 4x for RTX 5090 32GB
            listwise_max_docs=32,
            trust_remote_code=True,
        ),
        # jina-reranker-v2 (multilingual, strong)
        RerankerConfig(
            name="jina-reranker-v2",
            model_id="jinaai/jina-reranker-v2-base-multilingual",
            reranker_type="cross-encoder",
            max_length=1024,
            batch_size=192,  # 4x for RTX 5090 32GB
            trust_remote_code=True,
        ),
        # mxbai-rerank-base-v2 (fast strong open baseline)
        RerankerConfig(
            name="mxbai-rerank-base-v2",
            model_id="mixedbread-ai/mxbai-rerank-base-v2",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=256,  # 4x for RTX 5090 32GB
        ),
        # mxbai-rerank-large-v2 (quality push)
        RerankerConfig(
            name="mxbai-rerank-large-v2",
            model_id="mixedbread-ai/mxbai-rerank-large-v2",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=128,  # 4x for RTX 5090 32GB
        ),
        # mxbai-rerank-base-v1 (legacy)
        RerankerConfig(
            name="mxbai-rerank-base-v1",
            model_id="mixedbread-ai/mxbai-rerank-base-v1",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=256,  # 4x for RTX 5090 32GB
        ),
        # mxbai-rerank-large-v1 (legacy)
        RerankerConfig(
            name="mxbai-rerank-large-v1",
            model_id="mixedbread-ai/mxbai-rerank-large-v1",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=128,  # 4x for RTX 5090 32GB
        ),
        # Qwen3-Reranker-0.6B (instruction-aware, Qwen3-based)
        RerankerConfig(
            name="qwen3-reranker-0.6b",
            model_id="Qwen/Qwen3-Reranker-0.6B",
            reranker_type="listwise",  # Uses custom tokenizer handling
            max_length=1024,
            batch_size=64,  # 4x for RTX 5090 32GB
            listwise_max_docs=32,
            trust_remote_code=True,
            query_instruction="Instruct: Given a criterion, determine if the passage provides supporting evidence.\nQuery: ",
        ),
        # Qwen3-Reranker-4B (instruction-aware, Qwen3-based)
        RerankerConfig(
            name="qwen3-reranker-4b",
            model_id="Qwen/Qwen3-Reranker-4B",
            reranker_type="listwise",  # Uses custom tokenizer handling
            max_length=1024,
            batch_size=32,  # 4x for RTX 5090 32GB
            listwise_max_docs=16,
            trust_remote_code=True,
            query_instruction="Instruct: Given a criterion, determine if the passage provides supporting evidence.\nQuery: ",
        ),
        # === Strong baselines ===
        # BGE-reranker-v2-m3
        RerankerConfig(
            name="bge-reranker-v2-m3",
            model_id="BAAI/bge-reranker-v2-m3",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=256,  # 4x for RTX 5090 32GB
        ),
        # BGE-reranker-v2.5-gemma2-lightweight
        RerankerConfig(
            name="bge-reranker-gemma2-lightweight",
            model_id="BAAI/bge-reranker-v2.5-gemma2-lightweight",
            reranker_type="lightweight",
            max_length=512,
            batch_size=64,  # 4x for RTX 5090 32GB
            cutoff_layers=28,  # Use layer 28 for speed/quality tradeoff
        ),
        # MS MARCO MiniLM (fast baseline)
        RerankerConfig(
            name="ms-marco-minilm",
            model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=512,  # 4x for RTX 5090 32GB (small model)
        ),
        # === Additional rerankers from comprehensive list ===
        # BGE Reranker Large (original)
        RerankerConfig(
            name="bge-reranker-large",
            model_id="BAAI/bge-reranker-large",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=128,
        ),
        # BGE Reranker v2 Gemma
        RerankerConfig(
            name="bge-reranker-v2-gemma",
            model_id="BAAI/bge-reranker-v2-gemma",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=32,
            trust_remote_code=True,
        ),
        # BGE Reranker v2 MiniCPM Layerwise
        RerankerConfig(
            name="bge-reranker-v2-minicpm",
            model_id="BAAI/bge-reranker-v2-minicpm-layerwise",
            reranker_type="lightweight",
            max_length=512,
            batch_size=32,
            cutoff_layers=28,
            trust_remote_code=True,
        ),
        # Jina Reranker m0 (smaller/faster)
        RerankerConfig(
            name="jina-reranker-m0",
            model_id="jinaai/jina-reranker-m0",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=128,
            trust_remote_code=True,
        ),
        # GTE Multilingual Reranker Base
        RerankerConfig(
            name="gte-reranker-base",
            model_id="Alibaba-NLP/gte-multilingual-reranker-base",
            reranker_type="cross-encoder",
            max_length=512,
            batch_size=128,
            trust_remote_code=True,
        ),
        # RankZephyr 7B (LLM listwise reranker)
        RerankerConfig(
            name="rank-zephyr-7b",
            model_id="castorini/rank_zephyr_7b_v1_full",
            reranker_type="listwise",
            max_length=4096,
            batch_size=1,  # LLM-based, memory intensive
            listwise_max_docs=20,
            trust_remote_code=True,
        ),
        # Qwen3 Reranker 8B
        RerankerConfig(
            name="qwen3-reranker-8b",
            model_id="Qwen/Qwen3-Reranker-8B",
            reranker_type="listwise",
            max_length=8192,
            batch_size=8,
            listwise_max_docs=32,
            trust_remote_code=True,
        ),
    ]

    def __init__(
        self,
        configs: Optional[List[RerankerConfig]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = configs or self.DEFAULT_RERANKERS
        self._rerankers: Dict[str, BaseReranker] = {}

    def get_reranker(self, name: str) -> BaseReranker:
        """Get or create a reranker by name."""
        if name in self._rerankers:
            return self._rerankers[name]

        config = None
        for c in self.configs:
            if c.name == name:
                config = c
                break

        if config is None:
            raise ValueError(f"Unknown reranker: {name}. Available: {self.list_rerankers()}")

        reranker = self._create_reranker(config)
        self._rerankers[name] = reranker
        return reranker

    def _create_reranker(self, config: RerankerConfig) -> BaseReranker:
        """Create a reranker from config."""
        if config.reranker_type == "cross-encoder":
            return CrossEncoderReranker(config, self.device)
        elif config.reranker_type == "listwise":
            return ListwiseReranker(config, self.device)
        elif config.reranker_type == "lightweight":
            return BGELightweightReranker(config, self.device)
        else:
            raise ValueError(f"Unknown reranker type: {config.reranker_type}")

    def list_rerankers(self) -> List[str]:
        """List available reranker names."""
        return [c.name for c in self.configs]

    def get_config(self, name: str) -> RerankerConfig:
        """Get config for a reranker."""
        for c in self.configs:
            if c.name == name:
                return c
        raise ValueError(f"Unknown reranker: {name}")
