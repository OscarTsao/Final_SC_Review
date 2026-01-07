"""Retriever + reranker pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from final_sc_review.data.schemas import Sentence
from final_sc_review.retriever.bge_m3 import BgeM3Retriever
from final_sc_review.reranker.jina_v3 import JinaV3Reranker
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    bge_model: str = "BAAI/bge-m3"
    jina_model: str = "jinaai/jina-reranker-v3"
    bge_query_max_length: int = 128
    bge_passage_max_length: int = 256
    bge_use_fp16: bool = True
    bge_batch_size: int = 64
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    colbert_weight: float = 0.0
    fusion_method: str = "weighted_sum"
    score_normalization: str = "none"
    rrf_k: int = 60
    use_sparse: bool = True
    use_colbert: bool = True
    top_k_retriever: int = 50
    top_k_colbert: int = 50  # Deprecated: use top_k_rerank instead
    top_k_rerank: Optional[int] = None  # If None, defaults to top_k_retriever
    top_k_final: int = 20
    reranker_max_length: int = 512
    reranker_chunk_size: int = 64
    reranker_dtype: str = "auto"
    reranker_use_listwise: bool = True
    device: Optional[str] = None

    def get_top_k_rerank(self) -> int:
        """Get effective rerank pool size with backward compatibility."""
        if self.top_k_rerank is not None:
            return self.top_k_rerank
        # Backward compat: use top_k_colbert if set, else top_k_retriever
        return self.top_k_colbert if self.top_k_colbert else self.top_k_retriever


class ThreeStagePipeline:
    """Three-stage pipeline: dense+sparse -> ColBERT -> Jina rerank."""

    def __init__(
        self,
        sentences: List[Sentence],
        cache_dir: Path,
        config: PipelineConfig,
        rebuild_cache: bool = False,
    ):
        self.config = config
        self.retriever = BgeM3Retriever(
            sentences=sentences,
            cache_dir=cache_dir,
            model_name=config.bge_model,
            device=config.device,
            query_max_length=config.bge_query_max_length,
            passage_max_length=config.bge_passage_max_length,
            use_fp16=config.bge_use_fp16,
            batch_size=config.bge_batch_size,
            rebuild_cache=rebuild_cache,
        )
        self.reranker = JinaV3Reranker(
            model_name=config.jina_model,
            device=config.device,
            max_length=config.reranker_max_length,
            listwise_chunk_size=config.reranker_chunk_size,
            dtype=config.reranker_dtype,
            use_listwise=config.reranker_use_listwise,
        )

    def retrieve(self, query: str, post_id: str) -> List[Tuple[str, str, float]]:
        """Retrieve and rerank sentences within a post."""
        retrieved = self.retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k_retriever=self.config.top_k_retriever,
            top_k_colbert=self.config.top_k_retriever,  # Not used for slicing anymore
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            colbert_weight=self.config.colbert_weight,
            use_sparse=self.config.use_sparse,
            use_colbert=self.config.use_colbert,
            fusion_method=self.config.fusion_method,
            score_normalization=self.config.score_normalization,
            rrf_k=self.config.rrf_k,
        )
        if not retrieved:
            return []

        # Use decoupled rerank pool size
        top_k_rerank = self.config.get_top_k_rerank()
        candidates = retrieved[:top_k_rerank]
        candidate_ids = [sid for sid, _, _ in candidates]
        candidate_texts = [text for _, text, _ in candidates]

        rerank_scores = self.reranker.score_pairs(query, candidate_texts)
        return self._align_and_sort(candidate_ids, candidate_texts, rerank_scores, self.config.top_k_final)

    @staticmethod
    def _align_and_sort(
        candidate_ids: List[str],
        candidate_texts: List[str],
        scores: List[float],
        top_k: int,
    ) -> List[Tuple[str, str, float]]:
        if not (len(candidate_ids) == len(candidate_texts) == len(scores)):
            raise ValueError("candidate_ids/texts/scores must be aligned")
        reranked = list(zip(candidate_ids, candidate_texts, scores))
        reranked.sort(key=lambda x: (-x[2], x[0]))
        return reranked[:top_k]
