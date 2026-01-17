"""Zoo-based pipeline using retriever and reranker zoos.

This pipeline supports dynamic selection of retrievers and rerankers from the zoo,
enabling use of the best HPO-discovered model combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from final_sc_review.data.io import Sentence
from final_sc_review.retriever.zoo import RetrieverZoo, BaseRetriever
from final_sc_review.reranker.zoo import RerankerZoo, BaseReranker
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ZooPipelineConfig:
    """Configuration for zoo-based pipeline."""
    # Model selection
    retriever_name: str = "nv-embed-v2"
    reranker_name: str = "jina-reranker-v3"

    # Retrieval parameters
    top_k_retriever: int = 24
    top_k_final: int = 10

    # Fusion parameters (for hybrid retrievers)
    use_sparse: bool = False
    use_colbert: bool = False
    dense_weight: float = 1.0
    sparse_weight: float = 0.0
    colbert_weight: float = 0.0
    fusion_method: str = "rrf"  # "weighted_sum" or "rrf"
    score_normalization: str = "none"
    rrf_k: int = 60

    # Device
    device: Optional[str] = None


class ZooPipeline:
    """Pipeline using retriever and reranker from zoo."""

    def __init__(
        self,
        sentences: List[Sentence],
        cache_dir: Path,
        config: ZooPipelineConfig,
    ):
        self.sentences = sentences
        self.cache_dir = cache_dir
        self.config = config

        # Build lookup maps
        self.sent_uid_to_idx: Dict[str, int] = {
            s.sent_uid: i for i, s in enumerate(sentences)
        }
        self.post_to_indices: Dict[str, List[int]] = {}
        for i, s in enumerate(sentences):
            self.post_to_indices.setdefault(s.post_id, []).append(i)

        # Initialize zoos
        self._retriever_zoo = RetrieverZoo(
            sentences=sentences,
            cache_dir=cache_dir,
            device=config.device,
        )
        self._reranker_zoo = RerankerZoo()

        # Lazy-loaded models
        self._retriever: Optional[BaseRetriever] = None
        self._reranker: Optional[BaseReranker] = None

    @property
    def retriever(self) -> BaseRetriever:
        """Get or load the retriever."""
        if self._retriever is None:
            logger.info(f"Loading retriever: {self.config.retriever_name}")
            self._retriever = self._retriever_zoo.get_retriever(self.config.retriever_name)
        return self._retriever

    @property
    def reranker(self) -> BaseReranker:
        """Get or load the reranker."""
        if self._reranker is None:
            logger.info(f"Loading reranker: {self.config.reranker_name}")
            self._reranker = self._reranker_zoo.get_reranker(self.config.reranker_name)
        return self._reranker

    def retrieve(
        self,
        query: str,
        post_id: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Retrieve and rerank sentences for a query.

        Args:
            query: The query text (criterion text)
            post_id: The post ID to retrieve from
            top_k: Number of results to return (default: config.top_k_final)

        Returns:
            List of (sent_uid, sentence_text, score) tuples
        """
        top_k = top_k or self.config.top_k_final

        # Get candidate indices for this post
        candidate_indices = self.post_to_indices.get(post_id, [])
        if not candidate_indices:
            logger.warning(f"No sentences found for post_id: {post_id}")
            return []

        # Stage 1: Retrieval
        retrieval_results = self.retriever.retrieve(
            query=query,
            post_id=post_id,
            top_k=self.config.top_k_retriever,
        )

        if not retrieval_results:
            return []

        # Extract candidate info for reranking
        candidate_uids = [r.sent_uid for r in retrieval_results]
        candidate_texts = [r.text for r in retrieval_results]

        # Stage 2: Reranking
        rerank_results = self.reranker.rerank(
            query=query,
            documents=candidate_texts,
            doc_ids=candidate_uids,
            top_k=top_k,
        )

        # Format results
        results = [
            (r.sent_uid, r.text, r.score)
            for r in rerank_results
        ]

        return results

    def retrieve_batch(
        self,
        queries: List[Tuple[str, str]],  # List of (query, post_id)
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[str, str, float]]]:
        """Batch retrieve for multiple queries.

        Args:
            queries: List of (query_text, post_id) tuples
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        return [
            self.retrieve(query, post_id, top_k)
            for query, post_id in queries
        ]


def load_zoo_pipeline_from_config(config_path: Path) -> ZooPipeline:
    """Load zoo pipeline from YAML config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configured ZooPipeline instance
    """
    import yaml
    from final_sc_review.data.io import load_sentence_corpus

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sentences = load_sentence_corpus(Path(cfg["paths"]["sentence_corpus"]))
    cache_dir = Path(cfg["paths"]["cache_dir"])

    # Extract model names from config
    models_cfg = cfg.get("models", {})
    retriever_cfg = cfg.get("retriever", {})

    # Support both new and legacy config formats
    retriever_name = models_cfg.get("retriever_name", "bge-m3")
    reranker_name = models_cfg.get("reranker_name", "jina-reranker-v3")

    pipeline_config = ZooPipelineConfig(
        retriever_name=retriever_name,
        reranker_name=reranker_name,
        top_k_retriever=retriever_cfg.get("top_k_retriever", 24),
        top_k_final=retriever_cfg.get("top_k_final", 10),
        use_sparse=retriever_cfg.get("use_sparse", False),
        use_colbert=retriever_cfg.get("use_colbert", False),
        dense_weight=retriever_cfg.get("dense_weight", 1.0),
        sparse_weight=retriever_cfg.get("sparse_weight", 0.0),
        colbert_weight=retriever_cfg.get("colbert_weight", 0.0),
        fusion_method=retriever_cfg.get("fusion_method", "rrf"),
        score_normalization=retriever_cfg.get("score_normalization", "none"),
        rrf_k=retriever_cfg.get("rrf_k", 60),
        device=cfg.get("device"),
    )

    return ZooPipeline(
        sentences=sentences,
        cache_dir=cache_dir,
        config=pipeline_config,
    )
