"""Retriever package - embedding models and retrieval interfaces."""

from final_sc_review.retriever.bge_m3 import BgeM3Retriever, BGEM3HybridEncoder
from final_sc_review.retriever.zoo import (
    RetrieverZoo,
    BaseRetriever,
    DenseRetriever,
    BM25Retriever,
    BGEM3ZooRetriever,
    RetrieverConfig,
    RetrievalResult,
)

__all__ = [
    "BgeM3Retriever",
    "BGEM3HybridEncoder",
    "RetrieverZoo",
    "BaseRetriever",
    "DenseRetriever",
    "BM25Retriever",
    "BGEM3ZooRetriever",
    "RetrieverConfig",
    "RetrievalResult",
]
