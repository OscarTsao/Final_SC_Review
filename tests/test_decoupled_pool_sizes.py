"""Test decoupled retrieval pool and rerank pool sizes."""

import pytest

from final_sc_review.pipeline.three_stage import PipelineConfig


def test_get_top_k_rerank_explicit():
    """When top_k_rerank is set explicitly, use it."""
    config = PipelineConfig(
        top_k_retriever=100,
        top_k_colbert=50,
        top_k_rerank=20,
    )
    assert config.get_top_k_rerank() == 20


def test_get_top_k_rerank_backward_compat_colbert():
    """When top_k_rerank is None, fall back to top_k_colbert."""
    config = PipelineConfig(
        top_k_retriever=100,
        top_k_colbert=50,
        top_k_rerank=None,
    )
    assert config.get_top_k_rerank() == 50


def test_get_top_k_rerank_backward_compat_retriever():
    """When both top_k_rerank and top_k_colbert are default, use top_k_retriever."""
    config = PipelineConfig(
        top_k_retriever=100,
    )
    # Default top_k_colbert is 50, so it should return that
    assert config.get_top_k_rerank() == 50


def test_decoupled_pool_constraint():
    """Verify that top_k_rerank can be smaller than top_k_retriever."""
    config = PipelineConfig(
        top_k_retriever=128,
        top_k_rerank=32,
        top_k_final=10,
    )
    assert config.top_k_retriever == 128
    assert config.get_top_k_rerank() == 32
    assert config.top_k_final == 10
    # Verify constraint: final <= rerank <= retriever
    assert config.top_k_final <= config.get_top_k_rerank() <= config.top_k_retriever
