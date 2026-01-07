"""Constraint helpers for HPO."""

from __future__ import annotations

from typing import Dict


def validate_inference_params(params: Dict, max_candidates: int) -> None:
    top_k_retriever = int(params["top_k_retriever"])
    top_k_final = int(params["top_k_final"])
    if top_k_final > top_k_retriever:
        raise ValueError("top_k_final must be <= top_k_retriever")
    if top_k_retriever > max_candidates:
        raise ValueError("top_k_retriever exceeds cached candidate count")
