"""Jina-v3 reranker wrapper with listwise rerank support."""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class JinaV3Reranker:
    """Listwise reranker with fallback pairwise scoring."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        listwise_chunk_size: int = 64,
        dtype: str = "auto",
        use_listwise: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.listwise_chunk_size = listwise_chunk_size
        self.use_listwise = use_listwise

        logger.info("Loading reranker %s on %s", model_name, device)
        self.tokenizer = None
        self.model = None
        torch_dtype = _resolve_dtype(dtype)

        if use_listwise:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            if not hasattr(self.model, "rerank"):
                logger.warning("Model has no rerank() API; falling back to pairwise scoring")
                self.model = None
                self.use_listwise = False
            else:
                self._rerank_params = _get_rerank_params(self.model.rerank)
        else:
            self._rerank_params = {}

        if not self.use_listwise:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            if self.model.config.pad_token_id is None and self.tokenizer is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.model is None:
            raise ValueError("Failed to load reranker model")
        self.model.to(device)
        self.model.eval()

    def score_pairs(self, query: str, sentences: List[str]) -> List[float]:
        """Score a list of sentences against a query."""
        if not sentences:
            return []
        if self.use_listwise:
            return self._score_listwise(query, sentences)
        scores: List[float] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
                if logits.shape[-1] == 1:
                    batch_scores = torch.sigmoid(logits.squeeze(-1))
                else:
                    batch_scores = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(batch_scores.detach().cpu().float().tolist())
        return scores

    def _score_listwise(self, query: str, sentences: List[str]) -> List[float]:
        scores = [0.0] * len(sentences)
        for start in range(0, len(sentences), self.listwise_chunk_size):
            chunk = sentences[start : start + self.listwise_chunk_size]
            kwargs = _build_rerank_kwargs(self._rerank_params, len(chunk))
            with torch.inference_mode():
                results = self.model.rerank(query=query, documents=chunk, **kwargs)
            for item in results:
                idx, score = _extract_rerank_result(item)
                if idx is None:
                    continue
                scores[start + idx] = float(score)
        return scores


def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
    key = (dtype or "auto").lower()
    if key == "auto":
        return None
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    if key == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_rerank_params(rerank_fn) -> Dict[str, inspect.Parameter]:
    try:
        sig = inspect.signature(rerank_fn)
    except (TypeError, ValueError):
        return {}
    return dict(sig.parameters)


def _build_rerank_kwargs(params: Dict[str, inspect.Parameter], chunk_size: int) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if "top_n" in params:
        kwargs["top_n"] = chunk_size
    elif "top_k" in params:
        kwargs["top_k"] = chunk_size
    if "return_documents" in params:
        kwargs["return_documents"] = False
    if "return_document" in params:
        kwargs["return_document"] = False
    if "return_docs" in params:
        kwargs["return_docs"] = False
    return kwargs


def _extract_rerank_result(item) -> tuple[Optional[int], float]:
    if isinstance(item, dict):
        idx = item.get("index")
        score = item.get("relevance_score", item.get("score", 0.0))
        return (idx, score)
    idx = getattr(item, "index", None)
    score = getattr(item, "relevance_score", getattr(item, "score", 0.0))
    return (idx, score)
