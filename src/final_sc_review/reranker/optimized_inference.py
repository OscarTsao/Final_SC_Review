"""Optimized reranker inference for maximum GPU utilization.

Key optimizations:
1. Multi-query batch processing - Process N queries in a single forward pass
2. Prefetching dataloader - Prepare next batch while GPU is processing
3. Length bucketing - Group sequences by similar lengths to minimize padding
4. Score caching - Cache results for HPO speedup

Usage:
    from final_sc_review.reranker.optimized_inference import BatchReranker

    batch_reranker = BatchReranker(zoo, reranker_name)
    all_results = batch_reranker.rerank_batch(queries_and_candidates)
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from final_sc_review.reranker.zoo import (
    BaseReranker,
    RerankerResult,
    RerankerZoo,
    get_optimal_dtype,
)
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryCandidates:
    """A query with its candidate documents for reranking."""
    query_key: str  # Unique identifier for this query
    query_text: str
    candidates: List[Tuple[str, str]]  # [(sent_uid, text), ...]
    gold_uids: Optional[set] = None  # For evaluation
    has_evidence: bool = True


@dataclass
class BatchedInput:
    """A batch of (query, document) pairs ready for model inference."""
    query_keys: List[str]  # Which query each pair belongs to
    pair_indices: List[int]  # Index within the query's candidates
    input_texts: List[Tuple[str, str]]  # (query, doc) pairs
    sent_uids: List[str]  # Document identifiers


class RerankerDataset(Dataset):
    """Dataset for reranker inference with efficient batching."""

    def __init__(
        self,
        queries: List[QueryCandidates],
        max_candidates_per_query: int = 100,
    ):
        """
        Args:
            queries: List of queries with their candidates
            max_candidates_per_query: Maximum candidates to consider per query
        """
        self.queries = queries
        self.max_candidates = max_candidates_per_query

        # Flatten to (query_key, pair_index, query_text, doc_text, sent_uid)
        self.pairs = []
        for q in queries:
            for i, (sent_uid, text) in enumerate(q.candidates[:max_candidates_per_query]):
                self.pairs.append((q.query_key, i, q.query_text, text, sent_uid))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, int, str, str, str]:
        return self.pairs[idx]


class LengthBucketedSampler(Sampler):
    """Sampler that groups sequences by similar lengths to minimize padding."""

    def __init__(
        self,
        dataset: RerankerDataset,
        batch_size: int,
        bucket_boundaries: List[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Number of pairs per batch
            bucket_boundaries: Length boundaries for buckets (default: [64, 128, 256, 512])
            shuffle: Whether to shuffle within buckets
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries or [64, 128, 256, 512, 1024]
        self.shuffle = shuffle

        # Compute approximate lengths and bucket indices
        self._bucket_indices()

    def _bucket_indices(self):
        """Group sample indices by sequence length bucket."""
        self.buckets = defaultdict(list)

        for idx, (_, _, query_text, doc_text, _) in enumerate(self.dataset.pairs):
            # Approximate token count (4 chars per token is rough estimate)
            approx_len = (len(query_text) + len(doc_text)) // 4

            # Find bucket
            bucket_id = 0
            for boundary in self.bucket_boundaries:
                if approx_len < boundary:
                    break
                bucket_id += 1

            self.buckets[bucket_id].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices from same-length buckets."""
        # Shuffle within buckets if requested
        if self.shuffle:
            for bucket_id in self.buckets:
                np.random.shuffle(self.buckets[bucket_id])

        # Create batches from each bucket
        all_batches = []
        for bucket_id, indices in self.buckets.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                all_batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.buckets.values()
        )


class ScoreCache:
    """LRU-style cache for reranker scores to speed up HPO.

    Uses JSON for safe serialization (scores are just floats).
    """

    def __init__(self, cache_dir: Path = None, max_memory_mb: int = 1000):
        """
        Args:
            cache_dir: Directory for persistent cache (None = memory only)
            max_memory_mb: Maximum memory cache size in MB
        """
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        self._memory_cache: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(
        self,
        reranker_name: str,
        query_text: str,
        max_length: int,
    ) -> str:
        """Create cache key for a reranker + query combo."""
        content = f"{reranker_name}:{max_length}:{query_text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self,
        reranker_name: str,
        query_text: str,
        max_length: int,
    ) -> Optional[Dict[str, float]]:
        """Get cached scores for a query. Returns {sent_uid: score} or None."""
        key = self._make_key(reranker_name, query_text, max_length)

        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]

        # Check disk cache (using JSON for safety)
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        scores = json.load(f)
                    with self._lock:
                        self._memory_cache[key] = scores
                    return scores
                except (json.JSONDecodeError, IOError):
                    # Corrupted cache file, ignore
                    pass

        return None

    def put(
        self,
        reranker_name: str,
        query_text: str,
        max_length: int,
        scores: Dict[str, float],
    ) -> None:
        """Cache scores for a query."""
        key = self._make_key(reranker_name, query_text, max_length)

        with self._lock:
            self._memory_cache[key] = scores

        # Write to disk cache using JSON
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump(scores, f)
            except IOError:
                pass  # Ignore write errors


class PrefetchingLoader:
    """Wrapper that prefetches the next batch while current is processing."""

    def __init__(self, dataloader: DataLoader, prefetch_count: int = 2):
        """
        Args:
            dataloader: The base DataLoader
            prefetch_count: Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.prefetch_count = prefetch_count
        self._queue: Queue = None
        self._thread: threading.Thread = None
        self._stop_event: threading.Event = None

    def _prefetch_worker(self):
        """Worker thread that prefetches batches."""
        try:
            for batch in self.dataloader:
                if self._stop_event.is_set():
                    break
                self._queue.put(batch)
            self._queue.put(None)  # Signal end
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        """Iterate with prefetching."""
        self._queue = Queue(maxsize=self.prefetch_count)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()

        while True:
            batch = self._queue.get()
            if batch is None:
                break
            if isinstance(batch, Exception):
                raise batch
            yield batch

        self._thread.join()

    def __len__(self):
        return len(self.dataloader)


class BatchReranker:
    """Optimized batch reranker for maximum GPU utilization."""

    def __init__(
        self,
        zoo: RerankerZoo,
        reranker_name: str,
        cache_dir: Optional[Path] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        """
        Args:
            zoo: RerankerZoo instance
            reranker_name: Name of reranker to use
            cache_dir: Directory for score caching (None = no caching)
            num_workers: Number of data loading workers
            prefetch_factor: Batches to prefetch per worker
        """
        self.zoo = zoo
        self.reranker_name = reranker_name
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Initialize reranker
        self.reranker = zoo.get_reranker(reranker_name)
        self.config = zoo.get_config(reranker_name)

        # Score cache
        self.cache = ScoreCache(cache_dir) if cache_dir else None

    def rerank_batch(
        self,
        queries: List[QueryCandidates],
        top_k: int = 10,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_bucketing: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, List[RerankerResult]]:
        """Rerank multiple queries efficiently.

        Args:
            queries: List of queries with their candidates
            top_k: Return top-k results per query
            max_length: Override max sequence length
            batch_size: Override batch size
            use_bucketing: Use length-based bucketing
            show_progress: Show progress bar

        Returns:
            Dict mapping query_key to sorted RerankerResult list
        """
        from tqdm import tqdm

        # Load model if needed
        self.reranker.load_model()

        max_length = max_length or self.config.max_length
        batch_size = batch_size or self.config.batch_size

        # Check cache for each query
        results: Dict[str, Dict[str, float]] = {}
        queries_to_process = []

        for q in queries:
            if self.cache:
                cached = self.cache.get(self.reranker_name, q.query_text, max_length)
                if cached:
                    results[q.query_key] = cached
                    continue
            queries_to_process.append(q)

        if not queries_to_process:
            # All cached
            return self._scores_to_results(results, queries, top_k)

        # Create dataset and dataloader
        dataset = RerankerDataset(queries_to_process)

        if use_bucketing:
            sampler = LengthBucketedSampler(dataset, batch_size, shuffle=False)
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=min(self.num_workers, 4),
                pin_memory=True,
                collate_fn=self._collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(self.num_workers, 4),
                pin_memory=True,
                collate_fn=self._collate_fn,
            )

        # Prefetching loader
        loader = PrefetchingLoader(dataloader, prefetch_count=self.prefetch_factor)

        # Accumulate scores per query
        query_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        iterator = tqdm(loader, desc=f"Reranking ({self.reranker_name})", disable=not show_progress)

        for batch in iterator:
            query_keys, pair_indices, pairs, sent_uids = batch

            # Score batch
            scores = self._score_batch(pairs, max_length)

            # Accumulate scores
            for qk, sent_uid, score in zip(query_keys, sent_uids, scores):
                query_scores[qk][sent_uid] = score

        # Merge with cached results
        for qk, scores in query_scores.items():
            results[qk] = scores

            # Update cache
            if self.cache:
                # Find query text for this key
                for q in queries_to_process:
                    if q.query_key == qk:
                        self.cache.put(self.reranker_name, q.query_text, max_length, scores)
                        break

        return self._scores_to_results(results, queries, top_k)

    def _collate_fn(
        self,
        batch: List[Tuple[str, int, str, str, str]],
    ) -> Tuple[List[str], List[int], List[Tuple[str, str]], List[str]]:
        """Collate batch of pairs."""
        query_keys = [item[0] for item in batch]
        pair_indices = [item[1] for item in batch]
        pairs = [(item[2], item[3]) for item in batch]
        sent_uids = [item[4] for item in batch]
        return query_keys, pair_indices, pairs, sent_uids

    def _score_batch(
        self,
        pairs: List[Tuple[str, str]],
        max_length: int,
    ) -> List[float]:
        """Score a batch of (query, doc) pairs."""
        if hasattr(self.reranker, 'model') and hasattr(self.reranker.model, 'predict'):
            # CrossEncoder from sentence-transformers
            # Temporarily override max_length if needed
            original_max_length = self.reranker.model.max_length
            self.reranker.model.max_length = max_length

            scores = self.reranker.model.predict(
                pairs,
                batch_size=len(pairs),  # Already batched
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            self.reranker.model.max_length = original_max_length
            return scores.tolist()

        elif hasattr(self.reranker, 'tokenizer') and hasattr(self.reranker, 'model'):
            # Transformers-based model (listwise)
            queries = [p[0] for p in pairs]
            docs = [p[1] for p in pairs]

            inputs = self.reranker.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.reranker.device)

            with torch.no_grad():
                outputs = self.reranker.model(**inputs)
                logits = outputs.logits.float().cpu().numpy()

                if logits.ndim == 1:
                    scores = logits.tolist()
                elif logits.ndim == 2 and logits.shape[1] == 1:
                    scores = logits.squeeze(-1).tolist()
                else:
                    scores = logits[:, 0].tolist()

            return scores

        else:
            # Fallback to sequential scoring
            return [
                self.reranker.rerank(q, [(f"doc", d)])[0].score
                for q, d in pairs
            ]

    def _scores_to_results(
        self,
        scores: Dict[str, Dict[str, float]],
        queries: List[QueryCandidates],
        top_k: int,
    ) -> Dict[str, List[RerankerResult]]:
        """Convert score dicts to sorted RerankerResult lists."""
        results = {}

        # Build sent_uid -> text mapping
        uid_to_text = {}
        for q in queries:
            for sent_uid, text in q.candidates:
                uid_to_text[sent_uid] = text

        for qk, score_dict in scores.items():
            # Sort by score descending
            sorted_items = sorted(score_dict.items(), key=lambda x: -x[1])[:top_k]

            query_results = []
            for rank, (sent_uid, score) in enumerate(sorted_items, 1):
                query_results.append(RerankerResult(
                    sent_uid=sent_uid,
                    text=uid_to_text.get(sent_uid, ""),
                    score=score,
                    rank=rank,
                ))
            results[qk] = query_results

        return results


def benchmark_batch_reranker(
    zoo: RerankerZoo,
    reranker_name: str,
    queries: List[QueryCandidates],
    batch_sizes: List[int] = [16, 32, 64, 128],
) -> Dict[str, Any]:
    """Benchmark batch reranker with different batch sizes.

    Returns timing and throughput metrics.
    """
    import time

    results = {}

    for bs in batch_sizes:
        batch_reranker = BatchReranker(zoo, reranker_name)

        start = time.time()
        _ = batch_reranker.rerank_batch(
            queries,
            batch_size=bs,
            show_progress=False,
        )
        elapsed = time.time() - start

        total_pairs = sum(len(q.candidates) for q in queries)
        throughput = total_pairs / elapsed

        results[f"batch_size_{bs}"] = {
            "elapsed_seconds": elapsed,
            "total_pairs": total_pairs,
            "throughput_pairs_per_sec": throughput,
        }

        logger.info(f"Batch size {bs}: {throughput:.1f} pairs/sec, {elapsed:.2f}s total")

    return results
