"""Hashing utilities for cache validation."""

from __future__ import annotations

import hashlib
from typing import Iterable, Tuple


def corpus_fingerprint(records: Iterable[Tuple[str, int, str]]) -> str:
    """Compute a stable SHA256 fingerprint from ordered sentence records.

    Args:
        records: Iterable of (post_id, sid, sentence_text) in deterministic order.
    """
    hasher = hashlib.sha256()
    for post_id, sid, text in records:
        line = f"{post_id}\t{sid}\t{text}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()
