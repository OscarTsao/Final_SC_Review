"""Data schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Post:
    post_id: str
    text: str


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    text: str


@dataclass(frozen=True)
class Sentence:
    post_id: str
    sid: int
    sent_uid: str
    text: str


@dataclass(frozen=True)
class GroundTruthRow:
    post_id: str
    criterion_id: str
    sid: int
    sent_uid: str
    sentence_text: str
    groundtruth: int
    evidence_sentence_id: Optional[int] = None
    evidence_sentence_text: Optional[str] = None
