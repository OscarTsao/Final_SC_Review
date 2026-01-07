"""Data IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from final_sc_review.data.schemas import Criterion, GroundTruthRow, Post, Sentence
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_posts(posts_path: Path) -> List[Post]:
    """Load posts from a CSV with columns [post_id, text]."""
    df = pd.read_csv(posts_path)
    if "post_id" not in df.columns or "text" not in df.columns:
        raise ValueError("posts CSV must contain columns: post_id, text")
    posts = [Post(post_id=str(row["post_id"]), text=str(row["text"])) for _, row in df.iterrows()]
    logger.info("Loaded %d posts", len(posts))
    return posts


def load_criteria(criteria_path: Path) -> List[Criterion]:
    """Load criteria from MDD_Criteira.json."""
    with open(criteria_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    criteria = []
    for item in data.get("criteria", []):
        cid = item.get("id") or item.get("criterion_id")
        text = item.get("text") or item.get("description") or ""
        if cid is None:
            continue
        criteria.append(Criterion(criterion_id=str(cid), text=str(text)))
    logger.info("Loaded %d criteria", len(criteria))
    return criteria


def load_groundtruth(gt_path: Path) -> List[GroundTruthRow]:
    """Load groundtruth CSV produced by build_groundtruth.py."""
    df = pd.read_csv(gt_path)
    required = {"post_id", "criterion", "sid", "sent_uid", "sentence", "groundtruth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"groundtruth CSV missing columns: {sorted(missing)}")
    rows: List[GroundTruthRow] = []
    for _, row in df.iterrows():
        rows.append(
            GroundTruthRow(
                post_id=str(row["post_id"]),
                criterion_id=str(row["criterion"]),
                sid=int(row["sid"]),
                sent_uid=str(row["sent_uid"]),
                sentence_text=str(row["sentence"]),
                groundtruth=int(row["groundtruth"]),
                evidence_sentence_id=int(row["evidence_sentence_id"]) if not pd.isna(row.get("evidence_sentence_id")) else None,
                evidence_sentence_text=str(row.get("evidence_sentence")) if not pd.isna(row.get("evidence_sentence")) else None,
            )
        )
    logger.info("Loaded %d groundtruth rows", len(rows))
    return rows


def load_sentence_corpus(corpus_path: Path) -> List[Sentence]:
    """Load sentence corpus from JSONL."""
    sentences: List[Sentence] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sentences.append(
                Sentence(
                    post_id=str(obj["post_id"]),
                    sid=int(obj["sid"]),
                    sent_uid=str(obj["sent_uid"]),
                    text=str(obj["text"]),
                )
            )
    logger.info("Loaded %d sentences", len(sentences))
    return sentences
