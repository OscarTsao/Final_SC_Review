#!/usr/bin/env python3
"""Build sentence corpus JSONL from posts using canonical splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from final_sc_review.utils.text import split_sentences
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/groundtruth/sentence_corpus.jsonl",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    posts_path = data_dir / "redsm5" / "redsm5_posts.csv"
    annotations_path = data_dir / "redsm5" / "redsm5_annotations.csv"
    output_path = Path(args.output)

    for path in [posts_path, annotations_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(posts_path)
    if "post_id" not in df.columns or "text" not in df.columns:
        raise ValueError("posts CSV must contain columns: post_id, text")

    evidence_by_post: Dict[str, Set[int]] = {}
    evidence_text_by_post_idx: Dict[str, Dict[int, str]] = {}
    annotations_df = pd.read_csv(annotations_path)
    for _, row in annotations_df.iterrows():
        post_id = str(row.get("post_id", ""))
        sentence_id_str = str(row.get("sentence_id", ""))
        parts = sentence_id_str.split("_")
        if len(parts) < 2:
            continue
        try:
            sentence_idx = int(parts[-1])
        except ValueError:
            continue
        evidence_by_post.setdefault(post_id, set()).add(sentence_idx)
        evidence_text_by_post_idx.setdefault(post_id, {})[sentence_idx] = str(
            row.get("sentence_text", "")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            post_id = str(row["post_id"])
            text = str(row["text"])
            sentences = split_sentences(text)
            max_idx = max(len(sentences) - 1, max(evidence_by_post.get(post_id, {-1})))
            for sid in range(max_idx + 1):
                if sid < len(sentences):
                    sent = sentences[sid]
                else:
                    sent = evidence_text_by_post_idx.get(post_id, {}).get(sid, "")
                sent_uid = f"{post_id}_{sid}"
                obj = {
                    "post_id": post_id,
                    "sid": sid,
                    "sent_uid": sent_uid,
                    "text": sent,
                }
                f.write(json.dumps(obj, ensure_ascii=True) + "\n")
                count += 1
    logger.info("Saved %d sentences to %s", count, output_path)


if __name__ == "__main__":
    main()
