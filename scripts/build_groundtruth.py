#!/usr/bin/env python3
"""Build sentence-level evidence groundtruth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from final_sc_review.constants import CRITERION_TO_SYMPTOM
from final_sc_review.utils.text import split_sentences
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_criteria(criteria_path: Path) -> List[Dict]:
    with open(criteria_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("criteria", [])


def load_annotations(annotations_path: Path):
    df = pd.read_csv(annotations_path)
    evidence_sentences: Set[Tuple[str, str, int]] = set()
    annotated_posts: Set[str] = set()
    evidence_by_post: Dict[str, Set[int]] = {}
    evidence_text_lookup: Dict[Tuple[str, str, int], str] = {}
    evidence_text_by_post_idx: Dict[str, Dict[int, str]] = {}

    for _, row in df.iterrows():
        post_id = str(row["post_id"])
        annotated_posts.add(post_id)
        if int(row.get("status", 0)) != 1:
            continue
        sentence_id_str = str(row.get("sentence_id", ""))
        parts = sentence_id_str.split("_")
        if len(parts) < 2:
            continue
        try:
            sentence_idx = int(parts[-1])
        except ValueError:
            continue
        symptom = str(row.get("DSM5_symptom", ""))
        evidence_sentences.add((post_id, symptom, sentence_idx))
        evidence_by_post.setdefault(post_id, set()).add(sentence_idx)
        sent_text = str(row.get("sentence_text", ""))
        evidence_text_lookup[(post_id, symptom, sentence_idx)] = sent_text
        evidence_text_by_post_idx.setdefault(post_id, {})[sentence_idx] = sent_text

    return (
        annotated_posts,
        evidence_sentences,
        evidence_by_post,
        evidence_text_lookup,
        evidence_text_by_post_idx,
    )


def generate_groundtruth(
    posts_df: pd.DataFrame,
    criteria: List[Dict],
    annotated_posts: Set[str],
    evidence_sentences: Set[Tuple[str, str, int]],
    evidence_by_post: Dict[str, Set[int]],
    evidence_text_lookup: Dict[Tuple[str, str, int], str],
    evidence_text_by_post_idx: Dict[str, Dict[int, str]],
) -> pd.DataFrame:
    rows = []
    annotated_posts_df = posts_df[posts_df["post_id"].astype(str).isin(annotated_posts)]
    for _, post_row in annotated_posts_df.iterrows():
        post_id = str(post_row["post_id"])
        post_text = str(post_row["text"])
        sentences = split_sentences(post_text)
        max_needed_idx = max(
            len(sentences) - 1,
            max(evidence_by_post.get(post_id, {-1})),
        )
        for sid in range(max_needed_idx + 1):
            if sid < len(sentences):
                sentence_text = sentences[sid]
            else:
                sentence_text = evidence_text_by_post_idx.get(post_id, {}).get(sid, "")
            sent_uid = f"{post_id}_{sid}"
            for criterion in criteria:
                criterion_id = criterion.get("id") or criterion.get("criterion_id")
                if criterion_id is None:
                    continue
                criterion_id = str(criterion_id)
                symptom_name = CRITERION_TO_SYMPTOM.get(criterion_id, criterion_id)
                is_evidence = (post_id, symptom_name, sid) in evidence_sentences
                groundtruth = 1 if is_evidence else 0
                evidence_sent_text = None
                evidence_sent_id = None
                if is_evidence:
                    evidence_sent_text = evidence_text_lookup.get((post_id, symptom_name, sid))
                    evidence_sent_id = sid
                rows.append(
                    {
                        "post_id": post_id,
                        "criterion": criterion_id,
                        "sid": sid,
                        "sent_uid": sent_uid,
                        "sentence": sentence_text,
                        "groundtruth": groundtruth,
                        "evidence_sentence_id": evidence_sent_id,
                        "evidence_sentence": evidence_sent_text,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/groundtruth/evidence_sentence_groundtruth.csv",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    posts_path = data_dir / "redsm5" / "redsm5_posts.csv"
    criteria_path = data_dir / "DSM5" / "MDD_Criteira.json"
    annotations_path = data_dir / "redsm5" / "redsm5_annotations.csv"
    output_path = Path(args.output)

    for path in [posts_path, criteria_path, annotations_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    posts_df = pd.read_csv(posts_path)
    criteria = load_criteria(criteria_path)
    (
        annotated_posts,
        evidence_sentences,
        evidence_by_post,
        evidence_text_lookup,
        evidence_text_by_post_idx,
    ) = load_annotations(annotations_path)

    gt_df = generate_groundtruth(
        posts_df,
        criteria,
        annotated_posts,
        evidence_sentences,
        evidence_by_post,
        evidence_text_lookup,
        evidence_text_by_post_idx,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(output_path, index=False)
    logger.info("Saved groundtruth to %s (%d rows)", output_path, len(gt_df))


if __name__ == "__main__":
    main()
