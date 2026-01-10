#!/usr/bin/env python3
"""
Check gold label alignment with sentence corpus.

Verifies:
1. All gold sentence UIDs exist in the corpus
2. Post IDs in groundtruth match corpus
3. No orphaned gold labels

Usage:
    python scripts/retriever/check_gold_alignment.py
"""
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def main():
    repo_root = get_repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"

    print("="*60)
    print("CHECK GOLD ALIGNMENT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # Load data
    print("\n[1/4] Loading sentence corpus...")
    sentences = load_sentence_corpus(corpus_path)
    corpus_uids = {s.sent_uid for s in sentences}
    corpus_posts = {s.post_id for s in sentences}
    print(f"  Corpus sentences: {len(sentences)}")
    print(f"  Unique posts: {len(corpus_posts)}")

    print("\n[2/4] Loading groundtruth...")
    gt_rows = load_groundtruth(gt_path)
    print(f"  Groundtruth rows: {len(gt_rows)}")

    # Check alignment
    print("\n[3/4] Checking alignment...")
    missing_uids = []
    missing_posts = []
    positive_count = 0
    post_criterion_stats = defaultdict(lambda: {"total": 0, "positive": 0})

    for row in gt_rows:
        if row.sent_uid not in corpus_uids:
            missing_uids.append(row.sent_uid)

        if row.post_id not in corpus_posts:
            missing_posts.append(row.post_id)

        key = (row.post_id, row.criterion_id)
        post_criterion_stats[key]["total"] += 1
        if row.groundtruth == 1:
            positive_count += 1
            post_criterion_stats[key]["positive"] += 1

    # Report results
    print("\n[4/4] Results:")
    print(f"  Total groundtruth rows: {len(gt_rows)}")
    print(f"  Positive labels: {positive_count}")
    print(f"  Negative labels: {len(gt_rows) - positive_count}")
    print(f"  Unique (post, criterion) pairs: {len(post_criterion_stats)}")

    if missing_uids:
        print(f"\n  [ERROR] Missing UIDs in corpus: {len(missing_uids)}")
        for uid in missing_uids[:5]:
            print(f"    - {uid}")
        if len(missing_uids) > 5:
            print(f"    ... and {len(missing_uids) - 5} more")
        sys.exit(1)
    else:
        print("\n  [OK] All gold UIDs found in corpus")

    if missing_posts:
        print(f"\n  [ERROR] Missing posts in corpus: {len(set(missing_posts))}")
        sys.exit(1)
    else:
        print("  [OK] All gold posts found in corpus")

    # Positive label distribution
    positive_queries = sum(1 for k, v in post_criterion_stats.items() if v["positive"] > 0)
    empty_queries = len(post_criterion_stats) - positive_queries

    print(f"\n  Query statistics:")
    print(f"    Queries with positives: {positive_queries} ({100*positive_queries/len(post_criterion_stats):.1f}%)")
    print(f"    Queries with no positives: {empty_queries} ({100*empty_queries/len(post_criterion_stats):.1f}%)")

    print("\n" + "="*60)
    print("[PASS] Gold alignment check passed")
    print("="*60)


if __name__ == "__main__":
    main()
