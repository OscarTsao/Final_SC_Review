#!/usr/bin/env python3
"""
Check retriever coverage statistics.

Analyzes how well the retriever covers gold labels at various K values.

Usage:
    python scripts/retriever/check_retriever_coverage.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


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

    from final_sc_review.data.io import load_sentence_corpus, load_groundtruth, load_criteria
    from final_sc_review.data.splits import split_post_ids
    from final_sc_review.retriever.bge_m3 import BgeM3Retriever

    corpus_path = repo_root / "data" / "groundtruth" / "sentence_corpus.jsonl"
    gt_path = repo_root / "data" / "groundtruth" / "evidence_sentence_groundtruth.csv"
    criteria_path = repo_root / "data" / "DSM5" / "MDD_Criteira.json"
    cache_dir = repo_root / "data" / "cache" / "bge_m3"

    print("="*60)
    print("CHECK RETRIEVER COVERAGE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    sentences = load_sentence_corpus(corpus_path)
    gt_rows = load_groundtruth(gt_path)
    criteria = load_criteria(criteria_path)
    criterion_text = {c.criterion_id: c.text for c in criteria}

    # Build post -> sentences map
    post_sentences = defaultdict(list)
    for s in sentences:
        post_sentences[s.post_id].append(s.sent_uid)

    # Build queries
    query_gold = defaultdict(set)
    for row in gt_rows:
        if row.groundtruth == 1:
            key = (row.post_id, row.criterion_id)
            query_gold[key].add(row.sent_uid)

    all_posts = {row.post_id for row in gt_rows}

    # Statistics about data
    print("\n[2/4] Data statistics...")
    sentences_per_post = [len(v) for v in post_sentences.values()]
    gold_per_query = [len(v) for v in query_gold.values() if v]

    print(f"  Total posts: {len(all_posts)}")
    print(f"  Sentences/post: mean={np.mean(sentences_per_post):.1f}, median={np.median(sentences_per_post):.0f}, max={max(sentences_per_post)}")
    print(f"  Queries with gold: {len([v for v in query_gold.values() if v])}")
    print(f"  Gold/query: mean={np.mean(gold_per_query):.1f}, max={max(gold_per_query)}")

    # Initialize retriever
    print("\n[3/4] Initializing retriever...")
    retriever = BgeM3Retriever(
        sentences=sentences,
        cache_dir=cache_dir,
        rebuild_cache=False,
    )

    # Coverage analysis
    print("\n[4/4] Computing coverage...")
    k_values = [5, 10, 20, 50, 100, 200]
    coverage = {k: [] for k in k_values}
    perfect_coverage = {k: 0 for k in k_values}

    # Sample queries for analysis
    queries_with_gold = [(k, v) for k, v in query_gold.items() if v]
    sample_size = min(200, len(queries_with_gold))

    import random
    random.seed(42)
    sampled = random.sample(queries_with_gold, sample_size)

    for (post_id, criterion_id), gold_uids in sampled:
        query_text = criterion_text.get(criterion_id, criterion_id)

        results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k_retriever=max(k_values),
        )

        ranked_uids = [r[0] for r in results]
        gold_set = set(gold_uids)

        for k in k_values:
            hits = sum(1 for uid in ranked_uids[:k] if uid in gold_set)
            recall = hits / len(gold_uids) if gold_uids else 0.0
            coverage[k].append(recall)

            if hits == len(gold_uids):
                perfect_coverage[k] += 1

    # Report
    print("\n  Coverage (Recall) at K:")
    summary = {"timestamp": datetime.now().isoformat(), "n_queries_sampled": sample_size}

    for k in k_values:
        mean_recall = np.mean(coverage[k])
        perfect_pct = 100 * perfect_coverage[k] / sample_size
        print(f"    @{k}: mean={mean_recall:.3f}, perfect={perfect_pct:.1f}%")
        summary[f"recall@{k}"] = mean_recall
        summary[f"perfect@{k}"] = perfect_pct

    # Ceiling analysis
    print("\n  Ceiling analysis:")
    print(f"    If K >= max gold per query ({max(gold_per_query)}), perfect recall is possible")

    n_posts_lt_10 = sum(1 for v in sentences_per_post if v < 10)
    n_posts_lt_50 = sum(1 for v in sentences_per_post if v < 50)
    print(f"    Posts with <10 sentences: {n_posts_lt_10} ({100*n_posts_lt_10/len(sentences_per_post):.1f}%)")
    print(f"    Posts with <50 sentences: {n_posts_lt_50} ({100*n_posts_lt_50/len(sentences_per_post):.1f}%)")

    # Save results
    output_dir = repo_root / "outputs" / "retriever"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "coverage_stats.json"
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved to: {results_path}")

    print("\n" + "="*60)
    print("[SUCCESS] Coverage check complete")
    print("="*60)


if __name__ == "__main__":
    main()
