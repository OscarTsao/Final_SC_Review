#!/usr/bin/env python3
"""Generate comprehensive dataset profile as required by MAXOUT Plan Section 3.

Outputs:
- outputs/data_profile/data_profile.json
- outputs/data_profile/data_profile.md

Required stats:
- distribution of sentences per post: mean/median/p90/p95/max
- distribution of #gold evidence sentences per (post, criterion)
- prevalence of empty groups (No-Evidence rate)
- label noise indicators (duplicate sentences, conflicting labels)
- split sanity (no post_id overlap)
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_groundtruth, load_sentence_corpus


def load_split(split_name: str, splits_dir: str = "data/splits") -> List[str]:
    """Load a split by name."""
    path = Path(splits_dir) / f"{split_name}_post_ids.json"
    with open(path) as f:
        return json.load(f)


def compute_stats(values: List[float]) -> Dict:
    """Compute distribution statistics."""
    if not values:
        return {"count": 0}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def generate_data_profile():
    print("=" * 60)
    print("DATA PROFILING (Section 3)")
    print("=" * 60)

    output_dir = Path("outputs/data_profile")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    groundtruth = load_groundtruth(Path("data/groundtruth/evidence_sentence_groundtruth.csv"))
    corpus = load_sentence_corpus(Path("data/groundtruth/sentence_corpus.jsonl"))

    # Load splits
    try:
        dev_tune_ids = set(load_split("dev_tune"))
        dev_select_ids = set(load_split("dev_select"))
        test_ids = set(load_split("test"))
    except FileNotFoundError:
        print("WARNING: Splits not found, running without split analysis")
        dev_tune_ids = dev_select_ids = test_ids = set()

    # Basic counts
    all_post_ids = {row.post_id for row in groundtruth}
    all_criteria = {row.criterion_id for row in groundtruth}

    print(f"Total posts: {len(all_post_ids)}")
    print(f"Total criteria: {len(all_criteria)}")
    print(f"Total groundtruth rows: {len(groundtruth)}")
    print(f"Total sentences in corpus: {len(corpus)}")

    # 1. Distribution of sentences per post
    sents_per_post = defaultdict(set)
    for s in corpus:
        sents_per_post[s.post_id].add(s.sent_uid)
    sents_per_post_counts = [len(v) for v in sents_per_post.values()]
    sents_per_post_stats = compute_stats(sents_per_post_counts)

    print(f"\nSentences per post:")
    print(f"  Mean: {sents_per_post_stats['mean']:.1f}")
    print(f"  Median: {sents_per_post_stats['median']:.1f}")
    print(f"  P90: {sents_per_post_stats['p90']:.1f}")
    print(f"  P95: {sents_per_post_stats['p95']:.1f}")
    print(f"  Max: {sents_per_post_stats['max']:.0f}")

    # 2. Distribution of gold evidence per (post, criterion)
    gold_per_group = defaultdict(int)
    all_groups = set()
    for row in groundtruth:
        key = (row.post_id, row.criterion_id)
        all_groups.add(key)
        if row.groundtruth == 1:
            gold_per_group[key] += 1

    gold_counts = [gold_per_group.get(k, 0) for k in all_groups]
    gold_stats = compute_stats(gold_counts)

    # No-Evidence rate
    empty_groups = sum(1 for c in gold_counts if c == 0)
    positive_groups = sum(1 for c in gold_counts if c > 0)
    no_evidence_rate = empty_groups / len(all_groups) if all_groups else 0

    print(f"\nGold evidence per (post, criterion):")
    print(f"  Total queries: {len(all_groups)}")
    print(f"  Positive queries: {positive_groups} ({positive_groups/len(all_groups):.1%})")
    print(f"  Empty queries (No-Evidence): {empty_groups} ({no_evidence_rate:.1%})")
    print(f"  Mean gold: {gold_stats['mean']:.2f}")
    print(f"  Max gold: {gold_stats['max']:.0f}")

    # 3. Label noise indicators
    # Check for duplicate sentences within post
    sent_texts_per_post = defaultdict(list)
    for row in groundtruth:
        sent_texts_per_post[row.post_id].append((row.sentence_text, row.sent_uid))

    duplicate_sentences = 0
    for post_id, texts in sent_texts_per_post.items():
        unique_texts = set(t[0] for t in texts)
        if len(unique_texts) < len(texts):
            duplicate_sentences += len(texts) - len(unique_texts)

    # Check for conflicting labels (same sentence, same criterion, different labels)
    sent_labels = defaultdict(set)
    for row in groundtruth:
        key = (row.sent_uid, row.criterion_id)
        sent_labels[key].add(row.groundtruth)
    conflicting_labels = sum(1 for labels in sent_labels.values() if len(labels) > 1)

    print(f"\nLabel noise indicators:")
    print(f"  Duplicate sentences: {duplicate_sentences}")
    print(f"  Conflicting labels: {conflicting_labels}")

    # 4. Split sanity check
    split_sanity = {}
    if dev_tune_ids and dev_select_ids and test_ids:
        overlap_tune_select = len(dev_tune_ids & dev_select_ids)
        overlap_tune_test = len(dev_tune_ids & test_ids)
        overlap_select_test = len(dev_select_ids & test_ids)

        split_sanity = {
            "dev_tune_posts": len(dev_tune_ids),
            "dev_select_posts": len(dev_select_ids),
            "test_posts": len(test_ids),
            "overlap_tune_select": overlap_tune_select,
            "overlap_tune_test": overlap_tune_test,
            "overlap_select_test": overlap_select_test,
            "all_disjoint": (overlap_tune_select + overlap_tune_test + overlap_select_test) == 0,
        }

        print(f"\nSplit sanity:")
        print(f"  dev_tune: {split_sanity['dev_tune_posts']} posts")
        print(f"  dev_select: {split_sanity['dev_select_posts']} posts")
        print(f"  test: {split_sanity['test_posts']} posts")
        print(f"  All disjoint: {split_sanity['all_disjoint']}")

    # 5. Per-split statistics
    def stats_for_split(post_ids: Set[str]) -> Dict:
        """Compute stats for a specific split."""
        queries = [(p, c) for (p, c) in all_groups if p in post_ids]
        counts = [gold_per_group.get((p, c), 0) for (p, c) in queries]
        n_positive = sum(1 for c in counts if c > 0)
        n_empty = sum(1 for c in counts if c == 0)
        return {
            "n_queries": len(queries),
            "n_positive": n_positive,
            "n_empty": n_empty,
            "empty_rate": n_empty / len(queries) if queries else 0,
            "gold_stats": compute_stats(counts),
        }

    per_split_stats = {}
    if dev_tune_ids:
        per_split_stats["dev_tune"] = stats_for_split(dev_tune_ids)
    if dev_select_ids:
        per_split_stats["dev_select"] = stats_for_split(dev_select_ids)
    if test_ids:
        per_split_stats["test"] = stats_for_split(test_ids)

    # 6. K policy recommendation
    # Based on actual sentence distribution, recommend appropriate K values
    k_policy = {
        "recommended_ks": [1, 3, 5, 10],
        "max_meaningful_k": int(sents_per_post_stats["p95"]),
        "median_candidates": int(sents_per_post_stats["median"]),
        "note": f"K=200 meaningless when p95={sents_per_post_stats['p95']:.0f} sentences/post",
    }

    print(f"\nK Policy recommendation:")
    print(f"  Recommended Ks: {k_policy['recommended_ks']}")
    print(f"  Max meaningful K (p95): {k_policy['max_meaningful_k']}")
    print(f"  Note: {k_policy['note']}")

    # Compile results
    profile = {
        "timestamp": datetime.now().isoformat(),
        "basic_counts": {
            "n_posts": len(all_post_ids),
            "n_criteria": len(all_criteria),
            "n_groundtruth_rows": len(groundtruth),
            "n_sentences_corpus": len(corpus),
            "n_queries": len(all_groups),
        },
        "sentences_per_post": sents_per_post_stats,
        "gold_per_query": {
            "stats": gold_stats,
            "n_positive": positive_groups,
            "n_empty": empty_groups,
            "no_evidence_rate": no_evidence_rate,
        },
        "label_noise": {
            "duplicate_sentences": duplicate_sentences,
            "conflicting_labels": conflicting_labels,
        },
        "split_sanity": split_sanity,
        "per_split_stats": per_split_stats,
        "k_policy": k_policy,
    }

    # Save JSON
    json_path = output_dir / "data_profile.json"
    with open(json_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate markdown report
    md_content = f"""# Dataset Profile Report

**Generated:** {profile['timestamp']}

## Basic Counts

| Metric | Value |
|--------|-------|
| Posts | {profile['basic_counts']['n_posts']} |
| Criteria | {profile['basic_counts']['n_criteria']} |
| Groundtruth rows | {profile['basic_counts']['n_groundtruth_rows']} |
| Sentences in corpus | {profile['basic_counts']['n_sentences_corpus']} |
| Total queries (post, criterion) | {profile['basic_counts']['n_queries']} |

## Sentences per Post Distribution

| Statistic | Value |
|-----------|-------|
| Mean | {sents_per_post_stats['mean']:.1f} |
| Median | {sents_per_post_stats['median']:.1f} |
| P90 | {sents_per_post_stats['p90']:.1f} |
| P95 | {sents_per_post_stats['p95']:.1f} |
| Max | {sents_per_post_stats['max']:.0f} |

## No-Evidence Prevalence

| Metric | Value |
|--------|-------|
| Positive queries | {positive_groups} ({positive_groups/len(all_groups)*100:.1f}%) |
| Empty queries (No-Evidence) | {empty_groups} ({no_evidence_rate*100:.1f}%) |
| Mean gold per query | {gold_stats['mean']:.2f} |
| Max gold per query | {gold_stats['max']:.0f} |

**Critical insight:** {no_evidence_rate*100:.1f}% of queries have NO evidence sentences. This extreme class imbalance requires:
- Empty detection P/R/F1 metrics
- False evidence rate tracking
- Risk-coverage analysis

## Label Noise Indicators

| Issue | Count |
|-------|-------|
| Duplicate sentences | {duplicate_sentences} |
| Conflicting labels | {conflicting_labels} |

## Split Sanity

| Split | Posts | Queries | Positive | Empty | Empty Rate |
|-------|-------|---------|----------|-------|------------|
"""

    for split_name, stats in per_split_stats.items():
        md_content += f"| {split_name} | - | {stats['n_queries']} | {stats['n_positive']} | {stats['n_empty']} | {stats['empty_rate']*100:.1f}% |\n"

    md_content += f"""
**Disjoint check:** {split_sanity.get('all_disjoint', 'N/A')}

## K Policy Recommendation

Based on the sentence distribution (p95 = {sents_per_post_stats['p95']:.0f} sentences/post):

- **Recommended Ks:** {k_policy['recommended_ks']}
- **Max meaningful K:** {k_policy['max_meaningful_k']}
- **Note:** K=200 is meaningless for this dataset

## Key Takeaways

1. **Extreme No-Evidence prevalence ({no_evidence_rate*100:.1f}%)** - Most queries have no gold evidence
2. **Small candidate pools** - Median {sents_per_post_stats['median']:.0f} sentences/post means K>20 is rarely useful
3. **Must track:** Empty detection metrics, false evidence rate, selection micro P/R/F1
"""

    md_path = output_dir / "data_profile.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved: {md_path}")

    print("\n" + "=" * 60)
    print("DATA PROFILING COMPLETE")
    print("=" * 60)

    return profile


if __name__ == "__main__":
    generate_data_profile()
