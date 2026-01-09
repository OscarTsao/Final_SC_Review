#!/usr/bin/env python3
"""Gold count distribution analysis (Phase 2.2)."""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from final_sc_review.data.io import load_groundtruth
from create_splits import load_split

def compute_gold_count_stats():
    groundtruth = load_groundtruth(Path("data/groundtruth/evidence_sentence_groundtruth.csv"))
    
    dev_tune_ids = set(load_split("dev_tune"))
    dev_select_ids = set(load_split("dev_select"))
    test_ids = set(load_split("test"))
    
    # Count gold per (post_id, criterion_id)
    gold_counts = defaultdict(int)
    for row in groundtruth:
        key = (row.post_id, row.criterion_id)
        if row.groundtruth == 1:
            gold_counts[key] += 1
    
    # All queries (including empty)
    all_queries = set()
    for row in groundtruth:
        all_queries.add((row.post_id, row.criterion_id))
    
    def stats_for_split(post_ids):
        queries = [(p, c) for (p, c) in all_queries if p in post_ids]
        counts = [gold_counts.get((p, c), 0) for (p, c) in queries]
        n_positive = sum(1 for c in counts if c > 0)
        n_empty = sum(1 for c in counts if c == 0)
        return {
            "n_queries": len(counts),
            "n_positive": n_positive,
            "n_empty": n_empty,
            "empty_rate": n_empty / len(counts) if counts else 0,
            "mean_gold": float(np.mean(counts)) if counts else 0,
            "max_gold": int(max(counts)) if counts else 0,
        }
    
    results = {
        "dev_tune": stats_for_split(dev_tune_ids),
        "dev_select": stats_for_split(dev_select_ids),
        "test": stats_for_split(test_ids),
    }
    
    with open("outputs/analysis/gold_count_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("GOLD COUNT STATS:")
    for split, s in results.items():
        print(f"  {split}: n={s['n_queries']}, positive={s['n_positive']}, empty={s['n_empty']} ({s['empty_rate']:.1%})")
    
    return results

if __name__ == "__main__":
    compute_gold_count_stats()
