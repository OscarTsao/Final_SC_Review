#!/usr/bin/env python3
"""Post length distribution analysis (Phase 2.2)."""

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

def compute_post_length_stats():
    groundtruth = load_groundtruth(Path("data/groundtruth/evidence_sentence_groundtruth.csv"))
    
    dev_tune_ids = set(load_split("dev_tune"))
    dev_select_ids = set(load_split("dev_select"))
    test_ids = set(load_split("test"))
    
    all_groups = defaultdict(set)
    for row in groundtruth:
        key = (row.post_id, row.criterion_id)
        all_groups[key].add(row.sent_uid)
    
    def stats_for_split(post_ids):
        counts = [len(s) for (p, c), s in all_groups.items() if p in post_ids]
        if not counts: return {}
        return {
            "n_queries": len(counts),
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
        }
    
    results = {
        "dev_tune": stats_for_split(dev_tune_ids),
        "dev_select": stats_for_split(dev_select_ids),
        "test": stats_for_split(test_ids),
    }
    
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    with open("outputs/analysis/post_length_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("POST LENGTH STATS:")
    for split, s in results.items():
        print(f"  {split}: n={s['n_queries']}, mean={s['mean']:.1f}, median={s['median']:.1f}, p95={s['p95']:.1f}")
    
    return results

if __name__ == "__main__":
    compute_post_length_stats()
