#!/usr/bin/env python3
"""Run fine-tuning HPO for all retriever+reranker combinations.

This script runs fine-tuning HPO for ALL combinations from the model zoo.
Uses RetrieverZoo and RerankerZoo to get the complete list of models.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import csv

# Import zoo classes to get all available models
from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.reranker.zoo import RerankerZoo

# Get all retrievers and rerankers from zoo
def get_all_retrievers():
    """Get all retriever names from the zoo."""
    return [c.name for c in RetrieverZoo.DEFAULT_RETRIEVERS]

def get_all_rerankers():
    """Get all reranker names from the zoo."""
    return [c.name for c in RerankerZoo.DEFAULT_RERANKERS]

# Priority tiers for smart scheduling
TIER_1_RETRIEVERS = [
    "bge-m3",           # Hybrid, strong baseline
    "nv-embed-v2",      # SOTA dense
    "qwen3-embed-0.6b", # Fast instruction-aware
    "qwen3-embed-4b",   # Quality instruction-aware
    "llama-embed-8b",   # Large dense
]

TIER_1_RERANKERS = [
    "jina-reranker-v3",    # SOTA listwise
    "bge-reranker-v2-m3",  # Strong baseline
    "jina-reranker-v2",    # Multilingual
    "mxbai-rerank-base-v2", # Fast baseline
]

# Models to skip (known to fail or duplicate)
SKIP_RETRIEVERS = [
    "qwen3-embed-8b-4bit",  # Duplicate of qwen3-embed-8b
    "colbertv2",            # Requires special ColBERT installation
    "splade-cocondenser",   # SPLADE requires special handling
    "splade-v2-distil",     # SPLADE requires special handling
    "mxbai-colbert-large",  # ColBERT variant requires special handling
]

SKIP_RERANKERS = [
    "bge-reranker-gemma2-lightweight",  # Requires FlagEmbedding
    "bge-reranker-v2-minicpm",          # Requires FlagEmbedding
    "rank-zephyr-7b",                   # Very large LLM reranker
]

def check_existing_results(output_dir: Path) -> set:
    """Check which combinations already have results."""
    results_csv = output_dir / "hpo_finetuning_results.csv"
    completed = set()

    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("best_ndcg_at_10"):
                    completed.add((row["retriever"], row["reranker"]))

    return completed


def run_combo(retriever: str, reranker: str, n_trials: int, output_dir: Path) -> dict:
    """Run fine-tuning HPO for a single combination."""

    combo_dir = output_dir / f"{retriever}_{reranker}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Build cache
    print(f"\n{'='*60}")
    print(f"[CACHE] Building retrieval cache for {retriever}...")
    print(f"{'='*60}")

    cache_cmd = [
        sys.executable, "scripts/hpo_finetuning_combo.py",
        "--retriever", retriever,
        "--reranker", reranker,
        "--stage", "cache",
        "--output_dir", str(output_dir),
        "--include_no_evidence",
    ]

    result = subprocess.run(cache_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Cache building failed for {retriever}")
        return {"error": "cache_failed"}

    # Stage 2: Run HPO training
    print(f"\n{'='*60}")
    print(f"[TRAIN] Running HPO for {retriever} + {reranker}...")
    print(f"{'='*60}")

    train_cmd = [
        sys.executable, "scripts/hpo_finetuning_combo.py",
        "--retriever", retriever,
        "--reranker", reranker,
        "--stage", "train",
        "--n_trials", str(n_trials),
        "--output_dir", str(output_dir),
        "--include_no_evidence",
    ]

    start_time = datetime.now()
    result = subprocess.run(train_cmd, capture_output=False)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if result.returncode != 0:
        print(f"ERROR: Training failed for {retriever} + {reranker}")
        return {"error": "train_failed", "duration": duration}

    return {"success": True, "duration": duration}


def main():
    parser = argparse.ArgumentParser(description="Run all fine-tuning HPO combinations from model zoo")
    parser.add_argument("--n_trials", type=int, default=50, help="Trials per combination")
    parser.add_argument("--output_dir", type=str, default="outputs/hpo_finetuning")
    parser.add_argument("--retrievers", type=str, nargs="+", default=None,
                        help="Specific retrievers to run (default: all from zoo)")
    parser.add_argument("--rerankers", type=str, nargs="+", default=None,
                        help="Specific rerankers to run (default: all from zoo)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip combinations that already have results")
    parser.add_argument("--tier1_only", action="store_true",
                        help="Only run Tier-1 priority combinations")
    parser.add_argument("--list_models", action="store_true",
                        help="List all available models and exit")
    args = parser.parse_args()

    # List models mode
    if args.list_models:
        all_retrievers = get_all_retrievers()
        all_rerankers = get_all_rerankers()
        print(f"\n=== RETRIEVERS ({len(all_retrievers)}) ===")
        for r in all_retrievers:
            skip = " [SKIP]" if r in SKIP_RETRIEVERS else ""
            tier = " [TIER-1]" if r in TIER_1_RETRIEVERS else ""
            print(f"  {r}{tier}{skip}")
        print(f"\n=== RERANKERS ({len(all_rerankers)}) ===")
        for r in all_rerankers:
            skip = " [SKIP]" if r in SKIP_RERANKERS else ""
            tier = " [TIER-1]" if r in TIER_1_RERANKERS else ""
            print(f"  {r}{tier}{skip}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing results
    completed = check_existing_results(output_dir)
    print(f"Already completed: {len(completed)} combinations")

    # Get retrievers/rerankers
    if args.retrievers:
        retrievers = args.retrievers
    elif args.tier1_only:
        retrievers = TIER_1_RETRIEVERS
    else:
        # All from zoo except skipped
        retrievers = [r for r in get_all_retrievers() if r not in SKIP_RETRIEVERS]

    if args.rerankers:
        rerankers = args.rerankers
    elif args.tier1_only:
        rerankers = TIER_1_RERANKERS
    else:
        # All from zoo except skipped
        rerankers = [r for r in get_all_rerankers() if r not in SKIP_RERANKERS]

    print(f"\nRetrievers to test ({len(retrievers)}): {retrievers}")
    print(f"Rerankers to test ({len(rerankers)}): {rerankers}")
    print(f"Total combinations: {len(retrievers) * len(rerankers)}")

    # Build list of combos to run
    combos_to_run = []
    for retriever in retrievers:
        for reranker in rerankers:
            if args.skip_existing and (retriever, reranker) in completed:
                print(f"  Skipping {retriever} + {reranker} (already done)")
                continue
            combos_to_run.append((retriever, reranker))

    print(f"\nCombinations to run: {len(combos_to_run)}")
    for ret, rer in combos_to_run:
        print(f"  - {ret} + {rer}")

    # Run each combination
    results = []
    for i, (retriever, reranker) in enumerate(combos_to_run):
        print(f"\n\n{'#'*80}")
        print(f"# [{i+1}/{len(combos_to_run)}] {retriever} + {reranker}")
        print(f"{'#'*80}")

        result = run_combo(retriever, reranker, args.n_trials, output_dir)
        result["retriever"] = retriever
        result["reranker"] = reranker
        results.append(result)

        if result.get("success"):
            print(f"\n[OK] Completed {retriever} + {reranker} in {result['duration']:.1f}s")
        else:
            print(f"\n[FAIL] {retriever} + {reranker}: {result.get('error')}")

    # Summary
    print("\n\n" + "="*80)
    print("FINE-TUNING HPO SUMMARY")
    print("="*80)

    success_count = sum(1 for r in results if r.get("success"))
    print(f"Completed: {success_count}/{len(results)}")

    for r in results:
        status = "OK" if r.get("success") else f"FAIL ({r.get('error')})"
        print(f"  {r['retriever']:20s} + {r['reranker']:20s}: {status}")

    print(f"\nFull results in: {output_dir / 'hpo_finetuning_results.csv'}")


if __name__ == "__main__":
    main()
