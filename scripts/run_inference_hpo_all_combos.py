#!/usr/bin/env python3
"""Run inference-only HPO for all retriever x reranker combinations.

This establishes baselines before fine-tuning HPO.
Uses the full model zoo for comprehensive evaluation.
"""

import argparse
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
import csv

# Import zoo classes to get all available models
from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.reranker.zoo import RerankerZoo


def get_all_retrievers():
    """Get all retriever configs from the zoo."""
    return [(c.name, c.model_id) for c in RetrieverZoo.DEFAULT_RETRIEVERS]


def get_all_rerankers():
    """Get all reranker configs from the zoo."""
    return [(c.name, c.model_id) for c in RerankerZoo.DEFAULT_RERANKERS]


# Models to skip (known issues)
SKIP_RETRIEVERS = [
    "qwen3-embed-8b-4bit",  # Duplicate
    "colbertv2",            # Requires special installation
    "splade-cocondenser",   # SPLADE requires special handling
    "splade-v2-distil",     # SPLADE requires special handling
    "mxbai-colbert-large",  # ColBERT variant
]

SKIP_RERANKERS = [
    "bge-reranker-gemma2-lightweight",  # OOM issues
    "bge-reranker-v2-minicpm",          # Compatibility issues
    "rank-zephyr-7b",                    # Too large
]

# Legacy lists for backward compatibility
RETRIEVERS = [
    "bge-m3",
    "nv-embed-v2",
    "qwen3-embed-4b",
    "qwen3-embed-0.6b",
    "llama-embed-8b",
]

RERANKERS = [
    ("jina-reranker-v3", "jinaai/jina-reranker-v3"),
    ("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3"),
    ("jina-reranker-v2", "jinaai/jina-reranker-v2-base-multilingual"),
]

# Base config template
BASE_CONFIG = """
paths:
  data_dir: data
  groundtruth: data/groundtruth/evidence_sentence_groundtruth.csv
  sentence_corpus: data/groundtruth/sentence_corpus.jsonl
  criteria: data/DSM5/MDD_Criteira.json
  hpo_cache_dir: outputs/hpo_cache
  hpo_output_dir: outputs/hpo_inference_combos

models:
  bge_m3: BAAI/bge-m3
  retriever_name: {retriever}
  bge_query_max_length: 128
  bge_passage_max_length: 256
  bge_use_fp16: true
  bge_batch_size: 64

reranker:
  model_name: {reranker_model_id}
  chunk_size: 32
  dtype: auto
  max_length: 512

cache:
  dense_topk_max: 128
  sparse_topk_max: 128
  superset_max: 128
  use_sparse: true
  use_multiv: true
  rebuild_embeddings: false

split:
  seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  dev_split: val

evaluation:
  ks: [1, 5, 10, 20]
  skip_no_positives: true

hpo:
  objective_metric: ndcg@10_reranked
  prune_chunk_frac: 0.1
  prune_min_queries: 50

search_space:
  top_k_retriever: [8, 16, 24, 32, 48, 64]
  top_k_final: [1, 3, 5, 10]
  use_sparse: [true, false]
  use_multiv: [true, false]
  fusion_method: [weighted_sum, rrf]
  score_normalization: [minmax_per_query, zscore_per_query, none]
"""


def run_hpo_for_combo(retriever: str, reranker_name: str, reranker_model_id: str,
                       n_trials: int, output_dir: Path) -> dict:
    """Run inference HPO for a single retriever+reranker combination."""

    study_name = f"inference_{retriever}_{reranker_name}"
    combo_dir = output_dir / f"{retriever}_{reranker_name}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    # Create config file
    config_content = BASE_CONFIG.format(
        retriever=retriever,
        reranker_model_id=reranker_model_id,
    )

    config_path = combo_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    # Update paths in config
    config = yaml.safe_load(config_content)
    config["paths"]["hpo_output_dir"] = str(combo_dir)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    storage = f"sqlite:///{combo_dir}/optuna.db"

    print(f"\n{'='*60}")
    print(f"Running HPO: {retriever} + {reranker_name}")
    print(f"Study: {study_name}")
    print(f"Storage: {storage}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "scripts/hpo_inference.py",
        "--config", str(config_path),
        "--study_name", study_name,
        "--storage", storage,
        "--n_trials", str(n_trials),
    ]

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Get best value from Optuna
    best_value = None
    try:
        import optuna
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_value = study.best_value
        best_params = study.best_params
    except Exception as e:
        print(f"Warning: Could not load study results: {e}")
        best_params = {}

    return {
        "retriever": retriever,
        "reranker": reranker_name,
        "best_ndcg": best_value,
        "n_trials": n_trials,
        "duration_seconds": duration,
        "best_params": best_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference HPO for all combos")
    parser.add_argument("--n_trials", type=int, default=50, help="Trials per combo")
    parser.add_argument("--output_dir", type=str, default="outputs/hpo_inference_combos")
    parser.add_argument("--retrievers", type=str, nargs="+", default=None,
                        help="Specific retrievers to run (default: all)")
    parser.add_argument("--rerankers", type=str, nargs="+", default=None,
                        help="Specific rerankers to run (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip combinations that already have results")
    parser.add_argument("--all_zoo", action="store_true",
                        help="Use all models from the zoo (default: use legacy list)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model lists based on --all_zoo flag
    if args.all_zoo:
        # Use full zoo, filtering out problematic models
        all_retrievers = get_all_retrievers()
        all_rerankers = get_all_rerankers()

        retriever_list = [(name, model_id) for name, model_id in all_retrievers
                          if name not in SKIP_RETRIEVERS]
        reranker_full_list = [(name, model_id) for name, model_id in all_rerankers
                              if name not in SKIP_RERANKERS]

        # Apply user filters if specified
        if args.retrievers:
            retriever_list = [(n, m) for n, m in retriever_list if n in args.retrievers]
        if args.rerankers:
            reranker_full_list = [(n, m) for n, m in reranker_full_list if n in args.rerankers]

        retrievers = [name for name, _ in retriever_list]
        retriever_model_ids = {name: model_id for name, model_id in retriever_list}
        reranker_list = reranker_full_list

        print(f"Using full zoo: {len(retrievers)} retrievers x {len(reranker_list)} rerankers = {len(retrievers) * len(reranker_list)} combinations")
        print(f"Skipped retrievers: {SKIP_RETRIEVERS}")
        print(f"Skipped rerankers: {SKIP_RERANKERS}")
    else:
        # Use legacy lists for backward compatibility
        retrievers = args.retrievers if args.retrievers else RETRIEVERS
        reranker_list = [(n, m) for n, m in RERANKERS if args.rerankers is None or n in args.rerankers]

    results = []
    results_file = output_dir / "inference_hpo_results.csv"

    total_combos = len(retrievers) * len(reranker_list)
    current = 0

    for retriever in retrievers:
        for reranker_name, reranker_model_id in reranker_list:
            current += 1
            print(f"\n[{current}/{total_combos}] Processing {retriever} + {reranker_name}")

            # Check if already done
            if args.skip_existing:
                combo_dir = output_dir / f"{retriever}_{reranker_name}"
                db_path = combo_dir / "optuna.db"
                if db_path.exists():
                    print(f"  Skipping (already exists)")
                    continue

            try:
                result = run_hpo_for_combo(
                    retriever, reranker_name, reranker_model_id,
                    args.n_trials, output_dir
                )
                results.append(result)

                # Save results incrementally
                with open(results_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "retriever", "reranker", "best_ndcg", "n_trials", "duration_seconds"
                    ])
                    writer.writeheader()
                    for r in results:
                        writer.writerow({k: v for k, v in r.items() if k != "best_params"})

                print(f"  Best nDCG@10: {result['best_ndcg']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "retriever": retriever,
                    "reranker": reranker_name,
                    "best_ndcg": None,
                    "n_trials": args.n_trials,
                    "duration_seconds": 0,
                    "error": str(e),
                })

    # Final summary
    print("\n" + "="*60)
    print("INFERENCE HPO RESULTS SUMMARY")
    print("="*60)
    for r in results:
        ndcg = f"{r['best_ndcg']:.4f}" if r.get('best_ndcg') else "FAILED"
        print(f"{r['retriever']:20s} + {r['reranker']:20s}: {ndcg}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
