# Final S–C Evidence Sentence Retrieval

This repo contains the final, deployment-aligned sentence–criterion (S–C) evidence retrieval pipeline:
- **Retriever:** BGE-M3 dense retriever (within-post retrieval)
- **Reranker:** Jina-v3 reranker (`jinaai/jina-reranker-v3`)
- **Training:** Hybrid loss = listwise + pairwise + pointwise

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Required local data layout

Place data locally (not tracked in git):

```
data/
  redsm5/
    redsm5_posts.csv
    redsm5_annotations.csv
  DSM5/
    MDD_Criteira.json
```

## Commands

Build groundtruth (sentence-level labels):

```bash
python scripts/build_groundtruth.py --data_dir data --output data/groundtruth/evidence_sentence_groundtruth.csv
```

Build sentence corpus (canonical splitting + sent_uid):

```bash
python scripts/build_sentence_corpus.py --data_dir data --output data/groundtruth/sentence_corpus.jsonl
```

Train hybrid reranker:

```bash
python scripts/train_reranker_hybrid.py --config configs/reranker_hybrid.yaml
```

Evaluate retriever-only + reranked:

```bash
python scripts/eval_sc_pipeline.py --config configs/default.yaml --output outputs/eval_summary.json
```

Run a single query (post_id + criterion):

```bash
python scripts/run_single.py --config configs/default.yaml --post_id <POST_ID> --criterion_id <CRITERION_ID>
```

## Notes
- Evaluation is strict and uses groundtruth labels only (no gold-from-preds).
- All splits are **post_id-disjoint** to prevent leakage.
- Canonical sentence splitting and `sent_uid = f"{post_id}_{sid}"` are used everywhere.
- BGE-M3 hybrid retrieval (dense + sparse + ColBERT) requires `FlagEmbedding` (installed via `pip install -e .`).

## HPO (Cache-First, Dev-Only)

Stage A: build cache and run inference HPO (no model inference inside trials).

```bash
python scripts/precompute_hpo_cache.py --config configs/hpo_inference.yaml
python scripts/hpo_inference.py --config configs/hpo_inference.yaml --n_trials 200 --study_name sc_inference
```

Multi-GPU workers (one worker per GPU):

```bash
python scripts/launch_hpo_multi_gpu.py --config configs/hpo_inference.yaml --study_name sc_inference --n_trials_per_worker 200
```

Export best config and run a final test evaluation:

```bash
python scripts/export_best_config.py --study_name sc_inference --storage sqlite:///outputs/hpo/optuna.db
python scripts/final_eval.py --best_config outputs/hpo/sc_inference/best_config.yaml
```

Optional training HPO (disabled by default):

```bash
python scripts/hpo_training.py --config configs/hpo_training.yaml --n_trials 20 --study_name sc_training
```

Research protocol:
- HPO uses DEV only; TEST is evaluated once with the selected best config.
- Cache and Optuna studies are stored under `outputs/`.
