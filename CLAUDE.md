# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sentence-criterion (S-C) evidence retrieval pipeline for mental health research. Given a post and a DSM-5 criterion, the system retrieves evidence sentences supporting the criterion. The pipeline uses:
- **Retriever:** BGE-M3 hybrid retrieval (dense + sparse + ColBERT)
- **Reranker:** Jina-v3 reranker with listwise scoring
- **Training:** Hybrid loss combining listwise, pairwise, and pointwise objectives

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Build Data Artifacts
```bash
# Build groundtruth labels
python scripts/build_groundtruth.py --data_dir data --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus
python scripts/build_sentence_corpus.py --data_dir data --output data/groundtruth/sentence_corpus.jsonl
```

### Training
```bash
python scripts/train_reranker_hybrid.py --config configs/reranker_hybrid.yaml
```

### Evaluation
```bash
# Evaluate on test split
python scripts/eval_sc_pipeline.py --config configs/default.yaml --output outputs/eval_summary.json

# Final evaluation with best HPO config
python scripts/final_eval.py --best_config outputs/hpo/sc_inference/best_config.yaml
```

### Single Query Inference
```bash
python scripts/run_single.py --config configs/default.yaml --post_id <POST_ID> --criterion_id <CRITERION_ID>
```

### Hyperparameter Optimization
```bash
# Precompute cache then run HPO
python scripts/precompute_hpo_cache.py --config configs/hpo_inference.yaml
python scripts/hpo_inference.py --config configs/hpo_inference.yaml --n_trials 200 --study_name sc_inference

# Multi-GPU HPO
python scripts/launch_hpo_multi_gpu.py --config configs/hpo_inference.yaml --study_name sc_inference --n_trials_per_worker 200

# Export best config
python scripts/export_best_config.py --study_name sc_inference --storage sqlite:///outputs/hpo/optuna.db
```

### Testing
```bash
pytest                        # Run all tests
pytest tests/test_metrics.py  # Run single test file
pytest -k "test_name"         # Run tests matching pattern
```

## Architecture

### Three-Stage Pipeline (`src/final_sc_review/pipeline/three_stage.py`)
1. **Stage 1 - Hybrid Retrieval:** BGE-M3 encodes sentences with dense, sparse, and ColBERT representations. Retrieval is **within-post only** (candidates limited to sentences from the same post as the query).
2. **Stage 2 - Score Fusion:** Dense, sparse, and ColBERT scores are combined via weighted sum or RRF fusion with configurable normalization (minmax/zscore per query).
3. **Stage 3 - Reranking:** Top-k candidates are reranked by Jina-v3 using listwise scoring.

### Key Modules
- `src/final_sc_review/retriever/bge_m3.py` - BGE-M3 encoder with cached embeddings (dense, sparse, ColBERT vectors stored in `data/cache/bge_m3/`)
- `src/final_sc_review/reranker/jina_v3.py` - Listwise reranker with fallback to pairwise scoring
- `src/final_sc_review/reranker/losses.py` - Hybrid loss: `w_list * listwise + w_pair * pairwise + w_point * pointwise`
- `src/final_sc_review/data/splits.py` - Post-ID-disjoint splits to prevent leakage
- `src/final_sc_review/metrics/ranking.py` - Ranking metrics: Recall@K, MRR@K, MAP@K, nDCG@K

### Data Flow
- **Input:** Posts (`redsm5_posts.csv`) + Criteria (`MDD_Criteira.json`) + Annotations (`redsm5_annotations.csv`)
- **Groundtruth:** `evidence_sentence_groundtruth.csv` with columns: `post_id, criterion, sid, sent_uid, sentence, groundtruth`
- **Sentence corpus:** `sentence_corpus.jsonl` with canonical sentence splitting and `sent_uid = f"{post_id}_{sid}"`

### Key Constraints
- All splits are **post_id-disjoint** (no post appears in multiple splits)
- HPO uses DEV split only; TEST is evaluated once with the best config
- Retrieval is always **within-post** (candidate pool = sentences from the queried post)
- Cache fingerprint includes corpus hash + model config; auto-rebuilds on mismatch

## Configuration

YAML configs control all pipeline parameters. Key sections:
- `paths` - Data directories, groundtruth/corpus paths, cache location
- `models` - Model names, max lengths, batch sizes, dtype
- `retriever` - Fusion weights, top-k values, normalization method
- `split` - Train/val/test ratios and seed
- `evaluation` - Metrics at K values, which split to evaluate

The `top_k_retriever` controls retrieval pool size, `top_k_rerank` (or deprecated `top_k_colbert`) controls reranker input size, and `top_k_final` controls output size.
