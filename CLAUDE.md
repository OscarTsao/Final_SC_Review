# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sentence-criterion (S-C) evidence retrieval pipeline for mental health research. Given a post and a DSM-5 criterion, the system retrieves evidence sentences supporting the criterion.

### Best Model Configuration (HPO-Optimized)
- **Retriever:** NV-Embed-v2 (nvidia/NV-Embed-v2) - Best from 25 retrievers
- **Reranker:** Jina-Reranker-v3 (jinaai/jina-reranker-v3) - Best from 15 rerankers
- **Performance:** nDCG@10 = 0.8658 (from 324 model combinations tested)

### Pipeline Options
1. **Zoo Pipeline (RECOMMENDED)** - Uses best HPO model combo (nv-embed-v2 + jina-reranker-v3)
2. **Legacy Pipeline** - Uses BGE-M3 + Jina-v3 (for backward compatibility)

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

### Evaluation (RECOMMENDED - Zoo Pipeline)
```bash
# Evaluate with best HPO model combo (nv-embed-v2 + jina-reranker-v3)
python scripts/eval_zoo_pipeline.py --config configs/default.yaml --split test

# Single query inference with best models
python scripts/run_single_zoo.py --config configs/default.yaml --post_id <POST_ID> --criterion_id <CRITERION_ID>
```

### Evaluation (Legacy Pipeline)
```bash
# Evaluate with BGE-M3 (legacy)
python scripts/eval_sc_pipeline.py --config configs/default.yaml --output outputs/eval_summary.json

# Single query inference (legacy)
python scripts/run_single.py --config configs/default.yaml --post_id <POST_ID> --criterion_id <CRITERION_ID>
```

### Hyperparameter Optimization
```bash
# Run HPO for all retriever x reranker combinations
python scripts/run_inference_hpo_all_combos.py --all_zoo --n_trials 50

# Single combo HPO
python scripts/hpo_inference.py --config configs/hpo_inference.yaml --n_trials 200 --study_name sc_inference
```

### Testing
```bash
pytest                        # Run all tests
pytest tests/test_metrics.py  # Run single test file
pytest -k "test_name"         # Run tests matching pattern
```

## Architecture

### Zoo Pipeline (`src/final_sc_review/pipeline/zoo_pipeline.py`)
The recommended pipeline using retriever and reranker zoos:
1. **Stage 1 - Retrieval:** Uses any retriever from zoo (default: nv-embed-v2)
2. **Stage 2 - Reranking:** Uses any reranker from zoo (default: jina-reranker-v3)

### Three-Stage Pipeline (`src/final_sc_review/pipeline/three_stage.py`)
Legacy pipeline for backward compatibility:
1. **Stage 1 - Hybrid Retrieval:** BGE-M3 encodes sentences with dense, sparse, and ColBERT representations
2. **Stage 2 - Score Fusion:** Dense, sparse, and ColBERT scores combined via weighted sum or RRF
3. **Stage 3 - Reranking:** Top-k candidates reranked by Jina-v3

### Key Modules
- `src/final_sc_review/retriever/zoo.py` - Retriever zoo with 25+ models
- `src/final_sc_review/reranker/zoo.py` - Reranker zoo with 15+ models
- `src/final_sc_review/pipeline/zoo_pipeline.py` - Zoo-based pipeline (RECOMMENDED)
- `src/final_sc_review/retriever/bge_m3.py` - BGE-M3 encoder (legacy)
- `src/final_sc_review/reranker/jina_v3.py` - Listwise reranker
- `src/final_sc_review/data/splits.py` - Post-ID-disjoint splits to prevent leakage
- `src/final_sc_review/metrics/ranking.py` - Ranking metrics: Recall@K, MRR@K, MAP@K, nDCG@K

### GNN Enhancements (Research)
- `src/final_sc_review/gnn/` - Graph Neural Network models for:
  - P1: NE Gate (no-evidence detection)
  - P2: Dynamic-K selection
  - P3: Graph Reranker
  - P4: Criterion-Aware GNN (AUROC=0.8967)

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
- `models` - Model names (both zoo and legacy keys supported)
- `retriever` - Fusion weights, top-k values, normalization method
- `split` - Train/val/test ratios and seed
- `evaluation` - Metrics at K values, which split to evaluate

### Best HPO Parameters
```yaml
retriever:
  top_k_retriever: 24
  top_k_final: 10
  use_sparse: false
  use_colbert: false
  fusion_method: rrf
  score_normalization: none
  rrf_k: 60
```

### Config Keys
- **Zoo Pipeline:** `models.retriever_name`, `models.reranker_name`
- **Legacy Pipeline:** `models.bge_m3`, `models.jina_v3`

The `top_k_retriever` controls retrieval pool size, `top_k_rerank` controls reranker input size, and `top_k_final` controls output size.

## HPO Results Summary

| Rank | Retriever | Reranker | nDCG@10 |
|------|-----------|----------|---------|
| 1 | nv-embed-v2 | jina-reranker-v3 | 0.8658 |
| 2 | qwen3-embed-4b | jina-reranker-v3 | 0.8534 |
| 3 | llama-embed-8b | jina-reranker-v3 | 0.8489 |

Full results: `outputs/hpo_inference_combos/full_results.csv`
