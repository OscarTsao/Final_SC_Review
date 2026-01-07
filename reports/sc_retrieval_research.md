# S-C Evidence Retrieval Research Report

## 1. Repository Overview

### 1.1 Architecture Summary

The Final_SC_Review repository implements a **two-stage retrieval pipeline** for sentence-criterion (S-C) evidence retrieval:

1. **Stage 1 - Retriever (BGE-M3):** Hybrid retrieval using:
   - Dense embeddings (primary signal)
   - Sparse lexical matching (BM25-style)
   - ColBERT/MaxSim multi-vector scoring
   - Fusion methods: weighted_sum or RRF

2. **Stage 2 - Reranker (Jina-v3):** Cross-encoder reranking using:
   - `jinaai/jina-reranker-v3` with listwise API
   - Scores retrieved candidates from Stage 1

### 1.2 Key Files and Their Roles

| File | Role |
|------|------|
| `src/.../retriever/bge_m3.py` | BGE-M3 hybrid encoder + retriever with dense/sparse/colbert fusion |
| `src/.../pipeline/three_stage.py` | Orchestrates retriever -> reranker pipeline |
| `src/.../hpo/cache_builder.py` | Pre-computes all scores for HPO (cache-first approach) |
| `src/.../hpo/objective_inference.py` | Inference-HPO objective: replays cached scores with different fusion params |
| `src/.../reranker/trainer.py` | Hybrid loss training (listwise + pairwise + pointwise) |
| `scripts/eval_sc_pipeline.py` | Main evaluation script for retriever and reranked metrics |
| `scripts/final_eval.py` | Final TEST evaluation with manifest logging |

### 1.3 What is Optimized

**Inference-HPO (fast, ~seconds/trial):**
- `top_k_retriever`: Candidate pool size from retriever
- `top_k_final`: Final output size after reranking
- `use_sparse`, `use_multiv`: Which BGE-M3 signals to use
- `fusion_method`: weighted_sum vs RRF
- `score_normalization`: none/minmax/zscore
- `w_dense`, `w_sparse`, `w_multiv`: Fusion weights
- **Does NOT train model weights** - uses frozen pre-trained models

**Training-HPO (slow, hours/trial):**
- Learning rate, weight decay, batch size
- Loss weights: `w_list`, `w_pair`, `w_point`
- Temperature for listwise softmax
- **Actually trains model weights** via backpropagation

---

## 2. Baseline Metrics

### 2.1 Default Configuration (configs/default.yaml)

| Parameter | Value |
|-----------|-------|
| top_k_retriever | 50 |
| top_k_colbert | 50 |
| top_k_final | 20 |
| use_sparse | true |
| use_colbert | true |
| dense_weight | 0.6 |
| sparse_weight | 0.2 |
| colbert_weight | 0.2 |
| fusion_method | weighted_sum |
| score_normalization | minmax_per_query |

### 2.2 Baseline Results

*To be filled after running evaluations*

| Config | Split | nDCG@10 (Retriever) | nDCG@10 (Reranked) | Recall@10 | MRR@10 |
|--------|-------|---------------------|--------------------| ----------|--------|
| default | val | | | | |
| default | test | | | | |
| HPO-reranked | val | | | | |
| HPO-reranked | test | | | | |
| HPO-retriever-only | val | | | | |
| HPO-retriever-only | test | | | | |

---

## 3. Diagnostic Analysis

### 3.1 Candidate Ceiling Analysis

*To be computed: What % of queries have gold in top-K?*

| top-K | % queries with gold in pool |
|-------|----------------------------|
| 10 | |
| 20 | |
| 50 | |
| 100 | |

### 3.2 Failure Taxonomy

*To be analyzed:*
- Queries with no positives (should skip in eval)
- Per-criterion performance breakdown
- Best/worst performing criteria

### 3.3 Data Quality

*To be computed:*
- Sentence corpus statistics
- Backfilled sentence ratio
- Empty text ratio

---

## 4. Identified Issues and Improvements

### 4.1 Critical: Decoupled Pool Sizes

**Problem:** In `bge_m3.py:346-347`:
```python
pool_k = min(top_k_retriever, top_k_colbert)
```
This caps the returned pool, blocking the "big recall pool -> small rerank pool" pattern.

**Impact:** Cannot optimize retriever recall independently of reranker candidates.

**Solution:** Return `top_k_retriever` candidates, let pipeline slice for reranking.

### 4.2 HPO Cache Trim Bias

**Problem:** In `cache_builder.py:180`:
```python
union_indices = _trim_by_dense(union_indices, retriever, query_dense, limit)
```
Trims by dense score only, potentially dropping sparse-only good candidates.

**Solution:** Implement fair trim via RRF(dense_rank, sparse_rank) or max-fusion.

### 4.3 Missing top_k_rerank in HPO

**Problem:** HPO assumes `rerank_pool == top_k_retriever` (line 177 in objective_inference.py).

**Solution:** Add `top_k_rerank` as separate parameter in search space.

### 4.4 Training Uses Random Negatives Only

**Problem:** `dataset.py:47` samples random negatives from the same post.

**Solution:** Implement hard negative mining using retriever scores.

---

## 5. Experiment Results

### 5.1 Ablation: Pool Size Decoupling

*To be run after implementation*

| top_k_retriever | top_k_rerank | nDCG@10 (Reranked) |
|-----------------|--------------|-------------------|
| 32 | 8 | |
| 64 | 16 | |
| 128 | 32 | |

### 5.2 Ablation: Cache Trim Method

*To be run after implementation*

| Trim Method | nDCG@10 |
|-------------|---------|
| dense_only | |
| rrf_dense_sparse | |
| fused_dense_sparse | |

### 5.3 Training: Hard Negatives

*To be run after implementation*

| Negative Source | nDCG@10 |
|-----------------|---------|
| Random only | |
| Hard only | |
| Mixed (70/30) | |

---

## 6. Literature Review

### 6.1 Two-Stage Retrieval Best Practices

- **Recall-first, precision-second:** Retriever should maximize recall, reranker should optimize precision
- **Pool size ratio:** Common pattern is retriever_k >> rerank_k (e.g., 100 -> 10)

### 6.2 Rank Fusion

- **RRF:** Robust when component rankings have different scales
- **Score fusion:** Better when scores are calibrated, requires normalization

### 6.3 Hard Negative Mining

- **BM25 negatives:** Classic approach, finds lexically similar but semantically different
- **Dense retriever negatives:** Current best practice for dense retrieval training
- **In-batch negatives:** Efficient but may not be hard enough

### 6.4 Listwise Losses

- **ListNet:** Simple, but treats labels as soft targets
- **LambdaRank/LambdaLoss:** Directly optimizes nDCG, often better for ranking
- **ApproxNDCG:** Differentiable approximation to nDCG

---

## 7. Recommendations (Ranked by ROI)

1. **[HIGH] Decouple retriever pool from rerank pool** - Enables proper recall/precision tradeoff
2. **[HIGH] Add top_k_rerank to HPO search space** - Unlocks optimal pool size tuning
3. **[MEDIUM] Fix cache trim bias** - May improve candidate quality in HPO
4. **[MEDIUM] Implement hard negative mining** - Should improve reranker training
5. **[LOW] Add LambdaLoss option** - Direct nDCG optimization

---

## 8. Reproduction Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run tests
pytest -q

# Baseline evaluation
python scripts/eval_sc_pipeline.py --config configs/default.yaml --output outputs/baseline_test.json

# Inference HPO
python scripts/precompute_hpo_cache.py --config configs/hpo_inference.yaml
python scripts/hpo_inference.py --config configs/hpo_inference.yaml --n_trials 200 --study_name sc_inference

# Export and final eval
python scripts/export_best_config.py --study_name sc_inference --storage sqlite:///outputs/hpo/optuna.db
python scripts/final_eval.py --best_config outputs/hpo/sc_inference/best_config.yaml
```
