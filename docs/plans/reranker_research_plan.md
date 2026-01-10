# Reranker Research Plan
**File:** `reranker_research_plan.md`  
**Project:** SC Evidence Retrieval & Binding (sentence-level evidence retrieval inside a post)  
**Goal:** Exhaustively explore **reranker models × loss functions × training regimes × inference configs × fusion methods**, and select the **maximum end-to-end** system under a high compute budget, while controlling false-evidence on “no-evidence” queries.

---

## 0) What you already have (inputs)
- **Retriever leaderboard** on `dev_select` (you already ran this).
- A shortlist of **reranker model choices** (open SOTA-class + strong baselines).

This plan covers the missing part: **how to run the reranker research program end-to-end**.

---

## 1) Target outcomes (what “max performance” means)
You should optimize end-to-end performance (retriever → candidate set → reranker → final selection), not reranker-only.

### 1.1 Primary metric (optimize)
- **nDCG@10** (overall, all queries)

### 1.2 Secondary metrics (track + constrain)
- **False-evidence rate on no-evidence queries**  
  *Definition:* percent of no-evidence queries where the system returns ≥1 sentence above a chosen threshold.
- **Recall@K** (has-evidence queries) at the candidate sizes you actually rerank (K=50/100/200)
- **MRR@10**
- Optional: throughput / latency

> Recommendation: treat final system selection as **constrained optimization**: maximize nDCG@10 subject to false-evidence ≤ ε.

---

## 2) Components to explore (factorization)
To “try everything” without chaos, define a **factor graph**:

1) Candidate generator (retriever / fusion)  
2) Reranker model family  
3) Data construction regime  
4) Loss function  
5) Negative mining strategy  
6) Training hyperparameters  
7) Inference hyperparameters  
8) Calibration / abstention policy  
9) (Optional) ensembling policy  

This plan gives you a systematic sweep for each factor.

---

## 3) Candidate generation (retriever stage)
Even though your question is about rerankers, candidate generation strongly caps achievable reranker gains.

### 3.1 Candidate generators to include
Treat each of these as a discrete option `G`:

- **G1 Dense-best:** your best dense retriever (e.g., `qwen3-embed-0.6b`)
- **G2 Sparse-best:** your best sparse retriever (e.g., `splade-cocondenser`)
- **G3 Dense-recall:** the dense model with best Recall@K at your rerank pool size (possibly `qwen3-embed-4b`)
- **G4 Fusion-RRF:** RRF fusion of Dense-best + Sparse-best  
  Use standard RRF score:  
  \[
  \text{RRF}(d)=\sum_i \frac{1}{k_0+\text{rank}_i(d)}
  \]
  with `k0 ∈ {20, 60, 100}`.
- **G5 Fusion-weighted:** weighted sum of dense+sparse (only if per-query normalization is applied)

> RRF is robust and widely used for fusing ranked lists (Cormack et al., SIGIR 2009).

### 3.2 Candidate pool sizes
Predefine `top_k_retriever ∈ {50, 100, 200, 400}` and generate candidates for each.

---

## 4) Reranker model shortlist (the “must-run” set)
Split into: **SOTA-class open** vs **strong baselines**.

### 4.1 SOTA-class open rerankers (core)
Run these first; they are the most likely to win overall:

- **Qwen3-Reranker series**: 0.6B / 4B / 8B  
  Notes to exploit:
  - instruction-aware reranking
  - long context (32k)
  - model card notes typical gains from using tailored instructions (recommend English instructions)

- **jina-reranker-v3 (0.6B, listwise multi-doc reranker)**  
  Key properties:
  - listwise: processes *many docs in one forward* (up to ~64 docs in one context window)
  - high BEIR score reported on model card
  - built on Qwen3-0.6B; uses a “last but not late interaction” architecture

- **Mixedbread mxbai-rerank-v2**: base (0.5B) / large (1.5B)  
  Key properties:
  - open weights (Apache-2.0)
  - long context (8K / 32K compatible)
  - strong BEIR score reported in docs/blog

### 4.2 Strong baselines (include for sanity checks + ablations)
- **BGE reranker family**:
  - `bge-reranker-v2-m3` (fast strong baseline)
  - `bge-reranker-v2.5-gemma2-lightweight` (includes layerwise reduction + token compression)

- Optional: your current production baseline reranker (if any), for “delta vs baseline” reporting.

---

## 5) Data construction for reranker training
Your domain is “criterion → evidence sentence inside a post”.

### 5.1 Define the training unit
- **Query:** criterion text (+ optional instruction prompt)
- **Doc:** candidate sentence (optionally include small local context window)
- **Label:** evidence (1) vs not evidence (0), with optional graded labels

### 5.2 Multi-positive queries
If multiple evidence sentences exist for a query, preserve them as multiple positives.

### 5.3 No-evidence queries (crucial)
Many listwise losses assume at least one positive per list. For no-evidence queries:

- Don’t force them into listwise ranking training.
- Instead:
  1) Train ranking losses on **has-evidence** queries
  2) Calibrate thresholds (or train a small “has evidence?” head/model) on **no-evidence** queries

---

## 6) Negative mining strategies (treat as a sweep dimension)
Define a negative pool generator `N`:

- **N0 Random-only:** sample negatives uniformly from non-evidence sentences in the same post.
- **N1 Retriever-hard:** negatives from top ranks of the best retriever where label=0.
- **N2 Iterative-hard:** run a reranker checkpoint and take top-scoring false positives as hard negatives.
- **N3 Mixed (recommended):** mix random + hard negatives.

> Cross-encoder rerankers can get overly strict with only hard negatives; mixing random negatives can mitigate unexpected performance drops when reranking larger candidate pools.

---

## 7) Loss functions to sweep (SOTA-ish reranker training losses)
Use the SentenceTransformers CrossEncoder loss suite as your main standardized sweep surface.

### 7.1 Pointwise / binary
- **BinaryCrossEntropyLoss**  
  Strong baseline for reranking with binary labels.

### 7.2 Pairwise ranking
- **RankNetLoss** (pairwise logistic ranking)

### 7.3 Listwise learning-to-rank
- **ListNetLoss**
- **ListMLELoss**
- **PListMLELoss** (position-aware ListMLE)
- **LambdaLoss** (metric-driven framework; supports LambdaRank/NDCG weighting schemes)

### 7.4 In-batch negatives (contrastive-style)
- **MultipleNegativesRankingLoss / CachedMultipleNegativesRankingLoss**  
  (Useful when you can build large effective batches.)

### 7.5 Distillation losses (teacher → student)
- **MSELoss** (match teacher scores)
- **MarginMSELoss** (match score differences)

---

## 8) Training regimes (“methods & approaches”)
For each reranker model `M`, run:

### R0 — Off-the-shelf inference (no training)
- Sweep inference configs:
  - `top_k_rerank ∈ {20,50,100,200}`
  - `max_length ∈ {128,256,384,512}`
  - instruction templates (for instruction-aware rerankers)

### R1 — Supervised binary (fast + strong baseline)
- Loss: BCE
- Negatives: N3 mixed (random + hard)

### R2 — Pairwise ranking
- Loss: RankNetLoss
- Requires per-query doc lists with labels

### R3 — Listwise ranking
Run each listwise loss:
- ListNetLoss
- ListMLELoss
- PListMLELoss
- LambdaLoss (multiple weighting schemes / cutoff k)

### R4 — Curriculum (recommended stability)
- Warm-start with BCE (0.2–0.5 epoch)
- Switch to LambdaLoss / PListMLE / RankNet

### R5 — Distillation
- Pick best checkpoint as teacher
- Distill into smaller rerankers using MSE / MarginMSE
- Evaluate if student catches up while being faster

---

## 9) Inference & fusion config sweep (HPO search space)
Even with unlimited compute, you need *structured* HPO to avoid noise.

### 9.1 Inference knobs (always tune)
- `top_k_retriever` ∈ {50, 100, 200, 400}
- `top_k_rerank` ∈ {20, 50, 100, 200}
- `max_length` ∈ {128, 256, 384, 512}
- Context window around sentence: {none, ±1 sentence} (optional)

### 9.2 Prompting / instruction variants (especially for Qwen3)
- instruction text variants (2–10 templates)
- “English instruction only” vs localized (if multilingual)

### 9.3 Fusion knobs (if candidate generator is fusion)
- RRF `k0` ∈ {20, 60, 100}
- weighted fusion α ∈ [0,1] + per-query normalization choice

### 9.4 Model-family-specific knobs
- BGE lightweight reranker:
  - `cutoff_layers`
  - `compress_ratio`
  - `compress_layers`

- Jina listwise reranker:
  - group size (docs per forward): {16, 32, 64}
  - packing policy (how you concatenate many docs)

---

## 10) HPO orchestration (how to run “everything” efficiently)
### 10.1 Use Optuna with pruning
- Optuna supports define-by-run search spaces, pruning, and easy parallelization.
- Use pruners like MedianPruner / SuccessiveHalving / Hyperband where possible.

### 10.2 Three-tier optimization
**Tier A: cached inference-only HPO**
- With cached reranker scores, tune:
  - thresholds, pool sizes, fusion params, calibration
- This is cheap and should cover thousands of trials.

**Tier B: training HPO**
- For each (M, regime, loss):
  - tune LR, warmup, epochs, negative mix ratios, list sizes, loss hyperparams
  - prune aggressively

**Tier C: finalist end-to-end HPO**
- Re-run Tier A for the best fine-tuned checkpoints.

---

## 11) Evaluation protocol (avoid dev overfitting)
### 11.1 Splits
- train → dev_select (HPO) → dev_final (confirmation) → test (one-time)

If feasible:
- grouped k-fold by post_id, report mean±std.

### 11.2 Report these metrics
- nDCG@10, MRR@10, Recall@10 (overall)
- Recall@K for K={50,100,200} on has-evidence subset
- false-evidence rate on no-evidence subset
- (optional) calibration metrics (ECE / reliability plots)

### 11.3 Statistical testing
For top systems:
- paired bootstrap / randomization test over queries
- 3–5 seeds for training

---

## 12) Experiment matrix (the “all combinations” checklist)

For each candidate generator `G` in {G1..G5}  
for each reranker model `M`  
for each training regime `R` in {R0..R5}  
for each loss `L` compatible with `R`  
for each negative strategy `N` in {N0..N3}  

Run:
1) training (if R≠R0) with training HPO  
2) cache reranker scores on dev splits  
3) inference-only HPO over pool sizes, max_length, thresholds, fusion  
4) record best config + checkpoint per combination

Then:
- select top-N combos on dev_final
- run test once

---

## 13) Practical implementation (make it runnable)

### 13.1 Suggested folder layout
```
outputs/reranker_research/
  candidates/                  # cached candidate sets per G, per K
  reranker_scores/             # cached scores per (M checkpoint), per G/K
  train_runs/                  # checkpoints + configs
  hpo/                         # Optuna DB + trial logs
  leaderboards/                # csv/json summaries
  plots/
```

### 13.2 Minimal artifacts to save per run
- config JSON/YAML (full)
- best checkpoint
- per-query predictions (top-k sentences + scores)
- eval summary JSON
- cached reranker scores (if enabled)

---

## 14) Recommended starting defaults (even if you sweep)
- Candidate generator: **Fusion-RRF (dense + sparse)**
- Reranker contenders: **jina-reranker-v3**, **mxbai-rerank-large-v2**, **Qwen3-Reranker-4B**
- Training: **Curriculum (BCE warmup → LambdaLoss)**
- Negatives: **50% hard + 50% random**
- HPO: Optuna + pruning

---

## 15) Source links (for your writeup / citations)
- RRF paper (SIGIR 2009): https://doi.org/10.1145/1571941.1572114  
- RankNet (ICML 2005): https://doi.org/10.1145/1102351.1102363  
- ListNet TR: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/  
- ListMLE (ICML 2008): https://doi.org/10.1145/1390156.1390306  
- LambdaRank local optimality: https://www.microsoft.com/en-us/research/publication/on-the-local-optimality-of-lambdarank/  
- LambdaLoss (2018): https://www.tdcommons.org/dpubs_series/1216  
- SentenceTransformers CrossEncoder losses: https://www.sbert.net/docs/package_reference/cross_encoder/losses.html  
- SentenceTransformers training tips: https://www.sbert.net/docs/cross_encoder/training_overview.html  
- Optuna: https://github.com/optuna/optuna  
- Optuna pruners: https://optuna.readthedocs.io/en/stable/reference/pruners.html  
- Qwen3 reranker model card: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B  
- jina-reranker-v3 model card: https://huggingface.co/jinaai/jina-reranker-v3  
- Mixedbread rerank v2 docs:  
  - https://www.mixedbread.com/docs/models/reranking/mxbai-rerank-large-v2  
  - https://www.mixedbread.com/docs/models/reranking/mxbai-rerank-base-v2  
- BGE reranker docs: https://bge-model.com/tutorial/5_Reranking/5.2.html

---

## 16) Implementation Tracking

### 16.1 Models in Zoo (Available for Testing)

| Model | Model ID | Type | Status |
|-------|----------|------|--------|
| **jina-reranker-v3** | jinaai/jina-reranker-v3 | listwise | ✅ Tested (R0) |
| **jina-reranker-v2** | jinaai/jina-reranker-v2-base-multilingual | cross-encoder | ✅ Tested + HPO ⭐ |
| **mxbai-rerank-base-v2** | mixedbread-ai/mxbai-rerank-base-v2 | cross-encoder | ✅ Tested (R0) |
| **mxbai-rerank-large-v2** | mixedbread-ai/mxbai-rerank-large-v2 | cross-encoder | ✅ Tested (R0) |
| **mxbai-rerank-base-v1** | mixedbread-ai/mxbai-rerank-base-v1 | cross-encoder | ✅ Tested (R0) |
| **mxbai-rerank-large-v1** | mixedbread-ai/mxbai-rerank-large-v1 | cross-encoder | Available |
| **qwen3-reranker-0.6b** | Qwen/Qwen3-Reranker-0.6B | listwise | ✅ Tested (R0) |
| **qwen3-reranker-4b** | Qwen/Qwen3-Reranker-4B | listwise | ✅ Tested (R0) |
| **bge-reranker-v2-m3** | BAAI/bge-reranker-v2-m3 | cross-encoder | ✅ Tested + HPO |
| **bge-reranker-gemma2-lightweight** | BAAI/bge-reranker-v2.5-gemma2-lightweight | lightweight | Available |
| **ms-marco-minilm** | cross-encoder/ms-marco-MiniLM-L-12-v2 | cross-encoder | ✅ Tested (R0) |

### 16.2 R0 Off-the-Shelf Results (dev_select, fusion-rrf, K=20)

| Rank | Model | nDCG@10 | MRR@10 | Recall@10 | FalseEvid | Notes |
|------|-------|---------|--------|-----------|-----------|-------|
| 1 | **bge-reranker-v2-m3** | **0.6962** | 0.6331 | 0.8871 | 0.00 | Best overall |
| 2 | jina-reranker-v2 | 0.6940 | 0.6314 | 0.8769 | — | Close second |
| 3 | ms-marco-minilm | 0.6571 | 0.5827 | 0.8871 | — | Fastest baseline |
| 4 | mxbai-rerank-base-v1 | 0.6412 | 0.5648 | 0.8731 | — | DeBERTa-v3 |
| 5 | jina-reranker-v3 | 0.5236 | 0.4270 | 0.8165 | 0.86 | Listwise, high false-evid |
| 6 | qwen3-reranker-4b | 0.5118 | 0.4149 | 0.8169 | 0.98 | High false-evid |
| 7 | qwen3-reranker-0.6b | 0.4966 | 0.3964 | — | 0.75 | High false-evid |
| 8 | mxbai-rerank-large-v2 | 0.4917 | 0.3853 | — | 1.00 | All queries return evidence |
| 9 | mxbai-rerank-base-v2 | 0.4795 | 0.3846 | 0.7693 | 1.00 | All queries return evidence |

**Key Observations (2026-01-10):**
- **bge-reranker-v2-m3 remains the clear winner** with nDCG@10=0.6962 and 0% false evidence
- **New models (v2/v3 series) underperform on this domain** despite strong BEIR claims
- **False evidence rate is critical:** mxbai-v2 and qwen3 models return evidence for nearly all queries
- **Listwise rerankers (jina-v3, qwen3)** need calibration/thresholding to work in this domain
- **Cross-encoder baselines (bge, jina-v2, minilm)** better calibrated out-of-box

### 16.3 HPO Results (Tier A: Inference-Only)

| Model | Best nDCG@10 | MRR@10 | Recall@10 | FalseEvid | Best Params |
|-------|--------------|--------|-----------|-----------|-------------|
| **jina-reranker-v2** | **0.6968** | 0.6346 | 0.8899 | 0.00 | K=10, len=1024, thresh=1.53 |
| bge-reranker-v2-m3 | 0.6962 | 0.6331 | 0.8871 | 0.00 | K=20, len=384, thresh=4.7 |

**Key Finding (2026-01-10):** After HPO, **jina-reranker-v2 beats bge-reranker-v2-m3** by +0.06% nDCG@10!
- Optimal K is smaller (10 vs 20) - focusing on top candidates improves precision
- Both achieve 0% false evidence rate with proper thresholding
- jina-v2 achieves slightly better Recall@10 (0.8899 vs 0.8871)

### 16.4 Summary: Best Reranker Configuration

**Production recommendation: jina-reranker-v2**
```yaml
reranker:
  model: jinaai/jina-reranker-v2-base-multilingual
  top_k_rerank: 10
  max_length: 1024
  score_threshold: 1.53
```

Expected metrics (dev_select):
- nDCG@10: 0.6968
- MRR@10: 0.6346
- Recall@10: 0.8899
- False evidence rate: 0%

### 16.5 Inference Optimization (Implemented 2026-01-10)

**Optimized inference module:** `src/final_sc_review/reranker/optimized_inference.py`

Implements:
1. **Multi-query batch processing** - Process N queries in single forward pass
2. **Length bucketing** - Group sequences by similar lengths to minimize padding
3. **Prefetching dataloader** - Prepare next batch while GPU is processing
4. **Score caching** - Cache results for HPO speedup

**Benchmark results (jina-reranker-v2, 200 queries, K=20):**

| Method | Throughput | Speedup |
|--------|------------|---------|
| Baseline (sequential) | 1,863 pairs/sec | 1.0x |
| Optimized (bs=48, bucketing) | 4,258 pairs/sec | 2.3x |
| **Optimized (bs=96, bucketing)** | **5,045 pairs/sec** | **2.7x** |
| Optimized (bs=48, no bucket) | 4,404 pairs/sec | 2.4x |

**Usage:**
```python
from final_sc_review.reranker.optimized_inference import BatchReranker, QueryCandidates

batch_reranker = BatchReranker(zoo, "jina-reranker-v2")
results = batch_reranker.rerank_batch(queries, batch_size=96, use_bucketing=True)
```

### 16.6 Next Steps

1. **Train fine-tuned rerankers (R1-R4)** - Domain adaptation may push beyond 0.70
2. **Calibration experiments** - Add threshold tuning for high false-evidence models
3. **Test bge-reranker-gemma2-lightweight** - Potential speed/quality tradeoff
4. **Integrate optimized inference into HPO** - Use BatchReranker for faster HPO trials

---
