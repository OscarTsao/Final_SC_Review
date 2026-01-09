# docs/MODEL_INVENTORY.md

Last updated: 2026-01-09  
Repo: OscarTsao/Final_SC_Review  
Goal: MAXOUT deployable evidence extraction under extreme No-Evidence prevalence (paper-grade, no leakage).

This file defines:
1) **What “exhaustive” means** (scope + tiers)
2) The **model inventory** (retrievers + rerankers)
3) The **loss inventory** (pointwise/pairwise/listwise/hybrid + retriever contrastive)
4) The **postprocess inventory** (calibration + no-evidence + dynamic-K)
5) A **coverage contract**: what must be tried for every candidate system

---

## 0) What “SOTA included” means (scope)

We cannot guarantee “all SOTA models” globally. Instead we guarantee **SOTA coverage within a declared, auditable scope**:

### Scope S (must satisfy all)
- **Reproducible**: weights downloadable (HF/Git), version pinned, checksummed.
- **Academic-safe license**: license recorded; any non-commercial restriction flagged.
- **Single GPU**: must run on single RTX 5090. Training may use PEFT (LoRA/QLoRA) as needed.
- **Text-only**: multimodal models are excluded unless clearly beneficial (default: exclude).
- **Local-only**: no external APIs for training; Gemini only for evaluation/error analysis and fully cached.

### Tiers
- **Tier-0 (must-run)**: strong, widely used, and feasible on single GPU, plus any models already used in repo.
- **Tier-1 (run if accessible)**: gated or heavier but high-performing; run if license + access OK.
- **Tier-2 (optional)**: exploratory; run only if time remains after Tier-0/1 converge.

---

## 1) Retriever candidates (sentence candidates within a post)

> IMPORTANT: The dataset often has ~20 sentences/post. Retrieval “K=200” is meaningless.  
> Retriever evaluation must use **K ∈ {3, 5, 10}** plus **K = min(20, n_sent)** for sanity.  
> Also include a “no-retriever” baseline: score ALL sentences (since n is small).

### 1.1 Baselines (Tier-0)
- **ALL_SENTENCES** (no retrieval): candidates = all sentences in post.
- **BM25** within-post lexical scoring (rank_bm25 or Lucene-style).
- **BM25 + RM3** (optional; only if easy to implement).

### 1.2 Sparse neural retrievers (Tier-0/1)
- **SPLADE**:
  - `naver/splade-cocondenser-ensembledistil` (Tier-0)
  - `naver/splade_v2_distil` (Tier-1 baseline)
  Notes: sparse vectors; good lexical+semantic matching.

### 1.3 Dense embedding retrievers (Tier-0/1)
- **BAAI/bge-m3** (Tier-0)
  - Use dense + sparse + multi-vector variants (ablate all).
- **NVIDIA llama-nemotron-embed-1b-v2** (Tier-0/1 depending on license approval)
  - Note: supports long context + Matryoshka embeddings (dynamic dimension).
- **intfloat/e5-mistral-7b-instruct** (Tier-1)
- **Alibaba-NLP/gte-Qwen2-7B-instruct** (Tier-1)
- **nvidia/NV-Retriever-v1** (Tier-1; gated)
- **nvidia/NV-Embed-v2** (Tier-1; gated) — if you can access.

### 1.4 Late interaction retrievers (Tier-2 unless easy)
- **ColBERTv2**:
  - `colbert-ir/colbertv2.0`
  Caution: late interaction overhead may be overkill for within-post (small corpora), but worth as a “ceiling” check.

---

## 2) Reranker candidates (cross-encoder / listwise)

### 2.1 Baselines (Tier-0)
- **No reranker**: retrieval-only (or ALL_SENTENCES + heuristic)
- **BGE reranker**:
  - `BAAI/bge-reranker-v2-m3` (Tier-0)

### 2.2 Strong modern rerankers (Tier-0/1)
- **NVIDIA llama-nemotron-rerank-1b-v2** (Tier-0/1)
- **jinaai/jina-reranker-v3** (Tier-1; may be large)
- **mixedbread rerank v2 family** (Tier-1)
  - `mixedbread-ai/mxbai-rerank-base-v2`
  - `mixedbread-ai/mxbai-rerank-large-v2`
- **Qwen3 reranker family** (Tier-1)
  - `Qwen/Qwen3-Reranker-0.6B` (and larger if available)

### 2.3 Optional / gated / service-only (Tier-2)
- GTE rerank service exists commercially; only include if open weights exist and license fits.

---

## 3) Loss functions & training objectives inventory (EXHAUSTIVE requirement)

### 3.1 Reranker losses (must implement + HPO over)
You MUST implement and sweep at least these families:

#### Pointwise (binary relevance)
- BCEWithLogits (standard)
- **Focal loss** (handle extreme imbalance)
- Label smoothing / asymmetric losses (optional)

#### Pairwise
- RankNet (logistic pairwise)
- Margin ranking / hinge pairwise
- Hard-negative weighted pairwise (importance weights)

#### Listwise (ranking over a list per (post, criterion))
- **ListNet** (Plackett–Luce / top-k probability style)
- **ListMLE** (likelihood of permutation)
- **LambdaRank-style** (metric-sensitive gradients)
- **LambdaLoss** variants (metric-driven losses approximating NDCG)

#### Hybrid (must be supported)
- Weighted sum of:
  - α * pointwise + β * pairwise + γ * listwise
- α,β,γ are hyperparameters in HPO (subject to normalization α+β+γ=1 or free but constrained).

### 3.2 Retriever training losses (if finetuning retriever)
- Contrastive InfoNCE (in-batch negatives)
- MultipleNegativesRankingLoss (SentenceTransformers style)
- Hard-negative mining schedule (ANN/mined from current model)
- Matryoshka embedding loss (only if model supports dynamic dims, e.g., Nemotron embed)

### 3.3 Distillation (optional but high value)
- Distill a strong reranker into a retriever (student bi-encoder) to improve first-stage quality.

---

## 4) Postprocess policy inventory (must be tried for every promoted model-pair)

**Postprocess is part of the deployed system** and must be tuned on DEV only.

### 4.1 Calibration (must implement)
- Temperature scaling
- Platt scaling
- Isotonic regression
Selection criterion: dev ECE/Brier/NLL + downstream selection metrics.

### 4.2 No-Evidence detection (must implement ALL tracks)
Track A — **Sentinel (SQuAD2-style)**
- Add sentinel candidate to every group
- Train sentinel as positive for empty groups
- Decision rule based on **gap**: score_null − score_best > τ

Track B — **Abstention classifier**
- Group-level classifier over score distribution features:
  - top1, top2, gap, entropy, std, skew, topK mean, #above thresholds, etc.
- Output p_empty; empty if p_empty > τ

Track C — **Risk-coverage / selective prediction**
- Evaluate risk-coverage curves
- Provide coverage-constrained policy (e.g., “at most X false evidence rate on empty queries”)
- Optionally include conformal risk control variants if implementable.

### 4.3 Dynamic-K / threshold selection (must implement ALL families)
- Fixed K baseline: K ∈ {1,3,5,10}
- Per-criterion threshold τ_c: select all with p_cal >= τ_c, cap at Kmax, allow k=0
- Mass-based: select smallest k such that cumulative probability mass >= p0
- Gap / elbow heuristics
- Learned k-predictor (small MLP) (optional, only if stable)

---

## 5) Coverage contract (definition of “exhaustive done”)

A model system is defined by:
- Retriever choice + training state (frozen/LoRA/full)
- Reranker choice + loss family + training state
- Postprocess policy (calibration + no-evidence + dynamic-K)
- All configs + seeds

**Exhaustive Tier-0 completion requires:**
1) For EACH Tier-0 retriever, at least:
   - frozen baseline evaluation
   - finetuned (LoRA or full) HPO run (unless proven unnecessary by decision gate)
2) For EACH Tier-0 reranker, at least:
   - frozen baseline evaluation
   - finetuned HPO run sweeping pointwise/pairwise/listwise/hybrid losses
3) For EACH promoted (retriever, reranker) pair:
   - postprocess HPO covering all No-Evidence tracks (A/B/C) and dynamic-K families
4) For every stage:
   - per_query.csv + summary.json + manifest.json + logs.txt present

A script `scripts/check_coverage.py` must assert coverage, and fail if any required cell is missing.
