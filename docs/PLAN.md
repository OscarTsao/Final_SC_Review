# PLAN.md - MAXOUT Research Execution Plan

**Last Updated**: 2026-01-09
**Repo**: OscarTsao/Final_SC_Review
**Goal**: Paper-grade S-C evidence retrieval with SOTA coverage and reproducibility guarantees

---

## Phase Overview

| Phase | Name | Description | Gate Criteria |
|-------|------|-------------|---------------|
| 0 | **Audit** | Validate existing artifacts, check invariants | All invariants pass |
| 1 | **Verify** | Verify git state, run pytest, validate runs | Clean state, tests pass |
| 2 | **Profile** | Detect hardware, save system info | hw.json exists |
| 3 | **Baselines** | Run baseline models (frozen), establish floor | Per-query + summary saved |
| 4 | **Retriever Sweep** | HPO over retriever configs (frozen models) | Best config exported |
| 5 | **Reranker Sweep** | HPO over reranker configs (frozen models) | Best config exported |
| 6 | **Reranker Train** | Finetune reranker with hybrid loss HPO | Finetuned model saved |
| 7 | **Postprocess HPO** | Calibration + No-Evidence + Dynamic-K | Best postprocess config |
| 8 | **Retriever Train** | (Conditional) Finetune retriever if gate triggers | Model checkpoint |
| 9 | **GNN** | (Optional) Graph-based enhancement | Enhancement delta |
| 10 | **LLM Judge** | (Optional) Eval with external LLM | Judge scores |
| 11 | **Paper** | Final test eval + paper artifacts | Paper-ready outputs |

---

## Phase Details

### Phase 0: Audit

**Objective**: Validate all existing paper artifacts and check invariant integrity.

**Steps**:
1. Run `scripts/audit_pushed_results.py`
2. Run `scripts/verify_invariants.py`
3. Validate `outputs/paper_audit/` contents

**Gate**: Exit code 0 from both scripts

---

### Phase 1: Verify

**Objective**: Ensure clean repository state and passing tests.

**Steps**:
1. `git status` - must be clean or stashed
2. `pytest -q` - all tests pass
3. `scripts/validate_runs.py` - existing runs valid

**Gate**: All checks pass

---

### Phase 2: Profile

**Objective**: Detect and record hardware configuration.

**Steps**:
1. Run `scripts/hw_probe.py`
2. Save `outputs/system/hw.json`
3. Save `outputs/system/nvidia_smi.txt`
4. Save `outputs/system/pip_freeze.txt`

**Gate**: hw.json exists with valid GPU info

---

### Phase 3: Baselines

**Objective**: Establish performance floor with frozen models.

**Models to evaluate**:
- BGE-M3 (dense only)
- BGE-M3 (dense + sparse)
- BGE-M3 (dense + sparse + ColBERT)
- BM25 baseline
- No-retriever (all sentences)

**Metrics**: Recall@K, nDCG@K, MRR@K for K in {3, 5, 10}

**Gate**: summary.json + per_query.csv for each baseline

---

### Phase 4: Retriever Sweep

**Objective**: Find optimal retriever configuration via HPO.

**Search space**:
- Fusion method: weighted_sum, rrf
- Weight ratios: dense/sparse/colbert
- Top-K retrieval: 10, 20, 50, 100, 200
- Normalization: none, minmax, zscore

**Budget**:
- Standard: 200 trials
- Exhaustive: 1000+ trials with multi-fidelity

**Gate**: best_config.yaml exported, improvement over baseline

---

### Phase 5: Reranker Sweep (Frozen)

**Objective**: Evaluate reranker configurations without training.

**Search space**:
- Reranker model: jina-v3, bge-reranker-v2-m3, mixedbread
- Top-K for rerank: 10, 20, 50
- Score combination: replace, weighted_sum

**Gate**: best_config.yaml exported

---

### Phase 6: Reranker Training HPO

**Objective**: Finetune reranker with hybrid loss optimization.

**Loss families** (from MODEL_INVENTORY.md):
- Pointwise: BCE, Focal
- Pairwise: RankNet, Margin
- Listwise: ListNet, ListMLE, LambdaRank
- Hybrid: weighted combination

**Training config**:
- LoRA rank: [4, 8, 16]
- Learning rate: [1e-5, 5e-5, 1e-4]
- Hard negative mining: enabled
- Empty groups: included

**Budget**:
- Standard: 100 trials
- Exhaustive: 500+ trials

**Gate**: Best model checkpoint saved, val nDCG improved

---

### Phase 7: Postprocess HPO

**Objective**: Optimize postprocessing pipeline.

**Components** (from MODEL_INVENTORY.md):

**Calibration**:
- Temperature scaling
- Platt scaling
- Isotonic regression

**No-Evidence Detection**:
- Track A: Sentinel (SQuAD2-style)
- Track B: Abstention classifier
- Track C: Risk-coverage curves

**Dynamic-K**:
- Fixed K: {1, 3, 5, 10}
- Per-criterion threshold
- Mass-based selection
- Gap/elbow heuristics

**Gate**: All tracks evaluated, best policy selected

---

### Phase 8: Retriever Training (Conditional)

**Decision Gate**: Run only if:
1. Reranker improvement on finetuned model < 3% over frozen
2. Oracle retriever ceiling suggests >5% headroom
3. Explicit user request

**Training**:
- Contrastive InfoNCE loss
- Hard negative mining
- LoRA/full finetuning

**Gate**: Model checkpoint + val metrics

---

### Phase 9: GNN (Optional)

**Objective**: Graph-based sentence relationship modeling.

**Architecture**:
- GAT/GraphSAGE over sentence graph
- Edge features from retriever scores
- Node features from embeddings

**Gate**: Positive delta over non-GNN baseline

---

### Phase 10: LLM Judge (Optional)

**Objective**: Evaluate with external LLM for calibration.

**Requirements**:
- `--allow_external_api` flag
- API key configured
- Full caching to ensure reproducibility

**Gate**: Judge scores saved, correlation with human labels

---

### Phase 11: Paper

**Objective**: Final test evaluation and paper artifact generation.

**Steps**:
1. Load best config from all prior phases
2. Evaluate on TEST split (once only!)
3. Generate paper-ready tables
4. Save artifacts to `outputs/final_test_results/`

**Required outputs**:
- `summary.json` - aggregate metrics
- `per_query.csv` - per-query breakdown
- `manifest.json` - git hash, timestamp, config
- `paper_tables.tex` - LaTeX tables

**Gate**: All outputs present, metrics recorded

---

## Decision Gates

### Gate G1: Retriever Finetuning Decision

```
IF (val_ndcg_with_finetuned_reranker - val_ndcg_with_frozen_reranker) < 0.03:
    AND oracle_retriever_ceiling - current_val_ndcg > 0.05:
    TRIGGER Phase 8 (Retriever Training)
ELSE:
    SKIP Phase 8
```

### Gate G2: GNN Decision

```
IF Phase 6 (Reranker Training) showed diminishing returns:
    AND graph structure is meaningful (avg sentences per post > 10):
    SUGGEST Phase 9 (GNN)
ELSE:
    SKIP Phase 9
```

### Gate G3: External API Decision

```
IF --allow_external_api flag set:
    ENABLE Phase 10 (LLM Judge)
    ENABLE teacher distillation in Phase 6
ELSE:
    SKIP Phase 10
    USE local-only training
```

---

## Invariants (must hold at all times)

1. **I1 - Metrics Correctness**: Ranking metrics match reference implementation
2. **I2 - Split Leakage Prevention**: Train/val/test post_ids are disjoint
3. **I3 - Determinism**: Same config + seed = identical results
4. **I4 - Output Contract**: All eval outputs have summary.json, per_query.csv, manifest.json
5. **I5 - No Test Tuning**: HPO uses val split only; test evaluated once at end
6. **I6 - Empty Query Handling**: System handles no-evidence queries correctly

---

## Coverage Contract

A complete research run must satisfy:

1. **Tier-0 Models**: All Tier-0 retrievers and rerankers evaluated (frozen + finetuned)
2. **Loss Families**: All loss families from MODEL_INVENTORY.md swept
3. **Postprocess Tracks**: All No-Evidence tracks (A/B/C) evaluated
4. **Dynamic-K**: All Dynamic-K methods evaluated
5. **Artifacts**: Per-query + summary for every experiment

Use `scripts/check_coverage.py` to verify.

---

## Execution Commands

```bash
# Full pipeline (strict mode)
python scripts/research_driver.py --phase audit --strict
python scripts/research_driver.py --phase verify --strict
python scripts/research_driver.py --phase profile --strict
python scripts/research_driver.py --phase baselines --strict
python scripts/research_driver.py --phase retriever_sweep --budget exhaustive
python scripts/research_driver.py --phase reranker_sweep --budget exhaustive
python scripts/research_driver.py --phase reranker_train --budget exhaustive
python scripts/research_driver.py --phase postprocess_hpo --budget exhaustive
# Conditional phases
python scripts/research_driver.py --phase retriever_train --budget exhaustive  # if gate triggers
python scripts/research_driver.py --phase paper --strict

# Validation after each phase
python scripts/validate_runs.py
python scripts/verify_invariants.py
python scripts/check_coverage.py
```

---

## Troubleshooting: SOTA Underperforms

If a supposedly stronger model underperforms older models:

1. **Check Training Objective**: Did we actually train with the intended loss?
2. **Check Data Construction**: Are empty groups included? Are negatives correct?
3. **Check Evaluation**: Are K values appropriate? No test tuning?
4. **Run Ablations**: Same seed, same data, change one factor only
5. **Fix and Re-run**: Patch bug, re-run minimal reproduction
