# LLM Integration & Full E2E Evaluation: Implementation Plan

**Date**: 2026-01-17
**Status**: PLANNING
**Estimated Effort**: 3-5 days
**Estimated LLM API Cost**: $50-200 (depending on model and coverage)

---

## Executive Summary

This document outlines the plan to complete:
1. **Verification & Audit** (verification already largely complete)
2. **Full E2E Evaluation** with visualization
3. **LLM Integration Experiments** (novel research contribution)

**Current State**: GNN pipeline (P1-P4) trained with nv-embed-v2, achieving AUROC=0.9053 for NE detection.

**Goal**: Enhance with LLM-based components and provide deployment-ready evaluation.

---

## Phase 1: Verification & Audit ✅ (MOSTLY COMPLETE)

### A. Data Split Verification ✅
**Status**: Already verified via tests
**Evidence**:
- `tests/test_no_leakage_splits.py` - Post-ID disjoint tests
- 5-fold CV with post_id split confirmed in graph builder

**Action**: Run verification script to confirm
```bash
pytest tests/test_no_leakage_splits.py -v
```

### B. Leakage Detection ✅
**Status**: Comprehensive tests exist
**Evidence**:
- `tests/test_gnn_no_leakage.py` - 31 leakage tests
- Feature extraction verified independent of labels
- Graph construction verified independent of labels

**Action**: Run leakage tests
```bash
pytest tests/test_gnn_no_leakage.py -v
```

### C. Metric Verification ⚠️
**Status**: Needs independent recomputation
**Gap**: Cross-check reported metrics against independent implementation

**Action**: Run independent metric verification
```bash
python scripts/gnn/recompute_metrics_independent.py \
    --graph_dir data/cache/gnn_nvembed/20260117_215510 \
    --output outputs/verification/metric_check.json
```

### D. Dynamic-K Sanity ⚠️
**Status**: Needs debugging
**Gap**: Why do gamma values produce identical results?

**Action**: Already exists
```bash
python scripts/gnn/debug_dynamic_k_sanity.py \
    --graph_dir data/cache/gnn_nvembed/20260117_215510
```

---

## Phase 2: Full E2E Evaluation 🔨 (PARTIAL)

### A. Comprehensive Metrics ⚠️
**Status**: E2E script exists, needs full run

**Metrics to compute**:
1. **NE Detection**: AUROC, AUPRC, TPR@{1,3,5,10}%FPR, Precision, Recall, F1, MCC, Balanced Acc
2. **Ranking**: MRR, nDCG@{1,3,5,10,20}, Recall@{1,3,5,10,20}, Precision@{1,3,5,10}
3. **Multi-label**: Exact Match, Subset Accuracy, Hamming, Micro-F1, Macro-F1
4. **Dynamic-K**: Hit Rate, Evidence Recall, nDCG, AvgK at various gamma/tau

**Action**:
```bash
python scripts/gnn/run_e2e_eval_and_report.py \
    --graph_dir data/cache/gnn_nvembed/20260117_215510 \
    --output_dir outputs/e2e_full_eval/$(date +%Y%m%d_%H%M%S)
```

### B. Visualization 🔨
**Status**: Script exists, needs execution

**Required plots**:
1. ROC curve (P1, P4, baselines) with operating points
2. PR curve (P1, P4, baselines)
3. Calibration plot + ECE for P4
4. Dynamic-K tradeoffs:
   - AvgK vs Evidence Recall
   - AvgK vs Precision@5
   - AvgK vs Hit Rate
5. Per-criterion breakdown (heatmap)

**Action**:
```bash
python scripts/gnn/make_gnn_e2e_plots.py \
    --results_dir outputs/e2e_full_eval/<timestamp> \
    --output_dir outputs/e2e_full_eval/<timestamp>/plots
```

---

## Phase 3: LLM Integration 🚀 (NEW)

### Overview

Two main components:
1. **LLM Reranker**: Refine top-M candidates using LLM reasoning
2. **LLM Verifier**: Verify each candidate supports the criterion

### A. LLM Reranker (Post-P3)

**Architecture**:
```
Retriever → Reranker → P3 GNN Reranker → LLM Reranker → Dynamic-K
                         (top-24)         (top-10)        (final K)
```

**Input**: Top-M candidates (M=10 recommended for cost)
**Output**: Reordered candidates with LLM confidence scores

**Prompt Design**:
```
You are an expert in mental health assessment. Given a DSM-5 criterion and candidate evidence sentences from a social media post, rank the sentences by how well they support the criterion.

Criterion: {criterion_text}
Post context: {post_text}

Candidates (shuffled to reduce position bias):
1. {sentence_1}
2. {sentence_2}
...

Task: Provide a ranked list from most to least supportive.
Output format: {"rankings": [3, 1, 5, 2, 4, ...], "confidence": [0.9, 0.85, ...]}
```

**Implementation**:
```python
# scripts/llm_integration/llm_reranker.py

class LLMReranker:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.model = model
        self.temperature = temperature

    def rerank(self, candidates, criterion, context):
        # Shuffle candidates to mitigate position bias
        # Call LLM with structured output
        # Return reordered candidates
```

**Evaluation**:
- Run on DEV split only (cost control)
- Compare: P3-refined → LLM-reranked vs baseline
- Metrics: MRR, nDCG@{1,3,5,10}, Recall@{1,3,5,10}

**Cost Estimate**:
- DEV split: ~2,950 queries
- Avg 10 candidates × 50 tokens each = 500 tokens input
- Total: ~1.5M input tokens = $1.50 (gpt-4o-mini)
- Plus output: ~0.25M tokens = $0.75
- **Total per run**: ~$2.25
- **5-fold CV**: ~$11.25

### B. LLM Verifier/Judge

**Architecture**:
```
P4 (has_evidence?) → [if yes] → LLM Verifier (per candidate) → Final candidates
```

**Input**: Each candidate sentence
**Output**: {supports: bool, confidence: float, reasoning: str}

**Prompt Design**:
```
You are evaluating whether a sentence from a social media post supports a specific DSM-5 criterion.

Criterion: {criterion_text}
Sentence: {candidate_text}
Post context: {post_context}

Task: Does this sentence provide evidence supporting the criterion?
Output format: {"supports": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
```

**Decision Logic**:
1. If P4 predicts no_evidence AND high confidence → skip LLM verification
2. If P4 uncertain (0.3 < prob < 0.7) → run LLM verifier on top-K
3. If P4 predicts has_evidence → run LLM verifier to filter false positives

**Implementation**:
```python
# scripts/llm_integration/llm_verifier.py

class LLMVerifier:
    def __init__(self, model="gpt-4o-mini", threshold=0.6):
        self.model = model
        self.threshold = threshold

    def verify_candidates(self, candidates, criterion, p4_prob):
        if p4_prob < 0.2:  # High confidence no evidence
            return []

        # Verify each candidate
        verified = []
        for cand in candidates:
            result = self._verify_single(cand, criterion)
            if result["supports"] and result["confidence"] > self.threshold:
                verified.append(cand)
        return verified
```

**Evaluation**:
- Run on DEV split
- Compare:
  - Baseline: P4 → Dynamic-K
  - +Verifier: P4 → LLM Verifier → filtered candidates
- Metrics: Precision, Recall, F1 at evidence level

**Cost Estimate**:
- DEV split: ~2,950 queries × avg 5 candidates = ~14,750 verifications
- Avg 150 tokens per verification
- Total: ~2.2M input tokens = $2.20
- Plus output: ~0.5M tokens = $1.50
- **Total per run**: ~$3.70
- **5-fold CV**: ~$18.50

### C. Combined Pipeline (LLM-Enhanced)

**Full Architecture**:
```
Retriever (nv-embed-v2)
    ↓ (top-24)
Reranker (jina-reranker-v3)
    ↓
P3 Graph Reranker (GNN score refinement)
    ↓ (refined scores, top-10)
LLM Reranker (reasoning-based reordering) [OPTIONAL]
    ↓
P4 NE Gate (has_evidence?)
    ↓ (if yes)
LLM Verifier (per-candidate verification) [OPTIONAL]
    ↓
P2 Dynamic-K (final K selection)
```

**Ablation Matrix**:
| Variant | P3 | LLM-Reranker | P4 | LLM-Verifier | P2 |
|---------|----|--------------|----|--------------|-----|
| Baseline | ✓ | ✗ | ✓ | ✗ | ✓ |
| +Reranker | ✓ | ✓ | ✓ | ✗ | ✓ |
| +Verifier | ✓ | ✗ | ✓ | ✓ | ✓ |
| +Both | ✓ | ✓ | ✓ | ✓ | ✓ |

**Total Cost Estimate** (5-fold CV):
- LLM Reranker: $11.25
- LLM Verifier: $18.50
- Both: $29.75
- **With 3 ablations**: ~$60

### D. Optional: Distillation

**Goal**: Train a small verifier model using LLM outputs as weak labels

**Approach**:
1. Run LLM Verifier on TRAIN split → collect {sentence, criterion, supports, confidence}
2. Train small cross-encoder (e.g., deberta-v3-small):
   - Input: [CLS] sentence [SEP] criterion [SEP]
   - Output: binary classification + confidence
3. Evaluate distilled model vs LLM verifier

**Cost**: Training time only (no inference cost)

---

## Phase 4: Final Report 📊

### A. Comprehensive Documentation

**File**: `docs/llm_integration/FINAL_FULL_EVAL_AND_LLM_REPORT.md`

**Structure**:
1. **Verification Summary**
   - Data split verification
   - Leakage audit results
   - Metric verification
   - Dynamic-K analysis

2. **E2E Evaluation**
   - Component-wise performance
   - Full pipeline metrics
   - Per-criterion breakdown
   - Tradeoff analysis

3. **LLM Integration Results**
   - LLM Reranker ablation
   - LLM Verifier ablation
   - Combined pipeline results
   - Cost-benefit analysis

4. **Deployment Recommendations**
   - Recommended configuration
   - Operating points
   - Cost projections
   - Monitoring strategy

### B. Reproducibility Package

**Contents**:
```
outputs/llm_integration/<timestamp>/
├── config.yaml                    # Full configuration
├── report.md                      # Human-readable report
├── summary.json                   # Machine-readable metrics
├── leaderboard.csv               # All variants compared
├── plots/
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── calibration_p4.png
│   ├── dynamic_k_tradeoffs.png
│   └── per_criterion_heatmap.png
├── predictions/
│   ├── fold_0.parquet
│   ├── fold_1.parquet
│   ...
└── llm_outputs/
    ├── reranker_responses.jsonl
    └── verifier_responses.jsonl
```

---

## Execution Timeline

### Day 1: Verification & E2E Baseline
- [x] Run verification tests
- [ ] Run independent metric check
- [ ] Run full E2E evaluation
- [ ] Generate all plots

### Day 2-3: LLM Integration
- [ ] Implement LLM Reranker
- [ ] Run Reranker experiments (DEV split)
- [ ] Implement LLM Verifier
- [ ] Run Verifier experiments (DEV split)

### Day 4: Full Evaluation
- [ ] Run 5-fold CV for best LLM config
- [ ] Generate comprehensive results
- [ ] Cost analysis

### Day 5: Documentation
- [ ] Write final report
- [ ] Create deployment guide
- [ ] Package reproducibility artifacts

---

## Risk Assessment

### High Risk
1. **LLM API costs** - Mitigation: Start with DEV split only, use gpt-4o-mini
2. **LLM reliability** - Mitigation: Retry logic, structured output validation
3. **Position bias in LLM** - Mitigation: Shuffle candidates before ranking

### Medium Risk
1. **Metric discrepancies** - Mitigation: Independent recomputation
2. **Dynamic-K behavior** - Mitigation: Debug script exists

### Low Risk
1. **Data leakage** - Mitigation: Comprehensive tests already passing
2. **Split correctness** - Mitigation: Verified via tests

---

## Required Approvals

Before executing:
1. ✅ Budget approval for LLM API costs (~$60 estimated)
2. ✅ Priority confirmation: Full E2E + LLM integration vs other tasks
3. ✅ Timeline confirmation: 3-5 days acceptable?

---

## Next Steps

**Immediate**:
1. Get approval for LLM API budget
2. Confirm priority and timeline
3. Start with Phase 1 verification run

**Pending Approval**:
1. Implement LLM integration
2. Run experiments
3. Generate final report

---

**Questions for Review**:
1. Should we run LLM experiments on DEV split first, or full 5-fold CV?
2. Which LLM model: gpt-4o-mini (cheaper) vs gpt-4o (better)?
3. Should we implement distillation, or focus on direct LLM usage?
4. Any specific visualizations or analyses to prioritize?
