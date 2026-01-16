# Final Report: Supersearch Label Leakage Verification

**Date**: 2026-01-16
**Branch**: `verify_fix_supersearch_leakage_dynk`
**Status**: Leakage CONFIRMED and FIXED

---

## Executive Summary

The original supersearch NE (no-evidence) detection pipeline showed suspiciously high performance:
- **Original Claim**: ~99.93% TPR at ~3.05% FPR

Investigation revealed **severe label leakage** in the feature pipeline. After removing leaky features:
- **Clean Performance**: ~10.7% TPR at 5% FPR (AUROC ~0.60)

This represents a **~89 percentage point drop** in performance, confirming the leakage hypothesis.

---

## 1. Leakage Audit Results

### 1.1 Summary

| Category | Count |
|----------|-------|
| Total features audited | 33 |
| DEPLOYABLE (safe) | 15 |
| **LEAKY (forbidden)** | **8** |
| IDENTIFIER | 8 |
| LABEL | 2 |

### 1.2 Leaky Features Identified

| Feature | AUC vs Label | Description |
|---------|--------------|-------------|
| `mrr` | **0.9554** | Mean Reciprocal Rank - directly uses gold labels |
| `recall_at_10` | 0.8985 | Recall@10 - uses gold labels |
| `recall_at_5` | 0.8038 | Recall@5 - uses gold labels |
| `recall_at_3` | 0.7295 | Recall@3 - uses gold labels |
| `recall_at_1` | 0.5993 | Recall@1 - uses gold labels |
| `min_gold_rank` | 0.3114 | Minimum rank of gold items |
| `max_gold_rank` | 0.3276 | Maximum rank of gold items |
| `mean_gold_rank` | 0.3208 | Mean rank of gold items |

**Critical Finding**: `mrr` has AUC=0.9554 against the has_evidence label, making it nearly a perfect proxy for the label itself.

### 1.3 Leakage Root Cause

In `scripts/supersearch/build_feature_store.py`, the `compute_rank_features()` function:
```python
def compute_rank_features(candidate_ids, gold_ids, ...):  # <-- gold_ids is leakage
    for idx in sorted_indices:
        if candidate_ids[idx] in gold_set:  # <-- uses ground truth!
            gold_ranks.append(rank + 1)
    features["mrr"] = 1.0 / min(gold_ranks)  # <-- LEAKY FEATURE
```

This function was called during feature computation, creating features that depend on ground truth labels.

---

## 2. Clean Feature Pipeline

### 2.1 Deployable Features (No Gold Labels)

Created `scripts/verification/build_deployable_features.py` with explicit feature provenance:

| Feature Category | Examples | Input |
|-----------------|----------|-------|
| Score statistics | max, mean, std, median | reranker_scores |
| Gap features | top1_top2_gap, top1_mean_gap | reranker_scores |
| Concentration | SoftMRR, Mass@K | softmax(scores) |
| Entropy | entropy_top5, entropy_full | softmax(scores) |
| Cross-model | retriever_reranker_corr | both scores |

Total: **26 deployable features** that use only inference-time signals.

### 2.2 Feature Provenance Enforcement

```python
class FeatureMode(Enum):
    DEPLOYABLE = "deployable"   # Safe for deployment
    EVALUATION = "evaluation"   # Uses gold labels (eval only)

@dataclass
class FeatureSpec:
    name: str
    mode: FeatureMode
    inputs: List[str]  # Must NOT include gold_ids for DEPLOYABLE
    compute_fn: Callable
```

---

## 3. Clean Evaluation Results

### 3.1 5-Fold Cross-Validation (Nested, Post-Disjoint)

| Model | AUROC | TPR@5%FPR | Precision | F1 |
|-------|-------|-----------|-----------|-----|
| **rf_100** | 0.5963 | **0.1095** | 0.184 | 0.133 |
| hgb_100 | **0.6019** | 0.0964 | 0.166 | 0.124 |
| logreg | 0.5890 | 0.0914 | 0.151 | 0.109 |
| logreg_c01 | 0.5912 | 0.0906 | 0.161 | 0.118 |
| rf_200 | 0.5842 | 0.0812 | 0.152 | 0.103 |

**Best model by TPR@5%FPR**: Random Forest (100 trees) with TPR=10.95%
**Best model by AUROC**: HistGradientBoosting with AUROC=0.6019

### 3.2 Performance Comparison

| Metric | With Leakage | Clean | Drop |
|--------|--------------|-------|------|
| TPR@5%FPR | ~99.93% | ~10.95% | **-88.98pp** |
| AUROC | ~0.99+ | ~0.602 | **-0.39** |

This confirms the original performance was almost entirely due to label leakage.

---

## 4. Dynamic-K Implementation

### 4.1 Configuration

```python
k_min = 2        # Minimum results to return
hard_cap = 10    # Maximum results (deployment constraint)
k_max_ratio = 0.5  # Adaptive: k_max1 = min(hard_cap, ceil(0.5 * N))
```

### 4.2 K Selection Policies

**DK1: Mass Threshold**
```
K = min{k : cumsum(softmax(scores))_k >= gamma}
```
- gamma=0.9: Return enough candidates to capture 90% probability mass

**DK2: Score Gap / Knee Detection**
```
K = argmax_k (score_k - score_{k+1})
```
- Find natural "elbow" in score distribution

### 4.3 k_max1 Calculation

| N_candidates | k_max1 | Reasoning |
|--------------|--------|-----------|
| 20 | 10 | ceil(0.5*20)=10, min(10,10)=10 |
| 12 | 6 | ceil(0.5*12)=6, min(10,6)=6 |
| 40 | 10 | ceil(0.5*40)=20, min(10,20)=10 (capped) |
| 4 | 2 | ceil(0.5*4)=2, max(2,2)=2 (k_min floor) |

---

## 5. Test Coverage

### 5.1 Unit Tests Created

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_no_leaky_features.py` | 11 | ✅ Pass |
| `tests/test_dynamic_k_caps.py` | 15 | ✅ Pass |
| **Total** | **26** | **✅ All Pass** |

### 5.2 Leakage Prevention Tests

- `test_compute_deployable_features_no_leakage`: Verifies no forbidden patterns
- `test_deployable_feature_registry_no_gold_inputs`: Checks feature provenance
- `test_classify_feature_leaky`: Validates leaky feature detection

---

## 6. Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Leakage audit script | `scripts/verification/audit_feature_leakage.py` | ✅ |
| Clean feature builder | `scripts/verification/build_deployable_features.py` | ✅ |
| Clean evaluation script | `scripts/verification/run_clean_supersearch_eval.py` | ✅ |
| Unit tests | `tests/test_no_leaky_features.py`, `tests/test_dynamic_k_caps.py` | ✅ |
| Research notes | `docs/verification/research_notes_ne_and_dynk.md` | ✅ |
| Audit report | `outputs/verification_fix/*/leakage_audit_report.md` | ✅ |
| This final report | `docs/verification/FINAL_REPORT_leakage_verification.md` | ✅ |

---

## 7. Recommendations

### 7.1 Immediate Actions

1. **Do not deploy** the original supersearch NE classifier
2. **Use clean feature pipeline** from `build_deployable_features.py`
3. **Add CI check** for leaky features before training

### 7.2 Improving Clean Performance

Current clean performance (~60% AUROC) is near random. To improve:

1. **Better features**: Add query difficulty features, criterion-specific signals
2. **Calibration**: Temperature-scale reranker scores
3. **Ensemble**: Combine retriever and reranker confidence signals
4. **More data**: The NE detection task may need more training examples

### 7.3 Process Improvements

1. **Feature provenance tracking**: Every feature should declare its inputs
2. **Automated leakage detection**: Run `audit_feature_leakage.py` in CI
3. **Separate evaluation features**: Never compute leaky features during training

---

## 8. Conclusion

The investigation confirmed that the original supersearch NE detection pipeline suffered from severe label leakage. The `mrr` feature, which directly depends on gold labels, had an AUC of 0.9554 against the target label - explaining the ~99.93% TPR claimed.

After removing all leaky features and implementing a clean feature pipeline with explicit provenance tracking, the true performance is approximately:
- **AUROC: 0.602** (vs ~0.99 with leakage)
- **TPR@5%FPR: 10.95%** (vs ~99.93% with leakage)

This represents a fundamental capability limitation, not a modeling issue. The task of predicting whether evidence exists based solely on score distribution features is inherently difficult.

The codebase now includes:
- Proper feature separation (deployable vs evaluation-only)
- Dynamic-K implementation with deployment caps
- Comprehensive tests to prevent future leakage
- Documentation of the issue and fix

---

*Report generated: 2026-01-16*
*Author: Claude (verification task)*
