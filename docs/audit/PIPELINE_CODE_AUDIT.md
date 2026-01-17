# Pipeline Code Audit Report

**Audit Timestamp**: 20260117_203822
**Branch**: audit_full_eval_gold_standard
**Auditor**: Gold Standard Evaluation Audit

---

## 1. Executive Summary

This audit reviews the codebase for potential correctness issues related to:
- Data splitting and leakage prevention
- Metric computation accuracy
- Training/evaluation protocol
- Reproducibility guarantees

**Overall Assessment**: ✅ **PASS** with minor observations

---

## 2. Data Splitting Audit

### 2.1 Split Implementation (`src/final_sc_review/data/splits.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Post-ID disjoint | ✅ PASS | `k_fold_post_ids()` ensures no post appears in multiple folds |
| Deterministic | ✅ PASS | Uses `random.Random(seed)` for reproducible shuffling |
| Ratio validation | ✅ PASS | Validates `train + val + test = 1.0` |
| Empty handling | ✅ PASS | Handles empty input gracefully |

**Code Review**:
```python
# k_fold_post_ids (lines 31-46)
unique_ids = sorted(set(post_ids))  # ✅ Deduplicates
rng = random.Random(seed)           # ✅ Seeded RNG
rng.shuffle(unique_ids)             # ✅ In-place shuffle
# Round-robin assignment to folds
for idx, pid in enumerate(unique_ids):
    folds[idx % k].append(pid)      # ✅ Balanced distribution
```

### 2.2 Verified Properties

1. **Disjointness**: Verified via `tests/test_no_leakage_splits.py` and audit script
2. **Balance**: Folds differ by at most 1 post (verified in audit output)
3. **Determinism**: Same seed produces identical splits (verified)

---

## 3. Feature Leakage Prevention

### 3.1 GNN Feature Extraction (`src/final_sc_review/gnn/graphs/features.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Forbidden list | ✅ PASS | `LEAKAGE_FEATURES` set contains all gold-derived patterns |
| Runtime check | ✅ PASS | `check_leakage()` validates features at runtime |
| Assertion | ✅ PASS | `assert_no_leakage()` raises `ValueError` on violations |

**Verified Safe Features**:
- Embeddings (from pretrained model)
- Reranker scores (from inference)
- Rank positions (derived from scores)
- Score statistics (z-score, gaps)
- Graph structure (degree, similarity)

**Verified Blocked Features**:
- `is_gold`, `groundtruth`
- `mrr`, `recall_at_*`, `ndcg_at_*`, `map_at_*`
- `gold_rank`, `n_gold_sentences`, `gold_positions`

### 3.2 Test Coverage

- `tests/test_gnn_no_leakage.py`: Verifies feature extraction produces no leaky features
- `tests/test_no_leaky_features.py`: General leakage detection tests

---

## 4. Training Protocol Audit

### 4.1 GNN Trainer (`src/final_sc_review/gnn/training/trainer.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Train/val separation | ✅ PASS | Separate loaders, no cross-contamination |
| Early stopping | ✅ PASS | Based on validation metric only |
| Best model restore | ✅ PASS | Restores from validation-selected checkpoint |
| Gradient isolation | ✅ PASS | `torch.no_grad()` for validation |

**Training Flow**:
```
1. Train on train_loader
2. Validate on val_loader (torch.no_grad)
3. Track best validation metric
4. Early stop based on val metric
5. Restore best val model
```

### 4.2 Cross-Validation Protocol (`src/final_sc_review/gnn/evaluation/cv.py`)

| Check | Status | Notes |
|-------|--------|-------|
| Outer fold isolation | ✅ PASS | Each test fold never seen during training |
| Inner split | ✅ PASS | 70/30 train/tune split within training portion |
| Threshold tuning | ✅ PASS | Threshold selected on tune set, applied to test |
| No test tuning | ✅ PASS | Test predictions made before any test-based selection |

**Protocol Verification**:
```
Fold i:
  - Test set: fold_i posts (held out entirely)
  - Train/Tune: other folds split 70/30
  - Model trained on Train, stopped on Tune
  - Threshold selected on Tune
  - Final metrics computed on Test
```

---

## 5. Metric Computation Audit

### 5.1 Ranking Metrics (`src/final_sc_review/metrics/ranking.py`)

| Metric | Implementation | Status |
|--------|---------------|--------|
| Recall@K | Set intersection / gold count | ✅ Correct |
| MRR@K | 1 / (first relevant rank) | ✅ Correct |
| MAP@K | Mean precision at hits | ✅ Correct |
| nDCG@K | DCG / IDCG with binary rel | ✅ Correct |

**Verified Against**:
- sklearn implementations (AUROC, AUPRC)
- Manual calculations (ranking metrics)
- 37 unit tests covering edge cases

### 5.2 NE Gate Metrics (`src/final_sc_review/gnn/evaluation/metrics.py`)

| Metric | Implementation | Status |
|--------|---------------|--------|
| AUROC | sklearn.roc_auc_score | ✅ Correct |
| AUPRC | sklearn.average_precision_score | ✅ Correct |
| TPR@FPR | Interpolated from ROC curve | ✅ Correct |
| Threshold | From ROC curve at target FPR | ✅ Correct |

---

## 6. Reproducibility Audit

### 6.1 Seed Management

| Component | Seed Source | Status |
|-----------|-------------|--------|
| Data splits | `seed=42` (config) | ✅ Deterministic |
| PyTorch | Set in trainer | ✅ Deterministic |
| NumPy | Set in trainer | ✅ Deterministic |

### 6.2 Cache Fingerprinting

The caching system uses content-based fingerprints:
- Model config hash
- Corpus hash
- Parameter hash

This ensures cache invalidation on any input change.

---

## 7. Identified Issues (Minor)

### 7.1 Missing Seed in Some Configs

**Finding**: Some GNN experiment configs (P2, P3, P4) don't explicitly set seed.

**Impact**: Low - default seed is used, but non-explicit.

**Recommendation**: Always explicitly set seed in config.

### 7.2 Aggregated String Format

**Finding**: Some cv_results.json files use string format for aggregated metrics ("0.8911 +/- 0.0091").

**Impact**: None - parsing handled correctly in verification.

**Recommendation**: Standardize on numeric dict format.

---

## 8. Test Suite Verification

| Test File | Tests | Status |
|-----------|-------|--------|
| test_metrics.py | 1 | ✅ Pass |
| test_metrics_comprehensive.py | 37 | ✅ Pass |
| test_no_leakage_splits.py | 2 | ✅ Pass |
| test_gnn_no_leakage.py | 3 | ✅ Pass |
| test_no_leaky_features.py | 1 | ✅ Pass |
| test_dynamic_k_caps.py | 2 | ✅ Pass |

Total: **46 tests passing**

---

## 9. Conclusion

The pipeline implementation is **correct** with respect to:

1. ✅ Data splitting maintains post-ID disjointness
2. ✅ No label leakage in features
3. ✅ Training protocol properly isolates test data
4. ✅ Metrics computed correctly
5. ✅ Reproducibility ensured via seeding

**Minor Observations**:
- Some configs missing explicit seed (non-critical)
- Format inconsistency in result files (cosmetic)

**Recommendation**: Proceed with confidence in evaluation results.

---

*Audit completed: 2026-01-17*
