# Optimization Plan: Baseline Metrics

**Created:** 2026-01-14
**Branch:** `optimize_postproc_hpo_20260114`
**Source Reports:**
- `outputs/assessment_full/20260114_192708/report.md` (5-fold CV)
- `outputs/final_deployment_assessment/report.md` (train/val/test split)

---

## Executive Summary

This document records baseline performance metrics before optimization. The goal is to optimize the deployment pipeline for:
1. **Accurate no-evidence detection** (minimize false positives while maintaining high recall)
2. **Complete evidence extraction** (don't miss relevant evidence)
3. **Cost control** (minimize returned candidates)

### Current Issues

| Issue | Current Value | Target |
|-------|---------------|--------|
| Negative coverage (FPR) | **14.5%** | ≤ 5% |
| Positive coverage (TPR) | 89.8% | ≥ 90% |
| Evidence recall (dynamic-K) | 80.8% | ≥ 93% |
| Avg K returned | 1.71 | ≤ 5.0 |

**Main bottleneck:** The threshold-based NE detection predicts too many queries as "has evidence" (14.5% false positive rate), significantly above the 5% target.

---

## Dataset Statistics (5-fold CV)

| Statistic | Value |
|-----------|-------|
| Total posts | 1,477 |
| Total queries | 13,510 |
| Has-evidence queries | 1,276 (9.4%) |
| No-evidence queries | 12,234 (90.6%) |
| Queries per fold (val) | ~2,700 |

---

## 1. Retriever Performance (BGE-M3 Hybrid)

*Only on has-evidence queries (n=240.8 per fold)*

| Metric | Mean | Std |
|--------|------|-----|
| nDCG@1 | 0.5538 | 0.0411 |
| nDCG@3 | 0.6762 | 0.0286 |
| nDCG@5 | 0.7103 | 0.0307 |
| nDCG@10 | 0.7385 | 0.0253 |
| nDCG@20 | 0.7572 | 0.0233 |
| Recall@1 | 0.5149 | 0.0309 |
| Recall@3 | 0.7652 | 0.0229 |
| Recall@5 | 0.8452 | 0.0328 |
| Recall@10 | 0.9288 | 0.0109 |
| Recall@20 | 1.0000 | 0.0000 |
| MRR@10 | 0.6843 | 0.0312 |
| MAP@10 | 0.6764 | 0.0312 |

**Note:** Retriever achieves 100% recall@20 (ceiling for reranker).

---

## 2. Reranker Performance (Jina-v3 + NO_EVIDENCE)

*Only on has-evidence queries (n=240.8 per fold)*

| Metric | Mean | Std |
|--------|------|-----|
| nDCG@1 | 0.7167 | 0.0304 |
| nDCG@3 | 0.8053 | 0.0284 |
| nDCG@5 | 0.8326 | 0.0269 |
| nDCG@10 | 0.8502 | 0.0232 |
| nDCG@20 | 0.8552 | 0.0201 |
| Recall@1 | 0.6771 | 0.0226 |
| Recall@3 | 0.8666 | 0.0286 |
| Recall@5 | **0.9308** | 0.0245 |
| Recall@10 | 0.9815 | 0.0142 |
| Recall@20 | 1.0000 | 0.0000 |
| MRR@10 | 0.8119 | 0.0253 |
| MAP@10 | 0.8057 | 0.0273 |

**Key insight:** Reranker@5 achieves 93.08% recall - this is our evidence recall target.

---

## 3. No-Evidence Detection (Threshold-based)

*On all queries (n=2,702 per fold)*

| Metric | Mean | Std | Notes |
|--------|------|-----|-------|
| Best threshold | -2.30 | 0.19 | score_gap method |
| Accuracy | 0.8607 | 0.0145 | |
| Balanced accuracy | 0.8817 | 0.0068 | |
| **TPR (Recall)** | **0.9077** | 0.0090 | positive coverage |
| **TNR (Specificity)** | 0.8558 | 0.0165 | |
| **FPR** | **0.1442** | 0.0165 | negative coverage - **TOO HIGH** |
| Precision | 0.3979 | 0.0268 | low due to class imbalance |
| F1 | 0.5526 | 0.0242 | |
| NPV | 0.9889 | 0.0014 | excellent |
| MCC | 0.5431 | 0.0217 | |
| AUROC | 0.9417 | 0.0058 | excellent discrimination |
| AUPRC | 0.6438 | 0.0133 | |

### Confusion Matrix (per fold avg)
|  | Predicted Positive | Predicted Negative |
|--|---------------------|---------------------|
| **Actual Positive** | TP=231.6 | FN=23.6 |
| **Actual Negative** | FP=353.0 | TN=2093.8 |

**Problem:** 353 false positives per fold (14.4% FPR) - need to reduce to ≤5%.

---

## 4. Dynamic-K Performance

*Only on has-evidence queries*

| Method | nDCG | Recall | Avg K |
|--------|------|--------|-------|
| Fixed K=1 | 0.7167 | 0.6771 | 1.0 |
| Fixed K=3 | 0.8053 | 0.8666 | 3.0 |
| **Fixed K=5** | **0.8326** | **0.9308** | **5.0** |
| Fixed K=10 | 0.8502 | 0.9815 | 10.0 |
| Fixed K=20 | 0.8552 | 1.0000 | 20.0 |
| Dynamic (score_gap) | 0.7751 | 0.8080 | 1.71 |

**Analysis:** Current dynamic-K (avg K=1.71) achieves only 80.8% recall. To reach 93% recall target, we likely need avg K≈5.

---

## 5. E2E Deployment Performance

*On all queries (n=2,702 per fold)*

| Metric | Mean | Std |
|--------|------|-----|
| Avg returned K | 1.71 | 0.06 |
| **Positive coverage (TPR)** | **0.8983** | 0.0068 |
| **Negative coverage (FPR)** | **0.1454** | 0.0055 |
| Query accuracy | 0.8588 | 0.0052 |
| Query precision | 0.3918 | 0.0136 |
| Query F1 | 0.5454 | 0.0132 |
| Query MCC | 0.5345 | 0.0109 |
| Conditional nDCG | 0.7469 | 0.0323 |
| Conditional recall | 0.7730 | 0.0339 |

---

## 6. Per-Post Multi-Label Performance

| Metric | Mean | Std |
|--------|------|-----|
| Exact match rate | **0.2682** | 0.0281 |
| Subset accuracy | 0.8588 | 0.0052 |
| Hamming score | 0.8588 | 0.0052 |
| Micro precision | 0.3918 | 0.0136 |
| Micro recall | 0.8983 | 0.0068 |
| Micro F1 | 0.5454 | 0.0132 |

**Note:** Only 26.8% of posts have all 10 criteria predicted correctly. This is consistent with (0.86)^10 ≈ 22%.

---

## 7. Optimization Targets

Based on `configs/deployment_targets.yaml`:

### Profile: high_recall_low_hallucination (Default)

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Max FPR (neg coverage) | 14.5% | ≤ 5% | **-9.5pp** |
| Min TPR (pos coverage) | 89.8% | ≥ 90% | ✅ (close) |
| Min evidence recall | 80.8% | ≥ 93% | **-12.2pp** |
| Max avg K | 1.71 | ≤ 5.0 | ✅ |

### Profile: ultra_safe

| Metric | Target |
|--------|--------|
| Max FPR | ≤ 3% |
| Min TPR | ≥ 85% |
| Min evidence recall | ≥ 90% |
| Max avg K | ≤ 3.0 |

### Profile: cheap

| Metric | Target |
|--------|--------|
| Max FPR | ≤ 10% |
| Min TPR | ≥ 85% |
| Min evidence recall | ≥ 85% |
| Max avg K | ≤ 2.0 |

---

## 8. Optimization Strategy

### Phase 1: NE Gate Optimization
- **Goal:** Reduce FPR from 14.5% to ≤5% while maintaining TPR ≥90%
- **Methods to try:**
  1. Threshold sweep on score_gap
  2. Calibrated probability threshold
  3. NO_EVIDENCE rank-based decision
  4. Margin-based decision (best_real - NO_EVIDENCE score)

### Phase 2: Dynamic-K Optimization
- **Goal:** Increase evidence recall from 80.8% to ≥93%
- **Methods to try:**
  1. Fixed K (baseline comparison)
  2. Score gap / ratio (current method)
  3. Probability threshold (if calibrated)
  4. NO_EVIDENCE boundary (K = rank of NO_EVIDENCE - 1)

### Phase 3: Advanced (if needed)
- Per-criterion thresholds
- Lightweight post-level aggregator
- GNN (last resort)

---

## 9. Optimization Scripts (Added)

### Master Pipeline
```bash
# Run complete optimization pipeline
bash scripts/run_postproc_optimization.sh [profile]
# Profiles: high_recall_low_hallucination (default), ultra_safe, cheap
```

### Individual Scripts

1. **Build OOF Cache** (`scripts/build_oof_cache.py`)
   - Runs reranker inference on validation queries only (out-of-fold)
   - Stores per-query scores for fast HPO iteration
   - Includes NO_EVIDENCE pseudo-candidate scoring

2. **NE Gate HPO** (`scripts/hpo_ne_gate.py`)
   - Grid search + cross-validation for NE threshold
   - Methods: max_score, score_gap, margin, calibrated_prob, ne_rank
   - Target: FPR ≤ 5%, TPR ≥ 90%

3. **Dynamic-K HPO** (`scripts/hpo_dynamic_k.py`)
   - Grid search + cross-validation for K selection
   - Methods: fixed_k, score_gap, margin_threshold, ne_boundary, prob_threshold
   - Target: Evidence recall ≥ 93%, Avg K ≤ 5

4. **Generate Deployment Config** (`scripts/generate_deployment_config.py`)
   - Combines NE gate and Dynamic-K results
   - Runs E2E assessment with optimized config
   - Generates before/after comparison report

---

## 10. HPO Results (2026-01-15)

### Optimization Pipeline Completed
The optimization pipeline was executed successfully. Here are the findings:

#### NE Gate HPO Results

| Method | Threshold | TPR | FPR | Meets Targets |
|--------|-----------|-----|-----|---------------|
| max_score | -2.04 | 87.95% | 12.76% | ❌ |
| score_gap | 0.85 | 75.41% | 21.78% | ❌ |
| margin | -3.22 | 90.08% | 12.76% | ❌ |
| calibrated_prob | 0.11 | 88.67% | 13.53% | ❌ |
| ne_rank | 0.10 | 45.84% | 1.81% | ❌ |

**Best method:** `margin` (TPR=90.08%, FPR=12.76%)

#### Dynamic-K HPO Results

| Method | Threshold | Recall | Avg K | Meets Targets |
|--------|-----------|--------|-------|---------------|
| Fixed K=5 | - | 92.95% | 5.0 | ❌ (recall) |
| score_gap | 4.98 | 97.05% | 8.44 | ❌ (avg_k) |
| margin_threshold | -5.0 | 89.03% | 3.58 | ❌ (recall) |
| ne_boundary | 0.0 | 70.53% | 1.12 | ❌ (recall) |

**Best method:** `score_gap` (Recall=97.05%, Avg K=8.44)

#### Final E2E Performance

| Metric | Baseline | Optimized | Target | Status |
|--------|----------|-----------|--------|--------|
| FPR (neg coverage) | 14.50% | 12.76% | ≤ 5% | ❌ |
| TPR (pos coverage) | 89.80% | 90.08% | ≥ 90% | ✅ |
| Evidence Recall | 80.80% | 87.47% | ≥ 93% | ❌ |
| Avg K | 1.71 | 7.55 | ≤ 5.0 | ❌ |

#### Key Finding: Model Limitation

The NO_EVIDENCE pseudo-candidate analysis revealed:
- For **has-evidence queries**: Only 46% have any real candidate scoring above NO_EVIDENCE
- For **no-evidence queries**: 98% have NO_EVIDENCE ranked #1

This means the model was NOT trained with NO_EVIDENCE as a pseudo-candidate, so it doesn't properly discriminate. The `ne_rank` method achieves FPR=1.81% but only TPR=45.84%.

#### Recommended Next Steps

1. **Retrain reranker with NO_EVIDENCE pseudo-candidate** - This would allow proper discrimination
2. **Accept relaxed targets** - With current model, achievable: FPR≈5% with TPR≈64% (margin threshold=0)
3. **Per-criterion thresholds** (PHASE 3A) - May improve some criteria
4. **Ensemble classifier** (PHASE 3B) - Train classifier on score features

---

## 11. PHASE 3: Advanced Improvements (2026-01-15)

### PHASE 3A: Per-Criterion Threshold Optimization

Optimizing NE gate thresholds independently for each criterion (constrained to FPR ≤ 5% per criterion):

| Criterion | Threshold | TPR | FPR | Pos Rate | Meets Both |
|-----------|-----------|-----|-----|----------|------------|
| A.1 | -0.00 | 40.9% | 4.8% | 21.8% | ❌ |
| A.2 | -2.61 | 73.1% | 4.9% | 8.9% | ❌ |
| A.3 | -3.33 | 89.5% | 5.0% | 3.1% | ❌ |
| A.4 | -0.97 | 70.5% | 4.6% | 6.8% | ❌ |
| A.5 | -2.60 | 77.3% | 4.6% | 2.6% | ❌ |
| A.6 | -0.97 | 88.4% | 4.7% | 8.7% | ❌ |
| A.7 | 0.13 | 61.6% | 4.6% | 21.5% | ❌ |
| A.8 | -3.32 | 80.3% | 4.9% | 4.2% | ❌ |
| A.9 | 0.30 | **90.2%** | 4.7% | 10.7% | ✅ |
| A.10 | -2.21 | 23.6% | 4.8% | 6.0% | ❌ |

**Overall with per-criterion thresholds:**
- TPR: 63.86% ± 1.91% (target: ≥90%) ❌
- FPR: **4.76%** ± 0.25% (target: ≤5%) ✅

**Finding:** Per-criterion thresholds can achieve the FPR target but at significant cost to TPR. Only criterion A.9 meets both targets.

### PHASE 3B: Ensemble Classifier

Training various classifiers on score features (max_score, margin, ne_score, criterion_id, etc.):

| Classifier | TPR | FPR | Precision | AUROC | TPR@5%FPR |
|------------|-----|-----|-----------|-------|-----------|
| LogisticRegression | 88.6% | 10.7% | 46.3% | 0.951 | **74.5%** |
| LogisticRegression_L1 | 88.6% | 10.7% | 46.4% | 0.951 | 74.4% |
| RandomForest | 88.2% | 10.9% | 45.8% | 0.949 | 73.8% |
| RandomForest_Deep | 79.4% | 6.6% | 55.7% | 0.949 | 73.9% |
| GradientBoosting | 57.8% | **2.7%** | 68.7% | 0.949 | 74.0% |
| HistGradientBoosting | 57.4% | **2.8%** | 68.3% | 0.949 | 73.8% |

**Key metric: TPR@5%FPR** - The TPR achievable when constrained to 5% FPR (from ROC curve):
- All classifiers achieve ~73-74% TPR at 5% FPR
- This is consistent with per-criterion threshold results (~64%)

### PHASE 3 Summary: Fundamental Tradeoff

The experiments reveal a **fundamental discrimination limitation** with current model:

| Operating Point | TPR | FPR | Method |
|-----------------|-----|-----|--------|
| High recall | 90.1% | 12.8% | Global margin threshold |
| Balanced | 74.5% | 5.0% | Ensemble classifier (ROC operating point) |
| Low FPR | 63.9% | 4.8% | Per-criterion thresholds |
| Ultra-safe | 57.8% | 2.7% | GradientBoosting |

**Root cause:** The NO_EVIDENCE pseudo-candidate was not included during reranker training, so the model cannot properly distinguish between "has evidence with low scores" and "no evidence".

### Recommended Path Forward

1. **Accept relaxed targets** for current deployment:
   - Option A: TPR ≥ 90%, FPR ≤ 13% (global margin)
   - Option B: TPR ≥ 74%, FPR ≤ 5% (ensemble at ROC operating point)

2. **Retrain reranker with NO_EVIDENCE pseudo-candidate** for future improvement:
   - Include NO_EVIDENCE as a candidate during training
   - Use contrastive learning to push NO_EVIDENCE below real evidence
   - Target: proper discrimination enabling both FPR ≤ 5% AND TPR ≥ 90%

---

## 13. NO_EVIDENCE Retraining Results (2026-01-15)

### Training Summary

Retrained Jina-v3 with NO_EVIDENCE pseudo-candidate on expanded dataset:
- Training: 1,099 has-evidence + 10,711 no-evidence = 11,810 queries
- Validation: 140 has-evidence + 1,330 no-evidence = 1,470 queries

| Epoch | Train Loss | Val Loss | Saved |
|-------|-----------|----------|-------|
| 1 | 0.3656 | 0.3313 | ✓ |
| 2 | 0.3468 | 0.3378 | |
| 3 | 0.3344 | 0.3311 | ✓ |
| 4 | 0.3197 | **0.3157** | ✓ |
| 5 | 0.3139 | 0.3181 | |

Training time: 1.79h

### Assessment Results (Epoch 4 Model)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| TPR | 23.57% | ≥90% | ❌ |
| FPR | **1.13%** | ≤5% | ✅ |
| Precision | 68.75% | - | - |
| AUROC | 0.695 | - | - |

### ROC Operating Points

| Target FPR | Achievable TPR | Threshold |
|------------|----------------|-----------|
| 1% | 22.9% | 0.24 |
| 3% | 32.1% | -2.15 |
| **5%** | **35.7%** | -2.20 |
| 10% | 50.0% | -2.31 |
| 15% | 53.6% | -2.39 |

### Analysis

The retrained model learned to be **too conservative**:
- Achieves excellent FPR (1.13%) but poor TPR (23.57%)
- Margin distributions still overlap significantly
- Caused by severe class imbalance (10:1 no-evidence:has-evidence)

**Comparison:**
| Approach | TPR at 5% FPR | Notes |
|----------|--------------|-------|
| Baseline (per-criterion) | 64% | Best current method |
| Retrained model | 36% | Too conservative |
| Ensemble classifier | 74% | ROC operating point |

### Recommendations for Future Training

1. **Address class imbalance:**
   - Undersample no-evidence queries (e.g., 3:1 ratio)
   - Use weighted loss to upweight has-evidence examples
   - Use focal loss for hard example mining

2. **Loss function tuning:**
   - Increase `w_list` weight for better ranking
   - Add margin loss to separate has-evidence from no-evidence

3. **Training augmentation:**
   - Use hard negative mining more aggressively
   - Include more diverse has-evidence examples

---

## 14. Key References

- Model: `outputs/training/no_evidence_reranker`
- Candidates: `outputs/retrieval_candidates/retrieval_candidates.pkl`
- Assessment script: `scripts/run_full_assessment.py`
- Optimization pipeline: `scripts/run_postproc_optimization.sh`
- Deployment targets: `configs/deployment_targets.yaml`
- PHASE 3A script: `scripts/hpo_per_criterion_thresholds.py`
- PHASE 3B script: `scripts/hpo_ensemble_classifier.py`
- Git commit: `0da22ca991ec0fee9447a5c6fbaf1d7aa4b39256`
