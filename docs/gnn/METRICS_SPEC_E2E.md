# End-to-End Metrics Specification

**Document Version**: 1.0
**Date**: 2026-01-17
**Purpose**: Define all metrics used in the GNN E2E evaluation pipeline with exact formulas.

---

## Overview

This document specifies all metrics used in the end-to-end evaluation of the GNN-based evidence retrieval pipeline. All metrics are computed at the **query level** (post_id, criterion_id pair) and aggregated across folds.

### Pipeline Components
1. **Reranker**: Produces candidate scores (baseline or P3-refined)
2. **NE Gate (P4)**: Binary classification - predicts if ANY evidence exists
3. **Dynamic-K (P2)**: Selects K candidates to return based on node probabilities

### Evaluation Protocol
- **Outer 5-fold CV**: Post-ID disjoint splits
- **Inner split**: Within each outer fold, 70% TRAIN / 30% TUNE (also post-ID disjoint)
- **Threshold selection**: Done on TUNE set only
- **Final evaluation**: Done on EVAL (held-out fold) only

---

## A. NE Detection Metrics (Binary Classification)

### Population
All queries in the evaluation fold, regardless of ground-truth has_evidence status.

### A.1 Area Under ROC Curve (AUROC)

**Definition**: Probability that a randomly chosen positive query ranks higher than a randomly chosen negative query.

**Formula**:
```
AUROC = P(score(positive) > score(negative))
      = ∫₀¹ TPR(t) d(FPR(t))
```

**Implementation**: `sklearn.metrics.roc_auc_score(y_true, y_prob)`

**Range**: [0, 1], random classifier = 0.5

### A.2 Area Under Precision-Recall Curve (AUPRC)

**Definition**: Average precision across all recall thresholds.

**Formula**:
```
AUPRC = ∫₀¹ Precision(r) dr
      ≈ Σₙ (Rₙ - Rₙ₋₁) × Pₙ
```

**Implementation**: `sklearn.metrics.average_precision_score(y_true, y_prob)`

**Range**: [0, 1], baseline = positive class proportion

### A.3 True Positive Rate (TPR) / Recall / Sensitivity

**Definition**: Fraction of actual positives correctly identified.

**Formula**:
```
TPR = TP / (TP + FN) = TP / P
```
where:
- TP = True Positives (has_evidence=1 AND pred=1)
- FN = False Negatives (has_evidence=1 AND pred=0)
- P = Total positives = TP + FN

### A.4 False Positive Rate (FPR)

**Definition**: Fraction of actual negatives incorrectly classified as positive.

**Formula**:
```
FPR = FP / (FP + TN) = FP / N
```
where:
- FP = False Positives (has_evidence=0 AND pred=1)
- TN = True Negatives (has_evidence=0 AND pred=0)
- N = Total negatives = FP + TN

### A.5 Specificity (True Negative Rate)

**Definition**: Fraction of actual negatives correctly identified.

**Formula**:
```
Specificity = TN / (TN + FP) = 1 - FPR
```

### A.6 Precision (Positive Predictive Value)

**Definition**: Fraction of positive predictions that are correct.

**Formula**:
```
Precision = TP / (TP + FP)
```

### A.7 F1 Score

**Definition**: Harmonic mean of precision and recall.

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × TP / (2 × TP + FP + FN)
```

### A.8 Matthews Correlation Coefficient (MCC)

**Definition**: Correlation coefficient between observed and predicted binary classifications.

**Formula**:
```
MCC = (TP × TN - FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Range**: [-1, 1], 0 = random, 1 = perfect

### A.9 Balanced Accuracy

**Definition**: Average of recall for each class.

**Formula**:
```
Balanced_Accuracy = (TPR + TNR) / 2 = (Sensitivity + Specificity) / 2
```

### A.10 TPR at Fixed FPR (TPR@X%FPR)

**Definition**: True positive rate achieved when false positive rate is constrained to ≤X%.

**Procedure** (Gold Standard):
1. On **TUNE set**: Compute ROC curve (FPR, TPR, thresholds)
2. Find threshold `t*` where FPR ≤ X% and TPR is maximized
3. On **EVAL set**: Apply threshold `t*` and compute actual FPR and TPR

**Formula for threshold selection**:
```
t* = argmax_{t: FPR(t) ≤ X%} TPR(t)
```

**Reported variants**: TPR@3%FPR, TPR@5%FPR, TPR@10%FPR

**CRITICAL**: Threshold `t*` MUST be selected on TUNE set only. Never use EVAL labels for threshold selection.

---

## B. Ranking / Retrieval Metrics (Evidence Queries Only)

### Population
Queries where ground-truth `has_evidence = 1` (i.e., at least one gold sentence exists).

### B.1 Recall@K (Fraction of Gold)

**Definition**: Fraction of all gold sentences that appear in top-K predictions.

**Formula**:
```
Recall@K = |{gold ∩ top_K}| / |gold|
```
where:
- `gold` = set of gold evidence sentence UIDs
- `top_K` = set of top-K predicted sentence UIDs

**Range**: [0, 1]

### B.2 Hit Rate@K (Success@K)

**Definition**: Binary indicator - whether ANY gold sentence appears in top-K.

**Formula**:
```
HitRate@K = 1 if |{gold ∩ top_K}| ≥ 1 else 0
```

**Population average**: Fraction of queries with at least one hit in top-K.

### B.3 Mean Reciprocal Rank (MRR)

**Definition**: Inverse of the rank of the first relevant (gold) item.

**Formula**:
```
MRR = 1 / rank_first_gold

where:
- rank_first_gold = position of first gold item in ranked list (1-indexed)
- MRR = 0 if no gold item in list
```

### B.4 Normalized Discounted Cumulative Gain (nDCG@K)

**Definition**: DCG normalized by ideal DCG.

**Formula**:
```
DCG@K = Σᵢ₌₁ᴷ (2^relᵢ - 1) / log₂(i + 1)

IDCG@K = DCG@K for ideal ranking (all gold items first)

nDCG@K = DCG@K / IDCG@K
```

For **binary relevance** (rel ∈ {0, 1}):
```
DCG@K = Σᵢ₌₁ᴷ relᵢ / log₂(i + 1)
```

**Range**: [0, 1]

### B.5 Mean Average Precision (MAP@K)

**Definition**: Mean of precision at each relevant position.

**Formula**:
```
AP@K = (1/|gold_in_K|) × Σⱼ₌₁ᴷ [Precision@j × rel(j)]

MAP@K = average of AP@K across queries
```
where `rel(j) = 1` if item at position j is gold, else 0.

---

## C. Dynamic-K Metrics

### Population
Depends on the specific metric (see below).

### C.1 Average K (Predicted Positive Queries)

**Definition**: Mean number of candidates selected for queries predicted as "has evidence".

**Formula**:
```
avgK_pred_pos = Σ{q: pred(q)=1} K(q) / |{q: pred(q)=1}|
```

**Note**: Queries predicted as "no evidence" are NOT included (their K=0 by definition).

### C.2 Average K (All Queries)

**Definition**: Mean K across ALL queries, counting K=0 for predicted-negative queries.

**Formula**:
```
avgK_all = Σ_q K(q) / |Q|

where K(q) = 0 if pred(q) = 0 (no evidence predicted)
```

### C.3 Evidence Recall (Unconditional)

**Definition**: Recall of gold sentences across ALL evidence queries, treating predicted-no-evidence as returning 0 sentences.

**Population**: All queries where `has_evidence = 1` (ground-truth).

**Formula**:
```
EvidenceRecall_unconditional = Σ{q: has_evidence=1} |gold(q) ∩ returned(q)| / Σ{q: has_evidence=1} |gold(q)|
```

where `returned(q) = ∅` if NE gate predicts "no evidence".

### C.4 Evidence Recall (Conditional)

**Definition**: Recall among queries that BOTH have evidence AND pass the NE gate.

**Population**: Queries where `has_evidence = 1` AND `pred = 1`.

**Formula**:
```
EvidenceRecall_conditional = Σ{q: has_evidence=1 AND pred=1} |gold(q) ∩ returned(q)| / Σ{q: has_evidence=1 AND pred=1} |gold(q)|
```

### C.5 K Distribution

**Definition**: Histogram of selected K values.

**Reported statistics**:
- min, max, median, mean, std
- P25, P75 (quartiles)
- Histogram buckets: [k_min, k_min+1, ..., k_max]

---

## D. End-to-End Deployment Metrics (ALL Queries)

### Population
All queries in evaluation fold.

### D.1 Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

**Definitions**:
- TP: Ground-truth has evidence, system returns K≥1 candidates
- FN: Ground-truth has evidence, system returns 0 candidates (NE gate blocks)
- FP: Ground-truth no evidence, system returns K≥1 candidates (false alarm)
- TN: Ground-truth no evidence, system correctly returns 0

### D.2 False Positive Rate (Deployment)

**Definition**: Rate at which system returns evidence for no-evidence queries.

**Formula**:
```
FPR_deployment = FP / (FP + TN)
```

This equals NE gate FPR if Dynamic-K always returns ≥1 when gate passes.

### D.3 False Negative Rate (Deployment)

**Definition**: Rate at which system returns 0 for evidence queries.

**Formula**:
```
FNR_deployment = FN / (FN + TP) = 1 - Sensitivity
```

### D.4 Overall Precision

**Definition**: Fraction of queries with returned evidence that actually have evidence.

**Formula**:
```
Precision_deployment = TP / (TP + FP)
```

### D.5 Overall Recall

**Definition**: Fraction of evidence queries where system returns something.

**Formula**:
```
Recall_deployment = TP / (TP + FN) = Sensitivity
```

### D.6 Overall F1

**Formula**:
```
F1_deployment = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## E. Per-Post Multi-Label Metrics (Optional)

### Population
Unique posts, with 10 criteria per post.

### E.1 Exact Match Rate

**Definition**: Fraction of posts where ALL 10 criteria predictions match ground truth.

**Formula**:
```
ExactMatch = (1/|posts|) × Σ_p 𝟙[pred(p) = gold(p)]
```

### E.2 Hamming Score

**Definition**: Fraction of correctly predicted criteria per post, averaged.

**Formula**:
```
HammingScore = (1/|posts|) × Σ_p (|pred(p) ∩ gold(p)| + |¬pred(p) ∩ ¬gold(p)|) / 10
```

### E.3 Micro F1

**Definition**: F1 computed by aggregating TP/FP/FN across all (post, criterion) pairs.

### E.4 Macro F1

**Definition**: Average of per-criterion F1 scores.

---

## F. Aggregation Protocol

### F.1 Per-Fold Metrics
Each metric is computed separately on each held-out fold.

### F.2 Cross-Fold Aggregation
```
Mean = (1/5) × Σ_{k=0}^{4} metric_k
Std = √[(1/4) × Σ_{k=0}^{4} (metric_k - Mean)²]
```

### F.3 Reporting Format
All metrics reported as: `Mean ± Std`

Example: `AUROC = 0.8967 ± 0.0109`

---

## G. Operating Point Selection

### G.1 FPR-Budget Operating Point

**Purpose**: Choose a threshold that limits false positive rate.

**Procedure**:
1. On TUNE set, compute ROC curve
2. Find largest threshold `t*` such that `FPR(t*) ≤ budget` (e.g., 5%)
3. Report metrics at this threshold

### G.2 Metrics at Operating Point

At selected threshold `t*`:
- TPR (sensitivity)
- FPR (actual, may be < budget)
- Precision
- F1
- MCC

---

## H. Implementation Notes

### H.1 sklearn Functions Used
- `roc_auc_score`: AUROC
- `average_precision_score`: AUPRC
- `roc_curve`: ROC curve for threshold selection
- `confusion_matrix`: Confusion matrix counts
- `f1_score`, `precision_score`, `recall_score`: Classification metrics
- `matthews_corrcoef`: MCC

### H.2 Edge Cases

**Empty gold set** (Recall@K):
- Return 0.0 (no gold to retrieve)

**All same label** (AUROC):
- Return 0.5 (undefined, but sklearn returns 0.5)

**No positives above threshold** (Precision):
- Return 0.0 (or undefined/NaN)

### H.3 Tie Handling

For ranking metrics, ties in scores are broken by original rank order (stable sort).

---

## I. Verification Checklist

Before reporting final metrics:

1. [ ] Confirm post-ID disjoint splits
2. [ ] Confirm threshold selected on TUNE only
3. [ ] Confirm evaluation on EVAL only
4. [ ] Confirm no gold-derived features in model
5. [ ] Confirm metric formulas match this spec
6. [ ] Cross-check with independent recomputation

---

*Document created: 2026-01-17*
*For use with: GNN E2E Gold Standard Report*
