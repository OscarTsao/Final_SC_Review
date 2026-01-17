# Metric Specification Document

**Audit Timestamp**: 20260117_203822
**Purpose**: Define exact formulas for all metrics used in the evaluation pipeline to enable independent verification.

---

## 1. Ranking Metrics

All ranking metrics are implemented in `src/final_sc_review/metrics/ranking.py`.

### 1.1 Recall@K

**Definition**: Fraction of gold items appearing in top-K.

$$\text{Recall@K} = \frac{|\text{TopK} \cap \text{Gold}|}{|\text{Gold}|}$$

**Edge cases**:
- If `|Gold| = 0`: return `0.0`

**Implementation** (`ranking.py:9-15`):
```python
def recall_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = set(ranked_ids[:k]) & gold
    return len(hits) / len(gold)
```

---

### 1.2 MRR@K (Mean Reciprocal Rank)

**Definition**: Reciprocal rank of first relevant item in top-K.

$$\text{MRR@K} = \frac{1}{\text{rank of first gold in TopK}}$$

**Edge cases**:
- If `|Gold| = 0`: return `0.0`
- If no gold in top-K: return `0.0`

**Implementation** (`ranking.py:18-26`):
```python
def mrr_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            return 1.0 / idx
    return 0.0
```

---

### 1.3 MAP@K (Mean Average Precision)

**Definition**: Mean of precision values at each relevant hit in top-K.

$$\text{MAP@K} = \frac{1}{\min(|\text{Gold}|, K)} \sum_{i=1}^{K} \mathbf{1}[\text{ranked}_i \in \text{Gold}] \cdot \text{Precision@}i$$

Where Precision@i = (# hits in first i) / i

**Edge cases**:
- If `|Gold| = 0`: return `0.0`

**Implementation** (`ranking.py:29-40`):
```python
def map_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(len(gold), k)
```

---

### 1.4 nDCG@K (Normalized Discounted Cumulative Gain)

**Definition**: Normalized DCG with binary relevance.

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

$$\text{IDCG@K} = \sum_{i=1}^{\min(|\text{Gold}|, K)} \frac{1}{\log_2(i+1)}$$

$$\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Where $\text{rel}_i = 1$ if ranked_i ∈ Gold, else 0.

**Edge cases**:
- If `|Gold| = 0`: return `0.0`
- If `IDCG = 0`: return `0.0`

**Implementation** (`ranking.py:43-54`):
```python
def ndcg_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if sent_id in gold else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
```

---

## 2. No-Evidence (NE) Detection Metrics

All NE metrics are implemented in `src/final_sc_review/gnn/evaluation/metrics.py`.

### 2.1 AUROC (Area Under ROC Curve)

**Definition**: Area under the receiver operating characteristic curve.

$$\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx$$

**Implementation**: Uses `sklearn.metrics.roc_auc_score(y_true, y_prob)`

**Edge cases**:
- If only one class present: return `0.5`

---

### 2.2 AUPRC (Area Under Precision-Recall Curve)

**Definition**: Area under the precision-recall curve (average precision).

$$\text{AUPRC} = \sum_n (R_n - R_{n-1}) P_n$$

**Implementation**: Uses `sklearn.metrics.average_precision_score(y_true, y_prob)`

**Edge cases**:
- If only one class present: return `has_evidence_rate`

---

### 2.3 TPR@FPR (True Positive Rate at Fixed False Positive Rate)

**Definition**: Sensitivity (recall) achieved when operating at a fixed specificity level.

Given target FPR level (e.g., 5%):
1. Compute ROC curve: `fpr, tpr, thresholds = roc_curve(y_true, y_prob)`
2. Find largest index where `fpr <= target_fpr`
3. Return `tpr[index]`

**Implementation** (`metrics.py:83-94`):
```python
def get_tpr_at_fpr(target_fpr: float) -> Tuple[float, float]:
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, 1.0
    best_idx = idx[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])
```

**Reported levels**: TPR@FPR=3%, TPR@FPR=5%, TPR@FPR=10%

---

### 2.4 Threshold at Target FPR

**Definition**: The probability threshold that achieves the target FPR.

**Usage**: When applying NE gate at inference time, use this threshold to maintain the desired false positive rate.

**Implementation** (`metrics.py:197-214`):
```python
def compute_threshold_at_fpr(y_prob, y_true, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 1.0, 0.0, 0.0
    best_idx = idx[-1]
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]
```

---

## 3. Dynamic-K Metrics

### 3.1 Evidence Recall (with Dynamic K)

**Definition**: Recall when each query selects a variable number K of candidates.

$$\text{Evidence Recall} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Selected}_i \cap \text{Gold}_i|}{|\text{Gold}_i|}$$

**Edge cases**:
- If `|Gold_i| = 0`: contribute `1.0` if `|Selected_i| = 0`, else `0.0`

---

### 3.2 Evidence Precision (with Dynamic K)

**Definition**: Precision when each query selects a variable number K of candidates.

$$\text{Evidence Precision} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Selected}_i \cap \text{Gold}_i|}{|\text{Selected}_i|}$$

**Edge cases**:
- If `|Selected_i| = 0`: contribute `1.0` if `|Gold_i| = 0`, else `0.0`

---

### 3.3 Average K

**Definition**: Mean number of items selected across all queries.

$$\text{avg\_K} = \frac{1}{N} \sum_{i=1}^{N} |\text{Selected}_i|$$

---

### 3.4 Dynamic-K Constraints

The dynamic-K mechanism must satisfy these constraints:
- **k_min**: Minimum items to return (default: 2)
- **hard_cap**: Maximum absolute items (default: 10)
- **k_max_ratio**: Maximum fraction of candidates (default: 0.5)

Effective k_max = min(hard_cap, floor(n_candidates × k_max_ratio))

---

## 4. Calibration Metrics

Implemented in `src/final_sc_review/postprocessing/calibration.py`.

### 4.1 Expected Calibration Error (ECE)

**Definition**: Weighted average of absolute difference between predicted probability and observed accuracy, binned by confidence.

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
- $B_m$ = samples in bin m
- $\text{acc}(B_m)$ = mean label in bin
- $\text{conf}(B_m)$ = mean predicted probability in bin

**Implementation** (`calibration.py:151-179`):
```python
def compute_ece(probs, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_weight = mask.sum() / len(probs)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)
```

---

## 5. Multi-Label Metrics (Per-Criterion)

For criterion-aware evaluation, metrics are computed separately per criterion, then micro/macro averaged.

### 5.1 Micro-Average

Aggregate all (query, criterion) pairs, then compute metric once.

$$\text{Micro} = \text{Metric}(\bigcup \text{all predictions}, \bigcup \text{all golds})$$

### 5.2 Macro-Average

Compute metric per criterion, then average.

$$\text{Macro} = \frac{1}{C} \sum_{c=1}^{C} \text{Metric}_c$$

---

## 6. Aggregation Across Folds

For 5-fold cross-validation:

$$\text{Mean} = \frac{1}{5} \sum_{k=1}^{5} m_k$$

$$\text{Std} = \sqrt{\frac{1}{5} \sum_{k=1}^{5} (m_k - \text{Mean})^2}$$

Report format: `Mean ± Std`

---

## 7. Reference Values for Verification

Expected ranges for properly functioning pipeline:

| Metric | Baseline Range | Good Range |
|--------|----------------|------------|
| nDCG@10 | 0.70-0.80 | 0.85-0.90 |
| Recall@10 | 0.75-0.85 | 0.90-0.95 |
| MRR@10 | 0.65-0.75 | 0.80-0.90 |
| AUROC (NE) | 0.55-0.65 | 0.70-0.80 |
| TPR@5%FPR | 0.08-0.15 | 0.20-0.35 |

---

## 8. Verification Checksums

To verify metric implementations match sklearn/reference:

```python
# Test data
y_true = np.array([1, 1, 0, 0, 1])
y_prob = np.array([0.9, 0.7, 0.4, 0.2, 0.8])

# Expected AUROC
from sklearn.metrics import roc_auc_score
assert roc_auc_score(y_true, y_prob) == 1.0  # Perfect ranking
```

---

*Specification authored: 2026-01-17*
*For independent audit of Final_SC_Review evaluation pipeline*
