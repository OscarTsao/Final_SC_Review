# Research Notes: No-Evidence Detection and Dynamic-K Selection

## Overview

This document surveys state-of-the-art approaches for two related problems in evidence retrieval:
1. **No-Evidence (NE) Detection**: Predicting when a query has no relevant evidence in the corpus
2. **Dynamic-K Selection**: Adaptively selecting the number of results to return per query

## 1. No-Evidence Detection

### 1.1 Problem Definition

Given a query Q and retrieval results R = {(d_i, s_i)} ranked by score s_i, determine whether:
- **Has-Evidence**: At least one relevant document exists in the corpus
- **No-Evidence**: No relevant documents exist

This is distinct from "retrieval failure" (relevant doc exists but wasn't retrieved).

### 1.2 Approaches

#### 1.2.1 Score-Based Methods

**Maximum Score Threshold**
- Simplest approach: flag as NE if max(s_i) < threshold
- Pros: Simple, interpretable
- Cons: Threshold varies across query types, poorly calibrated scores

**Score Distribution Features**
- Use statistics of score distribution: mean, std, skewness, kurtosis
- Key insight: NE queries often have flatter score distributions
- Features used in our pipeline:
  - `max_reranker_score`, `mean_reranker_score`, `std_reranker_score`
  - `top1_top2_gap`, `top1_mean_gap`: Score gap between top and other results
  - `entropy_top5`, `entropy_full`: Entropy of softmax(scores)

#### 1.2.2 Softmax-Based Proxy Features

**SoftMRR (Soft Mean Reciprocal Rank)**
```
SoftMRR = sum_i [softmax(s_i) / rank(i)]
```
- Interpretable as "expected" MRR under softmax distribution
- High confidence in top-1 → high SoftMRR
- Does NOT use gold labels (deployable at inference)

**Mass@K (Probability Mass Concentration)**
```
Mass@K = sum_{i<=K} softmax(s_i)
```
- Measures how much probability mass is in top-K
- High mass@1 suggests confident prediction
- Related to calibration: well-calibrated model has mass@K ≈ P(relevant in top-K)

#### 1.2.3 Query Difficulty Estimation

**Query Performance Prediction (QPP)**
- Pre-retrieval: predict difficulty from query features alone
  - Query length, IDF statistics, query clarity
- Post-retrieval: use retrieval results to refine estimate
  - Score variance, result list similarity under perturbation

**Relevant Papers**:
- Cronen-Townsend et al. (2002): "Predicting query performance" - clarity score
- Carmel & Yom-Tov (2010): "Estimating the query difficulty for information retrieval"
- Roitman (2017): "An enhanced approach to query performance prediction"

#### 1.2.4 Neural Approaches

**Cross-Encoder Confidence**
- Modern rerankers (BERT, T5) output logits that can be calibrated
- Temperature scaling: s' = s / T, tune T on validation set
- Platt scaling: P(relevant) = sigmoid(a*s + b)

**Dedicated NE Classifier**
- Train classifier on retrieval features to predict NE
- Input: top-K scores, score gaps, query features
- Output: P(no_evidence)

### 1.3 Evaluation Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| AUROC | Area under ROC curve | Overall discrimination |
| TPR@X%FPR | True positive rate at X% false positive rate | Operational setting |
| Precision@K | Fraction of flagged NE queries correct | Cost of false alarms |

**Critical Consideration**: In deployment, flagging a has-evidence query as NE is worse than missing an NE query (potential missed evidence vs. wasted review effort).

### 1.4 Related Work

1. **Abstaining Classifiers**: Systems that can say "I don't know"
   - Chow (1970): "On optimum recognition error and reject tradeoff"
   - Herbei & Wegkamp (2006): "Classification with reject option"

2. **Selective Prediction**: Predict only when confident
   - Geifman & El-Yaniv (2017): "Selective prediction for deep neural networks"
   - Coverage-accuracy trade-off

3. **Retrieval with Fallback**: Return "no answer" when appropriate
   - Common in QA systems (SQuAD 2.0 unanswerable questions)
   - Rajpurkar et al. (2018): "Know What You Don't Know"

## 2. Dynamic-K Selection

### 2.1 Problem Definition

Given retrieval results R with scores, select K_i for each query i such that:
- Include all relevant documents (high recall)
- Minimize irrelevant documents (high precision)
- Respect deployment constraints (K_max)

### 2.2 Deployment Constraints

Our pipeline uses:
```python
k_min = 2        # Minimum results to review
hard_cap = 10    # Maximum results (deployment constraint)
k_max_ratio = 0.5  # Adaptive cap: k_max1 = min(hard_cap, ceil(0.5 * N_candidates))
```

Rationale:
- `k_min=2`: Always provide alternative even if top-1 is confident
- `hard_cap=10`: Human reviewer bandwidth limit
- `k_max_ratio`: Avoid returning 50% of candidates for small pools

### 2.3 Dynamic-K Policies

#### 2.3.1 DK1: Mass Threshold Policy

```
K = min{k : sum_{i<=k} softmax(s_i) >= gamma}
```

Parameters:
- `gamma`: Target probability mass (e.g., 0.9 = 90% cumulative probability)

Properties:
- Returns fewer results for confident predictions
- Returns more results for uncertain predictions
- Naturally calibrated if softmax outputs are calibrated

#### 2.3.2 DK2: Score Gap / Knee Detection

```
K = argmax_k (s_k - s_{k+1})  # Largest gap
```

Alternative formulations:
- **Elbow/Knee detection**: Find point of maximum curvature
- **Threshold-based**: K = min{k : s_k - s_{k+1} > delta}

Properties:
- Data-driven: adapts to score distribution shape
- Works well when there's natural cluster separation
- May fail if scores are uniformly distributed

#### 2.3.3 Hybrid Approaches

Combine multiple signals:
```python
K = max(K_mass, K_gap)  # Union: higher recall
K = min(K_mass, K_gap)  # Intersection: higher precision
K = alpha * K_mass + (1-alpha) * K_gap  # Weighted average
```

### 2.4 Calibration Methods

For softmax-based methods to work, scores must be calibrated:

**Temperature Scaling**
```
p_i = softmax(s_i / T)
T > 1: softer distribution (less confident)
T < 1: sharper distribution (more confident)
```
Tune T to minimize calibration error on validation set.

**Isotonic Regression**
- Non-parametric calibration
- Learn monotonic mapping from scores to probabilities
- More flexible but requires more data

**Expected Calibration Error (ECE)**
```
ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|
```
Measures gap between predicted confidence and actual accuracy.

### 2.5 Related Work

1. **Set-Valued Prediction**
   - Vovk et al. (2005): "Algorithmic learning in a random world" - conformal prediction
   - Return prediction sets with coverage guarantee

2. **Adaptive Retrieval**
   - Cormack & Grossman (2014): "Evaluation of machine-learning protocols for TAR"
   - Stop retrieval when marginal utility drops

3. **Recall-Oriented Ranking**
   - Medical/legal search: recall > precision
   - Total recall: when missing any relevant doc is costly

## 3. Comparable Datasets and Benchmarks

### 3.1 Evidence Retrieval Datasets

| Dataset | Domain | Size | Has NE? |
|---------|--------|------|---------|
| FEVER | Fact verification | 185K claims | Yes (NEI label) |
| HotpotQA | Multi-hop QA | 113K questions | No |
| MS MARCO | Web QA | 1M queries | Implicit |
| Natural Questions | Wikipedia QA | 307K | Yes (unanswerable) |
| SQuAD 2.0 | Reading comprehension | 150K | Yes (unanswerable) |

### 3.2 Medical/Clinical NLP

| Dataset | Task | NE Prevalence |
|---------|------|---------------|
| n2c2 (i2b2) | Clinical NER/RE | Variable |
| MIMIC-III | Clinical notes | N/A |
| BioASQ | Biomedical QA | ~20% |

### 3.3 Mental Health NLP

Limited public datasets for mental health evidence retrieval:
- Reddit-based datasets (CLPsych shared tasks)
- Crisis text line data (restricted access)
- Our RedSM5 dataset: post-criterion evidence annotation

## 4. Recommendations for Our Pipeline

### 4.1 Feature Engineering

Prioritize these deployable features (no gold label dependency):
1. **Score statistics**: max, mean, std, median
2. **Concentration metrics**: SoftMRR, Mass@K, entropy
3. **Gap features**: top1-top2, top1-mean, score range
4. **Cross-model**: retriever-reranker correlation

### 4.2 Dynamic-K Configuration

Recommended defaults:
```yaml
dynamic_k:
  policy: mass_threshold  # or score_gap, hybrid
  k_min: 2
  hard_cap: 10
  k_max_ratio: 0.5
  gamma: 0.9  # for mass_threshold
  gap_threshold: 0.1  # for score_gap
```

### 4.3 Evaluation Protocol

1. **Nested CV**: Outer 5-fold by post_id, inner 30% tune split
2. **Metrics**: AUROC, TPR@5%FPR, calibration error
3. **Leakage check**: Assert no gold-dependent features in deployable mode

### 4.4 Future Directions

1. **Better calibration**: Investigate temperature scaling for reranker scores
2. **Query features**: Add criterion-specific difficulty features
3. **Joint optimization**: Train retriever+reranker with NE-aware loss
4. **Confidence estimation**: Ensemble disagreement as uncertainty signal

## 5. References

Key papers for further reading:

1. Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
2. Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks." NeurIPS.
3. Rajpurkar et al. (2018). "Know What You Don't Know: Unanswerable Questions for SQuAD." ACL.
4. Carmel & Yom-Tov (2010). "Estimating the Query Difficulty for Information Retrieval."
5. Cormack & Grossman (2014). "Evaluation of Machine-Learning Protocols for Technology-Assisted Review."

---

*Generated: 2026-01-16*
*Author: Claude (verification task)*
