# Metric Policy for S-C Evidence Retrieval

**Document Version:** 1.0
**Last Updated:** 2026-01-09

## Overview

This document defines the evaluation metric policy for the Sentence-Criterion (S-C) evidence retrieval pipeline. The policy addresses deployment relevance, fairness for short posts, and interpretability.

## The Problem with Oracle@200

Previous evaluations used Oracle@200 (recall at K=200). This is **misleading** because:

1. **Posts are short:** Mean ~10 sentences, median ~8, p95 ~25
2. **K > n_candidates:** When K exceeds available candidates, Oracle@K collapses to 100%
3. **Not deployment-relevant:** Real applications use K=[1,3,5,10], not K=200

## Paper-Standard K Policy

### Primary K Values (Deployment Metrics)
```
K = [1, 3, 5, 10]
```

These values reflect real-world usage:
- **K=1:** Top-1 precision (most confident prediction)
- **K=3:** Typical assistant response (top 3 evidence sentences)
- **K=5:** Extended response
- **K=10:** Comprehensive evidence list

### Extended K Values (Long-tail Analysis)
```
K = [20]  # Only if p95(post_length) >= 20
```

### Ceiling K Values (Oracle Sanity Check)
```
K = ALL  # K = n_candidates for that query
```

This replaces Oracle@200 and represents the theoretical ceiling.

## K_eff: Fair Comparison for Short Posts

For any metric at K, we compute:
```
K_eff = min(K, n_candidates)
```

This ensures:
- Queries with 8 candidates aren't penalized at K=10
- Metrics are comparable across posts of different lengths
- No artificial inflation from K > n_candidates

## Metrics to Report

### Positives-Only Metrics (Queries with Evidence)
- **nDCG@{1,3,5,10,20}**: Ranking quality
- **MRR@{1,3,5,10,20}**: First relevant result position
- **Recall@{1,3,5,10,20}**: Coverage of gold evidence
- **Oracle@{1,3,5,10,20,ALL}**: Upper bound (ceiling)

### All-Queries Metrics (Including Empty)
- **Empty Detection P/R/F1**: Has-evidence classification
- **False Evidence Rate**: Evidence returned for empty queries
- **Micro P/R/F1**: Sentence-level precision/recall
- **Calibration**: ECE, Brier, NLL, AUPRC

## Implementation

The K policy is implemented in:
```python
from final_sc_review.metrics.k_policy import KPolicy, get_paper_k_values, compute_k_eff

# Get standard K values
k_values = get_paper_k_values()  # [1, 3, 5, 10, 20]

# Apply K_eff
k_eff = compute_k_eff(k=10, n_candidates=8)  # Returns 8
```

## Reporting Guidelines

1. **Primary tables:** Report metrics at K={1,3,5,10}
2. **Ceiling column:** Include Oracle@ALL (not Oracle@200)
3. **K_eff note:** State "using K_eff = min(K, n_candidates)"
4. **Distribution note:** Report n_candidates percentiles (p50, p90, p99)

## Example Table Format

| Metric | @1 | @3 | @5 | @10 | @ALL |
|--------|-----|-----|-----|------|------|
| nDCG | 0.48 | 0.62 | 0.67 | 0.69 | - |
| Recall | 0.44 | 0.72 | 0.83 | 0.89 | 1.0 |
| Oracle | 0.44 | 0.81 | 0.91 | 0.97 | 1.0 |

*Note: Using K_eff = min(K, n_candidates). Candidate pool: p50=8, p90=15, p99=25.*

## Rationale Summary

1. **Deployment relevance:** K=[1,3,5,10] matches real usage
2. **Fairness:** K_eff prevents penalizing short posts
3. **Interpretability:** Oracle@ALL is meaningful; Oracle@200 is not
4. **Reproducibility:** Clear policy enables fair comparison
