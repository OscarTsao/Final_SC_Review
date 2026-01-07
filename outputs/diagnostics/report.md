# S-C Retrieval Diagnostic Report

**Split:** val
**Total Queries:** 1470

## A) Candidate Ceiling Analysis

What percentage of queries have gold sentences in the retriever's top-K?

| Top-K | Count | Percentage |
|-------|-------|------------|
| top_10 | 126 | 90.0% |
| top_100 | 137 | 97.86% |
| top_20 | 132 | 94.29% |
| top_50 | 137 | 97.86% |

**Hard ceiling:** 3 queries (2.14%) have no gold in top-100

**Best gold rank distribution:**
- Min: 1, Max: 36, Mean: 3.55, Median: 1

## B) Failure Taxonomy

- Total queries: 1470
- Queries with positives: 140
- Queries with no positives: 1330

## C) Data Quality

- Total sentences in corpus: 30028
- Empty text: 46 (0.1532%)
- Average sentences per post: 20.23
- Sentence length: min=0, max=2069, mean=75.5
- Missing groundtruth sents in corpus: 0

## D) Per-Criterion Performance

| Criterion | #Queries | nDCG@10 | Recall@10 | MRR@10 |
|-----------|----------|---------|-----------|--------|
| A.5 | 1 | 1.0 | 1.0 | 1.0 |
| A.8 | 5 | 0.9 | 1.0 | 0.8667 |
| A.4 | 11 | 0.8653 | 1.0 | 0.8212 |
| A.6 | 11 | 0.7605 | 0.9091 | 0.7197 |
| A.2 | 12 | 0.7212 | 0.9167 | 0.6528 |
| A.3 | 7 | 0.7154 | 0.8571 | 0.6905 |
| A.9 | 11 | 0.7153 | 0.9091 | 0.6621 |
| A.7 | 34 | 0.6961 | 0.8824 | 0.6351 |
| A.1 | 38 | 0.6534 | 0.8421 | 0.5985 |
| A.10 | 10 | 0.5135 | 1.0 | 0.3718 |

**Macro-average nDCG@10:** 0.7541
**Best criterion:** A.5 (nDCG@10=1.0)
**Worst criterion:** A.10 (nDCG@10=0.5135)

## E) Key Takeaways

Based on this analysis:

1. **Ceiling Analysis:** Check if gold sentences are being missed at retrieval stage
2. **Per-Criterion Variance:** High variance suggests criterion-specific tuning may help
3. **Data Quality:** Missing/empty sentences can cause silent failures
