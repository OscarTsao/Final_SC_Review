# Retriever Zoo Evaluation Report

**Generated**: 2026-01-09T01:38:33.477526
**Split**: val
**Queries with positives**: 135

## Leaderboard (Paper-Standard K Values)

| Retriever | Oracle@20 | nDCG@10 | MRR@10 |
|-----------|-----------|---------|--------|
| e5-large-v2 | 0.9704 | 0.6377 | 0.5575 |
| bge-m3 | 0.9630 | 0.6589 | 0.5928 |
| bge-large-en-v1.5 | 0.9556 | 0.6693 | 0.6034 |
| bm25 | 0.9296 | 0.5433 | 0.4590 |

## Decision Gate D2 (Using Paper-Standard K=20)

Best Oracle@20 = 0.9704 >= 0.85
**Decision**: Retriever finetuning is OPTIONAL

## Top 3 Retrievers (Promoted)

1. **e5-large-v2**: Oracle@20=0.9704, nDCG@10=0.6377
2. **bge-m3**: Oracle@20=0.9630, nDCG@10=0.6589
3. **bge-large-en-v1.5**: Oracle@20=0.9556, nDCG@10=0.6693
