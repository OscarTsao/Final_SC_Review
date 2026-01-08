# Retriever Zoo Evaluation Report

**Generated**: 2026-01-09T00:20:52.323968
**Split**: val
**Queries with positives**: 135

## Leaderboard

| Retriever | Oracle@200 | nDCG@10 | MRR@10 |
|-----------|------------|---------|--------|
| bge-m3 | 1.0000 | 0.6589 | 0.5928 |
| bge-large-en-v1.5 | 1.0000 | 0.6693 | 0.6034 |
| e5-large-v2 | 1.0000 | 0.6377 | 0.5575 |

## Decision Gate D2

Best Oracle@200 = 1.0000 >= 0.97
**Decision**: Retriever finetuning is OPTIONAL

## Top 3 Retrievers (Promoted)

1. **bge-m3**: Oracle@200=1.0000, nDCG@10=0.6589
2. **bge-large-en-v1.5**: Oracle@200=1.0000, nDCG@10=0.6693
3. **e5-large-v2**: Oracle@200=1.0000, nDCG@10=0.6377
