# LLM Integration for Evidence Retrieval

This directory contains LLM-based enhancements for the GNN evidence retrieval pipeline:
- **LLM Reranker**: Listwise reranking using Gemini 1.5 Flash (post-P3)
- **LLM Verifier**: Evidence correctness verification (LLM-as-judge)

## Quick Start

### 1. Get Gemini API Key

Get your free API key from Google AI Studio:
https://makersuite.google.com/app/apikey

Free tier includes generous daily quota sufficient for pilot experiments.

### 2. Set API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

### 3. Run Pilot Experiment (FREE)

Test on 10-50 samples to validate implementation:

```bash
# Test both reranker and verifier on 10 samples
python scripts/llm_integration/run_llm_pilot.py --n_samples 10

# Test only reranker on 50 samples
python scripts/llm_integration/run_llm_pilot.py --n_samples 50 --test_reranker

# Test only verifier
python scripts/llm_integration/run_llm_pilot.py --n_samples 50 --test_verifier
```

Expected output:
```
outputs/llm_pilot/<timestamp>/
├── reranker_results.json  # Reranking results
└── verifier_results.json  # Verification results
```

### 4. Run Full 5-Fold CV (~$2-9)

After pilot succeeds:

```bash
python scripts/llm_integration/run_llm_full_eval.py \
  --use_reranker \
  --use_verifier \
  --output_dir outputs/llm_integration
```

## Cost Estimates (Gemini 1.5 Flash)

| Experiment | Queries | API Calls | Est. Cost |
|------------|---------|-----------|-----------|
| **Pilot (10 samples)** | 10 | 10-50 | **FREE** |
| **Pilot (50 samples)** | 50 | 50-250 | **$0.10-0.50** |
| **1 Fold (reranker)** | 2,950 | 2,950 | **$0.40** |
| **1 Fold (verifier, positives only)** | ~274 | 1,370 | **$0.12** |
| **1 Fold (verifier, all)** | 2,950 | 29,500 | **$1.33** |
| **5-Fold CV (optimized)** | 14,770 | ~23k | **$2.60** |
| **5-Fold CV (full)** | 14,770 | ~165k | **$8.65** |

Actual costs may be lower due to:
- Gemini Batch API (50% discount)
- Free tier quota
- Smart batching (multiple candidates per call)

## Architecture

### LLM Reranker

**Input**: Top-M candidates (M=10) post-P3 graph reranking

**Process**:
1. Randomize candidate order (reduce position bias)
2. Send to Gemini with listwise ranking prompt
3. Parse JSON response with relevance scores
4. Re-rank top-M, keep rest in original order

**Output**: Reranked candidate list + LLM scores

**Metrics**: MRR, nDCG@K, Recall@K improvements over P3

### LLM Verifier

**Input**: Top-K candidates for a query

**Process**:
1. Batch candidates into single API call (cost optimization)
2. Ask: "Does this sentence support the criterion?"
3. Parse JSON response: {supports: true/false, confidence: 0-1}
4. Apply confidence threshold
5. Final decision: has_evidence if ≥1 candidate supports

**Output**: Binary NE prediction + supported candidate indices

**Metrics**: AUROC, TPR@FPR, Precision, Recall improvements over P4

## Configuration

See `IMPLEMENTATION_PLAN.md` for:
- Prompt engineering details
- Ablation configurations
- Evaluation metrics
- Deployment recommendations

## Troubleshooting

**Error: GEMINI_API_KEY not found**
```bash
export GEMINI_API_KEY="your-key"
```

**Error: Failed to load graphs**
- Ensure GNN experiments have been run first
- Graph cache should exist at `data/cache/gnn/`

**Error: JSON decode error**
- Gemini occasionally returns non-JSON text
- Retry logic handles this automatically (3 retries)
- Check pilot results for success rate

**Error: Rate limit exceeded**
- Free tier: 60 requests/minute
- Add delays between requests (already implemented)
- Upgrade to paid tier for higher limits

## Next Steps

1. **Pilot Success** → Run full 5-fold CV
2. **Full CV Success** → Generate comprehensive report
3. **Report Complete** → Deploy best configuration

See `run_llm_full_eval.py` for full evaluation pipeline.
