# Retriever Comparison Results

**Date:** 2026-01-10
**Split:** dev_select (validation set)
**Task:** Sentence-Criterion evidence retrieval for MDD criteria

## Summary

Evaluated 14 retriever models on the dev_select split. NV-Embed-v2 achieves the best performance but requires a separate conda environment due to transformers version incompatibility.

## Results (nDCG@10)

| Rank | Model | nDCG@10 | Recall@10 | MRR@10 | Notes |
|------|-------|---------|-----------|--------|-------|
| 1 | **nv-embed-v2** | **0.7405** | 0.8853 | 0.6943 | Best overall, requires separate env |
| 2 | qwen3-embed-8b-4bit | 0.7088 | 0.8543 | 0.6715 | Best in main env, 4-bit quantized |
| 3 | qwen3-embed-0.6b | 0.7046 | 0.8665 | 0.6596 | Fastest, best efficiency |
| 4 | splade-cocondenser | 0.6953 | 0.8712 | 0.6502 | Sparse neural retriever |
| 5 | qwen3-embed-4b | 0.6855 | 0.8797 | 0.6301 | |
| 6 | mxbai-embed-large | 0.6721 | 0.8440 | 0.6242 | |
| 7 | bge-large-en-v1.5 | 0.6690 | 0.8647 | 0.6158 | |
| 8 | gte-large-en-v1.5 | 0.6635 | 0.8571 | 0.6068 | |
| 9 | bge-m3 | 0.6628 | 0.8637 | 0.6069 | Hybrid (dense+sparse+ColBERT) |
| 10 | llama-embed-8b | 0.6606 | 0.8540 | 0.6103 | NVIDIA, 4-bit quantized |
| 11 | e5-large-v2 | 0.6333 | 0.8477 | 0.5724 | |
| 12 | e5-mistral-7b | 0.6209 | 0.8202 | 0.5640 | |
| 13 | stella-1.5b | 0.6082 | 0.8227 | 0.5462 | |
| 14 | bm25 | 0.5761 | 0.8271 | 0.5021 | Lexical baseline |

## Key Findings

### Best Performers
1. **NV-Embed-v2** (+3.2% over runner-up): SOTA embedding model, but requires transformers 4.42.4
2. **Qwen3-Embedding-8B** (4-bit): Best performance in main environment with quantization
3. **Qwen3-Embedding-0.6B**: Excellent quality-speed tradeoff, only 0.6B params

### Model Categories
- **Dense embedders**: Qwen3, NV-Embed, BGE, GTE, E5, Stella, mxbai
- **Sparse neural**: SPLADE-cocondenser (competitive with dense)
- **Hybrid**: BGE-M3 (dense + sparse + ColBERT)
- **Lexical**: BM25 (baseline)

### Quantization Results
- 4-bit quantization (bitsandbytes) enables running 8B models on 32GB VRAM
- Performance loss from quantization is minimal (~1-2%)

## Environment Setup

### Main Environment (llmhe)
- transformers 4.57.3
- PyTorch 2.11.0+cu128
- All models except NV-Embed-v2

### NV-Embed-v2 Environment
```bash
mamba create -n nv-embed-v2 python=3.10 -y
mamba activate nv-embed-v2
pip install transformers==4.42.4 sentence-transformers==2.7.0
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Run evaluation:
```bash
mamba run -n nv-embed-v2 python scripts/retriever/eval_nv_embed_v2.py
```

## Recommendations

1. **For production**: Use Qwen3-Embedding-0.6B (best efficiency) or Qwen3-Embedding-8B-4bit (best quality in main env)
2. **For research**: NV-Embed-v2 achieves SOTA but has non-commercial license
3. **For hybrid retrieval**: Consider SPLADE + dense fusion

## Files

- Results: `outputs/retriever_comparison/`
- Scripts: `scripts/retriever/compare_all_retrievers.py`, `scripts/retriever/eval_nv_embed_v2.py`
- Config: `src/final_sc_review/retriever/zoo.py`
