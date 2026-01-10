With a single **RTX 5090 (32GB VRAM)** you can realistically test *everything up to ~8B embedding models* locally in BF16/FP16 (and even larger if you quantize / keep sequence lengths reasonable). ([NVIDIA][1])

Here are the **most “SOTA-ish” retriever (bi-encoder) models** worth trying *today* that make sense on one GPU, plus what I’d pick first for your evidence-retrieval setup.

## 1) Qwen3-Embedding (0.6B / 4B / 8B) — best “try this first”

If you want a modern, strong baseline that’s actually competitive on current MTEB-style retrieval leaderboards, **Qwen3-Embedding** is the most obvious next step.

**Why it’s worth it**

* Strong reported results on MTEB (English and multilingual) and built specifically for embedding + retrieval use. ([Hugging Face][2])
* **Instruction-aware** (you can give a task instruction); they report that *not using an instruction* can drop retrieval performance by ~1–5%. ([Hugging Face][3])
* **MRL / flexible embedding size**: you can choose smaller output dims (helps index size + speed) while keeping good quality. ([Hugging Face][2])
* Supported by Hugging Face **text-embeddings-inference** (TEI) ecosystem (practical for serving). ([Hugging Face][4])

**Which size to try (given 5090)**

* **Qwen3-Embedding-4B**: best “quality vs speed” sweet spot for one GPU. ([Hugging Face][2])
* **Qwen3-Embedding-8B**: if you want maximum retrieval quality and can tolerate slower embedding throughput. ([Qwen][5])
* **Qwen3-Embedding-0.6B**: fast baseline / ablation model (also surprisingly strong for its size). ([Hugging Face][2])

**Practical tip for *your* task (criteria → evidence sentence retrieval)**
Make your query embedding explicitly task-shaped, e.g.

> `Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: {criterion}`

That aligns with how instruction-aware embedders expect retrieval queries. ([Hugging Face][3])

## 2) NVIDIA llama-embed-nemotron-8b — “SOTA” but **research/non-commercial**

If you can use non-commercial/research models, this is a top-tier candidate.

**Why**

* NVIDIA’s model card states it was **SOTA on multilingual MTEB as of Oct 21, 2025**. ([Hugging Face][6])
* **32k context**, **4096-d embeddings**, instruction-aware. ([Hugging Face][6])

**Caveat**

* The model card explicitly says **non-commercial/research use only** (custom license). ([Hugging Face][6])

**On a 5090**

* 8B in BF16 should fit fine on 32GB for *inference* (especially if you keep max_length sane), but it’s heavier than the 4B options.

## 3) Stella (NovaSearch / dunzhang) — strong English retriever, smaller than 7–8B

If your data is mostly English and you want something very strong but not huge:

* **NovaSearch/stella_en_1.5B_v5** (Matryoshka / multi-dim, often used as a strong open retriever). ([Modal][7])

This is a nice “mid-size” check: often much stronger than classic 300–400M sentence-transformers, and far cheaper than 8B.

## 4) Fast + permissive “strong baselines” (great to include in your ablations)

These aren’t always *the* top of the leaderboard anymore, but they’re excellent baselines and very practical.

### Snowflake Arctic-Embed (open, lightweight)

* **Snowflake/snowflake-arctic-embed-l** (and related Arctic embed family) are designed for retrieval; Snowflake provides benchmark reporting and usage guidance. ([GitHub][8])

### Mixedbread mxbai-embed-large-v1 (open, widely used)

* **mixedbread-ai/mxbai-embed-large-v1** is a very common “strong open embedding baseline”; Mixedbread provides eval comparisons. ([Mixedbread][9])

### UAE-Large-V1 (MIT, classic strong universal embedder)

* **WhereIsAI/UAE-Large-V1**: model card README states MIT and strong MTEB positioning. ([Hugging Face][10])

## 5) Strong but non-commercial 7B-ish embedders (if license is OK)

These are still worth testing if your project is academic and licensing allows.

* **nvidia/NV-Embed-v2** (Mistral-7B-based embedding; model card indicates non-commercial). ([Hugging Face][11])
* **Salesforce/SFR-Embedding-Mistral** (model card indicates research-only). ([Hugging Face][12])

## What I’d actually run first (minimal, high-signal experiment set)

Given you already have **BGE-M3**, and you’re on one 5090:

1. **Qwen/Qwen3-Embedding-4B** (main contender) ([Hugging Face][2])
2. **Qwen/Qwen3-Embedding-0.6B** (speed/ablation) ([Hugging Face][2])
3. **Qwen/Qwen3-Embedding-8B** (upper bound) ([Hugging Face][3])
4. **stella_en_1.5B_v5** (English-focused mid-size check) ([Modal][7])
5. **mxbai-embed-large-v1** (strong permissive baseline) ([Mixedbread][9])
6. *(optional, research-only)* **nvidia/llama-embed-nemotron-8b** (SOTA ceiling) ([Hugging Face][6])

## Extra tip specific to your “criteria → evidence sentence” retrieval

Because your candidate pool per post is often not massive (sentences within a post), you’ll usually get more ROI from:

* **better query formulation (instruction + criterion templating)** and
* **top-k tuning** (e.g., retrieve top 20–60 sentences then rerank)

…than from swapping embedders endlessly. But Qwen3-Embedding is a genuinely meaningful “next gen” retriever family to test.

If you tell me what you’re currently using as retriever (e.g., BGE-M3 dense vs its multi-vector vs hybrid) and what your typical per-post sentence count/top-k is, I can suggest the most informative ablation grid (k values + dim sizes + hybrid combos) without adding much runtime.

[1]: https://www.nvidia.com/vi-vn/geforce/graphics-cards/50-series/rtx-5090/?utm_source=chatgpt.com "GeForce RTX 5090 Graphics Cards | NVIDIA"
[2]: https://huggingface.co/Qwen/Qwen3-Embedding-4B?utm_source=chatgpt.com "Qwen/Qwen3-Embedding-4B · Hugging Face"
[3]: https://huggingface.co/dengcao/Qwen3-Embedding-8B?utm_source=chatgpt.com "dengcao/Qwen3-Embedding-8B · Hugging Face"
[4]: https://huggingface.tw/docs/text-embeddings-inference/supported_models?utm_source=chatgpt.com "支援的模型和硬體 - Hugging Face 文件"
[5]: https://qwenlm.github.io/blog/qwen3-embedding/?utm_source=chatgpt.com "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | Qwen"
[6]: https://huggingface.co/nvidia/llama-embed-nemotron-8b?utm_source=chatgpt.com "nvidia/llama-embed-nemotron-8b · Hugging Face"
[7]: https://modal.com/blog/mteb-leaderboard-article?utm_source=chatgpt.com "Top embedding models on the MTEB leaderboard"
[8]: https://github.com/Snowflake-Labs/arctic-embed?utm_source=chatgpt.com "GitHub - Snowflake-Labs/arctic-embed"
[9]: https://www.mixedbread.com/docs/embeddings/models/mxbai-embed-large-v1?utm_source=chatgpt.com "mxbai-embed-large-v1 - Mixedbread"
[10]: https://huggingface.co/WhereIsAI/UAE-Large-V1/blob/main/README.md?utm_source=chatgpt.com "README.md · WhereIsAI/UAE-Large-V1 at main"
[11]: https://huggingface.co/nvidia/NV-Embed-v2?utm_source=chatgpt.com "nvidia/NV-Embed-v2 · Hugging Face"
[12]: https://huggingface.co/Salesforce/SFR-Embedding-Mistral?utm_source=chatgpt.com "Salesforce/SFR-Embedding-Mistral · Hugging Face"
