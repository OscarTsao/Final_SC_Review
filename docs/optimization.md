# Environment + Reproducibility + Performance Policy

Repo: `OscarTsao/Final_SC_Review`
Workstation: **single NVIDIA RTX 5090 (1×GPU)**
Conda env (fixed): **`llmhe`**
Last updated: **2026-01-09**

This document is the **authoritative policy** for:

1) **Environment management** (Conda + pip only; fixed env name `llmhe`)
2) **Paper-grade reproducibility** (auditable, no leakage, deterministic settings documented)
3) **Maxout performance** on a single RTX 5090 (AMP, SDPA/FlashAttention, TF32, `torch.compile`, quantization, checkpointing, etc.)

This doc must stay consistent with:
- `environment.yml` (if present) and/or a documented conda creation procedure
- `configs/*.yaml` (single source of truth for run parameters)
- `scripts/preflight_env.py` (must enforce this policy)
- every run's `outputs/**/manifest.json` (must record what actually ran)

---

## 0) Non-negotiables

### 0.1 Conda + pip only
- ✅ Use **Conda** to create/activate environments.
- ✅ Use **pip** *inside the active conda env* for editable installs and pip-only packages.
- ❌ Do not use `venv` / `.venv`, Poetry, uv, pipenv, mamba, or system-wide pip installs.

### 0.2 Fixed env name: `llmhe`
All experiments **must** run under:
```bash
conda activate llmhe
```

If `llmhe` doesn't exist, create it once (see Section 1.2).

### 0.3 Two run profiles (never mix silently)

Every run must declare one of these profiles and record it in `manifest.json`:

* **`paper_repro`**
  Target: auditability, stable comparisons, reduced nondeterminism (where feasible)

* **`maxout_speed`**
  Target: maximum throughput on RTX 5090; allows more aggressive kernel/precision choices
  (still must be logged; finalists require multi-seed)

### 0.4 "Docs are policy; code is enforcement"

A policy is only paper-grade if it's enforced:

* `scripts/preflight_env.py` must hard-fail if policy is violated.
* Every run must write a complete artifact set and manifest (Section 7).

---

## 1) Environment management (Conda + pip, env = `llmhe`)

### 1.1 Activate & verify (every session)

```bash
conda activate llmhe
conda info --envs
python -V
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi
```

**Hard requirement:** the active env must be `llmhe`.

### 1.2 One-time creation of `llmhe` (only if missing)

If repo provides `environment.yml`:

```bash
conda env create -f environment.yml -n llmhe -y || conda env update -f environment.yml -n llmhe
conda activate llmhe
python -m pip install -U pip
pip install -e .
```

If repo does **not** provide `environment.yml`:

```bash
conda create -n llmhe python=3.11 -y
conda activate llmhe
python -m pip install -U pip
pip install -e .
```

### 1.3 Allowed installs

* Conda:

  ```bash
  conda install <pkg> -y
  ```
* Pip (inside `llmhe` only):

  ```bash
  pip install <pkg>
  ```

---

## 2) Environment snapshots (mandatory for serious runs)

Even if you treat `llmhe` as "fixed," you must snapshot it regularly so paper artifacts remain auditable.

Create:

```bash
mkdir -p outputs/system
```

Snapshot commands (run at least once per day of experiments, and whenever deps change):

```bash
conda activate llmhe

conda -V > outputs/system/conda_version.txt
python -V > outputs/system/python_version.txt
pip -V > outputs/system/pip_version.txt

# Portable-ish: records only explicit installs from history
conda env export --from-history > outputs/system/conda_env_from_history.yml

# Exact lock for same platform/OS
conda list --explicit > outputs/system/conda_list_explicit.txt

# Pip lock for pip-installed packages
pip freeze > outputs/system/pip_freeze.txt

# Optional sanity check
python -m pip check || true
```

**Policy:** Every run's `manifest.json` must reference these snapshot files and record their checksums.

---

## 3) Hardware snapshot (single RTX 5090)

Before major experiment batches, record:

```bash
mkdir -p outputs/system
nvidia-smi > outputs/system/nvidia_smi.txt
```

The repo must provide:

* `scripts/hw_probe.py` → `outputs/system/hw.json`

`hw.json` must include:

* GPU name, total VRAM, driver version, CUDA runtime version (as reported), CPU cores, RAM, disk free

---

## 4) Performance knobs (maxout on RTX 5090)

All performance knobs must be:

1. **configurable** (YAML + CLI override),
2. **logged** (manifest),
3. **verified** (micro-benchmark + recorded throughput/memory),
4. **never silently enabled** for paper tables.

### 4.1 Mixed precision (AMP): BF16 preferred

**Recommended defaults**

* `maxout_speed`: enable AMP with **bf16** where supported
* `paper_repro`: allow bf16 only if documented; otherwise fp32

**Implementation**

* training/inference should use `torch.autocast("cuda", dtype=torch.bfloat16)` or fp16
* if fp16 training: use `torch.amp.GradScaler`

**Manifest must record**

* `precision_mode`: `fp32` | `bf16` | `fp16`
* `amp_enabled`: bool
* `grad_scaler_enabled`: bool

### 4.2 Attention acceleration: SDPA / FlashAttention backend

If the model uses Transformer attention, prefer PyTorch SDPA (scaled dot-product attention).

**Policy**

* `maxout_speed`: allow flash/mem-efficient SDPA kernels where available
* `paper_repro`: must record which kernel path was used; avoid "unknown auto" in final tables

**Implementation notes**

* Prefer `torch.nn.functional.scaled_dot_product_attention(...)` or model-native SDPA support
* Control kernel selection via a context manager if needed (varies by torch version):

  * `torch.nn.attention.sdpa_kernel(...)` (newer)
  * OR `torch.backends.cuda.sdp_kernel(...)` (older)
* Always implement a safe fallback to math attention.

**Manifest must record**

* `attention_impl`: `sdpa` | `eager` | `flash_attn_pkg` | `unknown`
* `sdpa_backends_allowed`: list
* `sdpa_backend_selected`: string (if detectable)
* `flash_attention_enabled`: bool (best-effort)

### 4.3 TF32 / float32 matmul precision

TF32 can accelerate matmul/conv at some precision cost.

**Policy**

* `paper_repro`: default `float32_matmul_precision="highest"` unless explicitly justified
* `maxout_speed`: default `float32_matmul_precision="high"` (or `"medium"` only if validated)

**Implementation**

* `torch.set_float32_matmul_precision("high")`
* optionally:

  * `torch.backends.cuda.matmul.allow_tf32 = True`
  * `torch.backends.cudnn.allow_tf32 = True`

**Manifest must record**

* `float32_matmul_precision`: `highest` | `high` | `medium`
* `allow_tf32_matmul`: bool
* `allow_tf32_cudnn`: bool

### 4.4 `torch.compile` (optional but recommended to test)

`torch.compile` can speed up stable loops; can also cause compile overhead or graph breaks.

**Policy**

* treat compile as an ablation: eager vs compiled
* do not mix results across modes without labeling

**Manifest must record**

* `torch_compile_enabled`: bool
* `torch_compile_mode`: string
* `torch_compile_backend`: string (if used)
* `torch_compile_options`: dict
* `compile_warmup_steps`: int

### 4.5 Gradient checkpointing (memory ↔ compute tradeoff)

**Policy**

* allowed in both profiles, but must be logged
* generally enable when:

  * model OOMs, or
  * you want to increase effective batch size

**Manifest must record**

* `gradient_checkpointing_enabled`: bool
* `checkpoint_preserve_rng_state`: bool

### 4.6 Quantization (bitsandbytes) for larger models (8-bit / 4-bit / QLoRA)

Quantization is allowed primarily to:

* fit larger rerankers into VRAM
* enable LoRA/QLoRA training efficiently

**Policy**

* keep quantized evaluation results in separate tables
* never "accidentally" compare quantized vs full precision without labeling

**Manifest must record**

* `quantization_enabled`: bool
* `quantization_scheme`: `none` | `int8` | `4bit_nf4` | `4bit_fp4` | ...
* `bnb_compute_dtype`: `bf16` | `fp16` | `fp32`
* `bnb_double_quant`: bool

### 4.7 DataLoader / input pipeline (often the hidden bottleneck)

**Recommendations**

* Use `pin_memory=True`
* Set `num_workers` based on CPU cores (benchmark)
* Use `persistent_workers=True` if workers > 0
* Use `prefetch_factor` sensibly
* Ensure tokenizer isn't the bottleneck: pre-tokenize if needed for large runs

**Manifest must record**

* `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`

---

## 5) Reproducibility controls (paper-grade)

### 5.1 Seeds

Every run must set and record:

* python `random` seed
* numpy seed
* torch CPU seed
* torch CUDA seed (and all GPUs seed, even if single GPU)
* dataloader worker seed function

**Manifest must record**

* `seed`: int
* `dataloader_seed_policy`: description

### 5.2 Determinism tradeoffs

GPU training may be nondeterministic depending on kernels, versions, and backend algorithm selection.

**Policy**

* `paper_repro`:

  * attempt deterministic settings
  * record any unavoidable nondeterminism
* `maxout_speed`:

  * allow nondeterminism for speed
  * finalists must run ≥ 3 seeds

**Recommended settings**

* `paper_repro`:

  * `torch.use_deterministic_algorithms(True)` (or warn_only if necessary, but record)
  * `torch.backends.cudnn.benchmark = False`
* `maxout_speed`:

  * `torch.backends.cudnn.benchmark = True` allowed (record)

**Manifest must record**

* `run_profile`: `paper_repro` | `maxout_speed`
* `deterministic_algorithms`: bool or `warn_only`
* `cudnn_benchmark`: bool
* `cublas_workspace_config`: string or null (if used)

---

## 6) Recommended defaults for RTX 5090

### 6.1 `maxout_speed` default (exploration/HPO)

Use these unless there is a known issue:

* AMP: bf16 on
* SDPA: allow flash/mem-efficient kernels
* TF32: matmul precision `"high"`
* checkpointing: on if needed to fit batch/list_size/seq_len
* `torch.compile`: try on after a warmup; keep an eager baseline for comparison

### 6.2 `paper_repro` default (paper tables)

* AMP: off by default (or bf16 only if explicitly justified and consistently used)
* TF32: `"highest"` unless justified
* avoid unknown auto kernel switching without logging
* run ≥ 3 seeds for the final system or at minimum for finalists

---

## 7) Required per-run artifacts and manifest fields

### 7.1 Artifact contract (mandatory)

Every run directory must contain:

* `summary.json`
* `per_query.csv`
* `manifest.json`
* `logs.txt`
* `resolved_config.yaml`

Missing any item => the run is **INVALID**.

### 7.2 Manifest required fields (minimum)

`manifest.json` must include:

**Git / code**

* `git_sha`, `branch`, `dirty_tree` (bool)

**Environment**

* `conda_env_name` (must be `llmhe`)
* snapshot filenames + checksums:

  * `conda_env_from_history.yml`
  * `conda_list_explicit.txt`
  * `pip_freeze.txt`

**Hardware**

* `gpu_name`, `vram_gb`, `driver_version`, `cuda_version`
* link to `outputs/system/hw.json` and `outputs/system/nvidia_smi.txt`

**Run profile**

* `run_profile`: `paper_repro` or `maxout_speed`

**Performance knobs (actual resolved)**

* precision / AMP
* SDPA/attention backend choice
* TF32/matmul precision
* `torch.compile` flags
* checkpointing flags
* quantization flags
* dataloader settings

**Reproducibility**

* seeds
* determinism flags
* cuDNN benchmark flag

**Data integrity**

* checksums of all dataset inputs
* split id/hash, and proof of post-level disjointness (or pointer to a split report)

---

## 8) Preflight enforcement (must exist and run first)

The repo must implement `scripts/preflight_env.py` (or equivalent) and call it at the start of the main orchestration entrypoint.

Preflight must hard-fail if:

* not in conda env `llmhe`
* CUDA is unavailable
* more than one GPU is visible and not explicitly pinned to GPU 0
* required snapshot files cannot be written
* forbidden tooling is detected (e.g., `.venv/` exists, venv activation is detected in logs)

Preflight must also:

* write/refresh `outputs/system/hw.json` and `outputs/system/nvidia_smi.txt`
* write/refresh environment snapshots (Section 2) unless explicitly disabled

---

## 9) Troubleshooting & guardrails

### 9.1 OOM (out of memory)

Preferred automatic fallback order (must log in manifest):

1. reduce batch size
2. enable gradient checkpointing
3. reduce sequence length
4. reduce list_size (if listwise)
5. switch to LoRA/QLoRA
6. enable quantization (4-bit) if training large models

### 9.2 "Speed knob made accuracy worse"

Do not assume it's a bug. First:

* run A/B with same seed, same data subset, same config except one knob
* compare per_query outputs and calibration metrics
* keep separate tables for compiled/quantized/precision variants

### 9.3 "Paper numbers must be comparable"

Never mix:

* compiled vs eager
* quantized vs full precision
* different precision modes (fp32 vs bf16 vs fp16)
  without labeling + manifest evidence.

---

## 10) Reference links (authoritative docs)

Conda:

* `conda env export`: [https://docs.conda.io/projects/conda/en/latest/commands/env/export.html](https://docs.conda.io/projects/conda/en/latest/commands/env/export.html)
* `conda env create`: [https://docs.conda.io/projects/conda/en/latest/commands/env/create.html](https://docs.conda.io/projects/conda/en/latest/commands/env/create.html)
* `conda list --explicit`: [https://docs.conda.io/projects/conda/en/latest/commands/list.html](https://docs.conda.io/projects/conda/en/latest/commands/list.html)

PyTorch:

* Reproducibility notes: [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)
* Deterministic algorithms: [https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
* AMP: [https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
* SDPA kernel controls (version-dependent): [https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
* torch.compile: [https://pytorch.org/docs/stable/generated/torch.compile.html](https://pytorch.org/docs/stable/generated/torch.compile.html)
* float32 matmul precision: [https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
* checkpointing: [https://pytorch.org/docs/stable/checkpoint.html](https://pytorch.org/docs/stable/checkpoint.html)

Transformers / bitsandbytes quantization:

* [https://huggingface.co/docs/transformers/quantization/bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
