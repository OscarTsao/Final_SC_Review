# docs/PLAN.md

Last updated: 2026-01-09  
Repo: OscarTsao/Final_SC_Review  
Hardware: single RTX 5090 workstation (auto-detect VRAM; optimize batch/seq/precision).  
Goal: MAXOUT deployable evidence extraction under extreme No-Evidence prevalence, paper-grade.

This plan is designed to be executed by Claude Code as a fully automated research assistant.

---

## 0) Paper-grade non-negotiables (blockers)

### R0: Test is sacred
- Absolutely no tuning (HPO, calibration, thresholds, model selection) on test.
- Test is used only for:
  - exact reproduction of already-committed paper artifacts, OR
  - ONE final locked evaluation after all dev decisions are frozen.

### R1: Reproducible artifacts contract
Every run directory MUST contain:
- `summary.json`
- `per_query.csv`  (mandatory; allows recomputation)
- `manifest.json`  (git SHA, branch, cmdline, full resolved config, dataset checksums, env, hardware, random seeds)
- `logs.txt`
Missing any => INVALID => auto-fix pipeline, re-run, regenerate.

### R2: No leakage
- Splits are post-level disjoint.
- If using CV: nested CV (inner tuning, outer reporting).
- Calibration and thresholding must be fit ONLY on dev-tune splits.

### R3: Determinism
- Fix seeds everywhere.
- Record TF32/cudnn settings in manifest.
- If non-deterministic kernels used, document and run 3 seeds for finalists.

### R4: Deployment realism
- Because No-Evidence dominates, dev metrics must include:
  - empty detection P/R/F1
  - false evidence rate on empty queries
  - risk-coverage curves
  - selection micro P/R/F1

---

## 1) One-time setup: repo + env + hardware + structure map

Claude Code must:
1) Pull latest, include tags and LFS:
   - `git fetch --all --prune --tags`
   - `git pull --ff-only || true`
   - `git lfs install || true`
   - `git lfs pull || true`
2) Record:
   - `git rev-parse HEAD`
   - `git branch -a`
   - `git log --oneline -n 50`
3) Create a locked environment:
   - `.venv` or conda per repo conventions
   - `pip install -e .`
   - `pip freeze > outputs/system/pip_freeze.txt`
4) Hardware probe:
   - write `scripts/hw_probe.py` → `outputs/system/hw.json` and `outputs/system/nvidia_smi.txt`
   - must include GPU name, VRAM, driver, CUDA, CPU cores, RAM, disk free
5) Repo map:
   - `git ls-files > outputs/paper_audit/git_ls_files.txt`
   - create `outputs/paper_audit/REPO_STRUCTURE.md` summarizing folders and entrypoints

STOP if any step fails; fix before continuing.

---

## 2) Audit + verify all committed results before new experiments

1) Inventory:
   - write `scripts/audit_pushed_results.py` to discover `outputs/**/{summary.json,manifest.json,per_query.csv}`
   - write `outputs/paper_audit/results_inventory.csv` + `.md`
2) Validation:
   - write `scripts/validate_runs.py`
   - checks: artifact contract, dataset checksum match, metric recompute from per_query, protocol (no test tuning)
3) If invalid:
   - patch eval scripts to always emit required artifacts
   - rerun exact reproduction and re-validate

No new research until audit passes.

---

## 3) Dataset reality check (critical for correct K, retrieval, and No-Evidence)

Claude Code must generate:
- `outputs/data_profile/data_profile.json` and `data_profile.md`

Required stats:
- distribution of sentences per post: mean/median/p90/p95/max
- distribution of #gold evidence sentences per (post, criterion)
- prevalence of empty groups (No-Evidence rate)
- label noise indicators (duplicate sentences, conflicting labels)
- split sanity (no post_id overlap)

**Key policy for K:**
- Default evaluation Ks: {1,3,5,10}
- Add `K=min(10, n_sent)` and `K=n_sent` for sanity only
- Do NOT use K=200 unless corpus size demands it (it doesn’t for within-post).

---

## 4) Build automation backbone (so “exhaustive” is actually enforced)

Create:
### 4.1 `scripts/research_driver.py`
Phases:
- `audit`, `verify`, `profile`, `baselines`, `retriever_sweep`, `retriever_train`,
  `reranker_sweep`, `reranker_train`, `postprocess_hpo`, `gnn`, `llm_judge`, `paper`, `all`

Key flags:
- `--budget {tiny,standard,exhaustive}`
- `--time_budget_hours` (default: 12 for exhaustive)
- `--resume`, `--strict`

### 4.2 Run registry
Maintain `outputs/run_registry.csv` with:
- run_id, phase, split, model_stack, config_hash, git_sha,
  key metrics, latency, mem, VALID/INVALID, artifact_dir.

### 4.3 Coverage checker
Implement `scripts/check_coverage.py` that reads `docs/MODEL_INVENTORY.md` requirements and fails if missing required runs.

### 4.4 Decision engine
Implement `scripts/decision_engine.py`:
- promotion gates (below)
- writes `configs/locked_best_<stage>.yaml`
- writes leaderboards under `outputs/decision/`

### 4.5 GPU-aware trainer utilities
Implement:
- auto batch size finder
- OOM handler (halve batch, enable grad checkpointing, reduce seq len, retry)
- always logs resolved config into manifest.

---

## 5) Stage 0 baselines (must be reproduced exactly)

Run dev baseline:
- ALL_SENTENCES + heuristic ranking (or current repo baseline)
- BM25 baseline
- current committed best config (if exists)

For each:
- compute selection metrics (micro P/R/F1)
- compute empty metrics (empty P/R/F1, false evidence rate)
- compute ranking metrics (nDCG@{3,5,10}, MRR@{3,5,10}, Recall@{3,5,10})
- produce per_query.csv

Lock baseline config to `configs/locked_baseline.yaml`.

---

## 6) Stage A — Retriever sweep (frozen, then finetune if justified)

### 6.1 Frozen sweep (Tier-0 then Tier-1)
For each retriever in `docs/MODEL_INVENTORY.md`:
- Evaluate using K ∈ {3,5,10} and also “ALL_SENTENCES”.
- Report:
  - oracle recall@K (does gold appear in top K?)
  - downstream reranker-free selection (threshold-only)
  - latency and mem

### 6.2 Decision gate: do we finetune retriever?
Finetune retriever ONLY if at least one is true:
- oracle_recall@5 < target (set target from baseline + desired margin)
- error analysis shows semantic mismatch (not lexical) dominates misses
- reranker gains are capped because candidates are missing

If NOT true:
- skip retriever finetuning; spend GPU on reranker and postprocess.

### 6.3 Retriever finetune (LoRA/QLoRA/full)
If gate passes:
- Train with contrastive + hard negatives mined from current retriever.
- HPO with Hyperband/ASHA on dev-tune.
- Run top 3 configs × 3 seeds.

Outputs:
- `configs/locked_best_retriever.yaml`
- `outputs/stageA/retriever_report.md`

---

## 7) Stage B — Reranker sweep + exhaustive reranker training HPO

### 7.1 Frozen reranker sweep
For each reranker (Tier-0 then Tier-1):
- Using top retriever candidates (or ALL_SENTENCES), compute ranking metrics and selection metrics.

### 7.2 Exhaustive reranker training HPO (this is the “why is SOTA underperforming?” killer step)
For each promising reranker:
- Training must sweep loss families:
  - pointwise (BCE, focal)
  - pairwise (RankNet, hinge)
  - listwise (ListNet, ListMLE, LambdaRank/LambdaLoss)
  - hybrid α/β/γ combination
- Must include hard negative mining + in-batch negatives.
- Must include empty groups if no-evidence policy needs them (sentinel or abstention).

Compute budget logic:
- Use ASHA/Hyperband to prune.
- Multi-fidelity:
  - short runs (few hundred steps) for screening
  - longer runs for top 10%
- For finalists: 3 seeds.

Outputs:
- `configs/locked_best_reranker.yaml`
- `outputs/stageB/reranker_training_report.md`

---

## 8) Stage C — Postprocess HPO (calibration + no-evidence + dynamic-K), per model-pair

This stage MUST run for every promoted (retriever, reranker) pair.

### 8.1 Correct protocol (no leakage)
Split dev into:
- dev_tune: fit calibration, tune thresholds, tune policies
- dev_select: select best policy/config
(Or nested CV if using CV)

### 8.2 Policies to tune (must include all from MODEL_INVENTORY.md)
- calibration: temp / platt / isotonic
- no-evidence: sentinel / abstention / risk-coverage
- dynamic-K: fixed-K / per-criterion threshold / mass-based / gap-elbow / learned-k

### 8.3 Objectives
Primary:
- All-queries micro-F1 (evidence selection)
- Empty precision + false evidence rate constraint

Secondary:
- positives-only nDCG@5/10 and MRR@5/10
- calibration metrics (ECE/Brier/NLL)

Outputs:
- `configs/locked_best_deploy_policy.yaml`
- `outputs/stageC/postprocess_report.md`

---

## 9) Stage D — Optional GNN between reranker and postprocess

Only if Stage C is strong and stable.

Graph:
- nodes = sentences (+ optional criterion node)
- edges = adjacency, semantic kNN, entity overlap (ablate)
Features:
- reranker logit/prob, retriever scores, position, length, embeddings (PCA64)

Models:
- GraphSAGE, GAT

HPO:
- depth/hidden/dropout/edge types/features

Gate: must improve deployment metrics significantly and overhead acceptable.

---

## 10) Stage E — Gemini LLM-as-judge (evaluation-only)

Only if explicitly approved.

Strict rules:
- never tune on test
- cache all calls (hash key: input text + prompt + model version)
- use judge for:
  - error categorization
  - label noise detection
  - qualitative case studies

---

## 11) Paper packaging

Generate:
- main table (baseline vs best)
- ablations (remove components)
- calibration & no-evidence tables
- efficiency table (latency, VRAM, throughput)
- statistical tests (paired bootstrap at post-level)

Modernize notebooks:
- notebooks should be thin wrappers calling scripts and loading artifacts
- execute headless and store executed notebooks in outputs/

---

## 12) Execution order (12-hour+ exhaustive run budget)

Claude Code must run with `--time_budget_hours 12` (minimum) and keep GPU saturated by:
1) audit → verify → profile
2) baselines
3) retriever_sweep (frozen)
4) reranker_sweep (frozen)
5) reranker_train (exhaustive HPO over loss families)
6) postprocess_hpo (per promoted model-pair)
7) optional retriever_train if gate triggers
8) optional GNN
9) optional Gemini judge
10) paper packaging

After every phase:
- validate_runs.py
- verify invariants
- check_coverage.py
- decision_engine.py

If any fails: auto-fix + rerun minimal reproduction.
