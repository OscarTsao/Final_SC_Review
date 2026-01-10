# CLEANUP + CLEAN RESTART PROMPT FOR CLAUDE CODE (Conda llmhe, single RTX 5090)

You are Claude Code. Repo: OscarTsao/Final_SC_Review.
Goal: restart the entire research from scratch with a clean repo state and a clean conda environment.
Hard requirements:
- Use ONLY conda + pip.
- Conda env name MUST be exactly: llmhe.
- Single GPU only: enforce CUDA_VISIBLE_DEVICES=0.
- Do not delete data irreversibly; archive outputs and experiment artifacts with timestamps.

============================================================
A) REPO: FETCH ALL, CLEAN TREE, NEW BRANCH
============================================================
1) Go to repo root. Record current state (save outputs):
   - mkdir -p outputs/_pre_restart
   - git status > outputs/_pre_restart/git_status.txt
   - git rev-parse HEAD > outputs/_pre_restart/git_head_sha.txt
   - git branch --show-current > outputs/_pre_restart/git_branch.txt
   - git log --oneline -n 50 > outputs/_pre_restart/git_log.txt

2) Fetch all branches/tags:
   - git fetch --all --prune --tags
   - git branch -a > outputs/_pre_restart/git_all_branches.txt

3) Ensure working tree is clean:
   - If uncommitted changes exist:
     Option A (preferred): git add -A && git commit -m "WIP snapshot before full restart"
     Option B: git stash push -u -m "pre_restart_snapshot_YYYYMMDD"
   - Do NOT proceed with a dirty tree.

4) Create a fresh restart branch off master:
   - git checkout master
   - git pull --ff-only || true
   - git checkout -b full_restart_YYYYMMDD

============================================================
B) OUTPUTS/CACHES: ARCHIVE EVERYTHING THAT CAN CONTAMINATE RUNS
============================================================
1) Create archive folder:
   - mkdir -p outputs/_archive/$(date +%Y%m%d_%H%M%S)

2) Archive prior outputs (do not delete):
   - If outputs/ contains old experiments, move them into the archive folder.
   - Preserve outputs/_archive itself.
   - Preserve outputs/_pre_restart as well.

3) Remove python cache cruft (safe):
   - find . -type d -name "__pycache__" -prune -exec rm -rf {} +
   - rm -rf .pytest_cache

4) Remove any accidental venv folders (forbidden):
   - rm -rf .venv venv .python-version || true

5) IMPORTANT: Do NOT delete the global HuggingFace cache unless disk pressure requires it.
   (Re-downloading slows everything.)

============================================================
C) CONDA ENV: RECREATE llmhe CLEANLY (BACKUP THEN REBUILD)
============================================================
1) Backup current llmhe env spec if it exists:
   - mkdir -p outputs/system/_env_backups/$(date +%Y%m%d_%H%M%S)
   - conda activate llmhe || true
   - conda env export --from-history > outputs/system/_env_backups/$(date +%Y%m%d_%H%M%S)/conda_env_from_history.yml || true
   - conda list --explicit > outputs/system/_env_backups/$(date +%Y%m%d_%H%M%S)/conda_list_explicit.txt || true
   - pip freeze > outputs/system/_env_backups/$(date +%Y%m%d_%H%M%S)/pip_freeze.txt || true

2) Remove llmhe completely:
   - conda deactivate || true
   - conda env remove -n llmhe -y || true

3) Recreate llmhe:
   If environment.yml exists:
     - conda env create -f environment.yml -n llmhe -y
   Else:
     - conda create -n llmhe python=3.11 -y

4) Activate and install repo with pip:
   - conda activate llmhe
   - python -m pip install -U pip
   - pip install -e .

5) Snapshot fresh environment (mandatory):
   - mkdir -p outputs/system
   - conda env export --from-history > outputs/system/conda_env_from_history.yml
   - conda list --explicit > outputs/system/conda_list_explicit.txt
   - pip freeze > outputs/system/pip_freeze.txt
   - conda -V > outputs/system/conda_version.txt
   - python -V > outputs/system/python_version.txt
   - pip -V > outputs/system/pip_version.txt

============================================================
D) HARDWARE + GPU PINNING
============================================================
1) Pin to single GPU:
   - export CUDA_VISIBLE_DEVICES=0

2) Record hardware:
   - nvidia-smi > outputs/system/nvidia_smi.txt

3) Ensure hw probe exists; create if missing; then run:
   - python scripts/hw_probe.py
   It must output outputs/system/hw.json

============================================================
E) PREFLIGHT + TESTS (MUST PASS BEFORE ANY RESEARCH)
============================================================
1) Ensure preflight script exists; create if missing:
   - scripts/preflight_env.py
   Requirements:
   - hard-fail unless conda env is llmhe
   - hard-fail unless CUDA is available and exactly one GPU visible
   - write/verify outputs/system snapshots exist
   - forbid .venv usage

2) Run:
   - python scripts/preflight_env.py --strict
   - pytest -q

Stop and fix if any fail.

============================================================
F) START RETRIEVER RESEARCH
============================================================
1) Create the plan file at repo root:
   - retriever_research_plan.md (use the exact content provided by the user)

2) Implement/verify retriever pipeline scripts:
   - scripts/retriever/retriever_driver.py
   - scripts/retriever/build_candidates.py
   - scripts/retriever/check_gold_alignment.py
   - scripts/retriever/model_zoo_smoke_test.py
   - scripts/retriever/build_cache.py
   - scripts/retriever/hpo_frozen.py
   - scripts/retriever/dev_select_eval.py
   - scripts/retriever/check_retriever_coverage.py
   - scripts/validate_runs.py
   - scripts/verify_invariants.py

3) Execute phases sequentially as specified in retriever_research_plan.md.
After each phase:
- validate_runs + invariants + coverage
- only proceed if all pass

END.
