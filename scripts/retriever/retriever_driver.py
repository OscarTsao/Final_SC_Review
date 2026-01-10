#!/usr/bin/env python3
"""
Retriever Research Driver - Orchestrates all retriever experiment phases.

Phases:
1. Build candidates and validate cache
2. Check gold alignment
3. Model zoo smoke test
4. HPO on frozen embeddings
5. Dev select results
6. Coverage check

Usage:
    python scripts/retriever/retriever_driver.py --phase all
    python scripts/retriever/retriever_driver.py --phase build_cache
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def run_script(script_name: str, args: list = None, cwd: Path = None) -> bool:
    """Run a Python script and return success status."""
    repo_root = get_repo_root()
    script_path = repo_root / "scripts" / "retriever" / script_name

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)] + (args or [])
    print(f"\n{'='*60}")
    print(f"[PHASE] Running: {script_name}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd or repo_root),
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {script_name} failed with exit code {e.returncode}")
        return False


def phase_build_cache(config_path: str = None) -> bool:
    """Phase 1: Build embedding caches."""
    args = ["--config", config_path] if config_path else []
    return run_script("build_cache.py", args)


def phase_check_gold_alignment() -> bool:
    """Phase 2: Verify gold labels align with corpus."""
    return run_script("check_gold_alignment.py")


def phase_model_zoo_smoke_test() -> bool:
    """Phase 3: Quick smoke test of retriever models."""
    return run_script("model_zoo_smoke_test.py", ["--n_samples", "50"])


def phase_build_candidates(config_path: str = None) -> bool:
    """Phase 4: Build candidate pools."""
    args = ["--config", config_path] if config_path else []
    return run_script("build_candidates.py", args)


def phase_hpo_frozen(n_trials: int = 100) -> bool:
    """Phase 5: Run HPO on frozen embeddings."""
    return run_script("hpo_frozen.py", ["--n_trials", str(n_trials)])


def phase_dev_select_check() -> bool:
    """Phase 6: Check on dev_select split."""
    return run_script("dev_select_eval.py")


def phase_check_coverage() -> bool:
    """Phase 7: Check retrieval coverage."""
    return run_script("check_retriever_coverage.py")


def run_validation_scripts(repo_root: Path) -> bool:
    """Run validation scripts after each phase."""
    validate_path = repo_root / "scripts" / "validate_runs.py"
    invariants_path = repo_root / "scripts" / "verify_invariants.py"

    success = True
    for script_path in [validate_path, invariants_path]:
        if script_path.exists():
            print(f"\n[VALIDATE] Running: {script_path.name}")
            try:
                subprocess.run([sys.executable, str(script_path)], check=True)
            except subprocess.CalledProcessError:
                print(f"[WARN] Validation script failed: {script_path.name}")
                success = False
    return success


def run_all_phases(config_path: str = None, n_trials: int = 100) -> dict:
    """Run all phases sequentially."""
    repo_root = get_repo_root()
    results = {}
    phases = [
        ("build_cache", lambda: phase_build_cache(config_path)),
        ("check_gold_alignment", phase_check_gold_alignment),
        ("model_zoo_smoke_test", phase_model_zoo_smoke_test),
        ("build_candidates", lambda: phase_build_candidates(config_path)),
        ("hpo_frozen", lambda: phase_hpo_frozen(n_trials)),
        ("dev_select_check", phase_dev_select_check),
        ("check_coverage", phase_check_coverage),
    ]

    for phase_name, phase_fn in phases:
        print(f"\n{'#'*60}")
        print(f"# Starting phase: {phase_name}")
        print(f"{'#'*60}")

        success = phase_fn()
        results[phase_name] = "PASS" if success else "FAIL"

        if success:
            run_validation_scripts(repo_root)
        else:
            print(f"\n[ABORT] Phase {phase_name} failed. Stopping execution.")
            break

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retriever Research Driver"
    )
    parser.add_argument(
        "--phase",
        choices=[
            "all", "build_cache", "check_gold_alignment", "model_zoo_smoke_test",
            "build_candidates", "hpo_frozen", "dev_select_check", "check_coverage"
        ],
        default="all",
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of HPO trials (default: 100)"
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    start_time = datetime.now()

    print("="*60)
    print("RETRIEVER RESEARCH DRIVER")
    print(f"Start time: {start_time.isoformat()}")
    print(f"Phase: {args.phase}")
    print("="*60)

    if args.phase == "all":
        results = run_all_phases(args.config, args.n_trials)
    else:
        phase_map = {
            "build_cache": lambda: phase_build_cache(args.config),
            "check_gold_alignment": phase_check_gold_alignment,
            "model_zoo_smoke_test": phase_model_zoo_smoke_test,
            "build_candidates": lambda: phase_build_candidates(args.config),
            "hpo_frozen": lambda: phase_hpo_frozen(args.n_trials),
            "dev_select_check": phase_dev_select_check,
            "check_coverage": phase_check_coverage,
        }
        success = phase_map[args.phase]()
        results = {args.phase: "PASS" if success else "FAIL"}

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    output_dir = repo_root / "outputs" / "retriever"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "driver_results.json"

    summary = {
        "timestamp": end_time.isoformat(),
        "duration_seconds": duration,
        "phase_results": results,
        "config": args.config,
    }
    results_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "="*60)
    print("RETRIEVER RESEARCH DRIVER - COMPLETE")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: {results}")
    print(f"Saved to: {results_path}")
    print("="*60)


if __name__ == "__main__":
    main()
