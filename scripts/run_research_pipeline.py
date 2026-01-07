#!/usr/bin/env python3
"""Main research pipeline driver for S-C evidence retrieval.

Orchestrates the full paper-grade reproducible research workflow:
- Phase B: Hardware setup and validation
- Phase C: Artifact audit
- Phase D: Baseline reproduction
- Phase E: Invariant verification
- Stage 2-4: Retriever, reranker, post-processing experiments

Usage:
    python scripts/run_research_pipeline.py --phase all
    python scripts/run_research_pipeline.py --phase baseline --split val
    python scripts/run_research_pipeline.py --phase retriever --budget quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def run_command(cmd: List[str], description: str, timeout: int = 3600) -> bool:
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def phase_hardware() -> bool:
    """Phase B: Hardware probe and environment validation."""
    print("\n" + "="*70)
    print("PHASE B: Hardware Setup and Validation")
    print("="*70)

    # Run hardware probe
    if not run_command(["python", "scripts/hw_probe.py"], "Hardware probe"):
        return False

    # Run tests
    if not run_command(["python", "-m", "pytest", "tests/", "-v"], "Test suite"):
        return False

    return True


def phase_audit() -> bool:
    """Phase C: Artifact audit."""
    print("\n" + "="*70)
    print("PHASE C: Artifact Audit")
    print("="*70)

    # Run artifact audit
    run_command(["python", "scripts/audit_pushed_results.py"], "Artifact audit")

    # Run validation
    run_command(["python", "scripts/validate_runs.py"], "Run validation")

    return True


def phase_baseline(split: str = "val", config: str = "default") -> bool:
    """Phase D: Baseline reproduction."""
    print("\n" + "="*70)
    print(f"PHASE D: Baseline Reproduction ({split})")
    print("="*70)

    config_path = CONFIGS_DIR / f"{config}.yaml"
    if not config_path.exists():
        config_path = CONFIGS_DIR / "default.yaml"

    cmd = [
        "python", "scripts/final_eval.py",
        "--config", str(config_path),
        "--split", split,
        "--dual_eval",
        "--output_dir", str(OUTPUTS_DIR / f"baseline_{config}_{split}"),
    ]

    return run_command(cmd, f"Baseline evaluation on {split}", timeout=3600)


def phase_invariants() -> bool:
    """Phase E: Invariant and safety checks."""
    print("\n" + "="*70)
    print("PHASE E: Invariant Verification")
    print("="*70)

    # Run specific invariant tests
    invariant_tests = [
        "tests/test_no_leakage_splits.py",
        "tests/test_hpo_never_uses_test_split.py",
        "tests/test_candidate_pool_within_post_only.py",
        "tests/test_id_mapping_no_collision.py",
    ]

    all_passed = True
    for test in invariant_tests:
        test_path = PROJECT_ROOT / test
        if test_path.exists():
            if not run_command(
                ["python", "-m", "pytest", test, "-v"],
                f"Invariant: {Path(test).stem}"
            ):
                all_passed = False

    return all_passed


def phase_retriever(budget: str = "standard") -> bool:
    """Stage 2: Retriever HPO."""
    print("\n" + "="*70)
    print("STAGE 2: Retriever Improvements")
    print("="*70)

    n_trials = {"quick": 20, "standard": 50, "extensive": 100}[budget]

    # Build HPO cache first
    run_command(
        ["python", "scripts/precompute_hpo_cache.py", "--config", "configs/hpo_inference_v2.yaml"],
        "Building HPO cache",
        timeout=7200,
    )

    # Run HPO
    return run_command(
        [
            "python", "scripts/hpo_inference.py",
            "--config", "configs/hpo_inference_v2.yaml",
            "--n_trials", str(n_trials),
            "--study_name", f"retriever_hpo_{datetime.now().strftime('%Y%m%d')}",
        ],
        f"Retriever HPO ({n_trials} trials)",
        timeout=14400,
    )


def phase_reranker(budget: str = "standard") -> bool:
    """Stage 3: Reranker experiments."""
    print("\n" + "="*70)
    print("STAGE 3: Reranker Improvements")
    print("="*70)

    # Run ablation study
    return run_command(
        ["python", "scripts/run_ablations.py", "--split", "val"],
        "Ablation study",
        timeout=7200,
    )


def phase_postprocessing() -> bool:
    """Stage 4: Post-processing experiments."""
    print("\n" + "="*70)
    print("STAGE 4: Post-Processing")
    print("="*70)

    print("Post-processing experiments not yet implemented.")
    print("Planned: calibration, no-evidence detection, dynamic-k")

    return True


def phase_reporting() -> bool:
    """Phase L: Generate paper tables and reports."""
    print("\n" + "="*70)
    print("PHASE L: Reporting")
    print("="*70)

    # Collect all results
    results_summary = collect_results_summary()

    # Save summary
    report_path = OUTPUTS_DIR / "paper_results_summary.json"
    with open(report_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"Results summary saved to: {report_path}")

    # Print key metrics
    print("\n--- Key Results ---")
    for name, metrics in results_summary.get("experiments", {}).items():
        if isinstance(metrics, dict) and "ndcg@10" in str(metrics):
            print(f"{name}: {metrics}")

    return True


def collect_results_summary() -> Dict:
    """Collect all experiment results into a summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    # Collect final eval results
    for eval_dir in OUTPUTS_DIR.glob("final_eval*"):
        if eval_dir.is_dir():
            summary_path = eval_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary["experiments"][eval_dir.name] = json.load(f)

    # Collect baseline results
    for baseline in OUTPUTS_DIR.glob("baseline_*.json"):
        if not baseline.name.endswith(".queries.csv"):
            with open(baseline) as f:
                summary["experiments"][baseline.stem] = json.load(f)

    # Collect HPO best params
    for hpo_dir in (OUTPUTS_DIR / "hpo").glob("*"):
        if hpo_dir.is_dir():
            best_path = hpo_dir / "best_params.json"
            if best_path.exists():
                with open(best_path) as f:
                    summary["experiments"][f"hpo_{hpo_dir.name}"] = json.load(f)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run research pipeline")
    parser.add_argument(
        "--phase",
        choices=["all", "hardware", "audit", "baseline", "invariants",
                 "retriever", "reranker", "postprocessing", "reporting"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split for baseline",
    )
    parser.add_argument(
        "--budget",
        choices=["quick", "standard", "extensive"],
        default="standard",
        help="Compute budget for HPO",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Config name for baseline",
    )
    args = parser.parse_args()

    phases = {
        "hardware": phase_hardware,
        "audit": phase_audit,
        "baseline": lambda: phase_baseline(args.split, args.config),
        "invariants": phase_invariants,
        "retriever": lambda: phase_retriever(args.budget),
        "reranker": lambda: phase_reranker(args.budget),
        "postprocessing": phase_postprocessing,
        "reporting": phase_reporting,
    }

    if args.phase == "all":
        # Run all phases in order
        phase_order = ["hardware", "audit", "invariants", "baseline", "retriever", "reranker", "reporting"]
        for phase_name in phase_order:
            if not phases[phase_name]():
                print(f"\nERROR: Phase {phase_name} failed!")
                return 1
    else:
        if not phases[args.phase]():
            print(f"\nERROR: Phase {args.phase} failed!")
            return 1

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
