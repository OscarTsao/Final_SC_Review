#!/usr/bin/env python3
"""Verify critical research invariants.

This script runs paper-grade invariant checks to ensure:
I1) Metrics correctness - ranking metrics match known values
I2) Split leakage - train/val/test post_ids are disjoint
I3) Determinism - same config produces identical results
I4) Output contract - all eval outputs have required artifacts
I5) Cache integrity - HPO uses val split, never test
I6) Empty-query validity - evaluation handles empty queries correctly
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def run_pytest_invariants() -> Tuple[bool, List[str]]:
    """Run pytest tests that verify invariants."""
    invariant_tests = [
        ("I1 - Metrics exactness", "tests/test_metrics.py"),
        ("I2 - Split leakage prevention", "tests/test_no_leakage_splits.py"),
        ("I3 - Determinism", "tests/test_hpo_cache_determinism.py"),
        ("I4 - Candidate pool within-post", "tests/test_candidate_pool_within_post_only.py"),
        ("I5 - HPO never uses test split", "tests/test_hpo_never_uses_test_split.py"),
        ("I5 - Cache integrity", "tests/test_cache_fingerprint.py"),
        ("I6 - Postprocessing correctness", "tests/test_postprocessing.py"),
    ]

    results = []
    all_passed = True

    for name, test_path in invariant_tests:
        result = subprocess.run(
            ["pytest", "-q", test_path],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        passed = result.returncode == 0
        if not passed:
            all_passed = False
        results.append((name, passed, result.stdout + result.stderr))

    return all_passed, results


def check_output_contract() -> Tuple[bool, str]:
    """I4: Check that final_eval outputs have required artifacts."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    issues = []

    # Check final_eval directories
    final_eval_dirs = list((outputs_dir / "final_eval").glob("*")) if (outputs_dir / "final_eval").exists() else []
    final_eval_dirs.extend(outputs_dir.glob("final_eval_*"))

    for run_dir in final_eval_dirs:
        if not run_dir.is_dir():
            continue
        required = ["summary.json", "per_query.csv", "manifest.json"]
        for artifact in required:
            if not (run_dir / artifact).exists():
                issues.append(f"{run_dir.name}: missing {artifact}")

    return len(issues) == 0, "\n".join(issues) if issues else "All outputs have required artifacts"


def check_hpo_uses_val_only() -> Tuple[bool, str]:
    """I5: Verify HPO caches and studies use val split only."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    hpo_dir = outputs_dir / "hpo"

    if not hpo_dir.exists():
        return True, "No HPO directory found"

    issues = []

    for study_dir in hpo_dir.iterdir():
        if not study_dir.is_dir():
            continue
        manifest_path = study_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            split = manifest.get("split") or manifest.get("config", {}).get("evaluation", {}).get("split")
            if split and split.lower() == "test":
                issues.append(f"{study_dir.name}: uses test split (INVALID)")

    return len(issues) == 0, "\n".join(issues) if issues else "All HPO studies use val/dev split only"


def check_cache_split() -> Tuple[bool, str]:
    """I5: Verify HPO cache uses val split."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    cache_dir = outputs_dir / "hpo_cache"

    if not cache_dir.exists():
        return True, "No HPO cache found"

    issues = []

    for cache_subdir in cache_dir.iterdir():
        if not cache_subdir.is_dir():
            continue
        manifest_path = cache_subdir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            split = manifest.get("split")
            if split and split.lower() == "test":
                issues.append(f"{cache_subdir.name}: cache uses test split (INVALID)")

    return len(issues) == 0, "\n".join(issues) if issues else "All HPO caches use val/dev split"


def main() -> int:
    """Run all invariant checks and generate report."""
    print("=" * 70)
    print("INVARIANTS VERIFICATION REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 70)

    all_passed = True
    report_lines = [
        "# Invariants Verification Report\n",
        f"**Generated**: {datetime.now().isoformat()}\n",
        "\n## Summary\n",
    ]

    # Run pytest invariants
    print("\n[1/4] Running pytest invariant tests...")
    pytest_passed, pytest_results = run_pytest_invariants()

    for name, passed, output in pytest_results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
            print(f"    {output[:200]}...")

    report_lines.append("\n### Pytest Invariant Tests\n")
    report_lines.append("| Invariant | Status |\n|-----------|--------|\n")
    for name, passed, _ in pytest_results:
        status = "PASS" if passed else "FAIL"
        report_lines.append(f"| {name} | {status} |\n")

    # Check output contract
    print("\n[2/4] Checking output contract (I4)...")
    contract_passed, contract_msg = check_output_contract()
    status = f"{GREEN}PASS{RESET}" if contract_passed else f"{RED}FAIL{RESET}"
    print(f"  {status}: Output contract")
    if not contract_passed:
        all_passed = False
        print(f"    {contract_msg}")

    report_lines.append(f"\n### Output Contract (I4)\n{contract_msg}\n")

    # Check HPO uses val only
    print("\n[3/4] Checking HPO split usage (I5)...")
    hpo_passed, hpo_msg = check_hpo_uses_val_only()
    status = f"{GREEN}PASS{RESET}" if hpo_passed else f"{RED}FAIL{RESET}"
    print(f"  {status}: HPO val-only usage")
    if not hpo_passed:
        all_passed = False
        print(f"    {hpo_msg}")

    report_lines.append(f"\n### HPO Split Usage (I5)\n{hpo_msg}\n")

    # Check cache split
    print("\n[4/4] Checking cache split (I5)...")
    cache_passed, cache_msg = check_cache_split()
    status = f"{GREEN}PASS{RESET}" if cache_passed else f"{RED}FAIL{RESET}"
    print(f"  {status}: Cache split")
    if not cache_passed:
        all_passed = False
        print(f"    {cache_msg}")

    report_lines.append(f"\n### Cache Split (I5)\n{cache_msg}\n")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(f"{GREEN}ALL INVARIANTS PASSED{RESET}")
        report_lines.insert(3, "**Status**: ALL INVARIANTS PASSED\n")
    else:
        print(f"{RED}SOME INVARIANTS FAILED{RESET}")
        report_lines.insert(3, "**Status**: SOME INVARIANTS FAILED\n")
    print("=" * 70)

    # Write report
    report_path = Path(__file__).parent.parent / "outputs" / "paper_audit" / "invariants_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"\nReport saved to: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
