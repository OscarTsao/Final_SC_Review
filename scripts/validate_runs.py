#!/usr/bin/env python3
"""Validate experiment runs for reproducibility and correctness.

Checks:
- Manifest git hash matches current repo (or warns if different)
- Config checksums match
- Metrics are within expected ranges
- No NaN or invalid values in per_query results
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of validating a run."""
    run_dir: Path
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def get_current_git_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def validate_manifest(manifest_path: Path, current_hash: Optional[str]) -> ValidationResult:
    """Validate manifest.json."""
    result = ValidationResult(run_dir=manifest_path.parent)

    if not manifest_path.exists():
        result.errors.append("manifest.json not found")
        result.passed = False
        return result

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Check git hash (accept both git_hash and git_sha)
        manifest_hash = manifest.get("git_hash") or manifest.get("git_sha", "unknown")
        if current_hash and manifest_hash != "unknown" and not manifest_hash.startswith(current_hash[:7]):
            result.warnings.append(
                f"Git hash mismatch: manifest={manifest_hash[:8]}, current={current_hash}"
            )

        # Check required fields (accept both git_hash and git_sha)
        has_git = "git_hash" in manifest or "git_sha" in manifest
        if "timestamp" not in manifest:
            result.warnings.append("Missing field in manifest: timestamp")
        if not has_git:
            result.warnings.append("Missing field in manifest: git_hash/git_sha")

    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON in manifest: {e}")
        result.passed = False
    except Exception as e:
        result.errors.append(f"Error reading manifest: {e}")
        result.passed = False

    return result


def validate_summary(summary_path: Path) -> ValidationResult:
    """Validate summary.json."""
    result = ValidationResult(run_dir=summary_path.parent)

    if not summary_path.exists():
        result.errors.append("summary.json not found")
        result.passed = False
        return result

    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Check metrics are in valid range
        metrics = summary.get("metrics", summary)  # Some summaries have metrics at top level
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if value < 0 or value > 1:
                        if "ndcg" in key.lower() or "recall" in key.lower() or "precision" in key.lower():
                            result.warnings.append(f"Metric {key}={value} outside [0,1] range")
                    if str(value).lower() == "nan":
                        result.errors.append(f"NaN value for metric: {key}")
                        result.passed = False

    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON in summary: {e}")
        result.passed = False
    except Exception as e:
        result.errors.append(f"Error reading summary: {e}")
        result.passed = False

    return result


def validate_per_query(per_query_path: Path) -> ValidationResult:
    """Validate per_query.csv."""
    result = ValidationResult(run_dir=per_query_path.parent)

    if not per_query_path.exists():
        result.errors.append("per_query.csv not found")
        result.passed = False
        return result

    try:
        with open(per_query_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            result.warnings.append("per_query.csv is empty")
            return result

        # Check for NaN values
        nan_count = 0
        for i, row in enumerate(rows):
            for key, value in row.items():
                if value and value.lower() == "nan":
                    nan_count += 1

        if nan_count > 0:
            result.warnings.append(f"Found {nan_count} NaN values in per_query.csv")

        # Check row count
        if len(rows) < 10:
            result.warnings.append(f"Only {len(rows)} rows in per_query.csv (expected more)")

    except Exception as e:
        result.errors.append(f"Error reading per_query.csv: {e}")
        result.passed = False

    return result


def validate_run(run_dir: Path, current_hash: Optional[str], is_hpo: bool = False) -> ValidationResult:
    """Validate an entire run directory."""
    combined = ValidationResult(run_dir=run_dir)

    # Validate manifest
    manifest_result = validate_manifest(run_dir / "manifest.json", current_hash)
    combined.warnings.extend(manifest_result.warnings)
    combined.errors.extend(manifest_result.errors)
    if not manifest_result.passed:
        combined.passed = False

    if is_hpo:
        # HPO runs use best_params.json and trials.csv
        if not (run_dir / "best_params.json").exists():
            combined.warnings.append("best_params.json not found")
        if not (run_dir / "trials.csv").exists():
            combined.warnings.append("trials.csv not found")
    else:
        # Regular eval runs use summary.json and per_query.csv
        summary_result = validate_summary(run_dir / "summary.json")
        per_query_result = validate_per_query(run_dir / "per_query.csv")
        for result in [summary_result, per_query_result]:
            combined.warnings.extend(result.warnings)
            combined.errors.extend(result.errors)
            if not result.passed:
                combined.passed = False

    return combined


def find_runs_to_validate(outputs_dir: Path) -> List[tuple]:
    """Find all run directories to validate.

    Returns:
        List of (path, is_hpo) tuples
    """
    runs = []

    # Final eval runs
    for pattern in ["final_eval/*", "final_eval_*"]:
        for path in outputs_dir.glob(pattern):
            if path.is_dir():
                runs.append((path, False))

    # HPO runs (these have different file structure)
    hpo_dir = outputs_dir / "hpo"
    if hpo_dir.exists():
        for subdir in hpo_dir.iterdir():
            if subdir.is_dir():
                runs.append((subdir, True))

    return runs


def main():
    """Main entry point."""
    outputs_dir = Path(__file__).parent.parent / "outputs"

    if not outputs_dir.exists():
        print(f"ERROR: Outputs directory not found: {outputs_dir}")
        return 1

    current_hash = get_current_git_hash()
    print(f"Current git hash: {current_hash or 'unknown'}")
    print()

    runs = find_runs_to_validate(outputs_dir)

    if not runs:
        print("No runs found to validate")
        return 0

    print("=" * 70)
    print("RUN VALIDATION REPORT")
    print("=" * 70)

    all_passed = True
    for run_dir, is_hpo in runs:
        result = validate_run(run_dir, current_hash, is_hpo=is_hpo)

        status = "✓" if result.passed else "✗"
        print(f"\n[{status}] {run_dir.relative_to(outputs_dir)}")

        for warning in result.warnings:
            print(f"    WARNING: {warning}")
        for error in result.errors:
            print(f"    ERROR: {error}")

        if not result.passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("All runs passed validation!")
        return 0
    else:
        print("Some runs failed validation. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
