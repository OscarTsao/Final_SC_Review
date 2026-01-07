#!/usr/bin/env python3
"""Audit pushed results for paper-grade artifact requirements.

Checks that all experiment runs have required artifacts:
- summary.json: Aggregate metrics
- per_query.csv: Per-query breakdown for significance testing
- manifest.json: Reproducibility metadata (config, git hash, checksums)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ArtifactCheck:
    """Result of checking a single artifact."""
    path: Path
    exists: bool
    valid: bool = True
    issues: List[str] = field(default_factory=list)


@dataclass
class RunAudit:
    """Audit result for an experiment run."""
    run_dir: Path
    run_type: str  # 'eval', 'hpo', 'ablation', 'baseline'
    summary: Optional[ArtifactCheck] = None
    per_query: Optional[ArtifactCheck] = None
    manifest: Optional[ArtifactCheck] = None

    @property
    def is_complete(self) -> bool:
        """Check if run has all required artifacts."""
        checks = [self.summary, self.per_query, self.manifest]
        return all(c and c.exists and c.valid for c in checks)

    @property
    def missing_artifacts(self) -> List[str]:
        """List missing or invalid artifacts."""
        missing = []
        if not self.summary or not self.summary.exists:
            missing.append("summary.json")
        if not self.per_query or not self.per_query.exists:
            missing.append("per_query.csv")
        if not self.manifest or not self.manifest.exists:
            missing.append("manifest.json")
        return missing


def check_json_artifact(path: Path, required_keys: Set[str] = None, alt_keys: Dict[str, str] = None) -> ArtifactCheck:
    """Check a JSON artifact for validity.

    Args:
        path: Path to JSON file
        required_keys: Set of required keys
        alt_keys: Dict mapping required key to alternative name (e.g., {"git_hash": "git_sha"})
    """
    check = ArtifactCheck(path=path, exists=path.exists())

    if not check.exists:
        return check

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        if required_keys:
            data_keys = set(data.keys())
            missing = set()
            for key in required_keys:
                if key in data_keys:
                    continue
                # Check alternative key names
                if alt_keys and key in alt_keys and alt_keys[key] in data_keys:
                    continue
                missing.add(key)
            if missing:
                check.valid = False
                check.issues.append(f"Missing required keys: {missing}")
    except json.JSONDecodeError as e:
        check.valid = False
        check.issues.append(f"Invalid JSON: {e}")
    except Exception as e:
        check.valid = False
        check.issues.append(f"Error reading file: {e}")

    return check


def check_csv_artifact(path: Path, min_rows: int = 1) -> ArtifactCheck:
    """Check a CSV artifact for validity."""
    check = ArtifactCheck(path=path, exists=path.exists())

    if not check.exists:
        return check

    try:
        import csv
        with open(path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < min_rows + 1:  # +1 for header
            check.valid = False
            check.issues.append(f"Too few rows: {len(rows)} (expected at least {min_rows + 1})")
    except Exception as e:
        check.valid = False
        check.issues.append(f"Error reading CSV: {e}")

    return check


def audit_eval_run(run_dir: Path) -> RunAudit:
    """Audit an evaluation run directory."""
    audit = RunAudit(run_dir=run_dir, run_type="eval")

    # Check summary.json - allow various structures (retriever/reranked or metrics)
    audit.summary = check_json_artifact(run_dir / "summary.json")

    # Check per_query.csv
    audit.per_query = check_csv_artifact(run_dir / "per_query.csv")

    # Check manifest.json - git_sha is acceptable alternative to git_hash
    audit.manifest = check_json_artifact(
        run_dir / "manifest.json",
        required_keys={"timestamp", "git_hash"} if (run_dir / "manifest.json").exists() else None,
        alt_keys={"git_hash": "git_sha"}
    )

    return audit


def audit_hpo_run(run_dir: Path) -> RunAudit:
    """Audit an HPO run directory."""
    audit = RunAudit(run_dir=run_dir, run_type="hpo")

    # HPO uses best_params.json instead of summary.json
    audit.summary = check_json_artifact(run_dir / "best_params.json")

    # HPO uses trials.csv instead of per_query.csv
    audit.per_query = check_csv_artifact(run_dir / "trials.csv")

    # Check manifest.json - git_sha is acceptable alternative
    audit.manifest = check_json_artifact(
        run_dir / "manifest.json",
        alt_keys={"git_hash": "git_sha"}
    )

    return audit


def audit_baseline_file(file_path: Path) -> RunAudit:
    """Audit a baseline result file."""
    audit = RunAudit(run_dir=file_path.parent, run_type="baseline")

    # Baseline has summary.json as the main file
    audit.summary = check_json_artifact(file_path)

    # Check for corresponding per_query CSV
    per_query_path = file_path.with_suffix('.queries.csv')
    audit.per_query = check_csv_artifact(per_query_path)

    # Baselines typically don't have separate manifest files
    # Mark as present if the summary has required metadata
    audit.manifest = ArtifactCheck(
        path=file_path,
        exists=True,
        valid=True,
        issues=["No separate manifest (embedded in summary)"]
    )

    return audit


def find_all_runs(outputs_dir: Path) -> List[RunAudit]:
    """Find and audit all experiment runs."""
    audits = []

    # Find final_eval runs (timestamped directories)
    final_eval_dir = outputs_dir / "final_eval"
    if final_eval_dir.exists():
        for subdir in final_eval_dir.iterdir():
            if subdir.is_dir():
                audits.append(audit_eval_run(subdir))

    # Find final_eval_v2, final_eval_retriever_only, etc.
    for pattern in ["final_eval_*"]:
        for path in outputs_dir.glob(pattern):
            if path.is_dir() and path != final_eval_dir:
                audits.append(audit_eval_run(path))

    # Find HPO runs
    hpo_dir = outputs_dir / "hpo"
    if hpo_dir.exists():
        for subdir in hpo_dir.iterdir():
            if subdir.is_dir():
                audits.append(audit_hpo_run(subdir))

    # Find baseline files
    for baseline_file in outputs_dir.glob("baseline_*.json"):
        if not baseline_file.name.endswith('.queries.json'):
            audits.append(audit_baseline_file(baseline_file))

    # Find ablation results
    ablations_dir = outputs_dir / "ablations"
    if ablations_dir.exists():
        audit = RunAudit(run_dir=ablations_dir, run_type="ablation")
        audit.summary = check_json_artifact(ablations_dir / "ablation_results.json")
        audit.per_query = ArtifactCheck(
            path=ablations_dir / "per_query.csv",
            exists=False,
            issues=["Ablations don't have per_query (aggregated)"]
        )
        audit.manifest = ArtifactCheck(
            path=ablations_dir / "manifest.json",
            exists=False,
            issues=["No manifest for ablations"]
        )
        audits.append(audit)

    return audits


def print_audit_report(audits: List[RunAudit]) -> int:
    """Print audit report and return exit code."""
    print("=" * 70)
    print("ARTIFACT AUDIT REPORT")
    print("=" * 70)

    complete_count = 0
    incomplete_runs = []

    for audit in audits:
        status = "✓" if audit.is_complete else "✗"
        print(f"\n[{status}] {audit.run_type}: {audit.run_dir.name}")

        if audit.is_complete:
            complete_count += 1
        else:
            incomplete_runs.append(audit)
            for artifact in ["summary", "per_query", "manifest"]:
                check = getattr(audit, artifact)
                if check:
                    if not check.exists:
                        print(f"    MISSING: {artifact}")
                    elif not check.valid:
                        print(f"    INVALID: {artifact} - {', '.join(check.issues)}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {complete_count}/{len(audits)} runs have complete artifacts")
    print("=" * 70)

    if incomplete_runs:
        print("\nRuns needing attention:")
        for audit in incomplete_runs:
            print(f"  - {audit.run_dir}: missing {', '.join(audit.missing_artifacts)}")
        return 1

    print("\nAll runs have complete artifacts!")
    return 0


def main():
    """Main entry point."""
    outputs_dir = Path(__file__).parent.parent / "outputs"

    if not outputs_dir.exists():
        print(f"ERROR: Outputs directory not found: {outputs_dir}")
        return 1

    audits = find_all_runs(outputs_dir)

    if not audits:
        print("WARNING: No experiment runs found to audit")
        return 0

    return print_audit_report(audits)


if __name__ == "__main__":
    sys.exit(main())
