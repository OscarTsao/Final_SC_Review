#!/usr/bin/env python3
"""Check model/loss/policy coverage against MODEL_INVENTORY.md requirements.

This script verifies that the research pipeline has exhaustively covered:
1. Tier-0 retriever models (frozen + finetuned)
2. Tier-0 reranker models (frozen + finetuned)
3. All loss families (pointwise/pairwise/listwise/hybrid)
4. All postprocess tracks (calibration/no-evidence/dynamic-K)

Usage:
    python scripts/check_coverage.py [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


# Coverage requirements from MODEL_INVENTORY.md

TIER0_RETRIEVERS = {
    "bge-m3": {"model_id": "BAAI/bge-m3", "variants": ["dense", "dense+sparse", "dense+sparse+colbert"]},
    "bm25": {"model_id": "bm25", "variants": ["default"]},
    "all_sentences": {"model_id": "all_sentences", "variants": ["default"]},
}

TIER0_RERANKERS = {
    "bge-reranker-v2-m3": {"model_id": "BAAI/bge-reranker-v2-m3"},
    "jina-reranker-v3": {"model_id": "jinaai/jina-reranker-v3"},
    "no-reranker": {"model_id": "none"},
}

LOSS_FAMILIES = {
    "pointwise": ["bce", "focal"],
    "pairwise": ["ranknet", "margin"],
    "listwise": ["listnet", "listmle", "lambdarank"],
    "hybrid": ["hybrid"],
}

POSTPROCESS_TRACKS = {
    "calibration": ["temperature_scaling", "platt_scaling", "isotonic"],
    "no_evidence_A": ["sentinel"],
    "no_evidence_B": ["abstention_classifier"],
    "no_evidence_C": ["risk_coverage"],
    "dynamic_k": ["fixed_k", "threshold", "mass_based", "gap_elbow"],
}


@dataclass
class CoverageStatus:
    """Coverage status for a category."""
    category: str
    required: Set[str]
    found: Set[str]
    missing: Set[str] = field(default_factory=set)

    @property
    def is_complete(self) -> bool:
        return len(self.missing) == 0

    @property
    def coverage_pct(self) -> float:
        if not self.required:
            return 100.0
        return 100.0 * len(self.found & self.required) / len(self.required)


def find_retriever_runs(outputs_dir: Path) -> Set[str]:
    """Find retriever models that have been evaluated."""
    found = set()

    # Check baseline outputs
    for pattern in ["baseline_*", "retriever_hpo/*", "zoo/*"]:
        for path in outputs_dir.glob(pattern):
            if path.is_dir():
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        retriever = manifest.get("retriever") or manifest.get("config", {}).get("models", {}).get("retriever")
                        if retriever:
                            found.add(retriever.lower().replace("/", "-").replace("_", "-"))
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Check for standard retriever names in summary files
    for summary_path in outputs_dir.glob("**/summary.json"):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            config = summary.get("config", {})
            retriever = config.get("retriever") or config.get("models", {}).get("retriever")
            if retriever:
                found.add(retriever.lower().replace("/", "-").replace("_", "-"))
        except (json.JSONDecodeError, KeyError):
            pass

    return found


def find_reranker_runs(outputs_dir: Path) -> Set[str]:
    """Find reranker models that have been evaluated."""
    found = set()

    # Check reranker HPO and training outputs
    for pattern in ["reranker_*", "final_eval*", "hpo/*"]:
        for path in outputs_dir.glob(pattern):
            if path.is_dir():
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        reranker = manifest.get("reranker") or manifest.get("config", {}).get("models", {}).get("reranker")
                        if reranker:
                            found.add(reranker.lower().replace("/", "-").replace("_", "-"))
                    except (json.JSONDecodeError, KeyError):
                        pass

    return found


def find_loss_runs(outputs_dir: Path) -> Set[str]:
    """Find loss functions that have been used in training."""
    found = set()

    # Check training logs and configs
    for pattern in ["reranker_*", "hpo/*", "training_*"]:
        for path in outputs_dir.glob(pattern):
            if path.is_dir():
                # Check training config
                for config_file in path.glob("*.yaml"):
                    try:
                        import yaml
                        with open(config_file) as f:
                            config = yaml.safe_load(f)
                        loss = config.get("training", {}).get("loss") or config.get("loss")
                        if loss:
                            found.add(loss.lower())
                    except Exception:
                        pass

                # Check manifest
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        loss = manifest.get("loss") or manifest.get("training", {}).get("loss")
                        if loss:
                            found.add(loss.lower())
                    except Exception:
                        pass

    # Also check trial logs for HPO
    for trials_path in outputs_dir.glob("**/trials.csv"):
        try:
            import csv
            with open(trials_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loss = row.get("loss") or row.get("loss_type")
                    if loss:
                        found.add(loss.lower())
        except Exception:
            pass

    return found


def find_postprocess_runs(outputs_dir: Path) -> Dict[str, Set[str]]:
    """Find postprocessing methods that have been evaluated."""
    found = {
        "calibration": set(),
        "no_evidence": set(),
        "dynamic_k": set(),
    }

    # Check postprocessing outputs
    for pattern in ["postprocessing/*", "postprocess_*", "stageE/*"]:
        for path in outputs_dir.glob(pattern):
            if path.is_dir():
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            manifest = json.load(f)

                        # Calibration
                        cal = manifest.get("calibration") or manifest.get("config", {}).get("calibration")
                        if cal:
                            found["calibration"].add(cal.lower())

                        # No-evidence
                        ne = manifest.get("no_evidence") or manifest.get("config", {}).get("no_evidence")
                        if ne:
                            found["no_evidence"].add(ne.lower())

                        # Dynamic-K
                        dk = manifest.get("dynamic_k") or manifest.get("config", {}).get("dynamic_k")
                        if dk:
                            found["dynamic_k"].add(dk.lower())
                    except Exception:
                        pass

    return found


def check_coverage(outputs_dir: Path, strict: bool = False) -> int:
    """Check coverage and return exit code."""
    print("=" * 70)
    print("COVERAGE CHECK REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 70)

    all_complete = True
    coverage_results = []

    # Check retrievers
    print("\n[1/4] Checking Tier-0 Retriever Coverage...")
    retriever_required = set()
    for name, info in TIER0_RETRIEVERS.items():
        retriever_required.add(name)
    retriever_found = find_retriever_runs(outputs_dir)
    retriever_status = CoverageStatus(
        category="Tier-0 Retrievers",
        required=retriever_required,
        found=retriever_found,
        missing=retriever_required - retriever_found,
    )
    coverage_results.append(retriever_status)

    if retriever_status.is_complete:
        print(f"  PASS: All {len(retriever_required)} Tier-0 retrievers covered")
    else:
        print(f"  INCOMPLETE: {retriever_status.coverage_pct:.0f}% coverage")
        print(f"  Missing: {', '.join(retriever_status.missing)}")
        if strict:
            all_complete = False

    # Check rerankers
    print("\n[2/4] Checking Tier-0 Reranker Coverage...")
    reranker_required = set(TIER0_RERANKERS.keys())
    reranker_found = find_reranker_runs(outputs_dir)
    reranker_status = CoverageStatus(
        category="Tier-0 Rerankers",
        required=reranker_required,
        found=reranker_found,
        missing=reranker_required - reranker_found,
    )
    coverage_results.append(reranker_status)

    if reranker_status.is_complete:
        print(f"  PASS: All {len(reranker_required)} Tier-0 rerankers covered")
    else:
        print(f"  INCOMPLETE: {reranker_status.coverage_pct:.0f}% coverage")
        print(f"  Missing: {', '.join(reranker_status.missing)}")
        if strict:
            all_complete = False

    # Check loss functions
    print("\n[3/4] Checking Loss Function Coverage...")
    loss_required = set()
    for family, losses in LOSS_FAMILIES.items():
        loss_required.update(losses)
    loss_found = find_loss_runs(outputs_dir)
    loss_status = CoverageStatus(
        category="Loss Functions",
        required=loss_required,
        found=loss_found,
        missing=loss_required - loss_found,
    )
    coverage_results.append(loss_status)

    if loss_status.is_complete:
        print(f"  PASS: All {len(loss_required)} loss functions covered")
    else:
        print(f"  INCOMPLETE: {loss_status.coverage_pct:.0f}% coverage")
        print(f"  Missing: {', '.join(loss_status.missing)}")
        if strict:
            all_complete = False

    # Check postprocessing
    print("\n[4/4] Checking Postprocess Policy Coverage...")
    postprocess_found = find_postprocess_runs(outputs_dir)

    for track_name, methods in POSTPROCESS_TRACKS.items():
        track_required = set(methods)
        if track_name.startswith("no_evidence"):
            track_found = postprocess_found.get("no_evidence", set())
        elif track_name == "dynamic_k":
            track_found = postprocess_found.get("dynamic_k", set())
        else:
            track_found = postprocess_found.get(track_name, set())

        track_status = CoverageStatus(
            category=f"Postprocess: {track_name}",
            required=track_required,
            found=track_found,
            missing=track_required - track_found,
        )
        coverage_results.append(track_status)

        if track_status.is_complete:
            print(f"  PASS: {track_name} fully covered")
        else:
            print(f"  INCOMPLETE: {track_name} - {track_status.coverage_pct:.0f}% coverage")
            print(f"    Missing: {', '.join(track_status.missing)}")
            if strict:
                all_complete = False

    # Summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)

    for status in coverage_results:
        pct = status.coverage_pct
        mark = "PASS" if status.is_complete else "INCOMPLETE"
        print(f"  [{mark}] {status.category}: {pct:.0f}%")

    # Save report
    report_path = outputs_dir / "paper_audit" / "coverage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "strict_mode": strict,
        "all_complete": all_complete,
        "categories": [
            {
                "name": s.category,
                "coverage_pct": s.coverage_pct,
                "required": list(s.required),
                "found": list(s.found),
                "missing": list(s.missing),
            }
            for s in coverage_results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    if all_complete:
        print("\nCOVERAGE: COMPLETE")
        return 0
    else:
        print("\nCOVERAGE: INCOMPLETE")
        if strict:
            print("(Strict mode: failing due to incomplete coverage)")
            return 1
        else:
            print("(Non-strict mode: returning success)")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Check model/loss/policy coverage")
    parser.add_argument("--strict", action="store_true", help="Fail if coverage is incomplete")
    parser.add_argument("--outputs_dir", type=Path, default=None, help="Outputs directory")
    args = parser.parse_args()

    outputs_dir = args.outputs_dir or Path(__file__).parent.parent / "outputs"

    if not outputs_dir.exists():
        print(f"ERROR: Outputs directory not found: {outputs_dir}")
        return 1

    return check_coverage(outputs_dir, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
