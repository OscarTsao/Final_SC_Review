#!/usr/bin/env python3
"""Verify phase completion per R1 (auditable runs).

Each phase must produce required artifacts with checksums.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def verify_phase_1():
    """Verify pre-flight audit complete."""
    required = [
        "outputs/paper_audit/git_state.txt",
        "outputs/system/hw.json",
        "outputs/system/nvidia_smi.txt",
        "data/splits/split_metadata.json",
        "data/splits/dev_tune_post_ids.json",
        "data/splits/dev_select_post_ids.json",
        "data/splits/test_post_ids.json",
        "configs/model_lists.yaml",
    ]

    missing = []
    for path in required:
        if not Path(path).exists():
            missing.append(path)

    if missing:
        print(f"PHASE 1 FAILED - Missing files: {missing}")
        return False

    print("PHASE 1 VERIFIED - Pre-flight complete")
    return True


def verify_phase_4():
    """Verify retriever HPO complete."""
    required = [
        "outputs/retriever_hpo/best_retriever.json",
        "outputs/retriever_hpo/top3_retrievers.json",
        "outputs/retriever_hpo/hpo_results.json",
    ]

    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print(f"PHASE 4 FAILED - Missing files: {missing}")
        return False

    # Check GPU time
    gpu_time = Path("outputs/system/gpu_time_total.json")
    if gpu_time.exists():
        with open(gpu_time) as f:
            data = json.load(f)
        print(f"  GPU hours so far: {data['total_gpu_hours']:.2f}h")

    print("PHASE 4 VERIFIED - Retriever HPO complete")
    return True


def verify_phase_5():
    """Verify reranker HPO complete."""
    required = [
        "outputs/reranker_hpo/best_reranker.json",
        "outputs/reranker_hpo/hpo_results.json",
    ]

    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print(f"PHASE 5 FAILED - Missing files: {missing}")
        return False

    print("PHASE 5 VERIFIED - Reranker HPO complete")
    return True


def verify_phase_6():
    """Verify interaction check complete."""
    required = [
        "outputs/interaction_check/interaction_results.json",
        "outputs/interaction_check/best_pair.json",
    ]

    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print(f"PHASE 6 FAILED - Missing files: {missing}")
        return False

    print("PHASE 6 VERIFIED - Interaction check complete")
    return True


def verify_gpu_time(min_hours: float = 12.0):
    """Verify GPU time requirement met."""
    gpu_time = Path("outputs/system/gpu_time_total.json")
    if not gpu_time.exists():
        print(f"GPU TIME CHECK FAILED - No tracking file")
        return False

    with open(gpu_time) as f:
        data = json.load(f)

    hours = data["total_gpu_hours"]
    if hours < min_hours:
        print(f"GPU TIME CHECK FAILED - {hours:.2f}h < {min_hours}h required")
        return False

    print(f"GPU TIME VERIFIED - {hours:.2f}h >= {min_hours}h")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=int, help="Phase number to verify")
    args = parser.parse_args()

    verifiers = {
        1: verify_phase_1,
        4: verify_phase_4,
        5: verify_phase_5,
        6: verify_phase_6,
        9: lambda: verify_gpu_time(12.0),
    }

    if args.phase in verifiers:
        success = verifiers[args.phase]()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown phase: {args.phase}")
        sys.exit(1)
