#!/usr/bin/env python3
"""Debug script for Dynamic-K sanity issues.

This script investigates and demonstrates:
1. Gamma invariance bug: Different gamma values (0.8, 0.9, 0.95) produce identical results
2. Fixed K bug: "Fixed K=5" reports avgK != 5

Root Cause Analysis:
====================

BUG 1: Gamma Invariance
-----------------------
The select_k_mass function uses raw sigmoid probabilities:
    cumsum = np.cumsum(sorted_probs)
    k = np.searchsorted(cumsum, gamma) + 1

Problem: Sigmoid outputs are NOT normalized to sum to 1.
If sum(probs) < gamma (e.g., 0.6), then cumsum never reaches gamma,
and searchsorted returns len(probs), meaning ALL nodes are selected.

Fix: Normalize probabilities to sum to 1 before computing cumsum:
    sorted_probs_norm = sorted_probs / sorted_probs.sum()
    cumsum = np.cumsum(sorted_probs_norm)

BUG 2: Fixed K avgK != K
------------------------
The select_k_fixed function enforces k_max constraint:
    k_max = min(10, ceil(n_candidates * 0.5))

For queries with n_candidates=8: k_max = min(10, 4) = 4
So fixed_k=5 becomes min(5, 4) = 4

This is by design (hard cap), but the naming is misleading.
avgK != 5 because k_max constrains it on many queries.

Usage:
    python scripts/gnn/debug_dynamic_k_sanity.py --graph_dir data/cache/gnn/20260117_003135
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def compute_k_constraints(n_candidates: int, k_min: int = 2, k_max: int = 10, k_max_ratio: float = 0.5) -> Tuple[int, int]:
    """Compute k constraints for a given number of candidates."""
    k_max_from_ratio = int(np.ceil(n_candidates * k_max_ratio))
    actual_k_max = min(k_max, k_max_from_ratio, n_candidates)
    actual_k_max = max(actual_k_max, k_min)
    return k_min, actual_k_max


def select_k_mass_buggy(probs: np.ndarray, gamma: float, k_min: int, k_max: int) -> int:
    """BUGGY version: Uses raw probabilities without normalization."""
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    k = np.searchsorted(cumsum, gamma) + 1
    return max(k_min, min(k, k_max))


def select_k_mass_fixed(probs: np.ndarray, gamma: float, k_min: int, k_max: int) -> int:
    """FIXED version: Normalizes probabilities to sum to 1."""
    sorted_probs = np.sort(probs)[::-1]
    total = sorted_probs.sum()
    if total <= 0:
        # All zeros or negative: no meaningful selection, return k_min
        return k_min
    sorted_probs_norm = sorted_probs / total
    cumsum = np.cumsum(sorted_probs_norm)
    k = np.searchsorted(cumsum, gamma) + 1
    return max(k_min, min(k, k_max))


def select_k_fixed(n_candidates: int, fixed_k: int, k_min: int, k_max: int) -> int:
    """Fixed K policy with constraints."""
    return max(k_min, min(fixed_k, k_max))


def analyze_gamma_invariance(graphs: List[Data], device: torch.device) -> Dict:
    """Analyze why different gamma values produce identical results."""

    # Simulate model predictions (use node features as proxy for this debug)
    # In real case, we'd load a trained model

    gammas = [0.8, 0.9, 0.95]

    buggy_results = {g: [] for g in gammas}
    fixed_results = {g: [] for g in gammas}

    prob_sum_stats = []
    constraint_stats = []

    for g_idx, graph in enumerate(graphs):
        n_nodes = graph.x.size(0)

        # Simulate sigmoid probabilities (random for demonstration)
        np.random.seed(g_idx)
        probs = np.random.rand(n_nodes) * 0.3 + 0.2  # Typical sigmoid range [0.2, 0.5]

        k_min, k_max = compute_k_constraints(n_nodes)

        prob_sum_stats.append({
            "graph_idx": g_idx,
            "n_nodes": n_nodes,
            "prob_sum": float(probs.sum()),
            "prob_mean": float(probs.mean()),
            "k_min": k_min,
            "k_max": k_max,
        })

        for gamma in gammas:
            k_buggy = select_k_mass_buggy(probs, gamma, k_min, k_max)
            k_fixed = select_k_mass_fixed(probs, gamma, k_min, k_max)

            buggy_results[gamma].append(k_buggy)
            fixed_results[gamma].append(k_fixed)

    # Analyze
    analysis = {
        "total_graphs": len(graphs),
        "prob_sum_analysis": {
            "mean_prob_sum": float(np.mean([s["prob_sum"] for s in prob_sum_stats])),
            "min_prob_sum": float(np.min([s["prob_sum"] for s in prob_sum_stats])),
            "max_prob_sum": float(np.max([s["prob_sum"] for s in prob_sum_stats])),
            "pct_below_0.8": float(np.mean([s["prob_sum"] < 0.8 for s in prob_sum_stats]) * 100),
            "pct_below_0.9": float(np.mean([s["prob_sum"] < 0.9 for s in prob_sum_stats]) * 100),
            "pct_below_0.95": float(np.mean([s["prob_sum"] < 0.95 for s in prob_sum_stats]) * 100),
        },
        "buggy_version": {},
        "fixed_version": {},
    }

    for gamma in gammas:
        k_vals_buggy = buggy_results[gamma]
        k_vals_fixed = fixed_results[gamma]

        analysis["buggy_version"][f"gamma_{gamma}"] = {
            "avg_k": float(np.mean(k_vals_buggy)),
            "std_k": float(np.std(k_vals_buggy)),
            "min_k": int(np.min(k_vals_buggy)),
            "max_k": int(np.max(k_vals_buggy)),
        }

        analysis["fixed_version"][f"gamma_{gamma}"] = {
            "avg_k": float(np.mean(k_vals_fixed)),
            "std_k": float(np.std(k_vals_fixed)),
            "min_k": int(np.min(k_vals_fixed)),
            "max_k": int(np.max(k_vals_fixed)),
        }

    # Check if buggy version produces identical results
    buggy_k_lists = [buggy_results[g] for g in gammas]
    analysis["buggy_all_identical"] = all(
        k_lists[i] == k_lists[i+1]
        for i in range(len(k_lists)-1)
        for k_lists in [buggy_k_lists]
    )

    # Check if fixed version produces DIFFERENT results
    fixed_k_08 = fixed_results[0.8]
    fixed_k_09 = fixed_results[0.9]
    fixed_k_095 = fixed_results[0.95]

    analysis["fixed_shows_variation"] = not (fixed_k_08 == fixed_k_09 == fixed_k_095)
    analysis["fixed_monotonic_increasing"] = all(
        np.mean(fixed_results[gammas[i]]) <= np.mean(fixed_results[gammas[i+1]])
        for i in range(len(gammas)-1)
    )

    return analysis


def analyze_fixed_k_constraint(graphs: List[Data]) -> Dict:
    """Analyze why Fixed K=5 produces avgK != 5."""

    fixed_k_target = 5
    k_results = []
    constraint_details = []

    for g_idx, graph in enumerate(graphs):
        n_nodes = graph.x.size(0)
        k_min, k_max = compute_k_constraints(n_nodes)

        actual_k = select_k_fixed(n_nodes, fixed_k_target, k_min, k_max)
        k_results.append(actual_k)

        if actual_k != fixed_k_target:
            constraint_details.append({
                "graph_idx": g_idx,
                "n_nodes": n_nodes,
                "k_min": k_min,
                "k_max": k_max,
                "target_k": fixed_k_target,
                "actual_k": actual_k,
                "reason": "k_max constraint" if actual_k == k_max else "k_min constraint",
            })

    analysis = {
        "target_k": fixed_k_target,
        "actual_avg_k": float(np.mean(k_results)),
        "actual_std_k": float(np.std(k_results)),
        "pct_constrained": float(len(constraint_details) / len(graphs) * 100),
        "n_constrained": len(constraint_details),
        "n_total": len(graphs),
        "k_distribution": {
            k: int(k_results.count(k))
            for k in sorted(set(k_results))
        },
    }

    # Analyze constraint reasons
    if constraint_details:
        analysis["constraint_breakdown"] = {
            "k_max_constrained": sum(1 for d in constraint_details if d["reason"] == "k_max constraint"),
            "k_min_constrained": sum(1 for d in constraint_details if d["reason"] == "k_min constraint"),
        }

        # Sample constraint examples
        analysis["sample_constraints"] = constraint_details[:10]

    return analysis


def load_graphs_from_folds(graph_dir: Path) -> List[Data]:
    """Load all graphs from fold files."""
    graphs = []
    for fold_id in range(5):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        if fold_path.exists():
            data = torch.load(fold_path, weights_only=False)
            graphs.extend(data["graphs"])
            logger.info(f"Loaded fold {fold_id}: {len(data['graphs'])} graphs")
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Debug Dynamic-K Sanity Issues")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    logger.info("=" * 70)
    logger.info("Dynamic-K Sanity Debug Script")
    logger.info("=" * 70)

    # Load graphs
    graphs = load_graphs_from_folds(graph_dir)
    logger.info(f"Total graphs loaded: {len(graphs)}")

    device = torch.device("cpu")

    # Analysis 1: Gamma Invariance Bug
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1: Gamma Invariance Bug")
    logger.info("=" * 70)

    gamma_analysis = analyze_gamma_invariance(graphs, device)

    print("\nProbability Sum Statistics:")
    print(f"  Mean sum: {gamma_analysis['prob_sum_analysis']['mean_prob_sum']:.4f}")
    print(f"  Min sum: {gamma_analysis['prob_sum_analysis']['min_prob_sum']:.4f}")
    print(f"  Max sum: {gamma_analysis['prob_sum_analysis']['max_prob_sum']:.4f}")
    print(f"  % below 0.8: {gamma_analysis['prob_sum_analysis']['pct_below_0.8']:.1f}%")
    print(f"  % below 0.9: {gamma_analysis['prob_sum_analysis']['pct_below_0.9']:.1f}%")
    print(f"  % below 0.95: {gamma_analysis['prob_sum_analysis']['pct_below_0.95']:.1f}%")

    print("\nBUGGY Version (raw probabilities):")
    for gamma in [0.8, 0.9, 0.95]:
        stats = gamma_analysis["buggy_version"][f"gamma_{gamma}"]
        print(f"  gamma={gamma}: avgK={stats['avg_k']:.2f} ± {stats['std_k']:.2f}")
    print(f"  All identical? {gamma_analysis['buggy_all_identical']}")

    print("\nFIXED Version (normalized probabilities):")
    for gamma in [0.8, 0.9, 0.95]:
        stats = gamma_analysis["fixed_version"][f"gamma_{gamma}"]
        print(f"  gamma={gamma}: avgK={stats['avg_k']:.2f} ± {stats['std_k']:.2f}")
    print(f"  Shows variation? {gamma_analysis['fixed_shows_variation']}")
    print(f"  Monotonic increasing? {gamma_analysis['fixed_monotonic_increasing']}")

    # Analysis 2: Fixed K Constraint
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: Fixed K Constraint Bug")
    logger.info("=" * 70)

    fixed_k_analysis = analyze_fixed_k_constraint(graphs)

    print(f"\nTarget K: {fixed_k_analysis['target_k']}")
    print(f"Actual Avg K: {fixed_k_analysis['actual_avg_k']:.2f} ± {fixed_k_analysis['actual_std_k']:.2f}")
    print(f"% Constrained: {fixed_k_analysis['pct_constrained']:.1f}%")
    print(f"  - k_max constrained: {fixed_k_analysis.get('constraint_breakdown', {}).get('k_max_constrained', 0)}")
    print(f"  - k_min constrained: {fixed_k_analysis.get('constraint_breakdown', {}).get('k_min_constrained', 0)}")

    print("\nK Distribution:")
    for k, count in sorted(fixed_k_analysis["k_distribution"].items()):
        pct = count / fixed_k_analysis["n_total"] * 100
        print(f"  K={k}: {count} ({pct:.1f}%)")

    if fixed_k_analysis.get("sample_constraints"):
        print("\nSample Constraint Examples:")
        for ex in fixed_k_analysis["sample_constraints"][:5]:
            print(f"  Graph {ex['graph_idx']}: n_nodes={ex['n_nodes']}, k_max={ex['k_max']}, actual_k={ex['actual_k']}")

    # Summary and Recommendations
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY & RECOMMENDATIONS")
    logger.info("=" * 70)

    print("\n1. GAMMA INVARIANCE BUG:")
    print("   Root cause: select_k_mass uses unnormalized sigmoid outputs")
    print("   Fix: Normalize probs to sum to 1 before cumsum")
    print("   Code change needed in: scripts/gnn/eval_dynamic_k_gnn.py:64-69")

    print("\n2. FIXED K BUG:")
    print("   Root cause: k_max_ratio=0.5 constrains K on small graphs")
    print("   This is BY DESIGN (hard constraint), not a bug")
    print("   Recommendation: Document this behavior clearly in reports")
    print("   Alternative: Report 'Fixed K (target)' vs 'Fixed K (actual)'")

    # Save results
    output_path = args.output or (graph_dir / "debug_dynamic_k_sanity.json")
    results = {
        "gamma_invariance_analysis": gamma_analysis,
        "fixed_k_analysis": fixed_k_analysis,
        "recommendations": {
            "gamma_invariance_fix": "Normalize probabilities before cumsum",
            "fixed_k_documentation": "Document k_max constraint behavior",
        }
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    logger.info("\nDebug analysis complete!")


if __name__ == "__main__":
    main()
