#!/usr/bin/env python3
"""Decision engine for research pipeline gates.

This script evaluates decision gates to determine whether to trigger
conditional phases (e.g., retriever finetuning, GNN, LLM judge).

Gates:
- G1: Retriever Finetuning Decision
- G2: GNN Enhancement Decision
- G3: External API Decision

Usage:
    python scripts/decision_engine.py --gate G1
    python scripts/decision_engine.py --gate all --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GateResult:
    """Result of evaluating a decision gate."""
    gate_name: str
    triggered: bool
    reason: str
    metrics: Dict[str, Any]
    recommendation: str


def load_best_metrics(outputs_dir: Path, pattern: str) -> Optional[Dict[str, float]]:
    """Load metrics from the best run matching pattern."""
    best_metrics = None
    best_ndcg = -1.0

    for path in outputs_dir.glob(pattern):
        if path.is_dir():
            summary_path = path / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    metrics = summary.get("metrics", summary)
                    ndcg = metrics.get("ndcg@5", metrics.get("ndcg_at_5", 0.0))
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_metrics = metrics
                except Exception:
                    pass

    return best_metrics


def load_oracle_ceiling(outputs_dir: Path) -> Optional[float]:
    """Load oracle retriever ceiling from evaluation."""
    oracle_path = outputs_dir / "oracle_ceiling.json"
    if oracle_path.exists():
        try:
            with open(oracle_path) as f:
                data = json.load(f)
            return data.get("ceiling_ndcg", data.get("oracle_ndcg"))
        except Exception:
            pass

    # Try to find in baseline outputs
    for pattern in ["baseline_*", "retriever_hpo/*"]:
        for path in outputs_dir.glob(pattern):
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    if "oracle_ceiling" in manifest:
                        return manifest["oracle_ceiling"]
                except Exception:
                    pass

    return None


def evaluate_G1(outputs_dir: Path, verbose: bool = False) -> GateResult:
    """G1: Retriever Finetuning Decision.

    TRIGGER if:
    - (val_ndcg_with_finetuned_reranker - val_ndcg_with_frozen_reranker) < 0.03
    - AND oracle_retriever_ceiling - current_val_ndcg > 0.05
    """
    gate_name = "G1: Retriever Finetuning"

    # Load metrics
    frozen_metrics = load_best_metrics(outputs_dir, "baseline_*")
    finetuned_metrics = load_best_metrics(outputs_dir, "reranker_*")
    oracle_ceiling = load_oracle_ceiling(outputs_dir)

    metrics = {
        "frozen_reranker_ndcg": frozen_metrics.get("ndcg@5") if frozen_metrics else None,
        "finetuned_reranker_ndcg": finetuned_metrics.get("ndcg@5") if finetuned_metrics else None,
        "oracle_ceiling": oracle_ceiling,
    }

    if verbose:
        print(f"  Frozen reranker nDCG@5: {metrics['frozen_reranker_ndcg']}")
        print(f"  Finetuned reranker nDCG@5: {metrics['finetuned_reranker_ndcg']}")
        print(f"  Oracle ceiling: {metrics['oracle_ceiling']}")

    # Check if we have enough data
    if frozen_metrics is None or finetuned_metrics is None:
        return GateResult(
            gate_name=gate_name,
            triggered=False,
            reason="Insufficient data: need frozen and finetuned reranker metrics",
            metrics=metrics,
            recommendation="Run baseline and reranker training phases first",
        )

    frozen_ndcg = metrics["frozen_reranker_ndcg"] or 0.0
    finetuned_ndcg = metrics["finetuned_reranker_ndcg"] or 0.0
    improvement = finetuned_ndcg - frozen_ndcg

    if verbose:
        print(f"  Reranker finetuning improvement: {improvement:.4f}")

    # Condition 1: Reranker improvement < 3%
    condition1 = improvement < 0.03

    # Condition 2: Oracle headroom > 5%
    if oracle_ceiling is not None:
        headroom = oracle_ceiling - finetuned_ndcg
        condition2 = headroom > 0.05
        metrics["oracle_headroom"] = headroom
        if verbose:
            print(f"  Oracle headroom: {headroom:.4f}")
    else:
        condition2 = False
        metrics["oracle_headroom"] = None

    triggered = condition1 and condition2

    if triggered:
        reason = f"Reranker improvement ({improvement:.3f}) < 0.03 AND oracle headroom ({metrics.get('oracle_headroom', 0):.3f}) > 0.05"
        recommendation = "Proceed with retriever finetuning (Phase 8)"
    elif not condition1:
        reason = f"Reranker finetuning improvement ({improvement:.3f}) >= 0.03 - sufficient"
        recommendation = "Skip retriever finetuning - reranker improvement is sufficient"
    else:
        reason = f"Oracle headroom ({metrics.get('oracle_headroom', 0):.3f}) <= 0.05 - limited upside"
        recommendation = "Skip retriever finetuning - limited improvement potential"

    return GateResult(
        gate_name=gate_name,
        triggered=triggered,
        reason=reason,
        metrics=metrics,
        recommendation=recommendation,
    )


def evaluate_G2(outputs_dir: Path, verbose: bool = False) -> GateResult:
    """G2: GNN Enhancement Decision.

    SUGGEST if:
    - Reranker training showed diminishing returns
    - AND graph structure is meaningful (avg sentences per post > 10)
    """
    gate_name = "G2: GNN Enhancement"

    # Load training history
    training_history_path = outputs_dir / "reranker_hybrid" / "training_history.json"
    diminishing_returns = False

    if training_history_path.exists():
        try:
            with open(training_history_path) as f:
                history = json.load(f)
            # Check if later epochs show < 1% improvement
            if len(history) >= 3:
                early_improvement = history[1].get("val_ndcg", 0) - history[0].get("val_ndcg", 0)
                late_improvement = history[-1].get("val_ndcg", 0) - history[-2].get("val_ndcg", 0)
                diminishing_returns = late_improvement < 0.01 and early_improvement > late_improvement * 2
        except Exception:
            pass

    # Check sentence stats
    data_dir = Path(__file__).parent.parent / "data"
    corpus_path = data_dir / "groundtruth" / "sentence_corpus.jsonl"
    avg_sentences_per_post = 0.0

    if corpus_path.exists():
        try:
            post_counts = {}
            with open(corpus_path) as f:
                for line in f:
                    record = json.loads(line)
                    post_id = record.get("post_id")
                    if post_id:
                        post_counts[post_id] = post_counts.get(post_id, 0) + 1
            if post_counts:
                avg_sentences_per_post = sum(post_counts.values()) / len(post_counts)
        except Exception:
            pass

    metrics = {
        "diminishing_returns": diminishing_returns,
        "avg_sentences_per_post": avg_sentences_per_post,
    }

    if verbose:
        print(f"  Diminishing returns detected: {diminishing_returns}")
        print(f"  Avg sentences per post: {avg_sentences_per_post:.1f}")

    meaningful_graph = avg_sentences_per_post > 10
    triggered = diminishing_returns and meaningful_graph

    if triggered:
        reason = "Reranker training shows diminishing returns AND graph structure is meaningful"
        recommendation = "Consider GNN enhancement (Phase 9)"
    elif not diminishing_returns:
        reason = "Reranker training still showing good improvements"
        recommendation = "Skip GNN - continue optimizing reranker"
    else:
        reason = f"Avg sentences per post ({avg_sentences_per_post:.1f}) <= 10 - graph may not help"
        recommendation = "Skip GNN - graph structure too sparse"

    return GateResult(
        gate_name=gate_name,
        triggered=triggered,
        reason=reason,
        metrics=metrics,
        recommendation=recommendation,
    )


def evaluate_G3(outputs_dir: Path, allow_external_api: bool, verbose: bool = False) -> GateResult:
    """G3: External API Decision.

    ENABLE if --allow_external_api flag set.
    """
    gate_name = "G3: External API"

    metrics = {
        "allow_external_api": allow_external_api,
    }

    if verbose:
        print(f"  External API allowed: {allow_external_api}")

    triggered = allow_external_api

    if triggered:
        reason = "External API access enabled via flag"
        recommendation = "Enable LLM Judge (Phase 10) and teacher distillation"
    else:
        reason = "External API access not enabled"
        recommendation = "Use local-only training and evaluation"

    return GateResult(
        gate_name=gate_name,
        triggered=triggered,
        reason=reason,
        metrics=metrics,
        recommendation=recommendation,
    )


def evaluate_all_gates(outputs_dir: Path, allow_external_api: bool = False, verbose: bool = False) -> List[GateResult]:
    """Evaluate all decision gates."""
    results = []

    print("\n[G1] Evaluating Retriever Finetuning Decision...")
    results.append(evaluate_G1(outputs_dir, verbose))

    print("\n[G2] Evaluating GNN Enhancement Decision...")
    results.append(evaluate_G2(outputs_dir, verbose))

    print("\n[G3] Evaluating External API Decision...")
    results.append(evaluate_G3(outputs_dir, allow_external_api, verbose))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate decision gates for research pipeline")
    parser.add_argument("--gate", choices=["G1", "G2", "G3", "all"], default="all", help="Gate to evaluate")
    parser.add_argument("--allow_external_api", action="store_true", help="Allow external API access")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--outputs_dir", type=Path, default=None, help="Outputs directory")
    args = parser.parse_args()

    outputs_dir = args.outputs_dir or Path(__file__).parent.parent / "outputs"

    print("=" * 70)
    print("DECISION ENGINE")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 70)

    if args.gate == "all":
        results = evaluate_all_gates(outputs_dir, args.allow_external_api, args.verbose)
    elif args.gate == "G1":
        results = [evaluate_G1(outputs_dir, args.verbose)]
    elif args.gate == "G2":
        results = [evaluate_G2(outputs_dir, args.verbose)]
    elif args.gate == "G3":
        results = [evaluate_G3(outputs_dir, args.allow_external_api, args.verbose)]
    else:
        print(f"Unknown gate: {args.gate}")
        return 1

    # Print results
    print("\n" + "=" * 70)
    print("GATE DECISIONS")
    print("=" * 70)

    for result in results:
        status = "TRIGGERED" if result.triggered else "NOT TRIGGERED"
        print(f"\n[{status}] {result.gate_name}")
        print(f"  Reason: {result.reason}")
        print(f"  Recommendation: {result.recommendation}")

    # Save decision report
    report_path = outputs_dir / "decision" / "gate_decisions.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "decisions": [
            {
                "gate": r.gate_name,
                "triggered": r.triggered,
                "reason": r.reason,
                "metrics": r.metrics,
                "recommendation": r.recommendation,
            }
            for r in results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDecision report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
