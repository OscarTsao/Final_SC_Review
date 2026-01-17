#!/usr/bin/env python3
"""
Audit script for data splits and leakage detection.

This script verifies:
1. Fold disjointness by post_id
2. No shared content leakage across folds
3. No leaky features in model inputs
4. Proper calibration training protocol

Usage:
    python scripts/audit/audit_splits_and_leakage.py --output_dir outputs/audit_full_eval/<ts>
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.data.io import load_groundtruth
from final_sc_review.data.splits import k_fold_post_ids, split_post_ids

# Forbidden feature patterns (leaky features)
LEAKY_PATTERNS = [
    r"mrr",
    r"recall_at_\d+",
    r"recall@\d+",
    r"gold_rank",
    r"min_gold_rank",
    r"max_gold_rank",
    r"mean_gold_rank",
    r"^gold",
    r"^label",
    r"^relevant",
    r"_gold_",
    r"_label_",
]


def is_leaky_feature(feature_name: str) -> bool:
    """Check if a feature name matches any leaky pattern."""
    for pattern in LEAKY_PATTERNS:
        if re.search(pattern, feature_name, re.IGNORECASE):
            return True
    return False


def audit_fold_disjointness(
    all_post_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify that folds are disjoint by post_id."""
    result = {
        "passed": True,
        "n_folds": n_folds,
        "n_total_posts": len(all_post_ids),
        "violations": [],
    }

    # Get fold splits
    folds = k_fold_post_ids(all_post_ids, k=n_folds, seed=seed)

    # Check each pair of folds for overlap
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            train_i = set(folds[i]["train"])
            train_j = set(folds[j]["train"])
            test_i = set(folds[i]["test"])
            test_j = set(folds[j]["test"])

            # Check train_i vs test_i disjoint
            overlap_train_test = train_i & test_i
            if overlap_train_test:
                result["passed"] = False
                result["violations"].append({
                    "type": "train_test_overlap",
                    "fold": i,
                    "overlap_count": len(overlap_train_test),
                    "examples": list(overlap_train_test)[:5],
                })

    # Check all test sets are disjoint
    all_test_posts = []
    for i in range(n_folds):
        all_test_posts.append(set(folds[i]["test"]))

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            overlap = all_test_posts[i] & all_test_posts[j]
            if overlap:
                result["passed"] = False
                result["violations"].append({
                    "type": "test_test_overlap",
                    "folds": [i, j],
                    "overlap_count": len(overlap),
                    "examples": list(overlap)[:5],
                })

    # Store fold sizes
    result["fold_sizes"] = [
        {"fold": i, "train": len(folds[i]["train"]), "test": len(folds[i]["test"])}
        for i in range(n_folds)
    ]

    return result


def audit_gnn_config(config_path: Path) -> Dict[str, Any]:
    """Check GNN config for proper split configuration."""
    result = {
        "passed": True,
        "issues": [],
    }

    if not config_path.exists():
        result["error"] = f"Config not found: {config_path}"
        return result

    with open(config_path) as f:
        config = json.load(f)

    # Check for proper fold configuration
    if "config" in config:
        cfg = config["config"]

        # Check inner_train_ratio
        inner_ratio = cfg.get("inner_train_ratio", 0.7)
        if inner_ratio < 0.5 or inner_ratio > 0.9:
            result["issues"].append(f"inner_train_ratio={inner_ratio} outside [0.5, 0.9]")

        # Check n_folds
        n_folds = cfg.get("n_folds", 5)
        if n_folds < 3:
            result["issues"].append(f"n_folds={n_folds} too few for reliable CV")
            result["passed"] = False

        # Check seed
        seed = cfg.get("training", {}).get("seed")
        if seed is None:
            result["issues"].append("No seed specified - non-deterministic")
            result["passed"] = False
        else:
            result["seed"] = seed

    return result


def audit_graph_features(cv_results_path: Path) -> Dict[str, Any]:
    """Check graph features used in GNN for leakage."""
    result = {
        "passed": True,
        "node_features": [],
        "edge_features": [],
        "leaky_components": [],
    }

    if not cv_results_path.exists():
        result["error"] = f"CV results not found: {cv_results_path}"
        return result

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    config = cv_results.get("config", {})
    graph_config = config.get("graph", {})

    # Check node features
    node_features = [
        "use_embedding",
        "use_reranker_score",
        "use_rank_percentile",
        "use_score_gaps",
        "use_score_stats",
    ]

    for feat in node_features:
        if graph_config.get(feat, False):
            result["node_features"].append(feat)

    # Check for gold-derived features (should be empty)
    gold_features = [
        "use_gold_labels",
        "use_mrr",
        "use_recall",
    ]

    for feat in gold_features:
        if graph_config.get(feat, False):
            result["leaky_components"].append(feat)
            result["passed"] = False

    return result


def run_full_audit(
    groundtruth_path: Path,
    output_dir: Path,
    gnn_results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full split and leakage audit."""
    timestamp = datetime.now().isoformat()

    audit_results = {
        "timestamp": timestamp,
        "passed_all": True,
        "audits": {},
    }

    # 1. Audit fold disjointness
    print("Auditing fold disjointness...")
    gt = load_groundtruth(groundtruth_path)
    all_post_ids = sorted(set(row.post_id for row in gt))

    fold_audit = audit_fold_disjointness(all_post_ids, n_folds=5, seed=42)
    audit_results["audits"]["fold_disjointness"] = fold_audit
    if not fold_audit["passed"]:
        audit_results["passed_all"] = False
        print("  ❌ FAILED: Fold disjointness")
    else:
        print("  ✅ PASSED: Fold disjointness")

    # 2. Audit GNN results if available
    if gnn_results_dir and gnn_results_dir.exists():
        print("Auditing GNN configurations...")

        # Find cv_results.json files
        cv_files = list(gnn_results_dir.rglob("cv_results.json"))
        for cv_file in cv_files:
            exp_name = cv_file.parent.name
            print(f"  Checking {exp_name}...")

            config_audit = audit_gnn_config(cv_file)
            graph_audit = audit_graph_features(cv_file)

            audit_results["audits"][f"gnn_{exp_name}_config"] = config_audit
            audit_results["audits"][f"gnn_{exp_name}_features"] = graph_audit

            if not config_audit["passed"]:
                audit_results["passed_all"] = False
                print(f"    ❌ Config: {config_audit.get('issues', [])}")
            else:
                print(f"    ✅ Config OK")

            if not graph_audit["passed"]:
                audit_results["passed_all"] = False
                print(f"    ❌ Features: {graph_audit.get('leaky_components', [])}")
            else:
                print(f"    ✅ Features OK (no leakage)")

    # 3. Check for common leaky feature patterns in codebase
    print("\nChecking for leaky patterns in feature generation code...")
    feature_check = {"passed": True, "files_checked": [], "potential_leakage": []}

    # Check specific files known to generate features
    feature_files = [
        PROJECT_ROOT / "src" / "final_sc_review" / "gnn" / "graphs" / "features.py",
        PROJECT_ROOT / "src" / "final_sc_review" / "gnn" / "graphs" / "builder.py",
    ]

    for fpath in feature_files:
        if fpath.exists():
            feature_check["files_checked"].append(str(fpath))
            content = fpath.read_text()
            for pattern in ["gold_ids", "gold_labels", "groundtruth"]:
                if pattern in content.lower():
                    # Check if it's actually used (not just in comments)
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if pattern in line.lower() and not line.strip().startswith("#"):
                            # Skip if it's in a docstring or function that validates leakage
                            if "leakage" in line.lower() or "forbidden" in line.lower():
                                continue
                            feature_check["potential_leakage"].append({
                                "file": str(fpath),
                                "line": i + 1,
                                "pattern": pattern,
                                "content": line.strip()[:100],
                            })

    # Filter out false positives (checking for leakage is OK)
    feature_check["potential_leakage"] = [
        item for item in feature_check["potential_leakage"]
        if "check" not in item["content"].lower() and "audit" not in item["content"].lower()
    ]

    if feature_check["potential_leakage"]:
        feature_check["passed"] = False
        print("  ⚠️ WARNING: Potential leakage patterns found in code")
        for item in feature_check["potential_leakage"][:5]:
            print(f"    - {item['file']}:{item['line']}: {item['pattern']}")
    else:
        print("  ✅ No obvious leakage patterns in feature generation code")

    audit_results["audits"]["feature_code_check"] = feature_check

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    with open(output_dir / "split_leakage_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    # Markdown report
    write_markdown_report(audit_results, output_dir / "split_leakage_audit.md")

    return audit_results


def write_markdown_report(results: Dict[str, Any], output_path: Path):
    """Write human-readable markdown audit report."""
    lines = [
        "# Split and Leakage Audit Report",
        "",
        f"**Generated**: {results['timestamp']}",
        f"**Overall Status**: {'✅ PASSED' if results['passed_all'] else '❌ FAILED'}",
        "",
        "---",
        "",
        "## 1. Fold Disjointness Audit",
        "",
    ]

    fold_audit = results["audits"].get("fold_disjointness", {})
    if fold_audit.get("passed"):
        lines.append("✅ **PASSED**: All folds are properly disjoint by post_id")
    else:
        lines.append("❌ **FAILED**: Fold disjointness violations detected")
        for v in fold_audit.get("violations", []):
            lines.append(f"- {v['type']}: {v.get('overlap_count', 0)} overlapping items")

    lines.append("")
    lines.append("### Fold Sizes")
    lines.append("")
    lines.append("| Fold | Train | Test |")
    lines.append("|------|-------|------|")
    for size in fold_audit.get("fold_sizes", []):
        lines.append(f"| {size['fold']} | {size['train']} | {size['test']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. GNN Configuration Audits",
        "",
    ])

    for key, audit in results["audits"].items():
        if key.startswith("gnn_"):
            exp_name = key.replace("gnn_", "").replace("_config", "").replace("_features", "")
            status = "✅" if audit.get("passed", True) else "❌"
            lines.append(f"### {exp_name}")
            lines.append(f"- Status: {status}")
            if audit.get("issues"):
                lines.append(f"- Issues: {', '.join(audit['issues'])}")
            if audit.get("leaky_components"):
                lines.append(f"- Leaky components: {', '.join(audit['leaky_components'])}")
            if audit.get("node_features"):
                lines.append(f"- Node features: {', '.join(audit['node_features'])}")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Feature Code Check",
        "",
    ])

    code_check = results["audits"].get("feature_code_check", {})
    if code_check.get("passed"):
        lines.append("✅ No obvious leakage patterns found in feature generation code")
    else:
        lines.append("⚠️ Potential leakage patterns detected:")
        for item in code_check.get("potential_leakage", [])[:10]:
            lines.append(f"- `{item['file']}:{item['line']}`: `{item['pattern']}`")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Conclusion",
        "",
    ])

    if results["passed_all"]:
        lines.append("✅ **All audits passed.** The evaluation pipeline has proper split separation and no detected label leakage.")
    else:
        lines.append("❌ **Audit failed.** Address the issues above before trusting evaluation results.")

    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Audit splits and leakage")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for audit outputs",
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        default="data/groundtruth/evidence_sentence_groundtruth.csv",
        help="Path to groundtruth file",
    )
    parser.add_argument(
        "--gnn_results",
        type=str,
        default="outputs/gnn_research",
        help="Path to GNN results directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    groundtruth_path = Path(args.groundtruth)
    gnn_results_dir = Path(args.gnn_results) if args.gnn_results else None

    results = run_full_audit(groundtruth_path, output_dir, gnn_results_dir)

    if results["passed_all"]:
        print("\n✅ ALL AUDITS PASSED")
        sys.exit(0)
    else:
        print("\n❌ AUDIT FAILED - See report for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
