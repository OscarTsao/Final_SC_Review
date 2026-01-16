#!/usr/bin/env python3
"""Audit feature store for label leakage.

This script:
1. Loads the feature store and categorizes features as DEPLOYABLE vs LEAKY
2. Flags suspicious features with automated heuristics
3. Reports correlation/AUC of each feature against the label
4. Outputs audit reports

Usage:
    python scripts/verification/audit_feature_leakage.py \
        --feature_store outputs/feature_store/<timestamp>/full_features.parquet \
        --output_dir outputs/verification_fix/<timestamp>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Feature Classification Rules
# ============================================================================

# Patterns that indicate LEAKY features (require gold labels)
LEAKY_PATTERNS = [
    "gold",
    "mrr",
    "recall_at",
    "recall@",
    "ndcg",
    "precision_at",
    "precision@",
    "map@",
    "map_at",
    "label",
    "relevant",
    "groundtruth",
    "gt_",
    "is_positive",
    "is_relevant",
    "hit_at",
    "hit@",
]

# Columns that are identifiers/metadata (not features)
IDENTIFIER_COLS = [
    "post_id",
    "query_id",
    "criterion_id",
    "criterion_text",
    "fold_id",
    "candidate_ids",
    "candidate_scores",
    "gold_sentence_ids",
]

# Columns that are labels (target variables)
LABEL_COLS = [
    "has_evidence",
    "n_gold_sentences",
]

# Known DEPLOYABLE features (computable at inference time)
# These are derived from retriever/reranker scores only
DEPLOYABLE_FEATURE_PREFIXES = [
    "max_reranker",
    "second_reranker",
    "mean_reranker",
    "std_reranker",
    "top1_top2_gap",
    "top1_top5_gap",
    "topk_sum",
    "entropy_top",
    "score_range",
    "score_skewness",
    "max_retriever",
    "mean_retriever",
    "retriever_reranker_corr",
    "n_candidates",
    # Calibrated probability features (if available)
    "calibrated_",
    "prob_",
    "softmax_",
    # Score distribution features
    "score_",
    "rank_position",  # candidate rank (not gold rank)
]


def classify_feature(col_name: str) -> str:
    """Classify a feature column as DEPLOYABLE, LEAKY, IDENTIFIER, or LABEL."""
    col_lower = col_name.lower()

    # Check identifiers
    if col_name in IDENTIFIER_COLS:
        return "IDENTIFIER"

    # Check labels
    if col_name in LABEL_COLS:
        return "LABEL"

    # Check for leaky patterns
    for pattern in LEAKY_PATTERNS:
        if pattern in col_lower:
            return "LEAKY"

    # Check for deployable patterns
    for prefix in DEPLOYABLE_FEATURE_PREFIXES:
        if col_lower.startswith(prefix.lower()):
            return "DEPLOYABLE"

    # Default: treat as potentially leaky if unknown
    return "UNKNOWN"


def compute_feature_auc(feature: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC of a single feature against binary labels."""
    try:
        # Handle constant features
        if np.std(feature) == 0:
            return 0.5
        # Handle missing values
        valid_mask = ~np.isnan(feature)
        if valid_mask.sum() < 10:
            return 0.5
        return roc_auc_score(labels[valid_mask], feature[valid_mask])
    except Exception:
        return 0.5


def compute_correlation(feature: np.ndarray, labels: np.ndarray) -> float:
    """Compute Pearson correlation of a feature with labels."""
    try:
        valid_mask = ~np.isnan(feature)
        if valid_mask.sum() < 10:
            return 0.0
        return float(np.corrcoef(feature[valid_mask], labels[valid_mask])[0, 1])
    except Exception:
        return 0.0


def audit_features(
    df: pd.DataFrame,
    label_col: str = "has_evidence",
    model_feature_cols: List[str] = None,
    auc_threshold: float = 0.95,
) -> Dict:
    """Run full feature audit.

    Args:
        df: Feature DataFrame
        label_col: Name of the label column
        model_feature_cols: If provided, list of columns actually used by the model
        auc_threshold: AUC threshold above which to flag as suspiciously high

    Returns:
        Dict with audit results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_columns": len(df.columns),
        "features": {},
        "summary": {},
        "leaky_in_model": [],
    }

    labels = df[label_col].values

    deployable = []
    leaky = []
    identifiers = []
    label_cols = []
    unknown = []
    high_auc_suspicious = []

    for col in df.columns:
        classification = classify_feature(col)

        feature_info = {
            "name": col,
            "classification": classification,
            "dtype": str(df[col].dtype),
        }

        # Compute stats for numeric columns
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            values = df[col].values.astype(float)
            feature_info["mean"] = float(np.nanmean(values))
            feature_info["std"] = float(np.nanstd(values))
            feature_info["min"] = float(np.nanmin(values))
            feature_info["max"] = float(np.nanmax(values))

            # Compute AUC and correlation with label
            auc = compute_feature_auc(values, labels)
            corr = compute_correlation(values, labels)
            feature_info["auc_vs_label"] = auc
            feature_info["corr_vs_label"] = corr

            # Flag suspiciously high AUC
            if auc > auc_threshold and classification not in ["LABEL", "IDENTIFIER"]:
                feature_info["suspicious"] = True
                feature_info["suspicious_reason"] = f"AUC={auc:.4f} > {auc_threshold}"
                high_auc_suspicious.append(col)

        results["features"][col] = feature_info

        # Categorize
        if classification == "DEPLOYABLE":
            deployable.append(col)
        elif classification == "LEAKY":
            leaky.append(col)
        elif classification == "IDENTIFIER":
            identifiers.append(col)
        elif classification == "LABEL":
            label_cols.append(col)
        else:
            unknown.append(col)

    # Check if model uses leaky features
    if model_feature_cols:
        leaky_in_model = [c for c in model_feature_cols if c in leaky]
        results["leaky_in_model"] = leaky_in_model
        results["model_uses_leaky_features"] = len(leaky_in_model) > 0

    # Summary
    results["summary"] = {
        "n_deployable": len(deployable),
        "n_leaky": len(leaky),
        "n_identifiers": len(identifiers),
        "n_labels": len(label_cols),
        "n_unknown": len(unknown),
        "n_high_auc_suspicious": len(high_auc_suspicious),
        "deployable_cols": deployable,
        "leaky_cols": leaky,
        "unknown_cols": unknown,
        "high_auc_suspicious_cols": high_auc_suspicious,
    }

    return results


def generate_audit_report(results: Dict, output_dir: Path) -> str:
    """Generate markdown audit report."""
    lines = [
        "# Feature Leakage Audit Report",
        f"\nGenerated: {results['timestamp']}",
        f"\n## Summary",
        f"\n| Category | Count |",
        f"|----------|-------|",
        f"| Total columns | {results['total_columns']} |",
        f"| DEPLOYABLE | {results['summary']['n_deployable']} |",
        f"| **LEAKY** | **{results['summary']['n_leaky']}** |",
        f"| IDENTIFIER | {results['summary']['n_identifiers']} |",
        f"| LABEL | {results['summary']['n_labels']} |",
        f"| UNKNOWN | {results['summary']['n_unknown']} |",
        f"| High AUC (suspicious) | {results['summary']['n_high_auc_suspicious']} |",
    ]

    # Leaky features section
    if results['summary']['n_leaky'] > 0:
        lines.extend([
            f"\n## ⚠️ LEAKY Features (MUST NOT use in deployment)",
            f"\nThese features require gold labels and are NOT available at inference time:",
            "",
        ])
        for col in results['summary']['leaky_cols']:
            info = results['features'][col]
            auc = info.get('auc_vs_label', 'N/A')
            if isinstance(auc, float):
                auc = f"{auc:.4f}"
            lines.append(f"- **{col}**: AUC={auc}")

    # Check if model uses leaky features
    if results.get('model_uses_leaky_features'):
        lines.extend([
            f"\n## 🚨 CRITICAL: Model Uses Leaky Features!",
            f"\nThe following leaky features are being used by the model:",
            "",
        ])
        for col in results['leaky_in_model']:
            info = results['features'][col]
            auc = info.get('auc_vs_label', 'N/A')
            if isinstance(auc, float):
                auc = f"{auc:.4f}"
            lines.append(f"- **{col}**: AUC={auc}")
        lines.append("\n**This explains the unrealistically high performance!**")

    # High AUC suspicious
    if results['summary']['n_high_auc_suspicious'] > 0:
        lines.extend([
            f"\n## ⚠️ Suspiciously High AUC Features",
            f"\nThese features have AUC > 0.95 and warrant investigation:",
            "",
        ])
        for col in results['summary']['high_auc_suspicious_cols']:
            info = results['features'][col]
            lines.append(f"- **{col}**: AUC={info.get('auc_vs_label', 'N/A'):.4f}, classification={info['classification']}")

    # Deployable features
    lines.extend([
        f"\n## ✅ DEPLOYABLE Features",
        f"\nThese features are safe to use at deployment (derived from inference-time signals):",
        "",
    ])
    for col in results['summary']['deployable_cols']:
        info = results['features'][col]
        auc = info.get('auc_vs_label', 'N/A')
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        lines.append(f"- {col}: AUC={auc}")

    # Feature provenance
    lines.extend([
        f"\n## Feature Provenance",
        f"\n### DEPLOYABLE features derive from:",
        f"- `retriever_score`: Dense/sparse/ColBERT scores from BGE-M3",
        f"- `reranker_score`: Logit/probability from reranker (Jina-v3)",
        f"- `candidate_rank`: Position in ranked list (1-indexed)",
        f"- `n_candidates`: Count of candidates for this query",
        f"- Derived statistics: max, mean, std, gaps, entropy of above",
        f"\n### LEAKY features derive from:",
        f"- `gold_sentence_ids`: Ground truth evidence sentence IDs",
        f"- `groundtruth`: Binary relevance labels",
        f"- Evaluation metrics: MRR, Recall@K, nDCG, gold_rank, etc.",
    ])

    # Recommendations
    lines.extend([
        f"\n## Recommendations",
        f"\n1. **Remove all LEAKY features** from model training",
        f"2. **Implement feature provenance checks** in training pipeline",
        f"3. **Re-run evaluation** with only DEPLOYABLE features",
        f"4. **Add unit tests** to prevent leakage regression",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit feature store for label leakage")
    parser.add_argument("--feature_store", type=str, required=True,
                        help="Path to feature store parquet file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for audit reports")
    parser.add_argument("--label_col", type=str, default="has_evidence",
                        help="Name of label column")
    parser.add_argument("--auc_threshold", type=float, default=0.95,
                        help="AUC threshold for suspicious feature flagging")
    parser.add_argument("--assert_no_leaky", action="store_true",
                        help="Fail with exit code 1 if leaky features found in model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature store
    logger.info(f"Loading feature store from {args.feature_store}")
    df = pd.read_parquet(args.feature_store)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Determine model feature columns (all numeric except excluded)
    exclude_cols = set(IDENTIFIER_COLS + LABEL_COLS)
    model_feature_cols = [c for c in df.columns
                         if c not in exclude_cols
                         and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    logger.info(f"Model feature columns: {len(model_feature_cols)}")

    # Run audit
    logger.info("Running feature audit...")
    results = audit_features(
        df,
        label_col=args.label_col,
        model_feature_cols=model_feature_cols,
        auc_threshold=args.auc_threshold,
    )

    # Generate outputs
    logger.info("Generating outputs...")

    # Save full results JSON
    with open(output_dir / "leakage_audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save report
    report = generate_audit_report(results, output_dir)
    with open(output_dir / "leakage_audit_report.md", "w") as f:
        f.write(report)

    # Save deployable features list
    deployable_df = pd.DataFrame([
        {"feature": col, **results["features"][col]}
        for col in results["summary"]["deployable_cols"]
    ])
    deployable_df.to_csv(output_dir / "deployable_features.csv", index=False)

    # Save leaky features list
    leaky_df = pd.DataFrame([
        {"feature": col, **results["features"][col]}
        for col in results["summary"]["leaky_cols"]
    ])
    leaky_df.to_csv(output_dir / "leaky_features.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE LEAKAGE AUDIT COMPLETE")
    print("=" * 60)
    print(f"\nTotal columns: {results['total_columns']}")
    print(f"DEPLOYABLE: {results['summary']['n_deployable']}")
    print(f"LEAKY: {results['summary']['n_leaky']}")
    print(f"UNKNOWN: {results['summary']['n_unknown']}")

    if results['summary']['n_leaky'] > 0:
        print(f"\n⚠️ LEAKY FEATURES FOUND:")
        for col in results['summary']['leaky_cols']:
            info = results['features'][col]
            auc = info.get('auc_vs_label', 'N/A')
            if isinstance(auc, float):
                print(f"  - {col}: AUC={auc:.4f}")
            else:
                print(f"  - {col}: AUC={auc}")

    if results.get('model_uses_leaky_features'):
        print(f"\n🚨 CRITICAL: Model uses {len(results['leaky_in_model'])} leaky features!")
        if args.assert_no_leaky:
            print("Exiting with error (--assert_no_leaky)")
            sys.exit(1)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - leakage_audit_report.md")
    print(f"  - leakage_audit_results.json")
    print(f"  - deployable_features.csv")
    print(f"  - leaky_features.csv")


if __name__ == "__main__":
    main()
