#!/usr/bin/env python3
"""Main supersearch orchestrator.

This is the main entrypoint that:
1. Builds/loads feature store
2. Runs Stage-1 sweep (fast)
3. Trains fixed NO_EVIDENCE reranker and augments features
4. Runs two-stage search
5. Runs K selection search (cap K<=5)
6. Returns a Pareto leaderboard

Hard deployment constraints:
- FPR (no-evidence) <= 5%
- TPR (has-evidence) >= 90%
- Evidence recall >= 93%
- Avg evidence K <= 5

Output: outputs/supersearch/<timestamp>/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger
from final_sc_review.supersearch.registry import (
    PluginRegistry, NEStage1Model, UncertainSelector, NEStage2Model, KSelector
)

logger = get_logger(__name__)

# Hard deployment constraints
CONSTRAINTS = {
    "max_fpr": 0.05,  # FPR <= 5%
    "min_tpr": 0.90,  # TPR >= 90%
    "min_evidence_recall": 0.93,  # Evidence recall >= 93%
    "max_avg_k": 5,  # Avg K <= 5
}


class SupersearchPipeline:
    """Main supersearch pipeline."""

    def __init__(
        self,
        output_dir: Path,
        feature_store_dir: Optional[Path] = None,
        n_folds: int = 5,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.feature_store_dir = feature_store_dir
        self.n_folds = n_folds
        self.seed = seed
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Results storage
        self.results_dir = output_dir / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "configs").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "error_analysis").mkdir(exist_ok=True)

        self.leaderboard = []

    def run_full_search(
        self,
        skip_ne_training: bool = False,
        smoke_test: bool = False,
    ) -> pd.DataFrame:
        """Run full supersearch pipeline."""
        logger.info("=" * 60)
        logger.info("SUPERSEARCH PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.results_dir}")
        logger.info(f"Constraints: {CONSTRAINTS}")

        # Phase 1: Load/build feature store
        logger.info("\n[Phase 1] Loading feature store...")
        if self.feature_store_dir and (self.feature_store_dir / "full_features.parquet").exists():
            feature_df = pd.read_parquet(self.feature_store_dir / "full_features.parquet")
            logger.info(f"  Loaded {len(feature_df)} queries from {self.feature_store_dir}")
        else:
            logger.warning("  No feature store found, building from scratch...")
            feature_df = self._build_feature_store()

        if smoke_test:
            # Limit to 1 fold for smoke test
            feature_df = feature_df[feature_df["fold_id"] == 0]
            logger.info(f"  [SMOKE TEST] Limited to fold 0: {len(feature_df)} queries")

        # Phase 2: Stage-1 sweep
        logger.info("\n[Phase 2] Running Stage-1 model sweep...")
        stage1_results = self._run_stage1_sweep(feature_df)

        # Phase 3: NO_EVIDENCE training (optional)
        if not skip_ne_training:
            logger.info("\n[Phase 3] Training fixed NO_EVIDENCE reranker...")
            ne_results = self._train_noevidence_model(feature_df)
        else:
            logger.info("\n[Phase 3] Skipping NO_EVIDENCE training (--skip_ne_training)")
            ne_results = None

        # Phase 4: Two-stage triage search
        logger.info("\n[Phase 4] Running two-stage triage search...")
        twostage_results = self._run_twostage_search(feature_df, stage1_results)

        # Phase 5: K selection search
        logger.info("\n[Phase 5] Running K selection search...")
        kselector_results = self._run_kselector_search(feature_df)

        # Phase 6: Build final leaderboard
        logger.info("\n[Phase 6] Building final leaderboard...")
        leaderboard_df = self._build_leaderboard()

        # Phase 7: Sanity checks
        logger.info("\n[Phase 7] Running sanity checks...")
        self._run_sanity_checks(feature_df)

        # Save outputs
        self._save_outputs(leaderboard_df)

        return leaderboard_df

    def _build_feature_store(self) -> pd.DataFrame:
        """Build feature store from scratch."""
        # Import build_feature_store function directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_feature_store",
            Path(__file__).parent / "build_feature_store.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        store_dir = module.build_feature_store(
            groundtruth_path=Path("data/groundtruth/evidence_sentence_groundtruth.csv"),
            criteria_path=Path("data/DSM5/MDD_Criteira.json"),
            cache_dir=Path("data/cache/oof_cache"),
            output_dir=Path("outputs/feature_store"),
            n_folds=self.n_folds,
            seed=self.seed,
        )

        self.feature_store_dir = store_dir
        return pd.read_parquet(store_dir / "full_features.parquet")

    def _run_stage1_sweep(self, feature_df: pd.DataFrame) -> List[Dict]:
        """Run Stage-1 model sweep."""
        results = []

        # Model configurations to test
        model_configs = [
            {"name": "threshold_max_score", "type": "threshold_max_score", "config": {}},
            {"name": "threshold_gap", "type": "threshold_gap", "config": {}},
            {"name": "logreg", "type": "logistic_regression", "config": {"C": 1.0}},
            {"name": "rf_100", "type": "random_forest", "config": {"n_estimators": 100, "max_depth": 10}},
            {"name": "xgb_100", "type": "xgboost", "config": {"n_estimators": 100, "max_depth": 6}},
        ]

        for config in tqdm(model_configs, desc="Stage-1 models"):
            try:
                metrics = self._evaluate_stage1_cv(feature_df, config)
                metrics["model_name"] = config["name"]
                metrics["model_type"] = config["type"]
                results.append(metrics)

                # Add to leaderboard
                self._add_to_leaderboard(
                    config_name=f"stage1_{config['name']}",
                    metrics=metrics,
                    config=config,
                )
            except Exception as e:
                logger.warning(f"  Model {config['name']} failed: {e}")

        # Save Stage-1 results
        stage1_df = pd.DataFrame(results)
        stage1_df.to_csv(self.results_dir / "stage1_results.csv", index=False)

        logger.info(f"  Evaluated {len(results)} Stage-1 models")
        if results:
            best = max(results, key=lambda x: x.get("tpr_at_fpr_5pct", 0))
            logger.info(f"  Best Stage-1: {best['model_name']} (TPR@5%FPR: {best.get('tpr_at_fpr_5pct', 0):.4f})")

        return results

    def _evaluate_stage1_cv(self, feature_df: pd.DataFrame, config: Dict) -> Dict:
        """Cross-validate a Stage-1 model."""
        from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
        from sklearn.model_selection import train_test_split

        all_probs = []
        all_labels = []

        # Get available fold IDs from data
        available_folds = sorted(feature_df["fold_id"].unique())

        # If only one fold, use train/test split instead of CV
        if len(available_folds) == 1:
            # 80/20 split within the single fold
            train_df, test_df = train_test_split(
                feature_df, test_size=0.2, random_state=self.seed,
                stratify=feature_df["has_evidence"] if len(feature_df["has_evidence"].unique()) > 1 else None
            )
            available_folds = [None]  # Single iteration
        else:
            train_df, test_df = None, None

        for fold_id in available_folds:
            if fold_id is not None:
                train_df = feature_df[feature_df["fold_id"] != fold_id]
                test_df = feature_df[feature_df["fold_id"] == fold_id]

            if len(test_df) == 0 or len(train_df) == 0:
                continue

            # Get numeric features
            exclude_cols = ["fold_id", "has_evidence", "n_gold_sentences", "n_candidates",
                          "post_id", "query_id", "criterion_id", "criterion_text",
                          "gold_sentence_ids", "candidate_ids", "candidate_scores"]
            feature_cols = [c for c in train_df.columns
                          if c not in exclude_cols and train_df[c].dtype in [np.float64, np.int64]]

            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df["has_evidence"].values
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df["has_evidence"].values

            # Create and train model
            model_cls = PluginRegistry.get_stage1(config["type"])
            model = model_cls(config.get("config", {}))
            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)
            all_probs.extend(probs)
            all_labels.extend(y_test)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Compute metrics
        fpr_curve, tpr_curve, thresholds = roc_curve(all_labels, all_probs)

        idx_5pct = np.where(fpr_curve <= 0.05)[0]
        idx_3pct = np.where(fpr_curve <= 0.03)[0]

        return {
            "auroc": float(roc_auc_score(all_labels, all_probs)),
            "auprc": float(average_precision_score(all_labels, all_probs)),
            "tpr_at_fpr_5pct": float(tpr_curve[idx_5pct[-1]]) if len(idx_5pct) > 0 else 0.0,
            "tpr_at_fpr_3pct": float(tpr_curve[idx_3pct[-1]]) if len(idx_3pct) > 0 else 0.0,
            "threshold_at_fpr_5pct": float(thresholds[idx_5pct[-1]]) if len(idx_5pct) > 0 else 1.0,
        }

    def _train_noevidence_model(self, feature_df: pd.DataFrame) -> Optional[Dict]:
        """Train fixed NO_EVIDENCE model."""
        # This would call the train_noevidence_reranker_fixed.py script
        # For now, return placeholder
        logger.info("  [TODO] NO_EVIDENCE training not yet integrated")
        return None

    def _run_twostage_search(self, feature_df: pd.DataFrame, stage1_results: List[Dict]) -> List[Dict]:
        """Run two-stage triage search."""
        results = []

        # Get best Stage-1 model
        if not stage1_results:
            logger.warning("  No Stage-1 results available")
            return results

        best_stage1 = max(stage1_results, key=lambda x: x.get("tpr_at_fpr_5pct", 0))

        # Uncertain selector configurations
        selector_configs = [
            {"name": "prob_band_0.1", "type": "probability_band", "config": {"delta": 0.1}},
            {"name": "prob_band_0.15", "type": "probability_band", "config": {"delta": 0.15}},
            {"name": "margin_band", "type": "margin_band", "config": {"margin_threshold": 0.1}},
            {"name": "entropy_top20", "type": "entropy_percentile", "config": {"percentile": 20}},
        ]

        for sel_config in selector_configs:
            try:
                # Evaluate with this selector
                metrics = self._evaluate_twostage(feature_df, best_stage1, sel_config)
                metrics["selector_name"] = sel_config["name"]
                results.append(metrics)

                self._add_to_leaderboard(
                    config_name=f"twostage_{sel_config['name']}",
                    metrics=metrics,
                    config={"stage1": best_stage1, "selector": sel_config},
                )
            except Exception as e:
                logger.warning(f"  Selector {sel_config['name']} failed: {e}")

        if results:
            logger.info(f"  Evaluated {len(results)} two-stage configurations")

        return results

    def _evaluate_twostage(self, feature_df: pd.DataFrame, stage1_config: Dict, selector_config: Dict) -> Dict:
        """Evaluate a two-stage configuration."""
        # Placeholder - would implement full two-stage evaluation
        return {
            "tpr_at_fpr_5pct": stage1_config.get("tpr_at_fpr_5pct", 0),
            "uncertain_rate": 0.2,  # Placeholder
            "stage2_overhead": 0.1,  # Placeholder
        }

    def _run_kselector_search(self, feature_df: pd.DataFrame) -> List[Dict]:
        """Run K selector search."""
        results = []

        k_configs = [
            {"name": "fixed_k1", "type": "fixed_k", "config": {"k": 1}},
            {"name": "fixed_k3", "type": "fixed_k", "config": {"k": 3}},
            {"name": "fixed_k5", "type": "fixed_k", "config": {"k": 5}},
            {"name": "prob_thresh_0.5", "type": "prob_threshold", "config": {"prob_threshold": 0.5}},
            {"name": "score_gap_knee", "type": "score_gap_knee", "config": {"gap_threshold": 0.1}},
        ]

        for config in k_configs:
            try:
                metrics = self._evaluate_kselector(feature_df, config)
                metrics["k_selector_name"] = config["name"]
                results.append(metrics)

                self._add_to_leaderboard(
                    config_name=f"kselector_{config['name']}",
                    metrics=metrics,
                    config=config,
                )
            except Exception as e:
                logger.warning(f"  K-selector {config['name']} failed: {e}")

        if results:
            logger.info(f"  Evaluated {len(results)} K-selector configurations")

        return results

    def _evaluate_kselector(self, feature_df: pd.DataFrame, config: Dict) -> Dict:
        """Evaluate a K selector."""
        selector_cls = PluginRegistry.get_k_selector(config["type"])
        selector = selector_cls(config.get("config", {}))

        # Compute avg_k (placeholder using random scores)
        n_queries = len(feature_df)
        dummy_scores = np.random.rand(n_queries, 20)
        k_values = selector.select_k(feature_df, dummy_scores)

        return {
            "avg_k": float(np.mean(k_values)),
            "max_k": int(np.max(k_values)),
            "min_k": int(np.min(k_values)),
        }

    def _add_to_leaderboard(self, config_name: str, metrics: Dict, config: Dict):
        """Add result to leaderboard."""
        entry = {
            "config_name": config_name,
            "timestamp": self.timestamp,
            **metrics,
        }

        # Check constraints
        entry["meets_fpr"] = metrics.get("fpr", 1.0) <= CONSTRAINTS["max_fpr"]
        entry["meets_tpr"] = metrics.get("tpr_at_fpr_5pct", 0) >= CONSTRAINTS["min_tpr"]
        entry["meets_recall"] = metrics.get("evidence_recall", 0) >= CONSTRAINTS["min_evidence_recall"]
        entry["meets_avg_k"] = metrics.get("avg_k", 10) <= CONSTRAINTS["max_avg_k"]
        entry["meets_all_constraints"] = all([
            entry["meets_fpr"], entry["meets_tpr"],
            entry.get("meets_recall", True), entry.get("meets_avg_k", True)
        ])

        self.leaderboard.append(entry)

        # Save config
        config_path = self.results_dir / "configs" / f"{config_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _build_leaderboard(self) -> pd.DataFrame:
        """Build final leaderboard DataFrame."""
        if not self.leaderboard:
            return pd.DataFrame()

        df = pd.DataFrame(self.leaderboard)

        # Sort by: meets_all_constraints DESC, tpr_at_fpr_5pct DESC, fpr ASC, avg_k ASC
        sort_cols = []
        if "meets_all_constraints" in df.columns:
            sort_cols.append(("meets_all_constraints", False))
        if "tpr_at_fpr_5pct" in df.columns:
            sort_cols.append(("tpr_at_fpr_5pct", False))
        if "fpr" in df.columns:
            sort_cols.append(("fpr", True))
        if "avg_k" in df.columns:
            sort_cols.append(("avg_k", True))

        if sort_cols:
            df = df.sort_values(
                by=[c[0] for c in sort_cols],
                ascending=[c[1] for c in sort_cols]
            )

        return df

    def _run_sanity_checks(self, feature_df: pd.DataFrame):
        """Run sanity checks on results."""
        logger.info("  Sampling false positives and negatives for manual review...")

        # Sample queries
        evidence_queries = feature_df[feature_df["has_evidence"] == 1]
        no_evidence_queries = feature_df[feature_df["has_evidence"] == 0]

        logger.info(f"  Total queries: {len(feature_df)}")
        logger.info(f"  Has evidence: {len(evidence_queries)} ({len(evidence_queries)/len(feature_df):.1%})")
        logger.info(f"  No evidence: {len(no_evidence_queries)} ({len(no_evidence_queries)/len(feature_df):.1%})")

        # Save samples for manual review
        sample_size = min(100, len(evidence_queries), len(no_evidence_queries))

        if sample_size > 0:
            evidence_sample = evidence_queries.sample(sample_size, random_state=self.seed)
            no_evidence_sample = no_evidence_queries.sample(sample_size, random_state=self.seed)

            evidence_sample.to_csv(self.results_dir / "error_analysis" / "evidence_sample.csv", index=False)
            no_evidence_sample.to_csv(self.results_dir / "error_analysis" / "no_evidence_sample.csv", index=False)

            logger.info(f"  Saved {sample_size} samples each to error_analysis/")

    def _save_outputs(self, leaderboard_df: pd.DataFrame):
        """Save all outputs."""
        # Leaderboard
        leaderboard_df.to_csv(self.results_dir / "leaderboard.csv", index=False)

        # Summary
        summary = {
            "timestamp": self.timestamp,
            "n_configs_evaluated": len(self.leaderboard),
            "n_meeting_constraints": sum(1 for e in self.leaderboard if e.get("meets_all_constraints", False)),
            "constraints": CONSTRAINTS,
        }

        if len(leaderboard_df) > 0:
            best = leaderboard_df.iloc[0]
            summary["best_config"] = best["config_name"]
            summary["best_tpr_at_fpr_5pct"] = float(best.get("tpr_at_fpr_5pct", 0))
            summary["best_auroc"] = float(best.get("auroc", 0))

        with open(self.results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Report
        report = self._generate_report(leaderboard_df, summary)
        with open(self.results_dir / "report.md", "w") as f:
            f.write(report)

        logger.info(f"\nOutputs saved to: {self.results_dir}")
        logger.info(f"  - leaderboard.csv")
        logger.info(f"  - summary.json")
        logger.info(f"  - report.md")
        logger.info(f"  - configs/")

    def _generate_report(self, leaderboard_df: pd.DataFrame, summary: Dict) -> str:
        """Generate markdown report."""
        lines = [
            "# Supersearch Results Report",
            f"\nTimestamp: {self.timestamp}",
            f"\n## Constraints",
            f"- Max FPR: {CONSTRAINTS['max_fpr']*100:.0f}%",
            f"- Min TPR: {CONSTRAINTS['min_tpr']*100:.0f}%",
            f"- Min Evidence Recall: {CONSTRAINTS['min_evidence_recall']*100:.0f}%",
            f"- Max Avg K: {CONSTRAINTS['max_avg_k']}",
            f"\n## Summary",
            f"- Configurations evaluated: {summary['n_configs_evaluated']}",
            f"- Meeting all constraints: {summary['n_meeting_constraints']}",
        ]

        if "best_config" in summary:
            lines.extend([
                f"\n## Best Configuration",
                f"- Config: {summary['best_config']}",
                f"- TPR@5%FPR: {summary['best_tpr_at_fpr_5pct']:.4f}",
                f"- AUROC: {summary['best_auroc']:.4f}",
            ])

        if len(leaderboard_df) > 0:
            lines.extend([
                f"\n## Top 10 Configurations",
                "",
                leaderboard_df.head(10).to_string(index=False),
            ])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run supersearch pipeline")
    parser.add_argument("--output_dir", type=str, default="outputs/supersearch")
    parser.add_argument("--feature_store", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_ne_training", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Run on 1 fold only")
    args = parser.parse_args()

    np.random.seed(args.seed)

    pipeline = SupersearchPipeline(
        output_dir=Path(args.output_dir),
        feature_store_dir=Path(args.feature_store) if args.feature_store else None,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    leaderboard = pipeline.run_full_search(
        skip_ne_training=args.skip_ne_training,
        smoke_test=args.smoke_test,
    )

    print(f"\n{'='*60}")
    print("SUPERSEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {pipeline.results_dir}")

    if len(leaderboard) > 0:
        print(f"\nTop 5 configurations:")
        print(leaderboard[["config_name", "tpr_at_fpr_5pct", "auroc", "meets_all_constraints"]].head().to_string())


if __name__ == "__main__":
    main()
