"""Plugin registry architecture for supersearch.

All plugins take OOF cache inputs and output decisions + debug signals.

Note: Uses pickle for internal model serialization (sklearn models).
This is safe as models are only saved/loaded by this system.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import numpy as np
import pandas as pd

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Standard prediction result from any plugin."""
    predictions: np.ndarray  # Binary or continuous predictions
    probabilities: Optional[np.ndarray] = None  # Calibrated probabilities
    debug_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Standard metrics for NE gate evaluation."""
    tpr: float  # True positive rate (recall for has-evidence)
    fpr: float  # False positive rate (1 - specificity for no-evidence)
    precision: float
    f1: float
    auroc: float
    auprc: float
    tpr_at_fpr_3pct: float  # TPR when FPR <= 3%
    tpr_at_fpr_5pct: float  # TPR when FPR <= 5%
    threshold_at_fpr_5pct: float  # Threshold to achieve FPR <= 5%

    def to_dict(self) -> Dict[str, float]:
        return {
            "tpr": self.tpr,
            "fpr": self.fpr,
            "precision": self.precision,
            "f1": self.f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "tpr_at_fpr_3pct": self.tpr_at_fpr_3pct,
            "tpr_at_fpr_5pct": self.tpr_at_fpr_5pct,
            "threshold_at_fpr_5pct": self.threshold_at_fpr_5pct,
        }


class PluginRegistry:
    """Central registry for all supersearch plugins."""

    _stage1_models: Dict[str, Type["NEStage1Model"]] = {}
    _uncertain_selectors: Dict[str, Type["UncertainSelector"]] = {}
    _stage2_models: Dict[str, Type["NEStage2Model"]] = {}
    _k_selectors: Dict[str, Type["KSelector"]] = {}
    _calibrators: Dict[str, Type["Calibrator"]] = {}

    @classmethod
    def register_stage1(cls, name: str):
        """Decorator to register a Stage-1 NE model."""
        def decorator(model_cls: Type["NEStage1Model"]):
            cls._stage1_models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def register_uncertain_selector(cls, name: str):
        """Decorator to register an uncertain selector."""
        def decorator(selector_cls: Type["UncertainSelector"]):
            cls._uncertain_selectors[name] = selector_cls
            return selector_cls
        return decorator

    @classmethod
    def register_stage2(cls, name: str):
        """Decorator to register a Stage-2 NE model."""
        def decorator(model_cls: Type["NEStage2Model"]):
            cls._stage2_models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def register_k_selector(cls, name: str):
        """Decorator to register a K selector."""
        def decorator(selector_cls: Type["KSelector"]):
            cls._k_selectors[name] = selector_cls
            return selector_cls
        return decorator

    @classmethod
    def register_calibrator(cls, name: str):
        """Decorator to register a calibrator."""
        def decorator(calibrator_cls: Type["Calibrator"]):
            cls._calibrators[name] = calibrator_cls
            return calibrator_cls
        return decorator

    @classmethod
    def get_stage1(cls, name: str) -> Type["NEStage1Model"]:
        if name not in cls._stage1_models:
            raise ValueError(f"Unknown Stage-1 model: {name}. Available: {list(cls._stage1_models.keys())}")
        return cls._stage1_models[name]

    @classmethod
    def get_uncertain_selector(cls, name: str) -> Type["UncertainSelector"]:
        if name not in cls._uncertain_selectors:
            raise ValueError(f"Unknown selector: {name}. Available: {list(cls._uncertain_selectors.keys())}")
        return cls._uncertain_selectors[name]

    @classmethod
    def get_stage2(cls, name: str) -> Type["NEStage2Model"]:
        if name not in cls._stage2_models:
            raise ValueError(f"Unknown Stage-2 model: {name}. Available: {list(cls._stage2_models.keys())}")
        return cls._stage2_models[name]

    @classmethod
    def get_k_selector(cls, name: str) -> Type["KSelector"]:
        if name not in cls._k_selectors:
            raise ValueError(f"Unknown K selector: {name}. Available: {list(cls._k_selectors.keys())}")
        return cls._k_selectors[name]

    @classmethod
    def get_calibrator(cls, name: str) -> Type["Calibrator"]:
        if name not in cls._calibrators:
            raise ValueError(f"Unknown calibrator: {name}. Available: {list(cls._calibrators.keys())}")
        return cls._calibrators[name]

    @classmethod
    def list_all(cls) -> Dict[str, List[str]]:
        return {
            "stage1_models": list(cls._stage1_models.keys()),
            "uncertain_selectors": list(cls._uncertain_selectors.keys()),
            "stage2_models": list(cls._stage2_models.keys()),
            "k_selectors": list(cls._k_selectors.keys()),
            "calibrators": list(cls._calibrators.keys()),
        }


class NEStage1Model(ABC):
    """Base class for Stage-1 NE gate models (fast).

    Stage-1 models make a quick decision on whether a query has evidence.
    They should be fast enough to run on all queries.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability of has_evidence."""
        pass

    def predict(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5,
    ) -> PredictionResult:
        """Predict binary has_evidence with threshold."""
        probs = self.predict_proba(features)
        preds = (probs >= threshold).astype(int)
        return PredictionResult(
            predictions=preds,
            probabilities=probs,
            debug_signals={"threshold": threshold},
        )

    def evaluate(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationMetrics:
        """Evaluate model performance."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_score, recall_score, f1_score, roc_curve
        )

        probs = self.predict_proba(features)
        preds = (probs >= threshold).astype(int)

        # Basic metrics at threshold
        tpr = recall_score(labels, preds, zero_division=0)
        fpr = 1 - recall_score(1 - labels, 1 - preds, zero_division=0)  # FPR = FP / (FP + TN)
        precision = precision_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # AUC metrics
        auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        auprc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

        # TPR at specific FPR levels
        fpr_curve, tpr_curve, thresholds = roc_curve(labels, probs)

        # Find TPR at FPR <= 3%
        idx_3pct = np.where(fpr_curve <= 0.03)[0]
        tpr_at_fpr_3pct = tpr_curve[idx_3pct[-1]] if len(idx_3pct) > 0 else 0.0

        # Find TPR at FPR <= 5%
        idx_5pct = np.where(fpr_curve <= 0.05)[0]
        tpr_at_fpr_5pct = tpr_curve[idx_5pct[-1]] if len(idx_5pct) > 0 else 0.0
        threshold_at_fpr_5pct = thresholds[idx_5pct[-1]] if len(idx_5pct) > 0 else 1.0

        return EvaluationMetrics(
            tpr=tpr,
            fpr=fpr,
            precision=precision,
            f1=f1,
            auroc=auroc,
            auprc=auprc,
            tpr_at_fpr_3pct=tpr_at_fpr_3pct,
            tpr_at_fpr_5pct=tpr_at_fpr_5pct,
            threshold_at_fpr_5pct=threshold_at_fpr_5pct,
        )

    def find_threshold_at_fpr(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        target_fpr: float = 0.05,
    ) -> Tuple[float, float]:
        """Find threshold that achieves target FPR, return (threshold, achieved_tpr)."""
        from sklearn.metrics import roc_curve

        probs = self.predict_proba(features)
        fpr_curve, tpr_curve, thresholds = roc_curve(labels, probs)

        # Find threshold for target FPR
        idx = np.where(fpr_curve <= target_fpr)[0]
        if len(idx) == 0:
            return 1.0, 0.0

        best_idx = idx[-1]  # Highest TPR at or below target FPR
        return thresholds[best_idx], tpr_curve[best_idx]

    def save(self, path: Path) -> None:
        """Save model to disk using joblib (safer than pickle for sklearn)."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "NEStage1Model":
        """Load model from disk."""
        import joblib
        return joblib.load(path)


class UncertainSelector(ABC):
    """Base class for uncertain query selectors.

    Selectors identify queries that should be routed to Stage-2
    for more expensive processing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def select_uncertain(
        self,
        features: pd.DataFrame,
        stage1_probs: np.ndarray,
        stage1_threshold: float,
    ) -> np.ndarray:
        """Return boolean mask of uncertain queries to route to Stage-2."""
        pass

    def get_uncertain_rate(
        self,
        features: pd.DataFrame,
        stage1_probs: np.ndarray,
        stage1_threshold: float,
    ) -> float:
        """Return fraction of queries routed to Stage-2."""
        mask = self.select_uncertain(features, stage1_probs, stage1_threshold)
        return mask.mean()


class NEStage2Model(ABC):
    """Base class for Stage-2 NE models (heavy).

    Stage-2 models are more expensive but more accurate.
    They only run on uncertain queries selected by UncertainSelector.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the model on training data."""
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability of has_evidence."""
        pass

    def predict(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5,
    ) -> PredictionResult:
        """Predict binary has_evidence with threshold."""
        probs = self.predict_proba(features)
        preds = (probs >= threshold).astype(int)
        return PredictionResult(
            predictions=preds,
            probabilities=probs,
            debug_signals={"threshold": threshold},
        )


class KSelector(ABC):
    """Base class for evidence count (K) selectors.

    K selectors determine how many evidence sentences to return
    for each query, capped at K=5 for deployment.
    """

    MAX_K = 5  # Hard deployment constraint

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def select_k(
        self,
        features: pd.DataFrame,
        candidate_scores: np.ndarray,  # [n_queries, n_candidates]
        candidate_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return K value for each query (capped at MAX_K)."""
        pass

    def get_avg_k(
        self,
        features: pd.DataFrame,
        candidate_scores: np.ndarray,
        candidate_probs: Optional[np.ndarray] = None,
    ) -> float:
        """Return average K across queries."""
        k_values = self.select_k(features, candidate_scores, candidate_probs)
        return k_values.mean()


class Calibrator(ABC):
    """Base class for probability calibrators.

    Calibrators transform raw scores into well-calibrated probabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit calibrator on held-out data."""
        pass

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transform scores to calibrated probabilities."""
        pass

    def fit_transform(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(scores, labels)
        return self.transform(scores)


# ============================================================================
# Built-in implementations
# ============================================================================

@PluginRegistry.register_stage1("threshold_max_score")
class ThresholdMaxScoreStage1(NEStage1Model):
    """Simple threshold on max reranker score."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.threshold = config.get("threshold", 0.5) if config else 0.5
        self.is_fitted = True  # No fitting needed

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        # Find optimal threshold on training data
        from sklearn.metrics import roc_curve
        scores = features["max_reranker_score"].values
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find threshold at 5% FPR
        idx = np.where(fpr <= 0.05)[0]
        if len(idx) > 0:
            self.threshold = thresholds[idx[-1]]
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return features["max_reranker_score"].values


@PluginRegistry.register_stage1("threshold_gap")
class ThresholdGapStage1(NEStage1Model):
    """Threshold on top1-top2 score gap."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.threshold = config.get("threshold", 0.1) if config else 0.1

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        from sklearn.metrics import roc_curve
        gaps = features["top1_top2_gap"].values
        fpr, tpr, thresholds = roc_curve(labels, gaps)
        idx = np.where(fpr <= 0.05)[0]
        if len(idx) > 0:
            self.threshold = thresholds[idx[-1]]
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return features["top1_top2_gap"].values


@PluginRegistry.register_stage1("logistic_regression")
class LogisticRegressionStage1(NEStage1Model):
    """Logistic regression with class weights."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model = None
        self.feature_cols = config.get("feature_cols", None) if config else None
        self.class_weight = config.get("class_weight", "balanced") if config else "balanced"
        self.C = config.get("C", 1.0) if config else 1.0

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Select features
        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            # Use all numeric columns
            X = features.select_dtypes(include=[np.number]).values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            class_weight=self.class_weight,
            C=self.C,
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(X_scaled, labels, sample_weight=sample_weights)
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


@PluginRegistry.register_stage1("random_forest")
class RandomForestStage1(NEStage1Model):
    """Random forest classifier."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model = None
        self.feature_cols = config.get("feature_cols", None) if config else None
        self.n_estimators = config.get("n_estimators", 100) if config else 100
        self.max_depth = config.get("max_depth", 10) if config else 10
        self.class_weight = config.get("class_weight", "balanced") if config else "balanced"

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        from sklearn.ensemble import RandomForestClassifier

        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, labels, sample_weight=sample_weights)
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values
        return self.model.predict_proba(X)[:, 1]


@PluginRegistry.register_stage1("xgboost")
class XGBoostStage1(NEStage1Model):
    """XGBoost classifier."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model = None
        self.feature_cols = config.get("feature_cols", None) if config else None
        self.n_estimators = config.get("n_estimators", 100) if config else 100
        self.max_depth = config.get("max_depth", 6) if config else 6
        self.learning_rate = config.get("learning_rate", 0.1) if config else 0.1
        self.scale_pos_weight = config.get("scale_pos_weight", None) if config else None

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        try:
            import xgboost as xgb
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            logger.warning("XGBoost not available, using HistGradientBoostingClassifier")

            if self.feature_cols:
                X = features[self.feature_cols].values
            else:
                X = features.select_dtypes(include=[np.number]).values

            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
            self.model.fit(X, labels, sample_weight=sample_weights)
            self.is_fitted = True
            self._use_sklearn = True
            return

        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values

        # Compute scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            neg_count = (labels == 0).sum()
            pos_count = (labels == 1).sum()
            self.scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X, labels, sample_weight=sample_weights)
        self.is_fitted = True
        self._use_sklearn = False

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values
        return self.model.predict_proba(X)[:, 1]


@PluginRegistry.register_uncertain_selector("probability_band")
class ProbabilityBandSelector(UncertainSelector):
    """Select queries within a probability band around threshold."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.delta = config.get("delta", 0.1) if config else 0.1

    def select_uncertain(
        self,
        features: pd.DataFrame,
        stage1_probs: np.ndarray,
        stage1_threshold: float,
    ) -> np.ndarray:
        lower = stage1_threshold - self.delta
        upper = stage1_threshold + self.delta
        return (stage1_probs >= lower) & (stage1_probs <= upper)


@PluginRegistry.register_uncertain_selector("entropy_percentile")
class EntropyPercentileSelector(UncertainSelector):
    """Select top percentile by entropy."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.percentile = config.get("percentile", 20) if config else 20

    def select_uncertain(
        self,
        features: pd.DataFrame,
        stage1_probs: np.ndarray,
        stage1_threshold: float,
    ) -> np.ndarray:
        # Use entropy from features if available, else compute from probs
        if "entropy_top5" in features.columns:
            entropy = features["entropy_top5"].values
        else:
            # Binary entropy from stage1 probs
            eps = 1e-10
            entropy = -stage1_probs * np.log(stage1_probs + eps) - (1 - stage1_probs) * np.log(1 - stage1_probs + eps)

        threshold = np.percentile(entropy, 100 - self.percentile)
        return entropy >= threshold


@PluginRegistry.register_uncertain_selector("margin_band")
class MarginBandSelector(UncertainSelector):
    """Select queries with small margin (top1 - top2 gap)."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.margin_threshold = config.get("margin_threshold", 0.1) if config else 0.1

    def select_uncertain(
        self,
        features: pd.DataFrame,
        stage1_probs: np.ndarray,
        stage1_threshold: float,
    ) -> np.ndarray:
        if "top1_top2_gap" in features.columns:
            margin = features["top1_top2_gap"].values
            return margin <= self.margin_threshold
        else:
            # Fallback to probability band
            return np.abs(stage1_probs - stage1_threshold) <= 0.1


@PluginRegistry.register_stage2("noevidence_margin")
class NoEvidenceMarginStage2(NEStage2Model):
    """Stage-2 using NO_EVIDENCE margin signal."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.margin_threshold = config.get("margin_threshold", 0.0) if config else 0.0
        self.rank_threshold = config.get("rank_threshold", 1) if config else 1
        self.is_fitted = True  # No training needed, uses pre-computed features

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        # Tune thresholds on training data
        if "margin_NE" in features.columns:
            from sklearn.metrics import roc_curve
            margins = features["margin_NE"].values
            fpr, tpr, thresholds = roc_curve(labels, margins)
            idx = np.where(fpr <= 0.05)[0]
            if len(idx) > 0:
                self.margin_threshold = thresholds[idx[-1]]
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if "margin_NE" in features.columns:
            return features["margin_NE"].values
        elif "rank_NE" in features.columns:
            # Convert rank to pseudo-probability
            return 1 - 1 / (features["rank_NE"].values + 1)
        else:
            raise ValueError("NO_EVIDENCE features not found in feature store")


@PluginRegistry.register_stage2("ensemble_stacker")
class EnsembleStackerStage2(NEStage2Model):
    """Ensemble stacker combining Stage-1 + NO_EVIDENCE features."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model = None
        self.feature_cols = config.get("feature_cols", None) if config else None

    def fit(self, features: pd.DataFrame, labels: np.ndarray, sample_weights=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(X_scaled, labels, sample_weight=sample_weights)
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_cols:
            X = features[self.feature_cols].values
        else:
            X = features.select_dtypes(include=[np.number]).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


@PluginRegistry.register_k_selector("fixed_k")
class FixedKSelector(KSelector):
    """Fixed K for all queries."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.k = min(config.get("k", 5) if config else 5, self.MAX_K)

    def select_k(
        self,
        features: pd.DataFrame,
        candidate_scores: np.ndarray,
        candidate_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.full(len(features), self.k)


@PluginRegistry.register_k_selector("prob_threshold")
class ProbThresholdKSelector(KSelector):
    """Dynamic K by probability threshold, capped at MAX_K."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.prob_threshold = config.get("prob_threshold", 0.5) if config else 0.5

    def select_k(
        self,
        features: pd.DataFrame,
        candidate_scores: np.ndarray,
        candidate_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if candidate_probs is None:
            # Use scores as pseudo-probabilities
            candidate_probs = candidate_scores

        # Count candidates above threshold, capped at MAX_K
        k_values = (candidate_probs >= self.prob_threshold).sum(axis=1)
        return np.clip(k_values, 1, self.MAX_K)


@PluginRegistry.register_k_selector("score_gap_knee")
class ScoreGapKneeSelector(KSelector):
    """Dynamic K by detecting score gap knee point."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.gap_threshold = config.get("gap_threshold", 0.1) if config else 0.1
        self.min_k = config.get("min_k", 1) if config else 1

    def select_k(
        self,
        features: pd.DataFrame,
        candidate_scores: np.ndarray,
        candidate_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_queries = candidate_scores.shape[0]
        k_values = np.zeros(n_queries, dtype=int)

        for i in range(n_queries):
            scores = candidate_scores[i]
            # Find knee point (largest gap)
            gaps = np.diff(scores)

            if len(gaps) == 0:
                k_values[i] = self.min_k
                continue

            # Find first position where gap exceeds threshold
            knee_positions = np.where(np.abs(gaps) > self.gap_threshold)[0]

            if len(knee_positions) > 0:
                k_values[i] = knee_positions[0] + 1
            else:
                k_values[i] = len(scores)

        return np.clip(k_values, self.min_k, self.MAX_K)


@PluginRegistry.register_calibrator("platt_scaling")
class PlattScalingCalibrator(Calibrator):
    """Platt scaling (logistic regression)."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(scores.reshape(-1, 1), labels)
        self.is_fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


@PluginRegistry.register_calibrator("isotonic")
class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibration."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(scores, labels)
        self.is_fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict(scores)


@PluginRegistry.register_calibrator("temperature_scaling")
class TemperatureScalingCalibrator(Calibrator):
    """Temperature scaling for neural network outputs."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.temperature = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        from scipy.optimize import minimize_scalar
        from sklearn.metrics import log_loss

        def objective(T):
            calibrated = 1 / (1 + np.exp(-scores / T))
            return log_loss(labels, calibrated)

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        self.is_fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-scores / self.temperature))
