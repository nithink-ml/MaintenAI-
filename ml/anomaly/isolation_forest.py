"""
MaintenAI— Isolation Forest Anomaly Detection Model

Unsupervised anomaly detector for machine sensor telemetry and IMS bearing degradation.
Maps raw decision function to standardized [0.0, 1.0] anomaly scores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """
    Isolation Forest anomaly detector with calibrated [0, 1] scoring.

    Attributes:
        n_estimators: Number of isolation trees in ensemble.
        contamination: Expected proportion of outliers in the dataset.
        max_features: Number/fraction of features to draw from X to train each base estimator.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state

        self.model: IsolationForest | None = None
        self.feature_names: list[str] = []
        self.score_min_: float = 0.0
        self.score_max_: float = 1.0
        self.threshold_: float = 0.5

    def fit(
        self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None
    ) -> IsolationForestAnomalyDetector:
        """
        Fit the Isolation Forest model on training features.

        Args:
            X: Training feature matrix (normal or mixed telemetry).
            feature_names: Optional list of column names if X is numpy array.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_mat = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_mat = X

        # Handle NaNs / infs safely
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_mat)

        # Calibrate score mapping based on training score distribution
        # decision_function gives negative for anomalies, positive for normal
        raw_scores = -self.model.decision_function(X_mat)  # Higher = more anomalous
        self.score_min_ = float(np.percentile(raw_scores, 1))
        self.score_max_ = float(np.percentile(raw_scores, 99))
        if self.score_max_ <= self.score_min_:
            self.score_max_ = self.score_min_ + 1.0

        # Calibration: 95th percentile or contamination quantile as warning threshold
        self.threshold_ = float(
            np.percentile(self._normalize_score(raw_scores), (1.0 - self.contamination) * 100)
        )
        logger.info(
            "IsolationForest fitted on %d samples with %d features", X_mat.shape[0], X_mat.shape[1]
        )
        return self

    def _normalize_score(self, raw_score: np.ndarray) -> np.ndarray:
        """Map raw inverted decision function to [0, 1]."""
        norm = (raw_score - self.score_min_) / (self.score_max_ - self.score_min_ + 1e-8)
        return np.clip(norm, 0.0, 1.0)

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly scores in range [0.0, 1.0].
        Higher values represent more anomalous behaviour.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                X_mat = X.reindex(columns=self.feature_names, fill_value=0.0).values
            else:
                X_mat = X.values
        else:
            X_mat = np.asarray(X)
            if self.feature_names and X_mat.shape[1] < len(self.feature_names):
                padded = np.zeros((X_mat.shape[0], len(self.feature_names)))
                padded[:, :X_mat.shape[1]] = X_mat
                X_mat = padded

        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        raw_scores = -self.model.decision_function(X_mat)
        return self._normalize_score(raw_scores)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Predict binary anomaly flags (0 = Normal, 1 = Anomaly).
        """
        scores = self.predict_score(X)
        th = threshold if threshold is not None else self.threshold_
        return (scores >= th).astype(int)

    def explain_instance(self, x: np.ndarray | pd.Series) -> dict[str, float]:
        """
        Approximate feature contributions to anomaly score using tree path depth differences.
        """
        if isinstance(x, pd.Series):
            df_aligned = pd.DataFrame([x]).reindex(columns=self.feature_names, fill_value=0.0)
            vals = df_aligned.values
        elif isinstance(x, pd.DataFrame):
            df_aligned = x.reindex(columns=self.feature_names, fill_value=0.0)
            vals = df_aligned.values[:1]
        else:
            vals = np.asarray(x).reshape(1, -1)
            if self.feature_names and vals.shape[1] < len(self.feature_names):
                padded = np.zeros((1, len(self.feature_names)))
                padded[0, :vals.shape[1]] = vals[0]
                vals = padded

        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        base_score = float(self.predict_score(vals)[0])

        importances: dict[str, float] = {}
        # Perturbation-based sensitivity for top contributors
        for idx, feat_name in enumerate(self.feature_names[: min(len(self.feature_names), 30)]):
            val_perturbed = vals.copy()
            val_perturbed[0, idx] = 0.0  # Zero baseline
            pert_score = float(self.predict_score(val_perturbed)[0])
            importances[feat_name] = round(abs(base_score - pert_score), 4)

        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5])

    def save(self, path: Path | str) -> None:
        """Save model and metadata to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "score_min": self.score_min_,
            "score_max": self.score_max_,
            "threshold": self.threshold_,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_features": self.max_features,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, target_path)
        logger.info("Saved Isolation Forest model to %s", target_path)

    @classmethod
    def load(cls, path: Path | str) -> IsolationForestAnomalyDetector:
        """Load model from disk."""
        target_path = Path(path)
        payload = joblib.load(target_path)
        detector = cls(**payload.get("params", {}))
        detector.model = payload["model"]
        detector.feature_names = payload.get("feature_names", [])
        detector.score_min_ = payload.get("score_min", 0.0)
        detector.score_max_ = payload.get("score_max", 1.0)
        detector.threshold_ = payload.get("threshold", 0.5)
        return detector
