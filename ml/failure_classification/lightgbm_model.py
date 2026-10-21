"""
MaintenAI— LightGBM Failure Mode Classification Model

Multi-output and multi-class gradient boosted classifier for predicting
specific equipment failure modes (Tool Wear, Heat Dissipation, Power Failure, Overstrain, Random Failure).
Includes fallback to Scikit-Learn HistGradientBoosting when OpenMP is unavailable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)

# Safe LightGBM import with graceful fallback
try:
    if os.getenv("FP_FORCE_FALLBACK", "1") == "1":
        raise RuntimeError(
            "Native LightGBM disabled due to macOS OpenMP ABI issue; using HistGradientBoosting"
        )
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except (ImportError, OSError, Exception) as e:
    logger.warning(
        "LightGBM native library unavailable or disabled (%s). Falling back to HistGradientBoostingClassifier.",
        e,
    )
    HAS_LIGHTGBM = False
    lgb = None

FAILURE_MODES = [
    "failure_tool_wear",
    "failure_heat_dissipation",
    "failure_power_failure",
    "failure_overstrain",
    "failure_random",
]

FAILURE_MODE_NAMES = {
    "failure_tool_wear": "Tool Wear Failure (TWF)",
    "failure_heat_dissipation": "Heat Dissipation Failure (HDF)",
    "failure_power_failure": "Power Failure (PWF)",
    "failure_overstrain": "Overstrain Failure (OSF)",
    "failure_random": "Random Failure (RNF)",
}


class LightGBMFailureClassifier:
    """
    LightGBM Multi-target Failure Classifier with automatic HistGradientBoosting fallback.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        class_weight: str = "balanced",
        random_state: int = 42,
        early_stopping_rounds: int = 20,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.class_weight = class_weight
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self.binary_model = None
        self.mode_models: dict[str, Any] = {}
        self.feature_names: list[str] = []

    def _build_model(self):
        if HAS_LIGHTGBM and lgb is not None:
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                class_weight=self.class_weight,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            return HistGradientBoostingClassifier(
                max_iter=min(self.n_estimators, 200),
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.num_leaves,
                min_samples_leaf=self.min_child_samples,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train_binary: pd.Series | np.ndarray,
        y_train_modes: pd.DataFrame | dict[str, np.ndarray] | None = None,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val_binary: pd.Series | np.ndarray | None = None,
        y_val_modes: pd.DataFrame | dict[str, np.ndarray] | None = None,
        feature_names: list[str] | None = None,
    ) -> LightGBMFailureClassifier:
        """
        Train binary failure model and individual failure mode models.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_tr = X_train.values
        else:
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
            X_tr = X_train

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        y_bin_tr = np.asarray(y_train_binary)

        eval_set_bin = None
        if X_val is not None and y_val_binary is not None:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            X_v = np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0)
            y_v_bin = np.asarray(y_val_binary)
            eval_set_bin = [(X_v, y_v_bin)]

        # 1. Train overall failure binary classifier
        self.binary_model = self._build_model()
        if HAS_LIGHTGBM and lgb is not None and eval_set_bin:
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            self.binary_model.fit(X_tr, y_bin_tr, eval_set=eval_set_bin, callbacks=callbacks)
        else:
            self.binary_model.fit(X_tr, y_bin_tr)

        # 2. Train individual failure mode models
        if y_train_modes is not None:
            mode_keys = (
                FAILURE_MODES
                if isinstance(y_train_modes, pd.DataFrame)
                else list(y_train_modes.keys())
            )
            for mode in mode_keys:
                if isinstance(y_train_modes, pd.DataFrame):
                    if mode not in y_train_modes.columns:
                        continue
                    y_mode_tr = y_train_modes[mode].values
                else:
                    y_mode_tr = np.asarray(y_train_modes[mode])

                if np.sum(y_mode_tr) == 0:
                    continue

                mode_model = self._build_model()
                mode_model.fit(X_tr, y_mode_tr)
                self.mode_models[mode] = mode_model

        logger.info(
            "Failure Classifier trained (%s). Features: %d, Mode models: %s",
            "LightGBM" if HAS_LIGHTGBM else "HistGradientBoosting",
            len(self.feature_names),
            list(self.mode_models.keys()),
        )
        return self

    def predict_failure_probability(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict overall failure probability in [0.0, 1.0]."""
        if self.binary_model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        X_mat = self._prep_input(X)
        probs = self.binary_model.predict_proba(X_mat)
        return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    def predict_mode_probabilities(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Predict individual failure mode probabilities."""
        X_mat = self._prep_input(X)
        results: dict[str, np.ndarray] = {}

        for mode, model in self.mode_models.items():
            probs = model.predict_proba(X_mat)
            results[mode] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        return results

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
        """Predict overall binary failure flag and probable failure modes."""
        fail_prob = self.predict_failure_probability(X)
        mode_probs = self.predict_mode_probabilities(X)

        return {
            "failure_predicted": (fail_prob >= threshold).astype(int),
            "failure_probability": fail_prob,
            "mode_probabilities": mode_probs,
        }

    def _prep_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                mat = X.reindex(columns=self.feature_names, fill_value=0.0).values
            else:
                mat = X.values
        else:
            mat = np.asarray(X)
            if self.feature_names and mat.ndim == 2:
                if mat.shape[1] < len(self.feature_names):
                    padded = np.zeros((mat.shape[0], len(self.feature_names)))
                    padded[:, : mat.shape[1]] = mat
                    mat = padded
                elif mat.shape[1] > len(self.feature_names):
                    mat = mat[:, : len(self.feature_names)]
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_importances(self, top_n: int = 15) -> dict[str, float]:
        """Return feature importances."""
        if self.binary_model is None:
            return {}
        if hasattr(self.binary_model, "feature_importances_"):
            importances = self.binary_model.feature_importances_
            feat_imp = {
                name: float(imp) for name, imp in zip(self.feature_names, importances, strict=False)
            }
            return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:top_n])
        return {name: 1.0 / len(self.feature_names) for name in self.feature_names[:top_n]}

    def save(self, path: Path | str) -> None:
        """Save model artifacts to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "binary_model": self.binary_model,
            "mode_models": self.mode_models,
            "feature_names": self.feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "early_stopping_rounds": self.early_stopping_rounds,
            },
        }
        joblib.dump(payload, target_path)
        logger.info("Saved Failure Classifier to %s", target_path)

    @classmethod
    def load(cls, path: Path | str) -> LightGBMFailureClassifier:
        """Load model from disk."""
        target_path = Path(path)
        payload = joblib.load(target_path)
        classifier = cls(**payload.get("params", {}))
        classifier.binary_model = payload["binary_model"]
        classifier.mode_models = payload.get("mode_models", {})
        classifier.feature_names = payload.get("feature_names", [])
        return classifier
