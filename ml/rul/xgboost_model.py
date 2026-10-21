"""
MaintenAI— XGBoost Remaining Useful Life (RUL) Prediction Model

Gradient boosted tree regressor for turbofan engine degradation trajectory
and remaining operational cycle estimation.
Includes graceful fallback to HistGradientBoostingRegressor when OpenMP is unavailable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

# Safe XGBoost import
try:
    if os.getenv("FP_FORCE_FALLBACK", "1") == "1":
        raise RuntimeError(
            "Native XGBoost disabled due to macOS OpenMP ABI issue; using HistGradientBoosting"
        )
    import xgboost as xgb

    HAS_XGBOOST = True
except (ImportError, OSError, Exception) as e:
    logger.warning(
        "XGBoost native library unavailable or disabled (%s). Falling back to HistGradientBoostingRegressor.",
        e,
    )
    HAS_XGBOOST = False
    xgb = None


class XGBoostRULRegressor:
    """
    XGBoost Regressor for Remaining Useful Life (RUL) prediction with HistGradientBoosting fallback.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 3.0,
        random_state: int = 42,
        early_stopping_rounds: int = 20,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> XGBoostRULRegressor:
        """
        Train Regressor on feature matrix and RUL targets.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_tr = X_train.values
        else:
            self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
            X_tr = X_train

        y_tr = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            X_v = np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0)
            y_v = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)
            eval_set = [(X_v, y_v)]

        if HAS_XGBOOST and xgb is not None:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
                early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
                objective="reg:squarederror",
                n_jobs=-1,
            )
            self.model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=min(self.n_estimators, 200),
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            self.model.fit(X_tr, y_tr)

        logger.info(
            "RUL model trained (%s) with %d features on %d samples",
            "XGBoost" if HAS_XGBOOST else "HistGradientBoosting",
            len(self.feature_names),
            len(X_tr),
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict Remaining Useful Life (cycles).
        Output is strictly non-negative (clipped at 0).
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
            if self.feature_names and X_mat.ndim == 2:
                if X_mat.shape[1] < len(self.feature_names):
                    padded = np.zeros((X_mat.shape[0], len(self.feature_names)))
                    padded[:, : X_mat.shape[1]] = X_mat
                    X_mat = padded
                elif X_mat.shape[1] > len(self.feature_names):
                    X_mat = X_mat[:, : len(self.feature_names)]

        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        raw_pred = self.model.predict(X_mat)
        return np.maximum(raw_pred, 0.0)

    def get_feature_importances(self, top_n: int = 15) -> dict[str, float]:
        """Return top N feature importances."""
        if self.model is None:
            return {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_imp = {
                name: float(imp) for name, imp in zip(self.feature_names, importances, strict=False)
            }
            return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:top_n])
        return {name: 1.0 / len(self.feature_names) for name in self.feature_names[:top_n]}

    def save(self, path: Path | str) -> None:
        """Save model and metadata to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "random_state": self.random_state,
                "early_stopping_rounds": self.early_stopping_rounds,
            },
        }
        joblib.dump(payload, target_path)
        logger.info("Saved RUL model to %s", target_path)

    @classmethod
    def load(cls, path: Path | str) -> XGBoostRULRegressor:
        """Load model from disk."""
        target_path = Path(path)
        payload = joblib.load(target_path)
        regressor = cls(**payload.get("params", {}))
        regressor.model = payload["model"]
        regressor.feature_names = payload.get("feature_names", [])
        return regressor
