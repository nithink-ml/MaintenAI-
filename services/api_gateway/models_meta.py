"""
MaintenAI— Authoritative ML Model Metadata Registry

Single source of truth for runtime ML model artifacts, architecture families,
versions, feature dimensions, and fallback disclosures.
"""

from __future__ import annotations

from typing import Any


def get_active_models_metadata() -> dict[str, Any]:
    """
    Return authoritative runtime ML model metadata.
    Enforces transparent disclosure of fallback implementations and synthetic data.
    """
    return {
        "anomaly": {
            "model_name": "Isolation Forest Anomaly Detector",
            "model_family": "IsolationForest",
            "model_version": "v1.0.0-dev",
            "artifact_path": "models/anomaly/isolation_forest.joblib",
            "feature_count": 53,
            "status": "active",
            "is_fallback": False,
            "fallback_reason": None,
            "training_dataset": "Turbofan & Industrial Sensor Simulation (Development)",
            "environment": "development",
            "score_interpretation": "Continuous score in [0.0, 1.0]; >= 0.70 Warning, >= 0.85 Critical",
        },
        "rul": {
            "model_name": "HistGradientBoosting RUL Regressor",
            "model_family": "HistGradientBoostingRegressor",
            "model_version": "v1.0.0-fallback",
            "artifact_path": "models/rul/xgboost.joblib",
            "feature_count": 133,
            "status": "active",
            "is_fallback": True,
            "fallback_reason": "Native XGBoost disabled on macOS arm64 due to OpenMP ABI segfault (exit code 139). HistGradientBoosting is the functional fallback.",
            "training_dataset": "CMAPSS Turbofan Synthetic Trajectories (Development)",
            "environment": "development",
            "score_interpretation": "Remaining operational cycles until recommended overhaul",
        },
        "failure_classifier": {
            "model_name": "HistGradientBoosting Multi-target Classifier",
            "model_family": "HistGradientBoostingClassifier",
            "model_version": "v1.0.0-fallback",
            "artifact_path": "models/failure/lightgbm.joblib",
            "feature_count": 13,
            "status": "active",
            "is_fallback": True,
            "fallback_reason": "Native LightGBM fallback to HistGradientBoosting for cross-platform stability.",
            "training_dataset": "AI4I 2020 Predictive Maintenance Dataset (Development)",
            "environment": "development",
            "score_interpretation": "Calibrated multi-class probabilities for tool wear, heat dissipation, power failure, and overstrain",
        },
    }
