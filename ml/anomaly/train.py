"""
MaintenAI— Anomaly Detection Model Training Pipeline (DVC Entrypoint)

Supports training and evaluating:
  - Isolation Forest (baseline unsupervised detector)
  - LSTM Autoencoder (production deep-learning temporal reconstruction detector)

Usage:
    python -m ml.anomaly.train --model isolation_forest
    python -m ml.anomaly.train --model lstm_autoencoder
    python -m ml.anomaly.train --model all
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from ml.anomaly.lstm_autoencoder import LSTMAutoencoderDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed/ims_features")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_OUTPUT_DIR = Path("models/anomaly")


def load_params(params_path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load training hyperparameters from params.yaml."""
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("train", {}).get("anomaly", {})


def load_ims_feature_data(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load IMS bearing processed features for train, val, and test splits."""
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    manifest_path = data_dir / "feature_manifest.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {data_dir}. Run 'make featurize-ims' first."
        )

    df_train = pd.read_parquet(train_path)
    df_val = (
        pd.read_parquet(val_path) if val_path.exists() else df_train.tail(int(len(df_train) * 0.2))
    )
    df_test = pd.read_parquet(test_path) if test_path.exists() else df_val

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        feature_cols = [
            item["name"]
            for item in manifest.get("features", [])
            if item["name"] in df_train.columns
        ]
    else:
        exclude_cols = {"machine_id", "timestamp", "anomaly_label", "operating_cycle"}
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    return df_train, df_val, df_test, feature_cols


def evaluate_anomaly_model(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Compute standard anomaly detection metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC if both classes present
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            metrics["roc_auc"] = 0.5

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["fpr"] = float(fp / (fp + tn + 1e-8))
    else:
        metrics["roc_auc"] = 1.0
        metrics["fpr"] = 0.0

    return metrics


def train_isolation_forest(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train and evaluate Isolation Forest anomaly detector."""
    iforest_params = params.get("isolation_forest", {})
    detector = IsolationForestAnomalyDetector(
        n_estimators=iforest_params.get("n_estimators", 200),
        contamination=iforest_params.get("contamination", 0.05),
        max_features=iforest_params.get("max_features", 1.0),
        random_state=iforest_params.get("random_state", 42),
    )

    X_train = df_train[feature_cols]
    detector.fit(X_train)

    # Evaluate on test set
    X_test = df_test[feature_cols]
    y_test = (
        df_test["anomaly_label"].values
        if "anomaly_label" in df_test.columns
        else np.zeros(len(df_test))
    )
    scores = detector.predict_score(X_test)
    preds = detector.predict(X_test)

    metrics = evaluate_anomaly_model(y_test, preds, scores)
    metrics["threshold"] = float(detector.threshold_)

    # Save model artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "isolation_forest.joblib"
    detector.save(model_path)

    # Save metrics for DVC
    metrics_path = Path("ml/anomaly/metrics_iforest.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Isolation Forest Test Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def train_lstm_autoencoder(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train and evaluate LSTM Autoencoder anomaly detector."""
    lstm_params = params.get("lstm_autoencoder", {})
    detector = LSTMAutoencoderDetector(
        sequence_length=lstm_params.get("sequence_length", 50),
        hidden_dim=lstm_params.get("hidden_dim", 64),
        num_layers=lstm_params.get("num_layers", 2),
        dropout=lstm_params.get("dropout", 0.2),
        learning_rate=lstm_params.get("learning_rate", 0.001),
        epochs=lstm_params.get("epochs", 50),
        batch_size=lstm_params.get("batch_size", 64),
        patience=lstm_params.get("patience", 10),
        reconstruction_percentile=lstm_params.get("reconstruction_percentile", 95.0),
    )

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]
    detector.fit(X_train, X_val)

    # Evaluate on test set
    X_test = df_test[feature_cols]
    y_test = (
        df_test["anomaly_label"].values
        if "anomaly_label" in df_test.columns
        else np.zeros(len(df_test))
    )
    scores = detector.predict_score(X_test)
    preds = detector.predict(X_test)

    metrics = evaluate_anomaly_model(y_test, preds, scores)
    metrics["threshold"] = float(detector.threshold_)

    # Save model artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lstm_autoencoder.joblib"
    detector.save(model_path)

    # Save metrics for DVC
    metrics_path = Path("ml/anomaly/metrics_lstm.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("LSTM Autoencoder Test Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


@click.command()
@click.option(
    "--model",
    "model_type",
    type=click.Choice(["isolation_forest", "lstm_autoencoder", "all"]),
    default="isolation_forest",
    help="Model type to train",
)
@click.option(
    "--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to IMS feature dataset directory"
)
@click.option("--params-file", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml")
@click.option(
    "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for model weights"
)
def main(model_type: str, data_dir: str, params_file: str, output_dir: str) -> None:
    """Train anomaly detection models."""
    params = load_params(Path(params_file))
    df_train, df_val, df_test, feature_cols = load_ims_feature_data(Path(data_dir))
    out_dir = Path(output_dir)

    logger.info(
        "Loaded dataset with %d features. Train: %d, Val: %d, Test: %d",
        len(feature_cols),
        len(df_train),
        len(df_val),
        len(df_test),
    )

    if model_type in ("isolation_forest", "all"):
        logger.info("--- Training Isolation Forest ---")
        train_isolation_forest(df_train, df_val, df_test, feature_cols, params, out_dir)

    if model_type in ("lstm_autoencoder", "all"):
        logger.info("--- Training LSTM Autoencoder ---")
        train_lstm_autoencoder(df_train, df_val, df_test, feature_cols, params, out_dir)


if __name__ == "__main__":
    main()
