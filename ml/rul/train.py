"""
MaintenAI— Remaining Useful Life (RUL) Training Pipeline (DVC Entrypoint)

Trains and evaluates:
  - XGBoost RUL Regressor (tabular baseline)
  - PyTorch Transformer RUL Regressor (sequence-to-scalar deep architecture)

Metrics tracked: RMSE, MAE, R-squared, NASA asymmetric scoring function (C-MAPSS benchmark).

Usage:
    python -m ml.rul.train --model xgboost
    python -m ml.rul.train --model transformer
    python -m ml.rul.train --model all
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.rul.transformer_model import TransformerRULRegressor
from ml.rul.xgboost_model import XGBoostRULRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed/cmapss_features")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_OUTPUT_DIR = Path("models/rul")


def load_params(params_path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load hyperparameters from params.yaml."""
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("train", {}).get("rul", {})


def load_cmapss_feature_data(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load C-MAPSS processed feature datasets."""
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    manifest_path = data_dir / "feature_manifest.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {data_dir}. Run 'make featurize-cmapss' first."
        )

    df_train = pd.read_parquet(train_path)
    df_val = (
        pd.read_parquet(val_path) if val_path.exists() else df_train.tail(int(len(df_train) * 0.15))
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
        exclude_cols = {"machine_id", "timestamp", "rul", "operating_cycle"}
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    return df_train, df_val, df_test, feature_cols


def compute_nasa_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric score metric.
    Penalizes late predictions (predicted > true) more heavily than early predictions.
    d = y_pred - y_true
    s_i = exp(-d_i / 13) - 1  if d_i < 0 (early prediction)
    s_i = exp(d_i / 10) - 1   if d_i >= 0 (late prediction)
    """
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def evaluate_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard RUL regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = compute_nasa_scoring(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "nasa_score": round(nasa_score, 2),
    }


def train_xgboost(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train and evaluate XGBoost RUL regressor."""
    xgb_params = params.get("xgboost", {})
    regressor = XGBoostRULRegressor(
        n_estimators=xgb_params.get("n_estimators", 500),
        max_depth=xgb_params.get("max_depth", 6),
        learning_rate=xgb_params.get("learning_rate", 0.05),
        subsample=xgb_params.get("subsample", 0.8),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
        min_child_weight=xgb_params.get("min_child_weight", 3.0),
        random_state=xgb_params.get("random_state", 42),
        early_stopping_rounds=xgb_params.get("early_stopping_rounds", 20),
    )

    X_train = df_train[feature_cols]
    y_train = df_train["rul"]
    X_val = df_val[feature_cols]
    y_val = df_val["rul"]

    regressor.fit(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    X_test = df_test[feature_cols]
    y_test = df_test["rul"].values
    preds = regressor.predict(X_test)

    metrics = evaluate_rul_predictions(y_test, preds)

    # Save model artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost.joblib"
    regressor.save(model_path)

    # Save metrics for DVC
    metrics_path = Path("ml/rul/metrics_xgboost.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("XGBoost RUL Test Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def train_transformer(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train and evaluate PyTorch Transformer RUL regressor."""
    tr_params = params.get("transformer", {})
    regressor = TransformerRULRegressor(
        sequence_length=tr_params.get("sequence_length", 30),
        d_model=tr_params.get("d_model", 64),
        nhead=tr_params.get("nhead", 4),
        num_encoder_layers=tr_params.get("num_encoder_layers", 3),
        dim_feedforward=tr_params.get("dim_feedforward", 128),
        dropout=tr_params.get("dropout", 0.1),
        learning_rate=tr_params.get("learning_rate", 0.0005),
        epochs=tr_params.get("epochs", 100),
        batch_size=tr_params.get("batch_size", 128),
        patience=tr_params.get("patience", 15),
        scheduler=tr_params.get("scheduler", "cosine"),
    )

    regressor.fit(df_train, feature_cols, df_val, target_col="rul")

    # Evaluate on test set
    y_test = df_test["rul"].values
    preds = regressor.predict(df_test)

    # Align lengths if needed
    if len(preds) != len(y_test):
        min_len = min(len(preds), len(y_test))
        preds = preds[:min_len]
        y_test = y_test[:min_len]

    metrics = evaluate_rul_predictions(y_test, preds)

    # Save model artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "transformer.joblib"
    regressor.save(model_path)

    # Save metrics for DVC
    metrics_path = Path("ml/rul/metrics_transformer.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Transformer RUL Test Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


@click.command()
@click.option(
    "--model",
    "model_type",
    type=click.Choice(["xgboost", "transformer", "all"]),
    default="xgboost",
    help="RUL model architecture",
)
@click.option(
    "--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to C-MAPSS feature dataset directory"
)
@click.option("--params-file", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml")
@click.option(
    "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for model weights"
)
def main(model_type: str, data_dir: str, params_file: str, output_dir: str) -> None:
    """Train RUL prediction models."""
    params = load_params(Path(params_file))
    df_train, df_val, df_test, feature_cols = load_cmapss_feature_data(Path(data_dir))
    out_dir = Path(output_dir)

    logger.info(
        "Loaded C-MAPSS feature dataset. Features: %d, Train: %d, Val: %d, Test: %d",
        len(feature_cols),
        len(df_train),
        len(df_val),
        len(df_test),
    )

    if model_type in ("xgboost", "all"):
        logger.info("--- Training XGBoost RUL Model ---")
        train_xgboost(df_train, df_val, df_test, feature_cols, params, out_dir)

    if model_type in ("transformer", "all"):
        logger.info("--- Training Transformer RUL Model ---")
        train_transformer(df_train, df_val, df_test, feature_cols, params, out_dir)


if __name__ == "__main__":
    main()
