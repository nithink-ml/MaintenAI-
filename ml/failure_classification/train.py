"""
MaintenAI— Failure Mode Classification Training Pipeline (DVC Entrypoint)

Trains and evaluates LightGBM multi-mode failure classifier on the AI4I 2020 dataset.
Tracks Precision, Recall, Macro-F1, ROC-AUC, and PR-AUC.

Usage:
    python -m ml.failure_classification.train --model lightgbm
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
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.failure_classification.lightgbm_model import (
    FAILURE_MODES,
    LightGBMFailureClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed/ai4i_features")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_OUTPUT_DIR = Path("models/failure")


def load_params(params_path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load hyperparameters from params.yaml."""
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("train", {}).get("failure", {})


def load_ai4i_feature_data(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load AI4I processed feature splits."""
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    manifest_path = data_dir / "feature_manifest.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {data_dir}. Run 'make featurize-ai4i' first."
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
        exclude_cols = {
            "machine_id",
            "timestamp",
            "failure_label",
            "failure_tool_wear",
            "failure_heat_dissipation",
            "failure_power_failure",
            "failure_overstrain",
            "failure_random",
            "failure_mode",
            "product_type",
        }
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    return df_train, df_val, df_test, feature_cols


def evaluate_failure_classifier(
    classifier: LightGBMFailureClassifier,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Evaluate overall failure detection and per-mode performance."""
    X_test = df_test[feature_cols]
    y_test_bin = df_test["failure_label"].values

    prob_bin = classifier.predict_failure_probability(X_test)
    pred_bin = (prob_bin >= 0.5).astype(int)

    metrics: dict[str, Any] = {
        "precision": float(precision_score(y_test_bin, pred_bin, zero_division=0)),
        "recall": float(recall_score(y_test_bin, pred_bin, zero_division=0)),
        "f1": float(f1_score(y_test_bin, pred_bin, zero_division=0)),
        "f1_macro": float(f1_score(y_test_bin, pred_bin, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_test_bin, pred_bin, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test_bin, prob_bin))
        if len(np.unique(y_test_bin)) > 1
        else 1.0,
        "pr_auc": float(average_precision_score(y_test_bin, prob_bin))
        if len(np.unique(y_test_bin)) > 1
        else 1.0,
        "modes": {},
    }

    mode_probs = classifier.predict_mode_probabilities(X_test)
    for mode, probs in mode_probs.items():
        if mode in df_test.columns:
            y_true_m = df_test[mode].values
            p_m = (probs >= 0.5).astype(int)
            metrics["modes"][mode] = {
                "f1": float(f1_score(y_true_m, p_m, zero_division=0)),
                "precision": float(precision_score(y_true_m, p_m, zero_division=0)),
                "recall": float(recall_score(y_true_m, p_m, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true_m, probs))
                if len(np.unique(y_true_m)) > 1
                else 1.0,
            }

    return metrics


def train_lightgbm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train and evaluate LightGBM failure classifier."""
    lgb_params = params.get("lightgbm", {})
    classifier = LightGBMFailureClassifier(
        n_estimators=lgb_params.get("n_estimators", 500),
        max_depth=lgb_params.get("max_depth", 8),
        learning_rate=lgb_params.get("learning_rate", 0.05),
        num_leaves=lgb_params.get("num_leaves", 63),
        min_child_samples=lgb_params.get("min_child_samples", 20),
        subsample=lgb_params.get("subsample", 0.8),
        colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
        class_weight=lgb_params.get("class_weight", "balanced"),
        random_state=lgb_params.get("random_state", 42),
        early_stopping_rounds=lgb_params.get("early_stopping_rounds", 20),
    )

    X_train = df_train[feature_cols]
    y_train_bin = df_train["failure_label"]
    y_train_modes = df_train[[m for m in FAILURE_MODES if m in df_train.columns]]

    X_val = df_val[feature_cols]
    y_val_bin = df_val["failure_label"]
    y_val_modes = df_val[[m for m in FAILURE_MODES if m in df_val.columns]]

    classifier.fit(
        X_train=X_train,
        y_train_binary=y_train_bin,
        y_train_modes=y_train_modes,
        X_val=X_val,
        y_val_binary=y_val_bin,
        y_val_modes=y_val_modes,
    )

    metrics = evaluate_failure_classifier(classifier, df_test, feature_cols)

    # Save model artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lightgbm.joblib"
    classifier.save(model_path)

    # Save metrics for DVC
    metrics_path = Path("ml/failure_classification/metrics_lgbm.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("LightGBM Failure Classifier Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


@click.command()
@click.option(
    "--model",
    "model_type",
    type=click.Choice(["lightgbm"]),
    default="lightgbm",
    help="Failure classification model type",
)
@click.option(
    "--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to AI4I feature dataset directory"
)
@click.option("--params-file", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml")
@click.option(
    "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for model weights"
)
def main(model_type: str, data_dir: str, params_file: str, output_dir: str) -> None:
    """Train failure mode classification model."""
    params = load_params(Path(params_file))
    df_train, df_val, df_test, feature_cols = load_ai4i_feature_data(Path(data_dir))
    out_dir = Path(output_dir)

    logger.info(
        "Loaded AI4I dataset. Features: %d, Train: %d, Val: %d, Test: %d",
        len(feature_cols),
        len(df_train),
        len(df_val),
        len(df_test),
    )
    train_lightgbm(df_train, df_val, df_test, feature_cols, params, out_dir)


if __name__ == "__main__":
    main()
