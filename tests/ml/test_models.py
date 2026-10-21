"""
MaintenAI— Machine Learning Models Test Suite

Tests training, scoring, persistence, and quality evaluation for:
  - Isolation Forest & LSTM Autoencoder (Anomaly Detection)
  - XGBoost & Transformer (RUL Prediction)
  - LightGBM (Failure Mode Classification)
  - Drift Detection & Evaluation Quality Gates
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from ml.anomaly.lstm_autoencoder import LSTMAutoencoderDetector
from ml.evaluation.drift import calculate_ks_test, calculate_psi
from ml.evaluation.metrics import verify_quality_gates
from ml.failure_classification.lightgbm_model import LightGBMFailureClassifier
from ml.rul.train import compute_nasa_scoring
from ml.rul.transformer_model import TransformerRULRegressor
from ml.rul.xgboost_model import XGBoostRULRegressor


@pytest.mark.ml
def test_isolation_forest_pipeline(tmp_path):
    np.random.seed(42)
    X = np.random.normal(loc=0, scale=1, size=(200, 5))

    detector = IsolationForestAnomalyDetector(n_estimators=50, contamination=0.05, random_state=42)
    detector.fit(X, feature_names=[f"sensor_{i}" for i in range(5)])

    scores = detector.predict_score(X)
    assert len(scores) == 200
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    preds = detector.predict(X)
    assert len(preds) == 200
    assert set(preds).issubset({0, 1})

    # Test explanation
    expl = detector.explain_instance(X[0])
    assert len(expl) > 0

    # Test Save & Load
    save_file = tmp_path / "iforest.joblib"
    detector.save(save_file)
    loaded = IsolationForestAnomalyDetector.load(save_file)
    loaded_scores = loaded.predict_score(X)
    np.testing.assert_allclose(scores, loaded_scores, rtol=1e-5)


@pytest.mark.ml
def test_lstm_autoencoder_pipeline(tmp_path):
    np.random.seed(42)
    X = np.random.normal(loc=0, scale=1, size=(120, 4))

    detector = LSTMAutoencoderDetector(
        sequence_length=10,
        hidden_dim=16,
        num_layers=1,
        epochs=3,
        batch_size=32,
    )
    detector.fit(X)

    scores = detector.predict_score(X)
    assert len(scores) == 120
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Test Save & Load
    save_file = tmp_path / "lstm.joblib"
    detector.save(save_file)
    loaded = LSTMAutoencoderDetector.load(save_file)
    loaded_scores = loaded.predict_score(X)
    np.testing.assert_allclose(scores, loaded_scores, rtol=1e-3)


@pytest.mark.ml
def test_xgboost_rul_regressor(tmp_path):
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 6), columns=[f"feat_{i}" for i in range(6)])
    y = np.linspace(125, 0, 100) + np.random.normal(0, 2, 100)

    reg = XGBoostRULRegressor(n_estimators=30, max_depth=3, random_state=42)
    reg.fit(X, y)

    preds = reg.predict(X)
    assert len(preds) == 100
    assert (preds >= 0.0).all()

    imps = reg.get_feature_importances()
    assert len(imps) > 0

    # Save & Load
    save_file = tmp_path / "xgb.joblib"
    reg.save(save_file)
    loaded = XGBoostRULRegressor.load(save_file)
    loaded_preds = loaded.predict(X)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)


@pytest.mark.ml
def test_transformer_rul_regressor(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(80, 5), columns=[f"feat_{i}" for i in range(5)])
    df["machine_id"] = "engine_1"
    df["rul"] = np.linspace(100, 0, 80)

    reg = TransformerRULRegressor(
        sequence_length=10,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        epochs=3,
        batch_size=16,
    )
    reg.fit(df, feature_cols=[f"feat_{i}" for i in range(5)], target_col="rul")

    preds = reg.predict(df)
    assert len(preds) == len(df)
    assert (preds >= 0.0).all()

    # Save & Load
    save_file = tmp_path / "trans.joblib"
    reg.save(save_file)
    loaded = TransformerRULRegressor.load(save_file)
    loaded_preds = loaded.predict(df)
    assert len(loaded_preds) == len(df)


@pytest.mark.ml
def test_lightgbm_failure_classifier(tmp_path):
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"sensor_{i}" for i in range(5)])
    y_bin = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    y_modes = pd.DataFrame(
        {
            "failure_tool_wear": np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
            "failure_heat_dissipation": np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
        }
    )

    clf = LightGBMFailureClassifier(n_estimators=30, max_depth=3, random_state=42)
    clf.fit(X, y_bin, y_modes)

    fail_probs = clf.predict_failure_probability(X)
    assert len(fail_probs) == 100
    assert (fail_probs >= 0.0).all() and (fail_probs <= 1.0).all()

    mode_probs = clf.predict_mode_probabilities(X)
    assert "failure_tool_wear" in mode_probs
    assert len(mode_probs["failure_tool_wear"]) == 100

    # Save & Load
    save_file = tmp_path / "lgbm.joblib"
    clf.save(save_file)
    loaded = LightGBMFailureClassifier.load(save_file)
    loaded_probs = loaded.predict_failure_probability(X)
    np.testing.assert_allclose(fail_probs, loaded_probs, rtol=1e-5)


@pytest.mark.ml
def test_evaluation_gates_and_drift():
    # NASA scoring
    y_true = np.array([50.0, 30.0, 10.0])
    y_pred = np.array([48.0, 32.0, 11.0])
    s = compute_nasa_scoring(y_true, y_pred)
    assert s >= 0.0

    # Quality Gate check
    summary = {
        "rul": {"xgboost": {"rmse": 14.2, "mae": 11.0}},
        "anomaly": {"isolation_forest": {"f1": 0.82, "fpr": 0.04}},
        "failure": {"lightgbm": {"f1_macro": 0.85, "precision_macro": 0.81}},
    }
    gates = {
        "rul": {"max_rmse": 20.0, "max_mae": 15.0},
        "anomaly": {"min_f1": 0.75, "max_fpr": 0.10},
        "failure": {"min_f1_macro": 0.80, "min_precision_macro": 0.75},
    }
    passed, violations = verify_quality_gates(summary, gates)
    assert passed is True
    assert len(violations) == 0

    # Drift check
    base_dist = np.random.normal(0, 1, 500)
    drifted_dist = np.random.normal(2, 1, 500)
    psi = calculate_psi(base_dist, drifted_dist)
    assert psi > 0.25  # Significant drift detected

    ks = calculate_ks_test(base_dist, drifted_dist)
    assert ks["p_value"] < 0.001
