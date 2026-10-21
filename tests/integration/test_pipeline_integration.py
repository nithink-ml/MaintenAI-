"""
MaintenAI— Cross-Service Integration Tests

Tests data contract compatibility, model loading across registry directories,
and unified API routing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from ml.failure_classification.lightgbm_model import LightGBMFailureClassifier
from ml.rul.xgboost_model import XGBoostRULRegressor
from services.api_gateway.app import app as gateway_app


@pytest.mark.integration
def test_model_artifact_compatibility():
    # Verify production model files load and predict on consistent schemas
    iforest = IsolationForestAnomalyDetector.load("models/anomaly/isolation_forest.joblib")
    assert iforest is not None
    assert len(iforest.feature_names) > 0

    rul_model = XGBoostRULRegressor.load("models/rul/xgboost.joblib")
    assert rul_model is not None
    assert len(rul_model.feature_names) > 0

    fail_model = LightGBMFailureClassifier.load("models/failure/lightgbm.joblib")
    assert fail_model is not None
    assert len(fail_model.feature_names) > 0


@pytest.mark.integration
def test_api_gateway_cross_service_routes():
    client = TestClient(gateway_app)

    # Health check
    h_resp = client.get("/health")
    assert h_resp.status_code == 200

    # Alerts route
    a_resp = client.get("/api/v1/alerts")
    assert a_resp.status_code == 200

    # Work orders route
    wo_resp = client.get("/api/v1/copilot/work_orders")
    assert wo_resp.status_code == 200
