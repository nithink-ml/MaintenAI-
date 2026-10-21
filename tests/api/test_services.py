"""
MaintenAI— Microservices API Test Suite

Tests Simulator, Ingestion Service, Inference Service, Alert Engine,
Copilot API, and API Gateway routing.
"""

from __future__ import annotations

from datetime import UTC

import pytest
from fastapi.testclient import TestClient

from services.alert_service.engine import AlertRuleEngine
from services.api_gateway.app import app as gateway_app
from services.copilot.app import app as copilot_app
from services.inference.app import app as inference_app
from services.ingestion.app import app as ingestion_app
from services.simulator.engine import MachineSimulator


@pytest.mark.api
def test_simulator_engine():
    sim = MachineSimulator("turbofan-001", equipment_type="turbofan", initial_cycle=10)
    reading = sim.generate_reading()
    assert reading["machine_id"] == "turbofan-001"
    assert reading["operating_cycle"] == 11
    assert "temperature" in reading
    assert "vibration" in reading

    # Fault injection test
    sim.inject_fault("thermal_runaway", severity=0.8)
    f_reading = sim.generate_reading()
    assert f_reading["fault_injected"] == "thermal_runaway"


@pytest.mark.api
def test_ingestion_api():
    client = TestClient(ingestion_app)

    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200

    from datetime import datetime

    now_str = datetime.now(UTC).isoformat()

    # Ingest single reading
    payload = {
        "machine_id": "turbofan-001",
        "timestamp": now_str,
        "operating_cycle": 100,
        "temperature": 620.5,
        "pressure": 515.2,
        "rpm": 2400.0,
        "vibration": 2.1,
        "current": 44.0,
    }
    resp = client.post("/api/v1/telemetry", json=payload)
    assert resp.status_code == 202

    # Fetch latest
    resp = client.get("/api/v1/telemetry/latest/turbofan-001")
    assert resp.status_code == 200
    assert resp.json()["machine_id"] == "turbofan-001"


@pytest.mark.api
def test_inference_api():
    client = TestClient(inference_app)

    payload = {
        "machine_id": "turbofan-001",
        "operating_cycle": 120,
        "features": {
            "temperature": 640.0,
            "vibration": 3.2,
            "pressure": 510.0,
        },
    }

    # Anomaly endpoint
    resp = client.post("/api/v1/predict/anomaly", json=payload)
    assert resp.status_code == 200
    assert "anomaly_score" in resp.json()

    # RUL endpoint
    resp = client.post("/api/v1/predict/rul", json=payload)
    assert resp.status_code == 200
    assert "predicted_rul_cycles" in resp.json()

    # Health Assessment consolidated endpoint
    resp = client.post("/api/v1/predict/health_assessment", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "anomaly_score" in data
    assert "predicted_rul_cycles" in data
    assert "probable_failure_modes" in data


@pytest.mark.api
def test_alert_engine_hysteresis_and_dedup():
    engine = AlertRuleEngine(
        anomaly_warning=0.70, anomaly_critical=0.85, hysteresis_cycles=2, dedup_window_minutes=15
    )

    # Cycle 1: High anomaly (should not trigger yet due to hysteresis=2)
    health_1 = {"machine_id": "turbofan-001", "anomaly_score": 0.88, "predicted_rul_cycles": 50.0}
    alerts_1 = engine.evaluate_machine(health_1)
    assert len(alerts_1) == 0

    # Cycle 2: High anomaly persisted for 2 cycles (now should trigger)
    health_2 = {"machine_id": "turbofan-001", "anomaly_score": 0.89, "predicted_rul_cycles": 50.0}
    alerts_2 = engine.evaluate_machine(health_2)
    assert len(alerts_2) == 1
    assert alerts_2[0]["severity"] == "CRITICAL"

    # Cycle 3: Should be deduplicated
    alerts_3 = engine.evaluate_machine(health_2)
    assert len(alerts_3) == 0


@pytest.mark.api
def test_copilot_service_and_work_orders():
    client = TestClient(copilot_app)

    # Chat endpoint
    resp = client.post("/api/v1/copilot/chat", json={"message": "Assess turbofan-001 health"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert data["work_order"] is not None

    wo_id = data["work_order"]["work_order_id"]

    # Approve work order
    appr_resp = client.post(
        "/api/v1/copilot/work_orders/approve",
        json={"work_order_id": wo_id, "technician_name": "Senior Engineer", "action": "APPROVE"},
    )
    assert appr_resp.status_code == 200
    assert appr_resp.json()["work_order"]["status"] == "APPROVED"


@pytest.mark.api
def test_api_gateway_routing():
    client = TestClient(gateway_app)

    # Root health
    resp = client.get("/health")
    assert resp.status_code == 200

    # Fleet status endpoint
    resp = client.get("/api/v1/fleet/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "fleet" in data
    assert data["total_machines"] > 0
