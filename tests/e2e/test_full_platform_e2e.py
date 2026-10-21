"""
MaintenAI— End-to-End (E2E) Platform Test Suite

Validates the complete unidirectional AI engineering lifecycle:
  Telemetry Stream -> Ingestion -> Online Feature Extraction -> ML Multi-Model Serving
  -> Alert Engine -> Hybrid RAG Retrieval -> LangGraph Agent -> Work Order Authorization
"""

from __future__ import annotations

import datetime

import pytest
from fastapi.testclient import TestClient

from services.alert_service.engine import AlertRuleEngine
from services.api_gateway.app import app as gateway_app
from services.copilot.app import app as copilot_app
from services.feature_service.calculator import OnlineFeatureStore
from services.inference.app import app as inference_app
from services.ingestion.app import app as ingestion_app
from services.simulator.engine import MachineSimulator


@pytest.mark.e2e
def test_full_predictive_maintenance_lifecycle():
    # 1. Edge / Simulator generates telemetry with injected fault
    sim = MachineSimulator("turbofan-001", equipment_type="turbofan", initial_cycle=180)
    sim.inject_fault("thermal_runaway", severity=0.9)
    raw_reading = sim.generate_reading()

    assert raw_reading["machine_id"] == "turbofan-001"
    assert raw_reading["fault_injected"] == "thermal_runaway"

    # 2. Ingestion Service receives and validates sensor reading
    ingest_client = TestClient(ingestion_app)
    now_str = datetime.datetime.now(datetime.UTC).isoformat()
    raw_reading["timestamp"] = now_str

    ingest_resp = ingest_client.post("/api/v1/telemetry", json=raw_reading)
    assert ingest_resp.status_code == 202

    # 3. Online Feature Service computes sliding features
    feature_store = OnlineFeatureStore(window_size=30)
    for _ in range(10):
        feature_store.append_reading(sim.generate_reading())

    features = feature_store.compute_features("turbofan-001")
    assert features is not None
    assert "temperature_rolling_mean_5" in features

    # 4. Inference Service evaluates multi-model health assessment
    inf_client = TestClient(inference_app)
    inf_payload = {
        "machine_id": "turbofan-001",
        "operating_cycle": raw_reading["operating_cycle"],
        "features": features,
    }

    inf_resp = inf_client.post("/api/v1/predict/health_assessment", json=inf_payload)
    assert inf_resp.status_code == 200
    health_data = inf_resp.json()

    assert health_data["anomaly_score"] > 0.0
    assert health_data["predicted_rul_cycles"] >= 0.0
    assert "probable_failure_modes" in health_data

    # 5. Deterministic Alert Engine evaluates health data
    alert_engine = AlertRuleEngine(hysteresis_cycles=1, dedup_window_minutes=15)
    alerts = alert_engine.evaluate_machine(health_data)
    # Alerts recorded if thresholds met
    assert isinstance(alerts, list)

    # 6. Maintenance Copilot Agent diagnoses machine & queries Hybrid RAG
    copilot_client = TestClient(copilot_app)
    chat_resp = copilot_client.post(
        "/api/v1/copilot/chat",
        json={
            "message": "Run emergency diagnosis on turbofan-001 and draft work order.",
            "machine_id": "turbofan-001",
        },
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()

    assert "response" in chat_data
    assert len(chat_data["retrieved_docs"]) > 0
    assert chat_data["work_order"] is not None
    assert chat_data["requires_human_approval"] is True

    wo_id = chat_data["work_order"]["work_order_id"]
    assert chat_data["work_order"]["status"] == "PENDING_APPROVAL"

    # 7. Human Authorization Gate (Chief Engineer approves work order)
    approval_resp = copilot_client.post(
        "/api/v1/copilot/work_orders/approve",
        json={
            "work_order_id": wo_id,
            "technician_name": "Chief Engineer (Nithin)",
            "action": "APPROVE",
            "notes": "Verified borescope inspection requirement per CF6 manual.",
        },
    )
    assert approval_resp.status_code == 200
    approved_wo = approval_resp.json()["work_order"]
    assert approved_wo["status"] == "APPROVED"
    assert approved_wo["approved_by"] == "Chief Engineer (Nithin)"

    # 8. API Gateway unified fleet verification
    gateway_client = TestClient(gateway_app)
    fleet_resp = gateway_client.get("/api/v1/fleet/status")
    assert fleet_resp.status_code == 200
    assert fleet_resp.json()["total_machines"] > 0
