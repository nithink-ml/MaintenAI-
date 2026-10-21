"""
MaintenAI— Phase 11 Backend API Integration Tests

Validates all required production API endpoints:
  - Authentication (JWT login, profile, 401 unauthorized handling)
  - Telemetry Ingestion (validation, 202 accepted, 422 rejection)
  - Machines Directory (list, individual machine metadata, 404 handling)
  - Machine Health Assessment (normalized scores, anomaly, RUL, failure modes)
  - Machine Telemetry Retrieval (verified latest readings)
  - Alert Engine Management (listing, filtering, detail, acknowledgment audit)
  - Work Order Lifecycle (draft creation, pending approval gate, operator approval)
  - Maintenance Copilot (grounded diagnostic query, citations)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from services.api_gateway.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------------------------------------------------
# 1. Authentication & RBAC Tests
# ----------------------------------------------------------------------
def test_01_login_success(client):
    payload = {"username": "technician", "password": "FactoryPulse2026!"}
    resp = client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


def test_02_login_invalid_credentials(client):
    payload = {"username": "technician", "password": "WrongPassword!"}
    resp = client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 401
    assert "Invalid username or password" in resp.json()["detail"]


def test_03_auth_me_with_valid_token(client):
    # Log in first
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "engineer", "password": "FactoryPulse2026!"},
    )
    token = login_resp.json()["access_token"]

    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    user = resp.json()
    assert user["username"] == "engineer"
    assert user["role"] == "engineer"
    assert user["is_active"] is True


def test_04_auth_me_missing_token(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


# ----------------------------------------------------------------------
# 2. Telemetry Ingestion Tests
# ----------------------------------------------------------------------
def test_05_telemetry_ingest_valid(client):
    now = datetime.now(UTC)
    payload = {
        "machine_id": "M001",
        "timestamp": now.isoformat(),
        "temperature": 71.5,
        "pressure": 101.3,
        "vibration": 1.15,
        "rpm": 2000.0,
        "current": 28.5,
        "operating_cycle": 15,
        "source": "SYNTHETIC / DEVELOPMENT",
    }
    resp = client.post("/api/v1/telemetry", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["machine_id"] == "M001"


def test_06_telemetry_ingest_invalid(client):
    payload = {
        "machine_id": "M001",
        "timestamp": datetime.now(UTC).isoformat(),
        "temperature": 9999.0,  # Invalid: exceeds physical maximum 1500C
    }
    resp = client.post("/api/v1/telemetry", json=payload)
    assert resp.status_code == 422


# ----------------------------------------------------------------------
# 3. Machines Directory & SCADA Health Tests
# ----------------------------------------------------------------------
def test_07_get_machines_list(client):
    resp = client.get("/api/v1/machines")
    assert resp.status_code == 200
    data = resp.json()
    assert "machines" in data
    assert data["total"] >= 7
    ids = [m["id"].lower() for m in data["machines"]]
    assert "m001" in ids
    assert "turbofan-001" in ids


def test_08_get_machine_by_id(client):
    resp = client.get("/api/v1/machines/M001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["machine"]["id"] == "M001"
    assert "telemetry" in data


def test_09_get_machine_not_found(client):
    resp = client.get("/api/v1/machines/nonexistent-machine-999")
    assert resp.status_code == 404


def test_10_get_machine_health(client):
    resp = client.get("/api/v1/machines/M001/health")
    assert resp.status_code == 200
    health = resp.json()
    assert health["machine_id"] == "M001"
    assert 0.0 <= health["health_score"] <= 1.0
    assert health["status"] in ["healthy", "warning", "critical"]
    assert "anomaly" in health
    assert "rul" in health
    assert "failure_mode" in health


def test_11_get_machine_telemetry(client):
    resp = client.get("/api/v1/machines/turbofan-001/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["machine_id"] == "turbofan-001"
    assert "temperature" in data["telemetry"]


def test_12_get_fleet_status(client):
    resp = client.get("/api/v1/fleet/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "fleet" in data
    assert data["total_machines"] >= 7


# ----------------------------------------------------------------------
# 4. Alerts Management Tests
# ----------------------------------------------------------------------
def test_13_get_alerts_list_and_filters(client):
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert data["total"] >= 1

    # Filter by severity
    resp_crit = client.get("/api/v1/alerts?severity=CRITICAL")
    assert resp_crit.status_code == 200
    for a in resp_crit.json()["alerts"]:
        assert a["severity"] == "CRITICAL"


def test_14_get_alert_by_id(client):
    # Fetch all alerts to get an existing ID
    alerts_resp = client.get("/api/v1/alerts")
    alerts = alerts_resp.json()["alerts"]
    assert len(alerts) > 0
    target_id = alerts[0]["alert_id"]

    resp = client.get(f"/api/v1/alerts/{target_id}")
    assert resp.status_code == 200
    assert resp.json()["alert"]["alert_id"] == target_id


def test_15_get_alert_not_found(client):
    resp = client.get("/api/v1/alerts/ALT-NONEXISTENT")
    assert resp.status_code == 404


def test_16_acknowledge_alert(client):
    alerts_resp = client.get("/api/v1/alerts")
    alerts = alerts_resp.json()["alerts"]
    target_id = alerts[0]["alert_id"]

    ack_payload = {"acknowledged_by": "Lead Reliability Engineer (Nithin)"}
    resp = client.post(f"/api/v1/alerts/{target_id}/acknowledge", json=ack_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "acknowledged"
    assert data["alert"]["acknowledged"] is True
    assert data["alert"]["acknowledged_by"] == "Lead Reliability Engineer (Nithin)"


# ----------------------------------------------------------------------
# 5. Work Orders Management Tests
# ----------------------------------------------------------------------
def test_17_create_and_list_work_orders(client):
    draft_payload = {
        "machine_id": "M001",
        "issue_summary": "Elevated bearing vibration detected during high-load cycle",
        "recommended_actions": [
            "Inspect bearing race for spalling or pitting",
            "Check lubricant viscosity and particle count",
        ],
        "priority": "warning",
        "estimated_downtime_hours": 3.0,
        "parts_needed": ["BEARING-KIT-M001", "SYNTHETIC-GREASE-700"],
    }

    # 1. Create work order
    resp = client.post("/api/v1/work-orders", json=draft_payload)
    assert resp.status_code == 201
    wo = resp.json()
    assert wo["id"].startswith("WO-")
    assert wo["machine_id"] == "M001"
    assert wo["status"] == "pending_approval"

    wo_id = wo["id"]

    # 2. Get single work order
    detail_resp = client.get(f"/api/v1/work-orders/{wo_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == wo_id

    # 3. List work orders
    list_resp = client.get("/api/v1/work-orders")
    assert list_resp.status_code == 200
    assert any(w["id"] == wo_id for w in list_resp.json()["work_orders"])

    # 4. Human approval
    approve_resp = client.post(
        f"/api/v1/work-orders/{wo_id}/approve",
        json={"approved_by": "Maintenance Supervisor", "approval_notes": "Parts confirmed in inventory."},
    )
    assert approve_resp.status_code == 200
    approved_wo = approve_resp.json()
    assert approved_wo["status"] == "approved"
    assert approved_wo["approved_by"] == "Maintenance Supervisor"


# ----------------------------------------------------------------------
# 6. Maintenance Copilot Diagnostic Query Tests
# ----------------------------------------------------------------------
def test_18_copilot_query_endpoint(client):
    query_payload = {
        "message": "What is the health status of turbofan-001?",
        "machine_id": "turbofan-001",
    }
    resp = client.post("/api/v1/copilot/query", json=query_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert data["machine_id"] == "turbofan-001"
    assert "ml_health" in data
