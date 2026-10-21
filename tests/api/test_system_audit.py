"""
MaintenAI— Final System Audit & All-Issues Verification Tests

Exhaustively verifies:
  1. Authoritative runtime model metadata (HistGradientBoostingRegressor fallback, IsolationForest, HistGradientBoostingClassifier)
  2. Dynamic live system status checks (Gateway, TimescaleDB hypertable, Redis stream, ML runtime)
  3. Machine health payload metadata and model versions
  4. Calibrated RAG relevance scores in [0.0, 1.0] and grounding guardrails
  5. Work order safety authorization gates with JWT RBAC enforcement
  6. Feature parity and zero data leakage across machine boundaries
  7. Copilot structured report demarcation (FACT vs ML PREDICTION vs RAG EVIDENCE vs RECOMMENDATION)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents.maintenance_agent.guardrails import validate_grounding
from agents.maintenance_agent.tools import search_maintenance_manuals
from rag.retrieval.hybrid import HybridRetriever
from services.api_gateway.app import app
from services.feature_service.calculator import OnlineFeatureStore


@pytest.fixture
def client():
    return TestClient(app)


def _get_auth_token(client: TestClient, username: str) -> str:
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "FactoryPulse2026!"},
    )
    assert login_resp.status_code == 200
    return login_resp.json()["access_token"]


# ======================================================================
# 1. Authoritative Model Metadata & Fallback Transparency (Issues #1, #7, #26, #29)
# ======================================================================
def test_models_metadata_authoritative_source(client):
    resp = client.get("/api/v1/models/metadata")
    assert resp.status_code == 200
    meta = resp.json()

    assert "anomaly" in meta
    assert "rul" in meta
    assert "failure_classifier" in meta

    # Anomaly
    assert meta["anomaly"]["model_family"] == "IsolationForest"
    assert meta["anomaly"]["status"] == "active"
    assert meta["anomaly"]["is_fallback"] is False

    # RUL Fallback
    assert meta["rul"]["model_family"] == "HistGradientBoostingRegressor"
    assert meta["rul"]["is_fallback"] is True
    assert "OpenMP" in meta["rul"]["fallback_reason"] or "segfault" in meta["rul"]["fallback_reason"]

    # Failure Classifier Fallback
    assert meta["failure_classifier"]["model_family"] == "HistGradientBoostingClassifier"
    assert meta["failure_classifier"]["is_fallback"] is True


# ======================================================================
# 2. Dynamic Live System Status Indicators (Issues #17, #18, #27)
# ======================================================================
def test_system_status_dynamic_checks(client):
    resp = client.get("/api/v1/system/status")
    assert resp.status_code == 200
    status_data = resp.json()

    assert status_data["simulation_mode"] == "DEVELOPMENT / SYNTHETIC SIMULATION"
    services = status_data["services"]

    # Gateway
    assert services["gateway"]["status"] == "online"
    assert services["gateway"]["port"] == 8000

    # TimescaleDB
    assert services["timescaledb"]["status"] in ("online", "offline")
    if services["timescaledb"]["status"] == "online":
        assert services["timescaledb"]["hypertable"] == "sensor_readings"
        assert services["timescaledb"]["total_records"] >= 0

    # Redis
    assert services["redis"]["status"] in ("online", "offline")
    if services["redis"]["status"] == "online":
        assert services["redis"]["authenticated"] is True
        assert services["redis"]["stream_state"] == "100Hz STREAM"

    # ML Runtime
    assert services["ml_runtime"]["status"] in ("active", "fallback_active")

    # Simulation Notice
    assert services["simulation"]["status"] == "active"
    assert "simulation" in services["simulation"]["notice"].lower()


# ======================================================================
# 3. Machine Health Payload & Model Versions (Issues #3, #4, #7)
# ======================================================================
def test_machine_health_includes_model_metadata(client):
    resp = client.get("/api/v1/machines/m001/health")
    assert resp.status_code == 200
    health = resp.json()

    assert health["machine_id"] == "M001"
    assert 0.0 <= health["health_score"] <= 1.0

    # Model versions
    assert health["anomaly"]["model_version"] == "isolation_forest_v1"
    assert health["rul"]["model_version"] == "hist_gradient_boosting_rul_v1"
    assert health["rul"]["is_fallback"] is True
    assert health["failure_mode"]["model_version"] == "hist_gradient_boosting_failure_v1"
    assert health["failure_mode"]["is_fallback"] is True

    # Model metadata block
    assert "model_metadata" in health
    assert health["model_metadata"]["rul"]["model_family"] == "HistGradientBoostingRegressor"


# ======================================================================
# 4. Calibrated RAG Relevance Scores in [0.0, 1.0] (Issue #8)
# ======================================================================
def test_rag_score_calibration():
    evidence = search_maintenance_manuals(
        query="turbofan high pressure compressor degradation",
        equipment_type="turbofan",
        top_k=3,
    )
    assert len(evidence) > 0

    for doc in evidence:
        score = doc["relevance_score"]
        # Score must be normalized in [0.0, 1.0]
        assert 0.0 <= score <= 1.0
        # Relevant turbofan manual docs should have strong calibrated score (>= 0.35)
        assert score >= 0.35

    # Test ungrounded or low-relevance query
    retriever = HybridRetriever()
    from rag.chunking.markdown_chunker import DocumentChunk

    dummy_chunks = [
        DocumentChunk(
            chunk_id="chk-1",
            source_file="manual.md",
            section_title="Turbofan Bearings",
            content="Turbofan bearing lubrication procedures",
            breadcrumbs=["Turbofan", "Bearings"],
            equipment_type="turbofan",
        )
    ]
    retriever.index(dummy_chunks)
    irrelevant_res = retriever.retrieve("completely unrelated topic about coffee brewing", top_k=1)
    if irrelevant_res:
        _, low_score = irrelevant_res[0]
        assert low_score < 0.35


# ======================================================================
# 5. Work Order Safety Authorization & RBAC (Issue #11)
# ======================================================================
def test_work_order_approval_rbac_enforcement(client):
    # 1. Create a draft work order (must be created as pending_approval)
    draft_payload = {
        "machine_id": "M001",
        "issue_summary": "Bearing high vibration spike in outer race",
        "recommended_actions": ["Replace bearing assembly", "Re-lubricate race"],
        "priority": "critical",
        "estimated_downtime_hours": 3.0,
    }
    create_resp = client.post("/api/v1/work-orders", json=draft_payload)
    assert create_resp.status_code == 201
    wo = create_resp.json()
    assert wo["status"] == "pending_approval"
    wo_id = wo["id"]

    # 2. Attempt approval as 'technician' (must be rejected with 403 Forbidden)
    tech_token = _get_auth_token(client, "technician")
    tech_approve = client.post(
        f"/api/v1/work-orders/{wo_id}/approve",
        headers={"Authorization": f"Bearer {tech_token}"},
        json={"approval_notes": "Attempting approval as technician"},
    )
    assert tech_approve.status_code == 403
    assert "Forbidden" in tech_approve.json()["detail"]

    # 3. Attempt approval as 'engineer' (must succeed with 200 OK)
    eng_token = _get_auth_token(client, "engineer")
    eng_approve = client.post(
        f"/api/v1/work-orders/{wo_id}/approve",
        headers={"Authorization": f"Bearer {eng_token}"},
        json={"approval_notes": "Authorized by Reliability Engineer."},
    )
    assert eng_approve.status_code == 200
    approved_wo = eng_approve.json()
    assert approved_wo["status"] == "approved"
    assert "engineer" in approved_wo["approved_by"].lower()


# ======================================================================
# 6. Feature Parity & No Temporal Data Leakage (Issues #5, #6)
# ======================================================================
def test_feature_parity_and_no_data_leakage():
    store = OnlineFeatureStore(window_size=10)

    # Ingest historical readings for M001
    for i in range(10):
        reading = {
            "machine_id": "M001",
            "temperature": 68.0 + (i * 0.5),
            "pressure": 101.3,
            "rpm": 2000.0,
            "vibration": 1.1 + (i * 0.1),
            "current": 40.0,
        }
        store.append_reading(reading)

    features = store.compute_features("M001")
    assert features is not None

    # Temporal feature checks
    assert "temperature_rolling_mean_5" in features
    assert "vibration_rolling_std_5" in features
    assert "temperature_delta_1" in features

    # Delta-1 must be current reading minus previous reading (no future leakage)
    assert features["temperature"] == pytest.approx(72.5, abs=1e-3)
    assert features["temperature_delta_1"] == pytest.approx(0.5, abs=1e-3)

    # Cross-machine isolation: machine M002 should have no features
    assert store.compute_features("M002") is None


# ======================================================================
# 7. Copilot Structured Report Demarcation (Issues #9, #10, #19)
# ======================================================================
def test_copilot_structured_report_demarcation(client):
    copilot_req = {
        "message": "Perform diagnostic evaluation on turbofan-001 and check HPC degradation.",
        "machine_id": "turbofan-001",
    }
    resp = client.post("/api/v1/copilot/query", json=copilot_req)
    assert resp.status_code == 200
    chat_data = resp.json()

    text = chat_data["response"]

    # Must explicitly demarcate sections
    assert "[MEASURED FACT" in text
    assert "[ML PREDICTION" in text
    assert "[RAG EVIDENCE" in text
    assert "[AI INTERPRETATION" in text
    assert "PENDING_APPROVAL" in text

    # Guardrails validation
    is_valid, violations = validate_grounding(
        response_text=text,
        ml_health=chat_data.get("ml_health"),
        retrieved_docs=chat_data.get("retrieved_docs", []),
    )
    assert is_valid, f"Guardrail violations: {violations}"
