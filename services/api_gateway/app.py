"""
MaintenAI— Production API Gateway Service (FastAPI)

Unified production entrypoint providing:
  - Role-based JWT authentication and authorization
  - Validated telemetry ingestion and TimescaleDB hypertable persistence
  - Machine directory, live SCADA state, and ML health assessments
  - Deterministic Alert Rule Engine lifecycle (list, inspect, acknowledge)
  - Work order creation, approval gates, and LangGraph Copilot integration
  - Comprehensive OpenAPI documentation and CORS support
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.maintenance_agent.tools import get_machine_telemetry, get_ml_health_assessment
from factorypulse.audit import read_audit_events, record_audit_event
from factorypulse.config import get_settings
from factorypulse.redis_state import RedisStateManager
from factorypulse.schemas import (
    AlertSeverity,
    MachineStatus,
    TokenResponse,
    UserResponse,
    UserRole,
    WorkOrder,
    WorkOrderDraft,
    WorkOrderStatus,
)
from services.alert_service.engine import AlertRuleEngine
from services.api_gateway.auth import (
    HTTPAuthorizationCredentials,
    LoginRequest,
    authenticate_user,
    create_access_token,
    get_current_user,
    security,
)
from services.api_gateway.models_meta import get_active_models_metadata
from services.copilot.app import ChatRequest, ChatResponse, chat_with_copilot
from services.inference.app import (
    get_health_assessment,
    load_models,
    predict_anomaly,
    predict_failure,
    predict_rul,
)
from services.ingestion.app import get_fleet_latest, get_latest, ingest_batch, ingest_reading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Registered Machine Metadata Directory
_REGISTERED_MACHINES: dict[str, dict[str, Any]] = {
    "m001": {
        "id": "M001",
        "name": "High-Load Primary Turbofan Engine M001",
        "equipment_type": "bearing",
        "location": "Test Cell Alpha - Primary Drive",
        "installed_at": "2025-06-15T08:00:00Z",
    },
    "turbofan-001": {
        "id": "turbofan-001",
        "name": "Commercial Turbofan Engine #1",
        "equipment_type": "turbofan",
        "location": "Aviation Propulsion Bay 1",
        "installed_at": "2024-03-10T12:00:00Z",
    },
    "turbofan-002": {
        "id": "turbofan-002",
        "name": "Commercial Turbofan Engine #2",
        "equipment_type": "turbofan",
        "location": "Aviation Propulsion Bay 2",
        "installed_at": "2024-03-12T12:00:00Z",
    },
    "bearing-004": {
        "id": "bearing-004",
        "name": "Heavy Industrial Drive Shaft Bearing Unit #4",
        "equipment_type": "bearing",
        "location": "Turbine Auxiliary Room B",
        "installed_at": "2023-11-20T09:30:00Z",
    },
    "bearing-005": {
        "id": "bearing-005",
        "name": "Heavy Industrial Drive Shaft Bearing Unit #5",
        "equipment_type": "bearing",
        "location": "Turbine Auxiliary Room B",
        "installed_at": "2024-01-05T09:30:00Z",
    },
    "cnc-012": {
        "id": "cnc-012",
        "name": "Precision 5-Axis Milling Machine CNC-012",
        "equipment_type": "cnc_mill",
        "location": "Machining Center Line 3",
        "installed_at": "2024-05-18T14:15:00Z",
    },
    "cnc-013": {
        "id": "cnc-013",
        "name": "Precision 5-Axis Milling Machine CNC-013",
        "equipment_type": "cnc_mill",
        "location": "Machining Center Line 3",
        "installed_at": "2024-05-20T14:15:00Z",
    },
}

# Work Orders Store (shared with copilot service so agent-generated drafts
# created via /api/v1/copilot/* are visible and approvable here)
from services.copilot.app import (
    _WORK_ORDERS as _COPILOT_WORK_ORDERS,
    approve_work_order as _approve_copilot_work_order_internal,
    reject_work_order as _reject_copilot_work_order_internal,
)

_WORK_ORDERS_DB: dict[str, WorkOrder] = {}


def _seed_initial_alerts(engine: AlertRuleEngine) -> None:
    """Seed clearly-labelled SIMULATED demo alerts consistent with the mock fleet state.

    These are deterministic fixtures derived from the in-memory simulated fleet
    (see agents/maintenance_agent/tools._MOCK_TELEMETRY); they are not model runs
    or real production alerts and are labelled as such.
    """
    now = datetime.now(UTC)
    initial_alerts = [
        {
            "alert_id": "ALT-789A12B0",
            "machine_id": "turbofan-001",
            "alert_type": "LOW_RUL_WARNING",
            "severity": "CRITICAL",
            "title": "CRITICAL Remaining Useful Life (RUL < 15 cycles)",
            "description": "Simulated fixture: mock fleet state reports RUL 11.5 cycles with HPC exhaust gas temperature above 640°C.",
            "metric_value": 11.5,
            "timestamp": (now - timedelta(minutes=2)).isoformat(),
            "acknowledged": False,
            "source": "SIMULATED / DEMO FIXTURE",
            "simulated": True,
        },
        {
            "alert_id": "ALT-44B19C34",
            "machine_id": "bearing-004",
            "alert_type": "ANOMALY_DETECTED",
            "severity": "WARNING",
            "title": "High Anomaly Score on Bearing Vibration",
            "description": "Simulated fixture: mock fleet state reports vibration 5.86g with elevated BPFO energy in band 2.5kHz-4.0kHz.",
            "metric_value": 5.86,
            "timestamp": (now - timedelta(minutes=8)).isoformat(),
            "acknowledged": False,
            "source": "SIMULATED / DEMO FIXTURE",
            "simulated": True,
        },
        {
            "alert_id": "ALT-12C98F56",
            "machine_id": "cnc-012",
            "alert_type": "ANOMALY_DETECTED",
            "severity": "WARNING",
            "title": "Heat Dissipation Warning (HDF Risk)",
            "description": "Simulated fixture: mock fleet state reports thermal differential under 8.6K with cumulative tool wear above 200 operating minutes.",
            "metric_value": 0.72,
            "timestamp": (now - timedelta(minutes=14)).isoformat(),
            "acknowledged": True,
            "acknowledged_by": "Chief Engineer (Nithin)",
            "acknowledged_at": (now - timedelta(minutes=10)).isoformat(),
            "source": "SIMULATED / DEMO FIXTURE",
            "simulated": True,
        },
    ]
    for a in initial_alerts:
        engine.active_alerts.append(a)


_ALERT_ENGINE = AlertRuleEngine()
_seed_initial_alerts(_ALERT_ENGINE)
_REDIS_MGR = RedisStateManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_models()
        logger.info("ML Models initialized in API Gateway runtime.")
    except Exception as e:
        logger.warning("Could not auto-load ML models at startup: %s", e)
    yield


app = FastAPI(
    title="MaintenAIAPI Gateway",
    version="1.0.0",
    description="Production API Gateway, SCADA Telemetry, ML Health, and Work Order Platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ============================================================
# 1. System Health & Catalog
# ============================================================
@app.get("/health")
def root_health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "api_gateway",
        "version": "1.0.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "routes": [
            "/api/v1/auth/login",
            "/api/v1/auth/me",
            "/api/v1/models/metadata",
            "/api/v1/system/status",
            "/api/v1/telemetry",
            "/api/v1/machines",
            "/api/v1/machines/{id}/health",
            "/api/v1/machines/{id}/telemetry",
            "/api/v1/alerts",
            "/api/v1/work-orders",
            "/api/v1/copilot/query",
        ],
    }


@app.get("/api/v1/models/metadata")
def get_models_metadata() -> dict[str, Any]:
    """Authoritative ML model metadata, architecture versions, and fallback disclosures."""
    return get_active_models_metadata()


@app.get("/api/v1/system/status")
async def get_system_status() -> dict[str, Any]:
    """
    Dynamic health and connectivity verification across all backend services:
    - API Gateway
    - TimescaleDB hypertable
    - Redis authenticated live stream
    - Active ML model runtime
    - Synthetic simulation environment
    """
    now = datetime.now(UTC)

    # 1. TimescaleDB Check
    timescale_status: dict[str, Any] = {"status": "offline"}
    try:
        import asyncpg

        from factorypulse.config import get_settings

        cfg = get_settings()
        conn = await asyncpg.connect(
            host=cfg.db.host,
            port=cfg.db.port,
            user=cfg.db.user,
            password=cfg.db.password,
            database=cfg.db.db,
            timeout=2.0,
        )
        record_count = await conn.fetchval("SELECT count(*) FROM sensor_readings")
        await conn.close()
        timescale_status = {
            "status": "online",
            "hypertable": "sensor_readings",
            "total_records": record_count,
            "sync_state": "SYNCHRONIZED",
        }
    except Exception as e:
        timescale_status = {"status": "offline", "error": str(e)}

    # 2. Redis Live State Check
    redis_status: dict[str, Any] = {"status": "offline"}
    try:
        import redis

        from factorypulse.config import get_settings

        cfg = get_settings()
        r = redis.Redis(
            host=cfg.redis.host,
            port=cfg.redis.port,
            password=cfg.redis.password,
            db=cfg.redis.db,
            socket_timeout=1.5,
        )
        if r.ping():
            key_count = r.dbsize()
            redis_status = {
                "status": "online",
                "ping": True,
                "authenticated": True,
                "stream_state": "100Hz STREAM",
                "cached_keys": key_count,
            }
    except Exception as e:
        redis_status = {"status": "offline", "error": str(e)}

    # 3. ML Runtime Check (actual loaded artifacts, not hardcoded claims)
    from services.inference.app import (
        _IFOREST_MODEL,
        _LGBM_FAIL_MODEL,
        _XGB_RUL_MODEL,
        loaded_model_info,
    )

    models_ok = _IFOREST_MODEL is not None and _XGB_RUL_MODEL is not None and _LGBM_FAIL_MODEL is not None
    models_meta = get_active_models_metadata()
    ml_status = {
        "status": "active" if models_ok else "fallback_active",
        "models": loaded_model_info(),
        "metadata": models_meta,
    }

    return {
        "timestamp": now.isoformat(),
        "environment": "development",
        "simulation_mode": "DEVELOPMENT / SYNTHETIC SIMULATION",
        "services": {
            "gateway": {"status": "online", "version": "1.0.0", "port": 8000},
            "timescaledb": timescale_status,
            "redis": redis_status,
            "ml_runtime": ml_status,
            "simulation": {
                "status": "active",
                "mode": "SYNTHETIC SIMULATION",
                "notice": "All machine telemetry is generated via physical degradation simulation for development and testing.",
            },
        },
    }


# ============================================================
# 2. Authentication & User Profile Routes
# ============================================================
@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    """Authenticate user credentials and issue an HS256 JWT access token."""
    try:
        user = authenticate_user(req)
    except HTTPException:
        record_audit_event(
            "auth.login",
            user_id=req.username,
            outcome="failure",
            details={"reason": "invalid credentials"},
        )
        raise
    token = create_access_token(user.id, user.username, user.role)
    record_audit_event("auth.login", user_id=user.username, outcome="success")
    logger.info("User logged in successfully: %s (Role: %s)", user.username, user.role)
    return token


@app.get("/api/v1/auth/me", response_model=UserResponse)
def get_current_user_profile(user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Return profile and role permissions of current authenticated user."""
    return user


# ============================================================
# 3. Telemetry Ingestion Routes
# ============================================================
app.add_api_route("/api/v1/telemetry", ingest_reading, methods=["POST"], status_code=status.HTTP_202_ACCEPTED)
app.add_api_route("/api/v1/telemetry/batch", ingest_batch, methods=["POST"], status_code=status.HTTP_202_ACCEPTED)
app.add_api_route("/api/v1/telemetry/latest/{machine_id}", get_latest, methods=["GET"])
app.add_api_route("/api/v1/telemetry/fleet/latest", get_fleet_latest, methods=["GET"])


# ============================================================
# 4. Machines Directory & SCADA Health Routes
# ============================================================
@app.get("/api/v1/machines")
def list_machines() -> dict[str, Any]:
    """Retrieve full directory of registered machines with live operational status."""
    machines_list = []
    for clean_id, meta in _REGISTERED_MACHINES.items():
        telemetry = get_machine_telemetry(clean_id)
        ml_health = get_ml_health_assessment(clean_id)
        current_status = telemetry.get("status") or (
            "critical"
            if ml_health.get("anomaly_status") == "CRITICAL"
            else "warning"
            if ml_health.get("anomaly_status") == "WARNING"
            else "healthy"
        )
        machines_list.append(
            {
                "id": meta["id"],
                "name": meta["name"],
                "equipment_type": meta["equipment_type"],
                "location": meta["location"],
                "status": current_status,
                "installed_at": meta["installed_at"],
                "operating_cycle": telemetry.get("operating_cycle", 1),
                "anomaly_score": ml_health.get("anomaly_score", 0.0),
                "predicted_rul": ml_health.get("predicted_rul_cycles", 100.0),
            }
        )
    return {"machines": machines_list, "total": len(machines_list)}


@app.get("/api/v1/machines/{machine_id}")
def get_machine_detail(machine_id: str) -> dict[str, Any]:
    """Retrieve metadata and current configuration for a single machine."""
    clean_id = machine_id.strip().lower()
    if clean_id not in _REGISTERED_MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found in registry")

    meta = _REGISTERED_MACHINES[clean_id]
    telemetry = get_machine_telemetry(clean_id)
    ml_health = get_ml_health_assessment(clean_id)

    return {
        "machine": meta,
        "telemetry": telemetry,
        "ml_health": ml_health,
    }


@app.get("/api/v1/machines/{machine_id}/health")
def get_machine_health_endpoint(machine_id: str) -> dict[str, Any]:
    """Return comprehensive deterministic MachineHealth object for a machine."""
    clean_id = machine_id.strip().lower()
    if clean_id not in _REGISTERED_MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found in registry")

    telemetry = get_machine_telemetry(clean_id)
    ml_health = get_ml_health_assessment(clean_id)
    anom_score = float(ml_health.get("anomaly_score", 0.1))
    rul = float(ml_health.get("predicted_rul_cycles", 100.0))
    models_meta = get_active_models_metadata()
    # Per-model fallback disclosure sourced from the authoritative metadata registry    anomaly_meta = models_meta.get("anomaly", {})
    rul_meta = models_meta.get("rul", {})
    failure_meta = models_meta.get("failure_classifier", {})
    runtime_flags = ml_health.get("model_runtime", {})

    anom_sev = (
        AlertSeverity.CRITICAL
        if anom_score >= 0.80
        else AlertSeverity.WARNING
        if anom_score >= 0.65
        else AlertSeverity.INFO
    )
    status_val = (
        MachineStatus.CRITICAL
        if anom_score >= 0.80 or rul <= 15.0
        else MachineStatus.WARNING
        if anom_score >= 0.65 or rul <= 35.0
        else MachineStatus.HEALTHY
    )

    rul_depletion = max(0.0, min(1.0, 1.0 - (rul / 100.0)))
    health_score = round(max(0.0, min(1.0, 1.0 - (0.50 * anom_score + 0.25 * rul_depletion))), 3)

    return {
        "machine_id": _REGISTERED_MACHINES[clean_id]["id"],
        "timestamp": datetime.now(UTC).isoformat(),
        "health_score": health_score,
        "status": status_val.value,
        "telemetry_source": telemetry.get("source", "SIMULATED / MOCK TELEMETRY"),
        "model_metadata": models_meta,
        "anomaly": {
            "machine_id": _REGISTERED_MACHINES[clean_id]["id"],
            "timestamp": datetime.now(UTC).isoformat(),
            "anomaly_score": anom_score,
            "severity": anom_sev.value,
            "status": status_val.value,
            "model_version": runtime_flags.get("anomaly", {}).get("model_version", "isolation_forest_v1"),
            "model_family": anomaly_meta.get("model_family", "IsolationForest"),
            "is_fallback": runtime_flags.get("anomaly", {}).get(
                "is_fallback", anomaly_meta.get("is_fallback", True)
            ),
            "fallback_reason": runtime_flags.get("anomaly", {}).get("fallback_reason"),
            "contributing_features": {"vibration": round(telemetry.get("vibration", 1.5) / 5.0, 3), "temperature": round(telemetry.get("temperature", 70.0) / 100.0, 3)},
        },
        "rul": {
            "machine_id": _REGISTERED_MACHINES[clean_id]["id"],
            "timestamp": datetime.now(UTC).isoformat(),
            "predicted_rul": rul,
            "confidence_lower": max(0.0, rul - 12.0),
            "confidence_upper": rul + 15.0,
            "model_version": runtime_flags.get("rul", {}).get("model_version", "hist_gradient_boosting_rul_v1"),
            "model_family": rul_meta.get("model_family", "HistGradientBoostingRegressor"),
            "is_fallback": runtime_flags.get("rul", {}).get(
                "is_fallback", rul_meta.get("is_fallback", True)
            ),
            "fallback_reason": runtime_flags.get("rul", {}).get("fallback_reason")
            or rul_meta.get("fallback_reason"),
        },
        "failure_mode": {
            "machine_id": _REGISTERED_MACHINES[clean_id]["id"],
            "timestamp": datetime.now(UTC).isoformat(),
            "predicted_mode": ml_health.get("predicted_failure_mode")
            or ("tool_wear" if "cnc" in clean_id else "bearing_spall" if ("bearing" in clean_id or "m001" in clean_id) else "thermal_runaway"),
            "probabilities": ml_health.get("probable_failure_modes", {}),
            "model_version": runtime_flags.get("failure_classifier", {}).get("model_version", "hist_gradient_boosting_failure_v1"),
            "model_family": failure_meta.get("model_family", "HistGradientBoostingClassifier"),
            "is_fallback": runtime_flags.get("failure_classifier", {}).get(
                "is_fallback", failure_meta.get("is_fallback", True)
            ),
            "fallback_reason": runtime_flags.get("failure_classifier", {}).get("fallback_reason")
            or failure_meta.get("fallback_reason"),
        },
        "latest_telemetry": telemetry,
    }


@app.get("/api/v1/machines/{machine_id}/telemetry")
def get_machine_telemetry_endpoint(machine_id: str) -> dict[str, Any]:
    """Return the latest verified telemetry record for a machine."""
    clean_id = machine_id.strip().lower()
    if clean_id not in _REGISTERED_MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found in registry")

    telemetry = get_machine_telemetry(clean_id)
    return {"machine_id": _REGISTERED_MACHINES[clean_id]["id"], "telemetry": telemetry}


@app.get("/api/v1/fleet/status")
def get_fleet_status() -> dict[str, Any]:
    """Return consolidated status across the fleet including telemetry, ML health, and active alerts."""
    fleet = []
    for clean_id in _REGISTERED_MACHINES:
        telemetry = get_machine_telemetry(clean_id)
        ml_health = get_ml_health_assessment(clean_id)
        alerts = _ALERT_ENGINE.evaluate_machine(ml_health)

        fleet.append(
            {
                "machine_id": _REGISTERED_MACHINES[clean_id]["id"],
                "equipment_type": telemetry.get("equipment_type"),
                "telemetry": telemetry,
                "ml_health": ml_health,
                "has_critical_alert": (
                    ml_health.get("anomaly_status") == "CRITICAL"
                    or any(a.get("severity") == "CRITICAL" for a in alerts)
                ),
            }
        )

    return {
        "fleet": fleet,
        "total_machines": len(fleet),
        "critical_count": sum(1 for m in fleet if m["ml_health"].get("anomaly_status") == "CRITICAL"),
        "warning_count": sum(1 for m in fleet if m["ml_health"].get("anomaly_status") == "WARNING"),
        "healthy_count": sum(1 for m in fleet if m["ml_health"].get("anomaly_status") == "NORMAL"),
    }


# ============================================================
# 5. Fault Injection Test Bench Routes
# ============================================================
class FaultInjectionRequest(BaseModel):
    fault_type: str = Field(..., description="Fault type: thermal_runaway, bearing_spall, tool_wear")
    severity: float = Field(default=1.0, ge=0.1, le=2.0)


@app.post("/api/v1/machines/{machine_id}/fault")
def inject_machine_fault(machine_id: str, req: FaultInjectionRequest) -> dict[str, Any]:
    from agents.maintenance_agent.tools import _MOCK_TELEMETRY

    clean_id = machine_id.strip().lower()
    if clean_id not in _MOCK_TELEMETRY:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found in fleet")

    m = _MOCK_TELEMETRY[clean_id]
    m["status"] = "critical"
    if req.fault_type == "thermal_runaway":
        m["temperature"] = round(m.get("temperature", 600.0) + 45.0 * req.severity, 1)
        m["vibration"] = round(m.get("vibration", 2.0) + 1.8 * req.severity, 2)
    elif req.fault_type == "bearing_spall":
        m["vibration"] = round(m.get("vibration", 2.0) + 3.5 * req.severity, 2)
        m["kurtosis"] = round(m.get("kurtosis", 3.0) + 2.5 * req.severity, 2)
    elif req.fault_type == "tool_wear":
        m["temperature"] = round(m.get("temperature", 295.0) + 15.0 * req.severity, 1)
        m["vibration"] = round(m.get("vibration", 38.0) + 25.0 * req.severity, 1)

    ml_health = get_ml_health_assessment(clean_id)
    alert = _ALERT_ENGINE._create_alert(
        machine_id=clean_id,
        alert_type="FAULT_INJECTED",
        severity="CRITICAL",
        title=f"CRITICAL: Fault Injected ({req.fault_type.upper()}) on {clean_id.upper()}",
        description=f"Operator test bench injected {req.fault_type} fault. Telemetry exceeded safety operating thresholds.",
        metric_value=m.get("vibration", 0.0),
        now=datetime.now(UTC),
    )
    return {"status": "fault_injected", "machine": m, "alert": alert, "ml_health": ml_health}


@app.post("/api/v1/machines/{machine_id}/reset")
def reset_machine_state(machine_id: str) -> dict[str, Any]:
    from agents.maintenance_agent.tools import _MOCK_TELEMETRY

    clean_id = machine_id.strip().lower()
    nominal_baselines = {
        "m001": {
            "temperature": 70.0,
            "vibration": 1.2,
            "pressure": 101.3,
            "status": "healthy",
            "operating_cycle": 20,
        },
        "turbofan-001": {
            "temperature": 605.0,
            "vibration": 1.6,
            "pressure": 524.0,
            "status": "healthy",
            "operating_cycle": 50,
        },
        "turbofan-002": {
            "temperature": 608.5,
            "vibration": 1.65,
            "pressure": 524.1,
            "status": "healthy",
            "operating_cycle": 64,
        },
        "bearing-004": {
            "temperature": 68.0,
            "vibration": 1.2,
            "pressure": 101.3,
            "status": "healthy",
            "kurtosis": 2.8,
            "operating_cycle": 50,
        },
        "bearing-005": {
            "temperature": 68.2,
            "vibration": 1.15,
            "pressure": 101.3,
            "status": "healthy",
            "kurtosis": 2.85,
            "operating_cycle": 110,
        },
        "cnc-012": {
            "temperature": 295.0,
            "vibration": 38.0,
            "pressure": 101.3,
            "status": "healthy",
            "temp_diff": 11.0,
            "operating_cycle": 50,
        },
        "cnc-013": {
            "temperature": 294.0,
            "vibration": 38.0,
            "pressure": 101.3,
            "status": "healthy",
            "temp_diff": 11.2,
            "operating_cycle": 92,
        },
    }
    if clean_id in _MOCK_TELEMETRY and clean_id in nominal_baselines:
        _MOCK_TELEMETRY[clean_id].update(nominal_baselines[clean_id])
        return {"status": "reset", "machine": _MOCK_TELEMETRY[clean_id]}
    return {"status": "reset", "machine_id": clean_id}


# ============================================================
# 6. Alert Engine Routes
# ============================================================
class AlertAcknowledgeRequest(BaseModel):
    acknowledged_by: str = Field(default="Chief Reliability Engineer")


@app.get("/api/v1/alerts")
def get_alerts(
    machine_id: str | None = Query(None, description="Filter by machine ID"),
    severity: str | None = Query(None, description="Filter by severity (CRITICAL, WARNING, INFO)"),
    acknowledged: bool | None = Query(None, description="Filter by acknowledgment status"),
) -> dict[str, Any]:
    """Retrieve all active or historical alerts with optional filters."""
    alerts = _ALERT_ENGINE.get_all_alerts(limit=100)

    if machine_id:
        clean_m_id = machine_id.strip().lower()
        alerts = [a for a in alerts if a.get("machine_id", "").lower() == clean_m_id]
    if severity:
        alerts = [a for a in alerts if a.get("severity", "").upper() == severity.upper()]
    if acknowledged is not None:
        alerts = [a for a in alerts if a.get("acknowledged") == acknowledged]

    return {"alerts": alerts, "total": len(alerts)}


@app.get("/api/v1/alerts/{alert_id}")
def get_alert_detail(alert_id: str) -> dict[str, Any]:
    """Retrieve details for a single alert by ID."""
    alert = _ALERT_ENGINE.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    return {"alert": alert}


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
def acknowledge_alert_endpoint(
    alert_id: str,
    req: AlertAcknowledgeRequest | None = None,
    user: UserResponse = Depends(get_current_user),
) -> dict[str, Any]:
    """Acknowledge alert and record acknowledging operator audit trail."""
    # Acknowledger identity always comes from the authenticated token
    ack_by = f"{user.full_name} ({user.username})"
    updated_alert = _ALERT_ENGINE.acknowledge_alert(alert_id, acknowledged_by=ack_by)
    if not updated_alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    record_audit_event(
        "alert.acknowledge",
        user_id=user.username,
        resource_type="alert",
        resource_id=alert_id,
        details={"machine_id": updated_alert.get("machine_id"), "severity": updated_alert.get("severity")},
    )
    return {"status": "acknowledged", "alert": updated_alert}


@app.post("/api/v1/alerts/{alert_id}/ack")
def acknowledge_alert_legacy_alias(
    alert_id: str,
    req: AlertAcknowledgeRequest | None = None,
    user: UserResponse = Depends(get_current_user),
) -> dict[str, Any]:
    """Backward-compatible endpoint for frontend alert acknowledgment (JWT required)."""
    return acknowledge_alert_endpoint(alert_id, req, user)


# ============================================================
# 7. Work Orders Management Routes
# ============================================================
class WorkOrderApprovalPayload(BaseModel):
    approved_by: str = Field(default="Chief Reliability Engineer")
    approval_notes: str | None = Field(default="Approved after diagnostic verification.")


@app.post("/api/v1/work-orders", response_model=WorkOrder, status_code=status.HTTP_201_CREATED)
def create_work_order(
    draft: WorkOrderDraft,
    user: UserResponse = Depends(get_current_user),
) -> WorkOrder:
    """Create a new work order draft requiring human technician authorization."""
    machine_key = draft.machine_id.strip().lower()
    if machine_key not in _REGISTERED_MACHINES:
        raise HTTPException(
            status_code=404, detail=f"Machine '{draft.machine_id}' not found in registry"
        )

    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(UTC)

    work_order = WorkOrder(
        id=wo_id,
        machine_id=draft.machine_id,
        status=WorkOrderStatus.PENDING_APPROVAL,
        created_at=now,
        created_by=f"{user.full_name} ({user.username})",
        issue_summary=draft.issue_summary,
        recommended_actions=draft.recommended_actions,
        priority=draft.priority,
        estimated_downtime_hours=draft.estimated_downtime_hours or 2.0,
        parts_needed=draft.parts_needed or [],
        safety_notes=draft.safety_notes or ["Ensure machine is locked out and tagged out."],
        evidence=draft.evidence or {},
    )
    _WORK_ORDERS_DB[wo_id] = work_order
    record_audit_event(
        "work_order.create",
        user_id=user.username,
        resource_type="work_order",
        resource_id=wo_id,
        details={"machine_id": draft.machine_id, "status": work_order.status.value},
    )
    logger.info("Created Work Order: %s for Machine %s (Status: PENDING_APPROVAL)", wo_id, draft.machine_id)
    return work_order


@app.get("/api/v1/work-orders")
def list_work_orders_endpoint(
    status_filter: str | None = Query(None, alias="status", description="Filter by status")
) -> dict[str, Any]:
    """List all work orders with optional status filter."""
    orders = list(_WORK_ORDERS_DB.values())
    if status_filter:
        orders = [wo for wo in orders if wo.status.value.lower() == status_filter.lower()]
    return {"work_orders": [wo.model_dump(mode="json") for wo in orders], "total": len(orders)}


@app.get("/api/v1/work-orders/{work_order_id}", response_model=WorkOrder)
def get_work_order_detail(work_order_id: str) -> WorkOrder:
    """Retrieve details for a single work order."""
    if work_order_id not in _WORK_ORDERS_DB:
        raise HTTPException(status_code=404, detail=f"Work order '{work_order_id}' not found")
    return _WORK_ORDERS_DB[work_order_id]


@app.post("/api/v1/work-orders/{work_order_id}/approve", response_model=WorkOrder)
def approve_work_order_endpoint(
    work_order_id: str,
    payload: WorkOrderApprovalPayload | None = None,
    user: UserResponse = Depends(get_current_user),
) -> WorkOrder:
    """
    Human-in-the-loop authorization gate approving a pending work order.

    Requires a valid JWT; the approving identity always comes from the token
    (never from the request body). Enforced roles: engineer, admin.
    """
    if user.role not in [UserRole.ENGINEER, UserRole.ADMIN]:
        record_audit_event(
            "work_order.approve",
            user_id=user.username,
            resource_type="work_order",
            resource_id=work_order_id,
            outcome="denied",
            details={"reason": f"role '{user.role.value}' not permitted"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Action requires role 'engineer' or 'admin'. Current role: '{user.role.value}'",
        )

    wo = _WORK_ORDERS_DB.get(work_order_id)
    if not wo:
        raise HTTPException(status_code=404, detail=f"Work order '{work_order_id}' not found")

    if wo.status == WorkOrderStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Work order '{work_order_id}' was already approved by '{wo.approved_by}'",
        )
    if wo.status != WorkOrderStatus.PENDING_APPROVAL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Work order '{work_order_id}' is in status '{wo.status.value}'; "
                "only 'pending_approval' work orders can be approved"
            ),
        )

    approved_by = f"{user.full_name} ({user.username})"
    wo.status = WorkOrderStatus.APPROVED
    wo.approved_by = approved_by
    wo.approved_at = datetime.now(UTC)
    wo.approval_notes = payload.approval_notes if payload else "Authorized by reliability engineering."

    record_audit_event(
        "work_order.approve",
        user_id=user.username,
        resource_type="work_order",
        resource_id=work_order_id,
        details={"machine_id": wo.machine_id, "approver_role": user.role.value},
    )
    logger.info("Work Order %s approved by %s", work_order_id, wo.approved_by)
    return wo


@app.post("/api/v1/work-orders/{work_order_id}/reject", response_model=WorkOrder)
def reject_work_order_endpoint(
    work_order_id: str,
    payload: WorkOrderApprovalPayload | None = None,
    user: UserResponse = Depends(get_current_user),
) -> WorkOrder:
    """Reject a pending work order (separate controlled action from approval)."""
    if user.role not in [UserRole.ENGINEER, UserRole.ADMIN]:
        record_audit_event(
            "work_order.reject",
            user_id=user.username,
            resource_type="work_order",
            resource_id=work_order_id,
            outcome="denied",
            details={"reason": f"role '{user.role.value}' not permitted"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Action requires role 'engineer' or 'admin'. Current role: '{user.role.value}'",
        )

    wo = _WORK_ORDERS_DB.get(work_order_id)
    if not wo:
        raise HTTPException(status_code=404, detail=f"Work order '{work_order_id}' not found")
    if wo.status != WorkOrderStatus.PENDING_APPROVAL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Work order '{work_order_id}' is in status '{wo.status.value}'; "
                "only 'pending_approval' work orders can be rejected"
            ),
        )

    wo.status = WorkOrderStatus.REJECTED
    wo.rejection_reason = (payload.approval_notes if payload else None) or "Rejected by reviewer."
    record_audit_event(
        "work_order.reject",
        user_id=user.username,
        resource_type="work_order",
        resource_id=work_order_id,
        details={"machine_id": wo.machine_id},
    )
    return wo


# Legacy routes for frontend compatibility
class LegacyApprovalRequest(BaseModel):
    work_order_id: str
    technician_name: str
    action: str = "APPROVE"
    notes: str | None = None


@app.get("/api/v1/copilot/work_orders")
def list_copilot_work_orders() -> dict[str, Any]:
    return {"work_orders": list(_COPILOT_WORK_ORDERS.values()), "total": len(_COPILOT_WORK_ORDERS)}


@app.post("/api/v1/copilot/work_orders/approve")
def approve_copilot_work_order(
    req: LegacyApprovalRequest,
    user: UserResponse = Depends(get_current_user),
) -> dict[str, Any]:
    """Approve or reject an agent-drafted work order (JWT + RBAC enforced)."""
    if user.role not in [UserRole.ENGINEER, UserRole.ADMIN]:
        record_audit_event(
            "work_order.approve",
            user_id=user.username,
            resource_type="work_order",
            resource_id=req.work_order_id,
            outcome="denied",
            details={"reason": f"role '{user.role.value}' not permitted"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Action requires role 'engineer' or 'admin'. Current role: '{user.role.value}'",
        )

    action = req.action.upper()
    if action == "APPROVE":
        wo = _approve_copilot_work_order_internal(
            req.work_order_id, approved_by=f"{user.full_name} ({user.username})", notes=req.notes
        )
    elif action == "REJECT":
        wo = _reject_copilot_work_order_internal(
            req.work_order_id, rejected_by=f"{user.full_name} ({user.username})", notes=req.notes
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action '{req.action}'")

    if wo is None:
        raise HTTPException(status_code=404, detail=f"Work Order {req.work_order_id} not found")
    if "error" in wo:
        raise HTTPException(status_code=409, detail=wo["error"])

    record_audit_event(
        f"work_order.{action.lower()}",
        user_id=user.username,
        resource_type="work_order",
        resource_id=req.work_order_id,
        details={"machine_id": wo.get("machine_id"), "approver_role": user.role.value},
    )
    return {"status": "success", "work_order": wo}


# ============================================================
# 8. Audit Log Route (admin only)
# ============================================================
@app.get("/api/v1/audit-log")
def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    action: str | None = Query(None, description="Filter by action type"),
    resource_id: str | None = Query(None, description="Filter by resource ID"),
    user: UserResponse = Depends(get_current_user),
) -> dict[str, Any]:
    """Read recent audit events (admin only)."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Audit log access requires the 'admin' role",
        )
    events = read_audit_events(limit=limit, action=action, resource_id=resource_id)
    return {"events": events, "total": len(events)}


# ============================================================
# 8. Maintenance Copilot Diagnostic Routes
# ============================================================
@app.post("/api/v1/copilot/query", response_model=ChatResponse)
def copilot_query_endpoint(req: ChatRequest) -> ChatResponse:
    """Grounded natural language query to the LangGraph Maintenance Copilot Agent."""
    return chat_with_copilot(req)


app.add_api_route("/api/v1/copilot/chat", chat_with_copilot, methods=["POST"], response_model=ChatResponse)


# ============================================================
# 9. Model Predictions Direct Routes
# ============================================================
app.add_api_route("/api/v1/predict/anomaly", predict_anomaly, methods=["POST"])
app.add_api_route("/api/v1/predict/rul", predict_rul, methods=["POST"])
app.add_api_route("/api/v1/predict/failure", predict_failure, methods=["POST"])
app.add_api_route("/api/v1/predict/health_assessment", get_health_assessment, methods=["POST"])
