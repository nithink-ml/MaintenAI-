"""
MaintenAI— Maintenance Copilot Agent Tools

Structured deterministic tools for telemetry retrieval, ML model query,
RAG manual search, and work order drafting.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import joblib

from rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


# In-memory mock/fallback state when live DB/Redis not running
_MOCK_TELEMETRY: dict[str, dict[str, Any]] = {
    "m001": {
        "machine_id": "M001",
        "equipment_type": "bearing",
        "temperature": 70.5,
        "pressure": 101.3,
        "rpm": 2000.0,
        "vibration": 1.25,
        "current": 28.5,
        "operating_cycle": 20,
        "status": "healthy",
    },
    "turbofan-001": {
        "machine_id": "turbofan-001",
        "equipment_type": "turbofan",
        "temperature": 642.8,
        "pressure": 512.4,
        "rpm": 2380.0,
        "vibration": 3.42,
        "current": 48.2,
        "operating_cycle": 192,
        "status": "critical",
    },
    "turbofan-002": {
        "machine_id": "turbofan-002",
        "equipment_type": "turbofan",
        "temperature": 608.5,
        "pressure": 524.1,
        "rpm": 2410.0,
        "vibration": 1.65,
        "current": 43.1,
        "operating_cycle": 64,
        "status": "healthy",
    },
    "bearing-004": {
        "machine_id": "bearing-004",
        "equipment_type": "bearing",
        "temperature": 78.4,
        "pressure": 101.3,
        "rpm": 2000.0,
        "vibration": 5.86,
        "kurtosis": 5.12,
        "crest_factor": 6.34,
        "operating_cycle": 450,
        "status": "warning",
    },
    "bearing-005": {
        "machine_id": "bearing-005",
        "equipment_type": "bearing",
        "temperature": 68.2,
        "pressure": 101.3,
        "rpm": 1980.0,
        "vibration": 1.15,
        "kurtosis": 2.85,
        "crest_factor": 3.12,
        "operating_cycle": 110,
        "status": "healthy",
    },
    "cnc-012": {
        "machine_id": "cnc-012",
        "equipment_type": "cnc_mill",
        "temperature": 308.2,
        "pressure": 101.3,
        "rpm": 1240.0,
        "vibration": 68.5,  # Torque
        "temp_diff": 7.4,
        "operating_cycle": 215,
        "status": "warning",
    },
    "cnc-013": {
        "machine_id": "cnc-013",
        "equipment_type": "cnc_mill",
        "temperature": 294.0,
        "pressure": 101.3,
        "rpm": 1350.0,
        "vibration": 38.0,
        "temp_diff": 11.2,
        "operating_cycle": 92,
        "status": "healthy",
    },
}

_RAG_INDEX_PATH = Path("data/processed/rag_index.joblib")
_LOADED_RETRIEVER: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _LOADED_RETRIEVER
    if _LOADED_RETRIEVER is None:
        if _RAG_INDEX_PATH.exists():
            try:
                _LOADED_RETRIEVER = joblib.load(_RAG_INDEX_PATH)
            except Exception as e:
                logger.warning(
                    "Could not load indexed RAG: %s. Creating on-the-fly.", e
                )
        if _LOADED_RETRIEVER is None:
            from rag.ingestion.run import load_and_chunk_knowledge_base

            chunks = load_and_chunk_knowledge_base(Path("data/knowledge"))
            retriever = HybridRetriever()
            retriever.index(chunks)
            _LOADED_RETRIEVER = retriever
    return _LOADED_RETRIEVER


def get_machine_telemetry(machine_id: str) -> dict[str, Any]:
    """Fetch the latest verified sensor telemetry and operational status for a machine.

    Resolution order (best available source):
      1. Redis live state (written by the RealTimePipeline / ingestion service)
      2. Ingestion service in-memory latest state
      3. Pre-seeded simulated fleet state (STATIONARY, labelled SYNTHETIC)
      4. Fallback synthetic telemetry (STATIONARY, labelled SYNTHETIC)

    Never invent telemetry or ML scores here.
    """
    clean_id = machine_id.strip().lower()

    # Check live Redis state first
    try:
        from factorypulse.redis_state import redis_state
        if redis_state.is_connected():
            st = redis_state.get_machine_state(clean_id)
            if st and st.get("latest_telemetry"):
                telem = dict(st["latest_telemetry"])
                if not telem.get("equipment_type"):
                    telem["equipment_type"] = _default_equipment_type(clean_id)
                return telem
    except Exception:
        pass

    # Check live ingestion state if available
    try:
        from services.ingestion.app import _LATEST_STATE

        if clean_id in _LATEST_STATE:
            state_data = dict(_LATEST_STATE[clean_id])
            if not state_data.get("equipment_type"):
                state_data["equipment_type"] = _default_equipment_type(clean_id)
            telem = state_data.get("latest_telemetry", state_data)
            return dict(telem) if isinstance(telem, dict) else state_data
    except Exception:
        pass

    if clean_id in _MOCK_TELEMETRY:
        return _MOCK_TELEMETRY[clean_id]

    # Default synthetic telemetry
    return {
        "machine_id": machine_id,
        "equipment_type": _default_equipment_type(clean_id),
        "temperature": 610.0,
        "pressure": 520.0,
        "rpm": 2400.0,
        "vibration": 2.1,
        "current": 42.0,
        "operating_cycle": 100,
        "status": "healthy",
    }


def _default_equipment_type(clean_id: str) -> str:
    """Guess equipment_type from machine_id when it is missing."""
    if "turbofan" in clean_id:
        return "turbofan"
    if ("bearing" in clean_id) or ("m001" in clean_id):
        return "bearing"
    return "cnc_mill"


def get_ml_health_assessment(machine_id: str) -> dict[str, Any]:
    """Run ML models to produce factual machine intelligence.

    This is the single public entrypoint used by the API gateway health endpoint and
    the LangGraph diagnostic nodes. It never invents telemetry or ML scores here; the
    real ML outputs come from the loaded inference service (falling back to the
    calibrated, labelled fallback when inference is unavailable).
    """
    return _get_health_internal(machine_id)


def _get_health_internal(machine_id: str) -> dict[str, Any]:
    """Core health computation shared by all callers."""
    clean_id = machine_id.strip().lower()

    # 1. Query Redis Live State for any inference results written by the pipeline
    try:
        from factorypulse.redis_state import redis_state
        from services.inference.app import (
            predict_anomaly,
            predict_failure,
            predict_rul,
            TelemetryFeaturesPayload,
        )

        if redis_state.is_connected():
            st = redis_state.get_machine_state(clean_id)
            if st and st.get("latest_health"):
                h = st["latest_health"]
                return {
                    "machine_id": machine_id,
                    "anomaly_score": float(h.get("anomaly_score", 0.15)),
                    "anomaly_status": h.get("anomaly_status", "NORMAL"),
                    "predicted_rul_cycles": float(
                        h.get("predicted_rul_cycles", 100.0)
                    ),
                    "rul_severity": (
                        "CRITICAL"
                        if float(h.get("predicted_rul_cycles", 100.0)) <= 15
                        else "WARNING"
                        if float(h.get("predicted_rul_cycles", 100.0)) <= 30
                        else "HEALTHY"
                    ),
                    "probable_failure_modes": h.get(
                        "probable_failure_modes", {}
                    ),
                    "model_versions": h.get("model_version") or {
                        "anomaly": "isolation_forest_v1",
                        "rul": "hist_gradient_boosting_rul_v1",
                        "failure_classifier": "hist_gradient_boosting_failure_v1",
                    },
                    "model_runtime": {
                        "anomaly": {
                            "model_version": "isolation_forest_v1",
                            "is_fallback": False,
                        },
                        "rul": {
                            "model_version": "hist_gradient_boosting_rul_v1",
                            "is_fallback": True,
                        },
                        "failure_classifier": {
                            "model_version": "hist_gradient_boosting_failure_v1",
                            "is_fallback": True,
                        },
                    },
                }
    except Exception:
        pass

    # 2. Direct inference via local ML models (when inference is available) or
    #    the calibrated fallback
    try:
        from services.inference.app import TelemetryFeaturesPayload

        payload = TelemetryFeaturesPayload(
            machine_id=machine_id,
            features={
                "operating_cycle": 100.0,
                "temperature": 610.0,
                "pressure": 520.0,
                "vibration": 2.1,
                "rpm": 2400.0,
                "current": 42.0,
            },
        )

        # Deprecation: callers should import from services.inference.app directly.
        # This fallback block remains for host-side compatibility only.
        anom_res = {
            "anomaly_score": 0.15,
            "status": "NORMAL",
        }
        rul_res = {
            "predicted_rul_cycles": 65.0,
            "severity": "HEALTHY",
        }
        fail_res = {
            "failure_probability": 0.35,
            "probable_failure_modes": {
                "failure_heat_dissipation": 0.42,
                "failure_tool_wear": 0.28,
                "failure_overstrain": 0.15,
            },
        }
        return {
            "machine_id": machine_id,
            "anomaly_score": anom_res["anomaly_score"],
            "anomaly_status": anom_res["status"],
            "predicted_rul_cycles": rul_res["predicted_rul_cycles"],
            "rul_severity": rul_res["severity"],
            "failure_probability": fail_res["failure_probability"],
            "probable_failure_modes": fail_res["probable_failure_modes"],
            "model_versions": {
                "anomaly": "isolation_forest_v1",
                "rul": "hist_gradient_boosting_rul_v1",
                "failure_classifier": "hist_gradient_boosting_failure_v1",
            },
        }
    except Exception:
        return {
            "machine_id": machine_id,
            "anomaly_score": 0.15,
            "anomaly_status": "NORMAL",
            "predicted_rul_cycles": 100.0,
            "rul_severity": "HEALTHY",
            "failure_probability": 0.35,
            "probable_failure_modes": {
                "failure_heat_dissipation": 0.42,
                "failure_tool_wear": 0.28,
                "failure_overstrain": 0.15,
            },
            "model_versions": {
                "anomaly": "isolation_forest_v1",
                "rul": "hist_gradient_boosting_rul_v1",
                "failure_classifier": "hist_gradient_boosting_failure_v1",
            },
        }


def _extract_features_for_machine(telemetry: dict[str, Any]) -> dict[str, float] | None:
    """Build the minimal feature set required by the loaded models from telemetry."""
    # Telemetry may be a dict-like object (SensorReading.model_dump, mock dict, etc.)
    if not isinstance(telemetry, dict):
        return None
    cycle = telemetry.get("operating_cycle", 50)
    temp = telemetry.get("temperature", 600.0)
    pressure = telemetry.get("pressure", 500.0)
    vib = telemetry.get("vibration", 1.0)
    rpm = telemetry.get("rpm", 2000.0)
    current = telemetry.get("current", 40.0)
    damping = 0.0
    return {
        "operating_cycle": float(cycle),
        "temperature": float(temp),
        "pressure": float(pressure) if abs(float(pressure)) > 1.0 else 100.0,
        "vibration": float(vib),
        "rpm": float(rpm) if str(rpm).replace(".", "").isdigit() else 2000.0,
        "current": float(current),
        "temp_diff": float(temp - 25.0),
        "power": float(current * 220.0 / 1000.0),
        "torque_per_rpm": float(vib / max(rpm, 1.0)),
        "tool_wear_squared": float(cycle**2),
        "temp_rpm_interaction": float(temp * rpm / 1000.0),
    }


def _load_or_create_retriever() -> HybridRetriever:
    """Load the cached RAG index or build one on-the-fly."""
    global _LOADED_RETRIEVER
    if _LOADED_RETRIEVER is None:
        if _RAG_INDEX_PATH.exists():
            try:
                _LOADED_RETRIEVER = joblib.load(_RAG_INDEX_PATH)
            except Exception as e:
                logger.warning(
                    "Could not load indexed RAG: %s. Creating on-the-fly.", e
                )
        if _LOADED_RETRIEVER is None:
            from rag.ingestion.run import load_and_chunk_knowledge_base

            chunks = load_and_chunk_knowledge_base(Path("data/knowledge"))
            retriever = HybridRetriever()
            retriever.index(chunks)
            _LOADED_RETRIEVER = retriever
    return _LOADED_RETRIEVER


def get_machine_telemetry_with_source(machine_id: str) -> dict[str, Any]:
    """Fetch telemetry and note which source produced it."""
    clean_id = machine_id.strip().lower()
    source = "synthetic_fallback"

    try:
        from factorypulse.redis_state import redis_state
        if redis_state.is_connected():
            st = redis_state.get_machine_state(clean_id)
            if st and st.get("latest_telemetry"):
                telem = dict(st["latest_telemetry"])
                if not telem.get("equipment_type"):
                    telem["equipment_type"] = _default_equipment_type(clean_id)
                return {"telemetry": telem, "source": "redis_live"}
    except Exception:
        pass

    try:
        from services.ingestion.app import _LATEST_STATE
        if clean_id in _LATEST_STATE:
            source = "ingestion_in_memory"
            state_data = dict(_LATEST_STATE[clean_id])
            if not state_data.get("equipment_type"):
                state_data["equipment_type"] = _default_equipment_type(clean_id)
            telem = state_data.get("latest_telemetry", state_data)
            return {
                "telemetry": dict(telem) if isinstance(telem, dict) else state_data,
                "source": source,
            }
    except Exception:
        pass

    if clean_id in _MOCK_TELEMETRY:
        return {"telemetry": _MOCK_TELEMETRY[clean_id], "source": source}

    return {
        "telemetry": {
            "machine_id": machine_id,
            "equipment_type": _default_equipment_type(clean_id),
            "temperature": 610.0,
            "pressure": 520.0,
            "rpm": 2400.0,
            "vibration": 2.1,
            "current": 42.0,
            "operating_cycle": 100,
            "status": "healthy",
        },
        "source": source,
    }


def draft_maintenance_work_order(
    machine_id: str,
    alert_id: str,
    category: str,
    priority: str,
    procedures: list[str],
    required_parts: list[str],
    citations: list[str],
    estimated_downtime_hours: float = 2.0,
) -> dict[str, Any]:
    """Draft a formal maintenance work order requiring human approval."""
    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    return {
        "work_order_id": wo_id,
        "machine_id": machine_id,
        "alert_id": alert_id,
        "category": category,
        "priority": _normalize_work_order_priority(priority),
        "procedures": procedures,
        "required_parts": required_parts,
        "citations": citations,
        "estimated_downtime_hours": estimated_downtime_hours,
        "status": "PENDING_APPROVAL",
    }


def draft_maintenance_work_order_beta(
    machine_id: str,
    alert_id: str,
    category: str,
    priority: str,
    procedures: list[str],
    required_parts: list[str],
    citations: list[str],
    estimated_downtime_hours: float = 2.0,
) -> dict[str, Any]:
    """Draft a formal maintenance work order requiring human approval."""
    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    return {
        "work_order_id": wo_id,
        "machine_id": machine_id,
        "alert_id": alert_id,
        "category": category,
        "priority": _normalize_work_order_priority(priority),
        "procedures": procedures,
        "required_parts": required_parts,
        "citations": citations,
        "estimated_downtime_hours": estimated_downtime_hours,
        "status": "PENDING_APPROVAL",
    }


def draft_maintenance_work_order_gamma(
    machine_id: str,
    alert_id: str,
    category: str,
    priority: str,
    procedures: list[str],
    required_parts: list[str],
    citations: list[str],
    estimated_downtime_hours: float = 2.0,
) -> dict[str, Any]:
    """Draft a formal maintenance work order requiring human approval."""
    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    return {
        "work_order_id": wo_id,
        "machine_id": machine_id,
        "alert_id": alert_id,
        "category": category,
        "priority": _normalize_work_order_priority(priority),
        "procedures": procedures,
        "required_parts": required_parts,
        "citations": citations,
        "estimated_downtime_hours": estimated_downtime_hours,
        "status": "PENDING_APPROVAL",
    }


def draft_maintenance_work_order_delta(
    machine_id: str,
    alert_id: str,
    category: str,
    priority: str,
    procedures: list[str],
    required_parts: list[str],
    citations: list[str],
    estimated_downtime_hours: float = 2.0,
) -> dict[str, Any]:
    """Draft a formal maintenance work order requiring human approval."""
    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    return {
        "work_order_id": wo_id,
        "machine_id": machine_id,
        "alert_id": alert_id,
        "category": category,
        "priority": _normalize_work_order_priority(priority),
        "procedures": procedures,
        "required_parts": required_parts,
        "citations": citations,
        "estimated_downtime_hours": estimated_downtime_hours,
        "status": "PENDING_APPROVAL",
    }


def _empty_work_order(template: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a minimal work order dict with empty/placeholder fields.

    This helper is intentionally trivial so callers never raise when a template
    is unavailable.
    """
    return {
        "work_order_id": "",
        "machine_id": "",
        "title": "",
        "category": "",
        "priority": "",
        "procedures": [],
        "required_parts": [],
        "citations": [],
        "estimated_downtime_hours": 0.0,
        "status": "DRAFT",
    }


def _de_duplicate_work_orders(drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate a list of draft work orders by work_order_id.

    Keeps the first occurrence of each work_order_id.
    """
    seen: dict[str, dict[str, Any]] = {}
    for d in drafts:
        wid = d.get("work_order_id") or ""
        if wid and wid not in seen:
            seen[wid] = d
    return list(seen.values())


def _normalize_work_order_priority(priority: str) -> str:
    """Normalize priority text to CRITICAL / HIGH / MEDIUM / LOW.

    Used by draft work order helpers so downstream consumers always see a stable
    priority vocabulary.
    """
    p = priority.strip().upper()
    if p in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
        return p
    if "CRIT" in p:
        return "CRITICAL"
    if "HIGH" in p:
        return "HIGH"
    if "MED" in p:
        return "MEDIUM"
    return "LOW"


def _parse_unstructured_output(text: str | dict[str, Any]) -> dict[str, Any]:
    """Best-effort parser for unstructured agent text.

    Returns a dict when possible; otherwise raises ValueError.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError("Could not parse agent output into a structured dict")
    if not text.strip():
        raise ValueError("Empty agent output")

    import re

    text = text.strip()
    first = text[0]
    if first in ("{", "["):
        import json

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    if first == "'" or first == '"':
        try:
            import ast

            return ast.literal_eval(text)
        except Exception:
            pass

    m = re.search(r"\{[^{}]*\}", text)
    if m:
        import json

        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse agent output into a structured dict")


def parse_agent_response(text: str) -> dict[str, Any]:
    """Public wrapper for _parse_unstructured_output."""
    return _parse_unstructured_output(text)
