"""
MaintenAI— Ingestion Service (FastAPI)

Validates incoming sensor telemetry payloads against physical range rules and Pydantic schemas.
Persists clean time-series data to TimescaleDB hypertable, updates Redis live state,
pushes to OnlineFeatureStore, and logs rejections.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import time
from datetime import datetime
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from factorypulse.config import get_settings
from factorypulse.redis_state import RedisStateManager
from factorypulse.schemas import SensorReading, SensorReadingBatch, TelemetryRejection
from services.feature_service.calculator import OnlineFeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# State stores
_LATEST_STATE: dict[str, dict[str, Any]] = {}
_TELEMETRY_BUFFER: list[dict[str, Any]] = []
_REJECTIONS: collections.deque = collections.deque(maxlen=500)
_LATENCY_SAMPLES: collections.deque = collections.deque(maxlen=1000)

_REDIS_STATE: RedisStateManager | None = None
_PG_POOL: asyncpg.Pool | None = None
_FEATURE_STORE: OnlineFeatureStore = OnlineFeatureStore(window_size=30)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _REDIS_STATE, _PG_POOL
    settings = get_settings()

    # 1. Initialize Redis State Manager
    try:
        _REDIS_STATE = RedisStateManager()
        if _REDIS_STATE.is_connected():
            logger.info("Connected to Redis Live State at %s:%s", settings.redis.host, settings.redis.port)
        else:
            logger.warning("Redis not reachable; using in-memory state store")
    except Exception as e:
        logger.warning("Redis initialization error: %s", e)
        _REDIS_STATE = None

    # 2. Initialize TimescaleDB Connection Pool
    try:
        db_url = f"postgresql://{settings.db.user}:{settings.db.password}@{settings.db.host}:{settings.db.port}/{settings.db.db}"
        _PG_POOL = await asyncpg.create_pool(
            db_url,
            min_size=2,
            max_size=10,
            timeout=5.0,
            command_timeout=5.0,
        )
        logger.info("Connected to TimescaleDB pool at %s:%s/%s", settings.db.host, settings.db.port, settings.db.db)
    except Exception as e:
        logger.warning("TimescaleDB connection pool unavailable (%s); operating in memory-only mode", e)
        _PG_POOL = None

    yield

    if _PG_POOL:
        await _PG_POOL.close()


app = FastAPI(
    title="MaintenAITelemetry Ingestion Service",
    version="0.2.0",
    description="High-throughput physical sensor telemetry validation, TimescaleDB hypertable persistence, and Redis live state API",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Intercept validation errors, log structured TelemetryRejection, and return 422."""
    reasons = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()]
    body_bytes = await request.body()
    try:
        raw_payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        raw_payload = {"raw": body_bytes.decode("utf-8", errors="replace")}

    machine_id = raw_payload.get("machine_id") if isinstance(raw_payload, dict) else None
    rejection = TelemetryRejection(
        machine_id=machine_id,
        raw_payload=raw_payload if isinstance(raw_payload, dict) else {"raw": str(raw_payload)},
        rejection_reasons=reasons,
    )
    _REJECTIONS.append(rejection.model_dump(mode="json"))
    logger.warning("Telemetry rejected for machine %s: %s", machine_id, reasons)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid telemetry payload",
            "machine_id": machine_id,
            "rejection_reasons": reasons,
            "rejected_at": rejection.rejected_at.isoformat(),
        },
    )


async def _persist_reading(data: dict[str, Any]) -> None:
    """Helper to persist single reading to TimescaleDB and Redis."""
    m_id = data["machine_id"]
    ts_str = data["timestamp"]
    ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str

    # Write to TimescaleDB if pool is available
    if _PG_POOL:
        try:
            async with _PG_POOL.acquire() as conn:
                # Ensure machine exists in directory
                eq_type = data.get("equipment_type") or "turbofan"
                try:
                    await conn.execute(
                        """
                        INSERT INTO machines (id, name, equipment_type)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (id) DO NOTHING;
                        """,
                        m_id,
                        f"Machine {m_id}",
                        eq_type,
                    )
                except Exception as e:
                    logger.debug("Machine upsert failed for %s: %s", m_id, e)

                # Insert into sensor_readings hypertable
                validation_meta = json.dumps({"fault_injected": data.get("fault_injected")})
                try:
                    await conn.execute(
                        """
                        INSERT INTO sensor_readings (
                            machine_id, timestamp, temperature, pressure,
                            vibration, rpm, current, operating_cycle,
                            source, validation_metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                        ON CONFLICT (machine_id, timestamp) DO NOTHING;
                        """,
                        m_id,
                        ts,
                        data.get("temperature"),
                        data.get("pressure"),
                        data.get("vibration"),
                        data.get("rpm"),
                        data.get("current"),
                        data.get("operating_cycle"),
                        data.get("source", "SYNTHETIC / DEVELOPMENT"),
                        validation_meta,
                    )
                except Exception as e:
                    logger.error("TimescaleDB insert failed for machine %s: %s", m_id, e)
        except Exception as e:
            logger.error("TimescaleDB insert failed for machine %s: %s", m_id, e)

    # Write to Redis if available
    if _REDIS_STATE and _REDIS_STATE.is_connected():
        try:
            _REDIS_STATE.update_telemetry(m_id, data)
        except Exception as e:
            logger.debug("Redis telemetry update failed: %s", e)


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "ingestion",
        "timescaledb_connected": _PG_POOL is not None,
        "redis_connected": _REDIS_STATE.is_connected() if _REDIS_STATE else False,
    }


@app.post("/api/v1/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def ingest_reading(reading: SensorReading) -> dict[str, Any]:
    """Ingest, validate, cache in Redis, and persist a single sensor reading."""
    t0 = time.perf_counter()
    data = reading.model_dump(mode="json")
    m_id = data["machine_id"]

    # Update in-memory stores
    _LATEST_STATE[m_id] = data
    _TELEMETRY_BUFFER.append(data)
    _FEATURE_STORE.append_reading(data)

    # Persist asynchronously
    await _persist_reading(data)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _LATENCY_SAMPLES.append(elapsed_ms)

    return {
        "status": "accepted",
        "machine_id": m_id,
        "source": data.get("source", "SYNTHETIC / DEVELOPMENT"),
        "ingest_latency_ms": round(elapsed_ms, 3),
    }


@app.post("/api/v1/telemetry/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_batch(batch: SensorReadingBatch) -> dict[str, Any]:
    """Ingest a batch of sensor readings."""
    t0 = time.perf_counter()
    count = 0
    for reading in batch.readings:
        data = reading.model_dump(mode="json")
        m_id = data["machine_id"]
        _LATEST_STATE[m_id] = data
        _TELEMETRY_BUFFER.append(data)
        _FEATURE_STORE.append_reading(data)
        await _persist_reading(data)
        count += 1

    if len(_TELEMETRY_BUFFER) > 10000:
        del _TELEMETRY_BUFFER[:-5000]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _LATENCY_SAMPLES.append(elapsed_ms)

    return {
        "status": "accepted",
        "processed_count": count,
        "batch_latency_ms": round(elapsed_ms, 3),
    }


@app.get("/api/v1/telemetry/latest/{machine_id}")
def get_latest(machine_id: str) -> dict[str, Any]:
    """Retrieve the latest verified sensor state for a machine."""
    # First attempt Redis
    if _REDIS_STATE and _REDIS_STATE.is_connected():
        st = _REDIS_STATE.get_machine_state(machine_id)
        if st and st.get("telemetry"):
            return st["telemetry"]

    if machine_id not in _LATEST_STATE:
        raise HTTPException(status_code=404, detail=f"No telemetry found for machine {machine_id}")
    raw = _LATEST_STATE[machine_id]
    telem = dict(raw.get("latest_telemetry", raw))
    return telem


@app.get("/api/v1/telemetry/fleet/latest")
def get_fleet_latest() -> dict[str, Any]:
    """Retrieve latest state for all active machines."""
    if _REDIS_STATE and _REDIS_STATE.is_connected():
        all_states = _REDIS_STATE.get_all_machines_state()
        if all_states:
            telemetry_list = [s.get("latest_telemetry") for s in all_states if s.get("latest_telemetry")]
            if telemetry_list:
                return {"machines": telemetry_list, "total_active": len(telemetry_list)}

    return {"machines": list(_LATEST_STATE.values()), "total_active": len(_LATEST_STATE)}


@app.get("/api/v1/telemetry/rejections")
def get_rejections(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve audit log of recently rejected telemetry payloads."""
    return list(_REJECTIONS)[-limit:]


@app.get("/api/v1/telemetry/metrics")
def get_telemetry_metrics() -> dict[str, Any]:
    """Return ingestion health and latency metrics."""
    avg_latency = float(sum(_LATENCY_SAMPLES) / len(_LATENCY_SAMPLES)) if _LATENCY_SAMPLES else 0.0
    return {
        "buffered_readings": len(_TELEMETRY_BUFFER),
        "tracked_machines": len(_LATEST_STATE),
        "rejected_payloads_total": len(_REJECTIONS),
        "average_latency_ms": round(avg_latency, 3),
        "feature_store_machines": list(_FEATURE_STORE.buffers.keys()),
    }

