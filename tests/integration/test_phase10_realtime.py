"""
MaintenAI— Phase 10 Real-Time Infrastructure & Stream Processing Integration Tests

Verifies all 14 requirements of Phase 10:
  1.  TimescaleDB connectivity & hypertable check.
  2.  Redis connectivity & auth check.
  3.  Telemetry schema validation (with source metadata).
  4.  Invalid telemetry rejection & audit logging.
  5.  Telemetry insertion into TimescaleDB hypertable.
  6.  Redis live state storage & retrieval under machine:{id}:state.
  7.  Online feature windowing without cross-machine leakage.
  8.  Feature consistency with Phase 5 model input spaces.
  9.  Phase 5 model loading directly from disk artifacts.
  10. Real-time multi-model inference execution.
  11. Normalized MachineHealth object creation & scoring.
  12. Deterministic alert threshold triggering.
  13. Alert hysteresis & deduplication window suppression.
  14. End-to-end latency measurement across all pipeline stages.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import asyncpg
import pandas as pd
import pytest
import redis

from factorypulse.config import get_settings
from factorypulse.redis_state import RedisStateManager
from factorypulse.schemas import (
    SensorReading,
    TelemetryRejection,
)
from ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from ml.failure_classification.lightgbm_model import LightGBMFailureClassifier
from ml.rul.xgboost_model import XGBoostRULRegressor
from services.alert_service.engine import AlertRuleEngine
from services.feature_service.calculator import OnlineFeatureStore
from services.realtime.pipeline import RealTimePipeline
from services.simulator.m001_simulator import M001Simulator


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def redis_client(settings):
    client = redis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password,
        decode_responses=True,
    )
    return client


# ----------------------------------------------------------------------
# 1. TimescaleDB Connectivity & Hypertable Check
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_01_timescaledb_hypertable(settings):
    db_url = f"postgresql://{settings.db.user}:{settings.db.password}@{settings.db.host}:{settings.db.port}/{settings.db.db}"
    conn = await asyncpg.connect(db_url)
    try:
        # Check hypertable registration
        row = await conn.fetchrow(
            "SELECT hypertable_schema, hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_readings';"
        )
        assert row is not None, "sensor_readings must be registered as a TimescaleDB hypertable"
        assert row["hypertable_name"] == "sensor_readings"
    finally:
        await conn.close()


# ----------------------------------------------------------------------
# 2. Redis Connectivity & Auth Check
# ----------------------------------------------------------------------
def test_02_redis_connectivity(redis_client):
    pong = redis_client.ping()
    assert pong is True, "Redis must respond with PONG to ping"


# ----------------------------------------------------------------------
# 3. Telemetry Schema Validation
# ----------------------------------------------------------------------
def test_03_telemetry_schema_validation():
    now = datetime.now(UTC)
    payload = {
        "machine_id": "M001",
        "timestamp": now,
        "temperature": 72.5,
        "pressure": 101.3,
        "vibration": 1.25,
        "rpm": 2000.0,
        "current": 28.5,
        "operating_cycle": 10,
        "source": "SYNTHETIC / DEVELOPMENT",
    }
    reading = SensorReading(**payload)
    assert reading.machine_id == "M001"
    assert reading.source == "SYNTHETIC / DEVELOPMENT"
    assert reading.temperature == 72.5


# ----------------------------------------------------------------------
# 4. Invalid Telemetry Rejection & Audit
# ----------------------------------------------------------------------
def test_04_invalid_telemetry_rejection():
    # Future timestamp (> 5 min)
    future_time = datetime.now(UTC) + timedelta(hours=2)
    with pytest.raises(ValueError, match="Timestamp cannot be more than 5 minutes in the future"):
        SensorReading(
            machine_id="M001",
            timestamp=future_time,
            temperature=75.0,
        )

    # Temperature out of physical bounds
    with pytest.raises(ValueError):
        SensorReading(
            machine_id="M001",
            timestamp=datetime.now(UTC),
            temperature=2000.0,  # Max is 1500
        )

    # TelemetryRejection audit schema
    audit = TelemetryRejection(
        machine_id="M001",
        raw_payload={"temperature": 2000.0},
        rejection_reasons=["temperature: Input should be less than or equal to 1500"],
    )
    assert audit.machine_id == "M001"
    assert len(audit.rejection_reasons) == 1


# ----------------------------------------------------------------------
# 5. Telemetry Insertion into TimescaleDB Hypertable
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_05_timescaledb_insert_telemetry(settings):
    db_url = f"postgresql://{settings.db.user}:{settings.db.password}@{settings.db.host}:{settings.db.port}/{settings.db.db}"
    conn = await asyncpg.connect(db_url)
    try:
        test_ts = datetime.now(UTC)
        await conn.execute(
            """
            INSERT INTO sensor_readings (
                machine_id, timestamp, temperature, pressure,
                vibration, rpm, current, operating_cycle, source
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (machine_id, timestamp) DO NOTHING;
            """,
            "M001",
            test_ts,
            68.4,
            101.2,
            1.15,
            2005.0,
            28.2,
            5,
            "SYNTHETIC / DEVELOPMENT",
        )

        row = await conn.fetchrow(
            "SELECT machine_id, temperature, vibration, source FROM sensor_readings WHERE machine_id = $1 AND timestamp = $2;",
            "M001",
            test_ts,
        )
        assert row is not None
        assert row["machine_id"] == "M001"
        assert row["temperature"] == 68.4
        assert row["source"] == "SYNTHETIC / DEVELOPMENT"
    finally:
        await conn.close()


# ----------------------------------------------------------------------
# 6. Redis Live State Storage & Retrieval
# ----------------------------------------------------------------------
def test_06_redis_state_storage():
    redis_mgr = RedisStateManager()
    assert redis_mgr.is_connected() is True

    machine_id = "M001"
    payload = {
        "machine_id": machine_id,
        "temperature": 70.0,
        "vibration": 1.2,
        "source": "SYNTHETIC / DEVELOPMENT",
    }
    redis_mgr.update_telemetry(machine_id, payload)
    state = redis_mgr.get_machine_state(machine_id)
    assert state is not None
    assert state["latest_telemetry"]["temperature"] == 70.0
    assert state["latest_telemetry"]["source"] == "SYNTHETIC / DEVELOPMENT"


# ----------------------------------------------------------------------
# 7. Online Feature Windowing Without Cross-Machine Leakage
# ----------------------------------------------------------------------
def test_07_temporal_features_no_cross_machine_leakage():
    store = OnlineFeatureStore(window_size=10)

    # Machine 1 has low vibration
    for i in range(5):
        store.append_reading({
            "machine_id": "M001",
            "vibration": 1.0 + (i * 0.1),
            "temperature": 60.0,
            "rpm": 2000.0,
        })

    # Machine 2 has high vibration
    for i in range(5):
        store.append_reading({
            "machine_id": "bearing-004",
            "vibration": 8.0 + (i * 0.2),
            "temperature": 95.0,
            "rpm": 1500.0,
        })

    feats_m1 = store.compute_features("M001")
    feats_m2 = store.compute_features("bearing-004")

    assert feats_m1 is not None
    assert feats_m2 is not None

    # M001 rolling mean vibration should be around 1.2
    assert 1.0 <= feats_m1["vibration_rolling_mean_5"] <= 1.5
    # bearing-004 rolling mean vibration should be around 8.4
    assert 8.0 <= feats_m2["vibration_rolling_mean_5"] <= 9.0

    # Ensure no contamination in buffer sizes
    assert len(store.buffers["M001"]) == 5
    assert len(store.buffers["bearing-004"]) == 5


# ----------------------------------------------------------------------
# 8. Feature Consistency with Model Requirements
# ----------------------------------------------------------------------
def test_08_feature_consistency():
    store = OnlineFeatureStore(window_size=20)
    for i in range(10):
        store.append_reading({
            "machine_id": "M001",
            "temperature": 70.0 + i,
            "pressure": 100.0,
            "vibration": 1.2 + (i * 0.05),
            "rpm": 2000.0,
            "current": 30.0,
            "operating_cycle": i + 1,
        })

    feats = store.compute_features("M001")
    assert feats is not None

    # Verify presence of expected model features
    required_keys = [
        "temperature", "vibration", "rpm", "current",
        "temperature_rolling_mean_5", "vibration_rolling_std_5",
        "power", "temp_diff", "torque_per_rpm", "raw_peak", "raw_std"
    ]
    for k in required_keys:
        assert k in feats, f"Expected feature '{k}' missing from online feature computation"


# ----------------------------------------------------------------------
# 9. Phase 5 Model Loading from Disk
# ----------------------------------------------------------------------
def test_09_model_loading():
    iso = IsolationForestAnomalyDetector.load("models/anomaly/isolation_forest.joblib")
    rul = XGBoostRULRegressor.load("models/rul/xgboost.joblib")
    fc = LightGBMFailureClassifier.load("models/failure/lightgbm.joblib")

    assert iso.model is not None
    assert len(iso.feature_names) > 0

    assert rul.model is not None
    assert len(rul.feature_names) > 0

    assert fc.binary_model is not None
    assert len(fc.feature_names) > 0


# ----------------------------------------------------------------------
# 10. Multi-Model Inference Execution
# ----------------------------------------------------------------------
def test_10_multimodel_inference():
    iso = IsolationForestAnomalyDetector.load("models/anomaly/isolation_forest.joblib")
    rul = XGBoostRULRegressor.load("models/rul/xgboost.joblib")
    fc = LightGBMFailureClassifier.load("models/failure/lightgbm.joblib")

    sample_df = pd.DataFrame([{
        "temperature": 70.0,
        "vibration": 1.2,
        "pressure": 101.3,
        "rpm": 2000.0,
        "current": 28.0,
        "operating_cycle": 10,
    }])

    # Anomaly
    score = iso.predict_score(sample_df)
    assert len(score) == 1
    assert 0.0 <= score[0] <= 1.0

    # RUL
    pred_rul = rul.predict(sample_df)
    assert len(pred_rul) == 1
    assert pred_rul[0] >= 0.0

    # Failure Mode
    pred_fc = fc.predict(sample_df)
    assert "failure_probability" in pred_fc
    assert 0.0 <= pred_fc["failure_probability"][0] <= 1.0


# ----------------------------------------------------------------------
# 11. MachineHealth Object Creation & Scoring
# ----------------------------------------------------------------------
def test_11_machine_health_object():
    pipeline = RealTimePipeline()
    now = datetime.now(UTC)
    reading = {
        "machine_id": "M001",
        "timestamp": now.isoformat(),
        "temperature": 70.0,
        "pressure": 101.3,
        "vibration": 1.1,
        "rpm": 2000.0,
        "current": 28.0,
        "operating_cycle": 1,
        "source": "SYNTHETIC / DEVELOPMENT",
    }

    out = pipeline.process_reading(reading)
    mh = out["machine_health"]

    assert mh["machine_id"] == "M001"
    assert 0.0 <= mh["health_score"] <= 1.0
    assert mh["status"] in ["healthy", "warning", "critical"]
    assert "anomaly" in mh
    assert "rul" in mh
    assert "failure_mode" in mh


# ----------------------------------------------------------------------
# 12. Alert Threshold Triggering
# ----------------------------------------------------------------------
def test_12_alert_triggering():
    engine = AlertRuleEngine(
        anomaly_critical=0.80,
        anomaly_warning=0.60,
        hysteresis_cycles=1,
        dedup_window_minutes=15,
    )

    health_data = {
        "machine_id": "M001",
        "anomaly_score": 0.88,
        "predicted_rul_cycles": 10.0,
        "failure_probability": 0.1,
    }

    alerts = engine.evaluate_machine(health_data)
    assert len(alerts) >= 1
    types = [a["alert_type"] for a in alerts]
    assert "ANOMALY_DETECTED" in types
    critical_alert = next(a for a in alerts if a["alert_type"] == "ANOMALY_DETECTED")
    assert critical_alert["severity"] == "CRITICAL"


# ----------------------------------------------------------------------
# 13. Alert Hysteresis & Deduplication
# ----------------------------------------------------------------------
def test_13_alert_hysteresis_and_deduplication():
    engine = AlertRuleEngine(
        anomaly_warning=0.70,
        anomaly_critical=0.85,
        hysteresis_cycles=2,
        dedup_window_minutes=15,
    )

    health_data = {
        "machine_id": "M001",
        "anomaly_score": 0.90,
        "predicted_rul_cycles": 100.0,
        "failure_probability": 0.0,
    }

    # Cycle 1: Exceeds threshold, but hysteresis is 2 cycles -> NO alert yet
    alerts_c1 = engine.evaluate_machine(health_data)
    assert len(alerts_c1) == 0

    # Cycle 2: Second consecutive violation -> Alert triggered!
    alerts_c2 = engine.evaluate_machine(health_data)
    assert len(alerts_c2) == 1
    assert alerts_c2[0]["severity"] == "CRITICAL"

    # Cycle 3: Still anomalous, but within 15-min dedup window -> Suppressed!
    alerts_c3 = engine.evaluate_machine(health_data)
    assert len(alerts_c3) == 0


# ----------------------------------------------------------------------
# 14. End-to-End Latency Measurement
# ----------------------------------------------------------------------
def test_14_end_to_end_latency():
    pipeline = RealTimePipeline()
    sim = M001Simulator(seed=100)

    reading = sim.generate_reading()
    res = pipeline.process_reading(reading)

    latencies = res["latencies"]
    assert "t_ingest_ms" in latencies
    assert "t_feature_ms" in latencies
    assert "t_infer_ms" in latencies
    assert "t_alert_ms" in latencies
    assert "t_e2e_ms" in latencies

    # All stage latencies should be non-negative numbers
    for k, v in latencies.items():
        assert v >= 0.0, f"Latency {k} must be non-negative"

    # Total pipeline end-to-end should be well under 1000ms
    assert latencies["t_e2e_ms"] < 1000.0
