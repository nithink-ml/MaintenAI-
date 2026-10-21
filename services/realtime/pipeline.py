"""
MaintenAI— Real-Time Telemetry & ML Inference Pipeline

Orchestrates the complete stream processing loop:
  Raw Telemetry
      ↓
  Sliding Window Buffer (OnlineFeatureStore)
      ↓
  Online Feature Extraction
      ↓
  Multi-Model Inference (IsolationForest, XGBoost RUL, LightGBM Failure Modes)
      ↓
  MachineHealth Object Creation (Normalized Health Score, MachineStatus)
      ↓
  Deterministic Alert Engine (Hysteresis, Deduplication)
      ↓
  Redis Live State Update (machine:{id}:state)
      ↓
  TimescaleDB Predictions & Alerts Persistence

Records high-resolution latencies for every phase (t_ingest, t_feature, t_infer, t_alert, t_e2e).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import pandas as pd

from factorypulse.config import get_settings
from factorypulse.redis_state import RedisStateManager
from factorypulse.schemas import (
    AlertSeverity,
    AnomalyResult,
    FailureMode,
    FailureModeResult,
    MachineHealth,
    MachineStatus,
    RULResult,
    SensorReading,
)
from ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from ml.failure_classification.lightgbm_model import LightGBMFailureClassifier
from ml.rul.xgboost_model import XGBoostRULRegressor
from services.alert_service.engine import AlertRuleEngine
from services.feature_service.calculator import OnlineFeatureStore

logger = logging.getLogger(__name__)

# Default model artifact locations
DEFAULT_ANOMALY_MODEL_PATH = Path("models/anomaly/isolation_forest.joblib")
DEFAULT_RUL_MODEL_PATH = Path("models/rul/xgboost.joblib")
DEFAULT_FAILURE_MODEL_PATH = Path("models/failure/lightgbm.joblib")


class RealTimePipeline:
    """
    Unified end-to-end real-time stream processing and inference pipeline.
    """

    def __init__(
        self,
        anomaly_model_path: Path | str = DEFAULT_ANOMALY_MODEL_PATH,
        rul_model_path: Path | str = DEFAULT_RUL_MODEL_PATH,
        failure_model_path: Path | str = DEFAULT_FAILURE_MODEL_PATH,
        window_size: int = 30,
        enable_redis: bool = True,
        pg_pool: asyncpg.Pool | None = None,
    ) -> None:
        self.settings = get_settings()
        self.feature_store = OnlineFeatureStore(window_size=window_size)
        self.alert_engine = AlertRuleEngine(
            anomaly_warning=0.65,
            anomaly_critical=0.80,
            rul_warning=35.0,
            rul_critical=15.0,
            failure_prob_warning=0.50,
            failure_prob_critical=0.75,
            hysteresis_cycles=2,
            dedup_window_minutes=15,
        )

        # Redis Live State
        self.redis_mgr: RedisStateManager | None = None
        if enable_redis:
            try:
                self.redis_mgr = RedisStateManager()
                if not self.redis_mgr.is_connected():
                    logger.warning("Redis is not connected; caching will be in-memory only")
            except Exception as e:
                logger.warning("RedisStateManager initialization failed: %s", e)
                self.redis_mgr = None

        # TimescaleDB Pool
        self.pg_pool = pg_pool

        # Load Phase 5 Model Artifacts
        logger.info("Loading Phase 5 ML model artifacts...")
        self.anomaly_detector = IsolationForestAnomalyDetector.load(anomaly_model_path)
        self.rul_regressor = XGBoostRULRegressor.load(rul_model_path)
        self.failure_classifier = LightGBMFailureClassifier.load(failure_model_path)
        logger.info(
            "Models loaded: Anomaly (%d feats), RUL (%d feats), Failure (%d feats)",
            len(self.anomaly_detector.feature_names),
            len(self.rul_regressor.feature_names),
            len(self.failure_classifier.feature_names),
        )

    def process_reading(self, raw_reading: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronously process a single sensor reading through the entire pipeline.
        """
        t_start = time.perf_counter()

        # ----------------------------------------------------
        # 1. Telemetry Ingestion & Window Update
        # ----------------------------------------------------
        t_ingest_start = time.perf_counter()

        # Standardize timestamp
        ts_val = raw_reading.get("timestamp")
        if isinstance(ts_val, str):
            ts = datetime.fromisoformat(ts_val)
        elif isinstance(ts_val, datetime):
            ts = ts_val
        else:
            ts = datetime.now(UTC)

        machine_id = str(raw_reading.get("machine_id", "M001")).strip().upper()
        reading_record = dict(raw_reading)
        reading_record["machine_id"] = machine_id
        reading_record["timestamp"] = ts.isoformat()
        if "source" not in reading_record:
            reading_record["source"] = "SYNTHETIC / DEVELOPMENT"

        # Validate with SensorReading schema
        sensor_obj = SensorReading.model_validate(reading_record)
        self.feature_store.append_reading(sensor_obj.model_dump(mode="json"))

        # Update Redis Telemetry Live State
        if self.redis_mgr and self.redis_mgr.is_connected():
            self.redis_mgr.update_telemetry(machine_id, sensor_obj.model_dump(mode="json"))

        t_ingest_ms = (time.perf_counter() - t_ingest_start) * 1000.0

        # ----------------------------------------------------
        # 2. Online Feature Computation
        # ----------------------------------------------------
        t_feat_start = time.perf_counter()
        features = self.feature_store.compute_features(machine_id)
        if not features:
            features = {
                "temperature": float(sensor_obj.temperature or 70.0),
                "pressure": float(sensor_obj.pressure or 100.0),
                "vibration": float(sensor_obj.vibration or 1.0),
                "rpm": float(sensor_obj.rpm or 2000.0),
                "current": float(sensor_obj.current or 25.0),
                "operating_cycle": float(sensor_obj.operating_cycle or 1),
            }

        df_feat = pd.DataFrame([features])
        t_feature_ms = (time.perf_counter() - t_feat_start) * 1000.0

        # ----------------------------------------------------
        # 3. ML Model Inference
        # ----------------------------------------------------
        t_infer_start = time.perf_counter()

        # 3a. Anomaly Detection
        anom_scores = self.anomaly_detector.predict_score(df_feat)
        anomaly_score = float(anom_scores[0])
        top_anom_features = self.anomaly_detector.explain_instance(df_feat.iloc[0])

        if anomaly_score >= 0.80:
            anom_sev = AlertSeverity.CRITICAL
            anom_status = MachineStatus.CRITICAL
        elif anomaly_score >= 0.65:
            anom_sev = AlertSeverity.WARNING
            anom_status = MachineStatus.WARNING
        else:
            anom_sev = AlertSeverity.INFO
            anom_status = MachineStatus.HEALTHY

        anomaly_res = AnomalyResult(
            machine_id=machine_id,
            timestamp=ts,
            anomaly_score=round(anomaly_score, 4),
            severity=anom_sev,
            status=anom_status,
            model_version="isolation_forest_v1",
            contributing_features=top_anom_features,
        )

        # 3b. Remaining Useful Life (RUL)
        rul_preds = self.rul_regressor.predict(df_feat)
        predicted_rul = float(rul_preds[0])
        rul_res = RULResult(
            machine_id=machine_id,
            timestamp=ts,
            predicted_rul=round(predicted_rul, 1),
            confidence_lower=round(max(0.0, predicted_rul - 12.0), 1),
            confidence_upper=round(predicted_rul + 15.0, 1),
            model_version="xgboost_rul_v1",
        )

        # 3c. Failure Mode Classification
        fc_out = self.failure_classifier.predict(df_feat)
        fail_prob = float(fc_out["failure_probability"][0])
        mode_probs_raw = fc_out["mode_probabilities"]
        mode_probs = {k: float(v[0]) for k, v in mode_probs_raw.items()}

        mode_mapping = {
            "failure_tool_wear": FailureMode.TOOL_WEAR,
            "failure_heat_dissipation": FailureMode.HEAT_DISSIPATION,
            "failure_power_failure": FailureMode.POWER_FAILURE,
            "failure_overstrain": FailureMode.OVERSTRAIN,
            "failure_random": FailureMode.RANDOM,
        }

        top_mode_key = max(mode_probs.items(), key=lambda x: x[1])[0] if mode_probs else None
        predicted_mode = (
            mode_mapping.get(top_mode_key, FailureMode.NONE)
            if fail_prob >= 0.40 and top_mode_key
            else FailureMode.NONE
        )

        fc_res = FailureModeResult(
            machine_id=machine_id,
            timestamp=ts,
            predicted_mode=predicted_mode,
            probabilities={k: round(v, 4) for k, v in mode_probs.items()},
            model_version="lightgbm_failure_v1",
        )

        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # ----------------------------------------------------
        # 4. Composite Machine Health Calculation
        # ----------------------------------------------------
        # Deterministic composite health formula in [0.0, 1.0]
        # Combines anomaly score (45%), failure probability (35%), and RUL depletion (20%)
        rul_depletion = max(0.0, min(1.0, 1.0 - (predicted_rul / 100.0)))
        raw_health = 1.0 - (0.45 * anomaly_score + 0.35 * fail_prob + 0.20 * rul_depletion)
        health_score = round(max(0.0, min(1.0, raw_health)), 3)

        if health_score >= 0.75 and anomaly_score < 0.65 and fail_prob < 0.50:
            machine_status = MachineStatus.HEALTHY
        elif health_score >= 0.45:
            machine_status = MachineStatus.WARNING
        else:
            machine_status = MachineStatus.CRITICAL

        machine_health = MachineHealth(
            machine_id=machine_id,
            timestamp=ts,
            health_score=health_score,
            status=machine_status,
            anomaly=anomaly_res,
            rul=rul_res,
            failure_mode=fc_res,
            latest_telemetry=sensor_obj,
        )

        # ----------------------------------------------------
        # 5. Deterministic Alert Engine Evaluation
        # ----------------------------------------------------
        t_alert_start = time.perf_counter()
        health_payload = {
            "machine_id": machine_id,
            "anomaly_score": anomaly_score,
            "predicted_rul_cycles": predicted_rul,
            "failure_probability": fail_prob,
            "probable_failure_modes": mode_probs,
        }
        triggered_alerts = self.alert_engine.evaluate_machine(health_payload)

        # If critical alert triggered, elevate status
        if any(a.get("severity") == "CRITICAL" for a in triggered_alerts):
            machine_status = MachineStatus.CRITICAL
            machine_health.status = machine_status

        # Update Redis Live State
        if self.redis_mgr and self.redis_mgr.is_connected():
            self.redis_mgr.update_health(machine_id, machine_health.model_dump(mode="json"))
            if triggered_alerts:
                self.redis_mgr.update_alerts(machine_id, self.alert_engine.active_alerts)

        t_alert_ms = (time.perf_counter() - t_alert_start) * 1000.0
        t_e2e_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "machine_id": machine_id,
            "timestamp": ts.isoformat(),
            "machine_health": machine_health.model_dump(mode="json"),
            "triggered_alerts": triggered_alerts,
            "latencies": {
                "t_ingest_ms": round(t_ingest_ms, 3),
                "t_feature_ms": round(t_feature_ms, 3),
                "t_infer_ms": round(t_infer_ms, 3),
                "t_alert_ms": round(t_alert_ms, 3),
                "t_e2e_ms": round(t_e2e_ms, 3),
            },
        }

    async def persist_to_timescaledb(self, output: dict[str, Any]) -> None:
        """Asynchronously persist telemetry, predictions, and alerts to TimescaleDB."""
        if not self.pg_pool:
            return

        m_id = output["machine_id"]
        ts = datetime.fromisoformat(output["timestamp"])
        mh = output["machine_health"]

        try:
            async with self.pg_pool.acquire() as conn:
                # 0. Insert raw sensor reading into hypertable
                telem = mh.get("latest_telemetry") or {}
                if telem:
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
                        telem.get("temperature"),
                        telem.get("pressure"),
                        telem.get("vibration"),
                        telem.get("rpm"),
                        telem.get("current"),
                        telem.get("operating_cycle"),
                        telem.get("source", "SYNTHETIC / DEVELOPMENT"),
                        json.dumps({"fault_injected": telem.get("fault_injected")}),
                    )

                # 1. Insert prediction record
                await conn.execute(
                    """
                    INSERT INTO predictions (
                        machine_id, timestamp, anomaly_score, anomaly_status,
                        predicted_rul_cycles, failure_probability, probable_failure_modes,
                        model_versions, latency_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9);
                    """,
                    m_id,
                    ts,
                    mh["anomaly"]["anomaly_score"],
                    mh["status"].upper(),
                    mh["rul"]["predicted_rul"],
                    mh["failure_mode"]["probabilities"].get("failure_tool_wear", 0.0),
                    json.dumps(mh["failure_mode"]["probabilities"]),
                    json.dumps({
                        "anomaly": mh["anomaly"]["model_version"],
                        "rul": mh["rul"]["model_version"],
                        "failure_mode": mh["failure_mode"]["model_version"],
                    }),
                    output["latencies"]["t_e2e_ms"],
                )

                # 2. Insert any newly triggered alerts
                for alert in output.get("triggered_alerts", []):
                    thresh_val = str(
                        self.alert_engine.anomaly_critical
                        if "ANOMALY" in alert["alert_type"]
                        else self.alert_engine.rul_critical
                    )
                    await conn.execute(
                        """
                        INSERT INTO alerts (
                            id, machine_id, timestamp, severity,
                            alert_type, title, description,
                            metric_value, threshold, acknowledged
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (id) DO NOTHING;
                        """,
                        alert["alert_id"],
                        m_id,
                        datetime.fromisoformat(alert["timestamp"]),
                        alert["severity"].upper(),
                        alert["alert_type"],
                        alert["title"],
                        alert["description"],
                        float(alert["metric_value"]),
                        thresh_val,
                        False,
                    )
        except Exception as e:
            logger.error("TimescaleDB persistence error for machine %s: %s", m_id, e)
