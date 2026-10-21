"""
MaintenAI— Deterministic Alert Engine

Evaluates multi-condition deterministic alert rules with:
  - Severity classification (WARNING, CRITICAL)
  - Hysteresis filter (e.g. condition must persist for N consecutive cycles)
  - Deduplication window (e.g. suppress duplicate alerts within 15 minutes)
"""

from __future__ import annotations

import collections
import datetime
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class AlertRuleEngine:
    """
    Deterministic Alert Rule Engine with hysteresis and deduplication.
    """

    def __init__(
        self,
        anomaly_warning: float = 0.70,
        anomaly_critical: float = 0.85,
        rul_warning: float = 30.0,
        rul_critical: float = 15.0,
        failure_prob_warning: float = 0.50,
        failure_prob_critical: float = 0.70,
        hysteresis_cycles: int = 3,
        dedup_window_minutes: int = 15,
    ) -> None:
        self.anomaly_warning = anomaly_warning
        self.anomaly_critical = anomaly_critical
        self.rul_warning = rul_warning
        self.rul_critical = rul_critical
        self.failure_prob_warning = failure_prob_warning
        self.failure_prob_critical = failure_prob_critical
        self.hysteresis_cycles = hysteresis_cycles
        self.dedup_window_minutes = dedup_window_minutes

        # State tracking: machine_id -> condition_name -> consecutive_count
        self.consecutive_violations: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        # machine_id:rule_type -> last_alert_datetime
        self.last_alert_time: dict[str, datetime.datetime] = {}
        self.active_alerts: list[dict[str, Any]] = []

    def evaluate_machine(self, health_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Evaluate health metrics for a single machine and trigger new alerts if warranted.
        """
        m_id = health_data.get("machine_id", "unknown")
        anom = health_data.get("anomaly_score", 0.0)
        rul = health_data.get("predicted_rul_cycles", 100.0)
        health_data.get("failure_probability", 0.0)
        health_data.get("probable_failure_modes", {})

        now = datetime.datetime.now(datetime.UTC)
        triggered_alerts: list[dict[str, Any]] = []

        # 1. Evaluate Anomaly Score
        if anom >= self.anomaly_critical:
            self.consecutive_violations[m_id]["anomaly"] += 1
            severity = "CRITICAL"
        elif anom >= self.anomaly_warning:
            self.consecutive_violations[m_id]["anomaly"] += 1
            severity = "WARNING"
        else:
            self.consecutive_violations[m_id]["anomaly"] = 0
            severity = None

        if severity and self.consecutive_violations[m_id]["anomaly"] >= self.hysteresis_cycles:
            alert = self._create_alert(
                machine_id=m_id,
                alert_type="ANOMALY_DETECTED",
                severity=severity,
                title=f"{severity} Anomaly Detected on {m_id.upper()}",
                description=f"Anomaly score reached {anom:.2f} (threshold: {self.anomaly_warning if severity == 'WARNING' else self.anomaly_critical:.2f}) for {self.consecutive_violations[m_id]['anomaly']} consecutive cycles.",
                metric_value=anom,
                now=now,
            )
            if alert:
                triggered_alerts.append(alert)

        # 2. Evaluate RUL Countdown
        if rul <= self.rul_critical:
            self.consecutive_violations[m_id]["rul"] += 1
            severity = "CRITICAL"
        elif rul <= self.rul_warning:
            self.consecutive_violations[m_id]["rul"] += 1
            severity = "WARNING"
        else:
            self.consecutive_violations[m_id]["rul"] = 0
            severity = None

        if severity and self.consecutive_violations[m_id]["rul"] >= self.hysteresis_cycles:
            alert = self._create_alert(
                machine_id=m_id,
                alert_type="LOW_RUL_WARNING",
                severity=severity,
                title=f"{severity} Remaining Useful Life on {m_id.upper()}",
                description=f"Predicted RUL is {rul:.1f} cycles (threshold: {self.rul_warning if severity == 'WARNING' else self.rul_critical:.1f} cycles).",
                metric_value=rul,
                now=now,
            )
            if alert:
                triggered_alerts.append(alert)

        # 3. Evaluate Failure Probability
        fail_prob = health_data.get("failure_probability", 0.0)
        if fail_prob >= self.failure_prob_critical:
            self.consecutive_violations[m_id]["failure_prob"] += 1
            severity = "CRITICAL"
        elif fail_prob >= self.failure_prob_warning:
            self.consecutive_violations[m_id]["failure_prob"] += 1
            severity = "WARNING"
        else:
            self.consecutive_violations[m_id]["failure_prob"] = 0
            severity = None

        if severity and self.consecutive_violations[m_id]["failure_prob"] >= self.hysteresis_cycles:
            alert = self._create_alert(
                machine_id=m_id,
                alert_type="HIGH_FAILURE_PROBABILITY",
                severity=severity,
                title=f"{severity} Failure Risk on {m_id.upper()}",
                description=f"Predicted failure probability is {fail_prob:.2f} (threshold: {self.failure_prob_warning if severity == 'WARNING' else self.failure_prob_critical:.2f}).",
                metric_value=fail_prob,
                now=now,
            )
            if alert:
                triggered_alerts.append(alert)

        return triggered_alerts

    def _create_alert(
        self,
        machine_id: str,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        metric_value: float,
        now: datetime.datetime,
    ) -> dict[str, Any] | None:
        dedup_key = f"{machine_id}:{alert_type}:{severity}"
        last_time = self.last_alert_time.get(dedup_key)

        if last_time and (now - last_time).total_seconds() < (self.dedup_window_minutes * 60):
            # Suppressed by deduplication window
            return None

        self.last_alert_time[dedup_key] = now
        alert = {
            "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
            "machine_id": machine_id,
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "metric_value": metric_value,
            "timestamp": now.isoformat(),
            "acknowledged": False,
        }
        self.active_alerts.append(alert)
        logger.warning("Triggered Alert: [%s] %s - %s", severity, machine_id, title)
        return alert

    def get_all_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self.active_alerts))[:limit]

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        """Find alert by unique alert_id."""
        for a in self.active_alerts:
            if a["alert_id"] == alert_id:
                return a
        return None

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str | None = None
    ) -> dict[str, Any] | None:
        """Acknowledge alert and record audit details."""
        now = datetime.datetime.now(datetime.UTC)
        for a in self.active_alerts:
            if a["alert_id"] == alert_id:
                a["acknowledged"] = True
                a["acknowledged_by"] = acknowledged_by or "operator"
                a["acknowledged_at"] = now.isoformat()
                return a
        return None
