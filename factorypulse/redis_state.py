"""
MaintenAI— Redis Live State Manager

Maintains volatile real-time machine telemetry, health scoring, and active alert state
in Redis with structured keys, enabling sub-millisecond lookups for SCADA dashboards and agents.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import redis

from factorypulse.config import get_settings

logger = logging.getLogger(__name__)


class RedisStateManager:
    """
    Manages live state caching in Redis under 'machine:{machine_id}:state' keys.
    """

    def __init__(self, host: str | None = None, port: int | None = None, password: str | None = None) -> None:
        settings = get_settings()
        self.host = host or settings.redis.host
        self.port = port or settings.redis.port
        self.password = password or settings.redis.password
        self._client: redis.Redis | None = None

    def get_client(self) -> redis.Redis:
        """Get or initialize Redis client connection pool."""
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
        return self._client

    def is_connected(self) -> bool:
        """Verify live connection to Redis."""
        try:
            return bool(self.get_client().ping())
        except Exception as e:
            logger.debug("Redis ping failed: %s", e)
            return False

    def _state_key(self, machine_id: str) -> str:
        return f"machine:{machine_id.strip().upper()}:state"

    def set_machine_state(
        self,
        machine_id: str,
        state: dict[str, Any],
        ttl_sec: int = 86400,
    ) -> bool:
        """Store the complete consolidated state document for a machine."""
        client = self.get_client()
        key = self._state_key(machine_id)
        payload = {
            **state,
            "machine_id": machine_id,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        try:
            client.set(key, json.dumps(payload, default=str), ex=ttl_sec)
            return True
        except Exception as e:
            logger.warning("Failed writing state to Redis for %s: %s", machine_id, e)
            return False

    def get_machine_state(self, machine_id: str) -> dict[str, Any] | None:
        """Retrieve live state for a machine."""
        client = self.get_client()
        key = self._state_key(machine_id)
        try:
            raw = client.get(key)
            if raw:
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning("Failed reading state from Redis for %s: %s", machine_id, e)
            return None

    def update_telemetry(self, machine_id: str, telemetry: dict[str, Any]) -> dict[str, Any]:
        """Update telemetry portion of machine state."""
        existing = self.get_machine_state(machine_id) or {
            "machine_id": machine_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        existing["latest_telemetry"] = telemetry
        existing["timestamp"] = telemetry.get("timestamp") or datetime.now(UTC).isoformat()
        self.set_machine_state(machine_id, existing)
        return existing

    def update_health(self, machine_id: str, health: dict[str, Any]) -> dict[str, Any]:
        """Update ML health and predictions portion of machine state."""
        existing = self.get_machine_state(machine_id) or {
            "machine_id": machine_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        existing["latest_health"] = health
        existing["latest_anomaly_score"] = health.get("anomaly_score")
        existing["latest_rul"] = health.get("predicted_rul_cycles")
        existing["latest_failure_probability"] = health.get("failure_probability")
        existing["model_version"] = health.get("model_versions")
        self.set_machine_state(machine_id, existing)
        return existing

    def update_alerts(self, machine_id: str, alerts: list[dict[str, Any]]) -> dict[str, Any]:
        """Update active alerts portion of machine state."""
        existing = self.get_machine_state(machine_id) or {
            "machine_id": machine_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        existing["active_alerts"] = alerts
        self.set_machine_state(machine_id, existing)
        return existing

    def get_all_machines_state(self) -> list[dict[str, Any]]:
        """Retrieve state for all active machines stored in Redis."""
        client = self.get_client()
        try:
            keys = client.keys("machine:*:state")
            if not keys:
                return []
            pipe = client.pipeline()
            for k in keys:
                pipe.get(k)
            values = pipe.execute()
            results = []
            for v in values:
                if v:
                    results.append(json.loads(v))
            return results
        except Exception as e:
            logger.warning("Failed retrieving all states from Redis: %s", e)
            return []


# Global singleton instance
redis_state = RedisStateManager()
