"""
MaintenAI— Persistent Audit Log

Records security-relevant actions (authentication, work-order lifecycle,
alert acknowledgment, fault injection) to an append-only JSONL file with an
in-memory ring buffer for fast API access.

Notes on tamper resistance:
  - The log is append-only from the application's perspective (no update or
    delete API is exposed).
  - It is NOT cryptographically tamper-evident; a hash chain is intentionally
    not claimed. Treat this file as an audit record, not as an immutable ledger.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_PATH = Path("data/audit/audit_log.jsonl")
_RING_MAX = 2000

_lock = threading.Lock()
_ring: deque[dict[str, Any]] = deque(maxlen=_RING_MAX)


def _resolve_path(path: Path | str | None) -> Path:
    return Path(path) if path else DEFAULT_AUDIT_PATH


def record_audit_event(
    action: str,
    *,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    outcome: str = "success",
    ip_address: str | None = None,
    details: dict[str, Any] | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Append one audit event to the JSONL log and the in-memory ring buffer.

    Never pass secrets (passwords, tokens, API keys) in `details`.
    """
    event = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "action": action,
        "user_id": user_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "outcome": outcome,
        "ip_address": ip_address,
        "details": details or {},
    }

    with _lock:
        _ring.append(event)
        target = _resolve_path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError:
            # Audit persistence must never take down the request path; the
            # in-memory ring still holds the event for this process.
            event["persistence_error"] = True

    return event


def read_audit_events(
    limit: int = 100,
    *,
    action: str | None = None,
    resource_id: str | None = None,
    path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Read the most recent audit events (newest first) from disk + ring buffer."""
    events: list[dict[str, Any]] = []
    target = _resolve_path(path)
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    with _lock:
        if not events:
            events = list(_ring)

    if action:
        events = [e for e in events if e.get("action") == action]
    if resource_id:
        events = [e for e in events if e.get("resource_id") == resource_id]
    return events[-limit:][::-1]
