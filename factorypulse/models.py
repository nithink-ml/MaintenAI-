"""
MaintenAI— SQLAlchemy ORM Models

Defines database table mappings for TimescaleDB. These models represent
the persistent storage schema for machines, telemetry, predictions,
alerts, work orders, users, and audit logs.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from factorypulse.database import Base
from factorypulse.schemas import (
    AlertSeverity,
    AlertStatus,
    MachineStatus,
    UserRole,
    WorkOrderStatus,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


# ============================================================
# Machines
# ============================================================


class MachineModel(Base):
    __tablename__ = "machines"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    equipment_type: Mapped[str | None] = mapped_column(String(100))
    location: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(
        Enum(MachineStatus, name="machine_status"),
        default=MachineStatus.UNKNOWN,
    )
    installed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)

    # Relationships
    sensor_readings: Mapped[list["SensorReadingModel"]] = relationship(
        back_populates="machine", lazy="dynamic"
    )
    alerts: Mapped[list["AlertModel"]] = relationship(back_populates="machine", lazy="dynamic")


# ============================================================
# Sensor Readings (TimescaleDB Hypertable)
# ============================================================


class SensorReadingModel(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    machine_id: Mapped[str] = mapped_column(String(50), ForeignKey("machines.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    temperature: Mapped[float | None] = mapped_column(Float)
    pressure: Mapped[float | None] = mapped_column(Float)
    vibration: Mapped[float | None] = mapped_column(Float)
    rpm: Mapped[float | None] = mapped_column(Float)
    current: Mapped[float | None] = mapped_column(Float)
    operating_cycle: Mapped[int | None] = mapped_column(Integer)

    # Relationships
    machine: Mapped["MachineModel"] = relationship(back_populates="sensor_readings")

    __table_args__ = (Index("ix_sensor_readings_machine_time", "machine_id", "timestamp"),)


# ============================================================
# Predictions
# ============================================================


class PredictionModel(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    machine_id: Mapped[str] = mapped_column(String(50), ForeignKey("machines.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    prediction_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # anomaly, rul, failure_mode
    result: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)

    __table_args__ = (
        Index("ix_predictions_machine_time", "machine_id", "timestamp"),
        Index("ix_predictions_type", "prediction_type"),
    )


# ============================================================
# Alerts
# ============================================================


class AlertModel(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    machine_id: Mapped[str] = mapped_column(String(50), ForeignKey("machines.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    severity: Mapped[str] = mapped_column(
        Enum(AlertSeverity, name="alert_severity"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum(AlertStatus, name="alert_status"), default=AlertStatus.ACTIVE
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text)
    trigger_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_value: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(100))
    acknowledged_by: Mapped[str | None] = mapped_column(String(50))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    machine: Mapped["MachineModel"] = relationship(back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_machine_status", "machine_id", "status"),
        Index("ix_alerts_severity", "severity"),
    )


# ============================================================
# Model Versions
# ============================================================


class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    stage: Mapped[str] = mapped_column(
        String(20), default="staging"
    )  # staging, shadow, production, archived
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100))
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    parameters: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_by: Mapped[str | None] = mapped_column(String(50))


# ============================================================
# Work Orders
# ============================================================


class WorkOrderModel(Base):
    __tablename__ = "work_orders"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    machine_id: Mapped[str] = mapped_column(String(50), ForeignKey("machines.id"), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(WorkOrderStatus, name="work_order_status"),
        default=WorkOrderStatus.DRAFT,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str] = mapped_column(String(50), nullable=False)
    issue_summary: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_actions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    priority: Mapped[str] = mapped_column(
        Enum(AlertSeverity, name="work_order_priority"), nullable=False
    )
    estimated_downtime_hours: Mapped[float | None] = mapped_column(Float)
    parts_needed: Mapped[dict | None] = mapped_column(JSONB)
    safety_notes: Mapped[dict | None] = mapped_column(JSONB)
    evidence: Mapped[dict | None] = mapped_column(JSONB)

    # Approval
    approved_by: Mapped[str | None] = mapped_column(String(50))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    approval_notes: Mapped[str | None] = mapped_column(Text)
    rejection_reason: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_work_orders_machine", "machine_id"),
        Index("ix_work_orders_status", "status"),
    )


# ============================================================
# Users
# ============================================================


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(Enum(UserRole, name="user_role"), default=UserRole.TECHNICIAN)
    full_name: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ============================================================
# Audit Logs
# ============================================================


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(100))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ip_address: Mapped[str | None] = mapped_column(String(50))
    details: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("ix_audit_logs_user", "user_id"),
        Index("ix_audit_logs_timestamp", "timestamp"),
    )


# ============================================================
# Maintenance Events (Historical)
# ============================================================


class MaintenanceEventModel(Base):
    __tablename__ = "maintenance_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    machine_id: Mapped[str] = mapped_column(String(50), ForeignKey("machines.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    performed_by: Mapped[str | None] = mapped_column(String(50))
    performed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    work_order_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("work_orders.id")
    )
    details: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (Index("ix_maintenance_events_machine", "machine_id"),)
