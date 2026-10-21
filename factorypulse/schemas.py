"""
MaintenAI— Canonical Data Schemas

These Pydantic models define the canonical data contracts used across
all services: ingestion, feature engineering, inference, alerts,
and work orders. Using a single shared schema set prevents
training-serving skew and ensures consistent validation.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================
# Enumerations
# ============================================================


class MachineStatus(StrEnum):
    """Machine operational status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(StrEnum):
    """Alert lifecycle states."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class WorkOrderStatus(StrEnum):
    """Work order lifecycle states."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class UserRole(StrEnum):
    """RBAC user roles."""

    TECHNICIAN = "technician"
    ENGINEER = "engineer"
    MANAGER = "manager"
    ADMIN = "admin"


class FailureMode(StrEnum):
    """Failure mode categories from AI4I dataset."""

    TOOL_WEAR = "tool_wear"
    HEAT_DISSIPATION = "heat_dissipation"
    POWER_FAILURE = "power_failure"
    OVERSTRAIN = "overstrain"
    RANDOM = "random"
    NONE = "none"


# ============================================================
# Telemetry Schemas
# ============================================================


class SensorReading(BaseModel):
    """
    Canonical sensor telemetry record.

    This is the universal schema for all telemetry flowing through FactoryPulse.
    Source datasets (C-MAPSS, IMS, AI4I) are mapped into this schema during
    data preparation. The same schema is used by the simulator, ingestion,
    feature engineering, and inference services.
    """

    machine_id: str = Field(
        ..., min_length=1, max_length=50, description="Unique machine identifier"
    )
    timestamp: datetime = Field(..., description="UTC timestamp of the reading")
    temperature: float | None = Field(
        None, ge=-50.0, le=1500.0, description="Temperature in Celsius"
    )
    pressure: float | None = Field(None, ge=0.0, le=1000.0, description="Pressure in bar")
    vibration: float | None = Field(None, ge=0.0, le=500.0, description="Vibration in mm/s RMS")
    rpm: float | None = Field(None, ge=0.0, le=50000.0, description="Rotational speed in RPM")
    current: float | None = Field(None, ge=0.0, le=1000.0, description="Current draw in Amps")
    operating_cycle: int | None = Field(None, ge=0, description="Cumulative operating cycle count")
    equipment_type: str | None = Field(None, description="Equipment type classification")
    source: str = Field(default="SYNTHETIC / DEVELOPMENT", description="Data provenance label")
    fault_injected: str | None = Field(None, description="Active fault condition if test mode")

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_future(cls, v: datetime) -> datetime:
        """Reject timestamps more than 5 minutes in the future (clock skew tolerance)."""

        now = datetime.now(UTC)
        if v.tzinfo is None:
            # Treat naive datetimes as UTC

            v = v.replace(tzinfo=UTC)
        if v > now.replace(microsecond=0) + __import__("datetime").timedelta(minutes=5):
            raise ValueError("Timestamp cannot be more than 5 minutes in the future")
        return v


class TelemetryRejection(BaseModel):
    """Audit log of invalid sensor telemetry rejected at ingestion gate."""

    machine_id: str | None = None
    raw_payload: dict[str, Any]
    rejection_reasons: list[str]
    rejected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SensorReadingBatch(BaseModel):
    """Batch of sensor readings for bulk ingestion."""

    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)
    source: str = Field(default="SYNTHETIC / DEVELOPMENT", description="Data provenance label")


# ============================================================
# Prediction / Inference Schemas
# ============================================================


class AnomalyResult(BaseModel):
    """Anomaly detection inference output."""

    machine_id: str
    timestamp: datetime
    anomaly_score: float = Field(
        ..., ge=0.0, le=1.0, description="Anomaly score 0.0 (normal) to 1.0 (anomalous)"
    )
    severity: AlertSeverity
    status: MachineStatus
    model_version: str
    contributing_features: dict[str, float] | None = Field(
        None, description="Top features contributing to anomaly score"
    )


class RULResult(BaseModel):
    """Remaining Useful Life prediction output."""

    machine_id: str
    timestamp: datetime
    predicted_rul: float = Field(
        ..., ge=0.0, description="Predicted remaining useful life in cycles"
    )
    confidence_lower: float | None = Field(None, description="Lower bound of prediction interval")
    confidence_upper: float | None = Field(None, description="Upper bound of prediction interval")
    model_version: str


class FailureModeResult(BaseModel):
    """Failure mode classification output."""

    machine_id: str
    timestamp: datetime
    predicted_mode: FailureMode
    probabilities: dict[str, float] = Field(..., description="Probability for each failure mode")
    model_version: str


class MachineHealth(BaseModel):
    """Composite machine health assessment combining all models."""

    machine_id: str
    timestamp: datetime
    health_score: float = Field(..., ge=0.0, le=1.0, description="Composite health score")
    status: MachineStatus
    anomaly: AnomalyResult | None = None
    rul: RULResult | None = None
    failure_mode: FailureModeResult | None = None
    latest_telemetry: SensorReading | None = None


# ============================================================
# Alert Schemas
# ============================================================


class Alert(BaseModel):
    """Alert generated by the deterministic alert engine."""

    id: str | None = None
    machine_id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    alert_type: str = Field(..., description="e.g. anomaly_score, rul_low, failure_probability")
    title: str
    description: str
    trigger_value: float
    threshold_value: float
    model_version: str | None = None
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


# ============================================================
# Work Order Schemas
# ============================================================


class WorkOrderDraft(BaseModel):
    """AI-generated work order draft awaiting human approval."""

    machine_id: str
    issue_summary: str
    recommended_actions: list[str]
    priority: AlertSeverity
    estimated_downtime_hours: float | None = None
    parts_needed: list[str] | None = None
    safety_notes: list[str] | None = None
    evidence: dict[str, Any] | None = Field(
        None, description="ML predictions and RAG citations backing this draft"
    )


class WorkOrder(BaseModel):
    """Full work order with approval lifecycle."""

    id: str | None = None
    machine_id: str
    status: WorkOrderStatus = WorkOrderStatus.DRAFT
    created_at: datetime
    created_by: str  # agent or user
    issue_summary: str
    recommended_actions: list[str]
    priority: AlertSeverity
    estimated_downtime_hours: float | None = None
    parts_needed: list[str] | None = None
    safety_notes: list[str] | None = None
    evidence: dict[str, Any] | None = None

    # Approval fields
    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_notes: str | None = None
    rejection_reason: str | None = None


# ============================================================
# User / Auth Schemas
# ============================================================


class UserCreate(BaseModel):
    """User registration schema."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.TECHNICIAN
    full_name: str | None = None


class UserResponse(BaseModel):
    """User response schema (never exposes password)."""

    id: str
    username: str
    email: str
    role: UserRole
    full_name: str | None = None
    is_active: bool = True
    created_at: datetime


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int


# ============================================================
# Audit Schemas
# ============================================================


class AuditEntry(BaseModel):
    """Immutable audit log entry."""

    id: str | None = None
    user_id: str
    action: str
    resource_type: str
    resource_id: str | None = None
    timestamp: datetime
    ip_address: str | None = None
    details: dict[str, Any] | None = None
