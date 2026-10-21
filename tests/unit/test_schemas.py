"""
MaintenAI— Unit Tests for Canonical Schemas
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


class TestSensorReading:
    """Test sensor telemetry schema validation."""

    def test_valid_reading(self):
        """Valid sensor readings should pass validation."""
        from factorypulse.schemas import SensorReading

        reading = SensorReading(
            machine_id="M001",
            timestamp=datetime.now(UTC),
            temperature=75.5,
            pressure=10.2,
            vibration=3.4,
            rpm=1500.0,
            current=12.5,
            operating_cycle=1000,
        )
        assert reading.machine_id == "M001"
        assert reading.temperature == 75.5

    def test_optional_fields(self):
        """Sensor fields should be optional (different datasets have different sensors)."""
        from factorypulse.schemas import SensorReading

        reading = SensorReading(
            machine_id="M002",
            timestamp=datetime.now(UTC),
        )
        assert reading.temperature is None
        assert reading.vibration is None

    def test_invalid_temperature_range(self):
        """Temperature outside physical bounds should be rejected."""
        from factorypulse.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(
                machine_id="M001",
                timestamp=datetime.now(UTC),
                temperature=2000.0,  # exceeds max 1500
            )

    def test_negative_vibration_rejected(self):
        """Negative vibration should be rejected (RMS is always positive)."""
        from factorypulse.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(
                machine_id="M001",
                timestamp=datetime.now(UTC),
                vibration=-5.0,
            )

    def test_empty_machine_id_rejected(self):
        """Empty machine ID should be rejected."""
        from factorypulse.schemas import SensorReading

        with pytest.raises(ValidationError):
            SensorReading(
                machine_id="",
                timestamp=datetime.now(UTC),
            )


class TestSensorReadingBatch:
    """Test batch telemetry validation."""

    def test_valid_batch(self):
        """Valid batch should pass validation."""
        from factorypulse.schemas import SensorReading, SensorReadingBatch

        batch = SensorReadingBatch(
            readings=[
                SensorReading(
                    machine_id="M001",
                    timestamp=datetime.now(UTC),
                    temperature=50.0,
                ),
                SensorReading(
                    machine_id="M002",
                    timestamp=datetime.now(UTC),
                    vibration=2.5,
                ),
            ]
        )
        assert len(batch.readings) == 2

    def test_empty_batch_rejected(self):
        """Empty batch should be rejected."""
        from factorypulse.schemas import SensorReadingBatch

        with pytest.raises(ValidationError):
            SensorReadingBatch(readings=[])


class TestAnomalyResult:
    """Test anomaly detection output schema."""

    def test_valid_anomaly_result(self):
        from factorypulse.schemas import AlertSeverity, AnomalyResult, MachineStatus

        result = AnomalyResult(
            machine_id="M001",
            timestamp=datetime.now(UTC),
            anomaly_score=0.91,
            severity=AlertSeverity.CRITICAL,
            status=MachineStatus.CRITICAL,
            model_version="isolation_forest_v1.2",
        )
        assert result.anomaly_score == 0.91

    def test_anomaly_score_bounds(self):
        """Anomaly score must be between 0.0 and 1.0."""
        from factorypulse.schemas import AlertSeverity, AnomalyResult, MachineStatus

        with pytest.raises(ValidationError):
            AnomalyResult(
                machine_id="M001",
                timestamp=datetime.now(UTC),
                anomaly_score=1.5,  # exceeds max 1.0
                severity=AlertSeverity.CRITICAL,
                status=MachineStatus.CRITICAL,
                model_version="v1",
            )


class TestWorkOrder:
    """Test work order schema."""

    def test_valid_work_order(self):
        from factorypulse.schemas import AlertSeverity, WorkOrder, WorkOrderStatus

        wo = WorkOrder(
            machine_id="M001",
            status=WorkOrderStatus.DRAFT,
            created_at=datetime.now(UTC),
            created_by="copilot_agent",
            issue_summary="Elevated vibration detected",
            recommended_actions=["Inspect bearing", "Check alignment"],
            priority=AlertSeverity.WARNING,
        )
        assert wo.status == WorkOrderStatus.DRAFT
        assert len(wo.recommended_actions) == 2


class TestEnumerations:
    """Test enum values."""

    def test_machine_status_values(self):
        from factorypulse.schemas import MachineStatus

        assert MachineStatus.HEALTHY.value == "healthy"
        assert MachineStatus.CRITICAL.value == "critical"

    def test_user_roles(self):
        from factorypulse.schemas import UserRole

        assert UserRole.TECHNICIAN.value == "technician"
        assert UserRole.ADMIN.value == "admin"

    def test_failure_modes(self):
        from factorypulse.schemas import FailureMode

        assert FailureMode.TOOL_WEAR.value == "tool_wear"
        assert FailureMode.HEAT_DISSIPATION.value == "heat_dissipation"
