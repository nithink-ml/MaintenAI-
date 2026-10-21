"""
MaintenAI— M001 Dedicated Operational Simulator

Generates deterministic, physically grounded synthetic telemetry for machine M001
across four operational stages:
  1. HEALTHY: Nominal operations, baseline physics, low vibration
  2. DEGRADING: Incipient wear, subtle thermal and vibration drift
  3. ANOMALOUS: Measurable harmonic disturbances, localized anomalies
  4. FAILURE-APPROACHING: Severe instability, critical thresholds exceeded

All generated readings are explicitly tagged with:
  "source": "SYNTHETIC / DEVELOPMENT"
"""

from __future__ import annotations

import datetime
import math
import random
from typing import Any


class M001Simulator:
    """
    Deterministic multi-phase telemetry simulator for machine M001.
    """

    def __init__(
        self,
        machine_id: str = "M001",
        equipment_type: str = "bearing",
        seed: int = 42,
        sampling_rate_hz: float = 1.0,
        noise_level: float = 0.05,
        degradation_start_cycle: int = 35,
        anomaly_start_cycle: int = 65,
        failure_start_cycle: int = 85,
    ) -> None:
        self.machine_id = machine_id
        self.equipment_type = equipment_type
        self.seed = seed
        self.sampling_rate_hz = sampling_rate_hz
        self.noise_level = noise_level
        self.degradation_start_cycle = degradation_start_cycle
        self.anomaly_start_cycle = anomaly_start_cycle
        self.failure_start_cycle = failure_start_cycle

        self.rng = random.Random(self.seed)
        self.current_cycle = 0
        self.active_fault: str | None = None
        self.fault_severity: float = 0.0

    def reset(self, cycle: int = 0) -> None:
        """Reset simulator state and RNG to initial seed."""
        self.rng = random.Random(self.seed)
        self.current_cycle = cycle
        self.active_fault = None
        self.fault_severity = 0.0

    def get_phase(self, cycle: int) -> str:
        """Determine operational phase based on cycle count."""
        if cycle >= self.failure_start_cycle:
            return "FAILURE-APPROACHING"
        if cycle >= self.anomaly_start_cycle:
            return "ANOMALOUS"
        if cycle >= self.degradation_start_cycle:
            return "DEGRADING"
        return "HEALTHY"

    def inject_fault(self, fault_type: str, severity: float = 1.0) -> None:
        """Inject specific simulated fault condition."""
        self.active_fault = fault_type
        self.fault_severity = max(0.1, min(1.0, severity))

    def clear_fault(self) -> None:
        """Clear active fault."""
        self.active_fault = None
        self.fault_severity = 0.0

    def generate_reading(
        self, timestamp: datetime.datetime | None = None
    ) -> dict[str, Any]:
        """
        Generate next deterministic sensor reading for M001.
        """
        self.current_cycle += 1
        cycle = self.current_cycle
        phase = self.get_phase(cycle)

        now = timestamp or datetime.datetime.now(datetime.UTC)

        # Baseline physical oscillations
        t_osc = math.sin(cycle * 0.12) * 1.5
        v_osc = abs(math.sin(cycle * 0.25)) * 0.15

        # Phase-dependent degradation factors
        if phase == "HEALTHY":
            temp_base = 68.0 + t_osc + self.rng.gauss(0, 0.4 * self.noise_level * 10)
            pressure_base = 101.3 + self.rng.gauss(0, 0.2 * self.noise_level * 10)
            rpm_base = 2000.0 + self.rng.gauss(0, 4.0 * self.noise_level * 10)
            vib_base = 1.05 + v_osc + self.rng.gauss(0, 0.04 * self.noise_level * 10)
            curr_base = 28.5 + self.rng.gauss(0, 0.25 * self.noise_level * 10)

        elif phase == "DEGRADING":
            deg_progress = (cycle - self.degradation_start_cycle) / 30.0
            temp_base = 72.0 + (deg_progress * 8.0) + t_osc + self.rng.gauss(0, 0.6)
            pressure_base = 101.1 - (deg_progress * 1.2) + self.rng.gauss(0, 0.3)
            rpm_base = 1995.0 - (deg_progress * 10.0) + self.rng.gauss(0, 6.0)
            vib_base = 1.4 + (deg_progress * 0.9) + v_osc + self.rng.gauss(0, 0.08)
            curr_base = 29.5 + (deg_progress * 2.0) + self.rng.gauss(0, 0.4)

        elif phase == "ANOMALOUS":
            anom_progress = (cycle - self.anomaly_start_cycle) / 20.0
            temp_base = 82.0 + (anom_progress * 12.0) + self.rng.gauss(0, 1.2)
            pressure_base = 99.5 - (anom_progress * 3.0) + self.rng.gauss(0, 0.5)
            rpm_base = 1970.0 - (anom_progress * 30.0) + self.rng.gauss(0, 12.0)
            vib_base = 2.6 + (anom_progress * 1.8) + (self.rng.random() * 0.6)
            curr_base = 32.0 + (anom_progress * 4.0) + self.rng.gauss(0, 0.8)

        else:  # FAILURE-APPROACHING
            fail_progress = (cycle - self.failure_start_cycle) / 15.0
            temp_base = 96.0 + (fail_progress * 22.0) + self.rng.gauss(0, 2.5)
            pressure_base = 95.0 - (fail_progress * 8.0) + self.rng.gauss(0, 1.0)
            rpm_base = 1900.0 - (fail_progress * 80.0) + self.rng.gauss(0, 25.0)
            vib_base = 4.8 + (fail_progress * 3.5) + (self.rng.random() * 1.5)
            curr_base = 38.0 + (fail_progress * 8.0) + self.rng.gauss(0, 1.5)

        # Fault injection modifiers if any
        active_fault = self.active_fault
        if active_fault == "bearing_spall":
            vib_base += 5.5 * self.fault_severity
            temp_base += 18.0 * self.fault_severity
        elif active_fault == "thermal_runaway":
            temp_base += 45.0 * self.fault_severity
            curr_base += 8.0 * self.fault_severity
        elif active_fault == "overstrain":
            vib_base += 4.0 * self.fault_severity
            curr_base += 12.0 * self.fault_severity

        return {
            "machine_id": self.machine_id,
            "timestamp": now.isoformat(),
            "operating_cycle": cycle,
            "temperature": round(max(0.0, temp_base), 2),
            "pressure": round(max(0.0, pressure_base), 2),
            "vibration": round(max(0.0, vib_base), 3),
            "rpm": round(max(0.0, rpm_base), 2),
            "current": round(max(0.0, curr_base), 2),
            "equipment_type": self.equipment_type,
            "source": "SYNTHETIC / DEVELOPMENT",
            "fault_injected": active_fault,
            "phase": phase,
        }

    def generate_batch(self, count: int) -> list[dict[str, Any]]:
        """Generate a consecutive sequence of readings."""
        return [self.generate_reading() for _ in range(count)]
