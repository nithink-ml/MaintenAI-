"""
MaintenAI— Sensor Telemetry Simulation Engine

Replays industrial multi-machine sensor telemetry with configurable:
  - Machines and operating cycles
  - Realistic sensor physics, drift, and degradation curves
  - Fault injections (thermal runaway, bearing spalling, tool wear, overstrain)
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class MachineSimulator:
    """Simulates a single industrial machine's operational lifecycle."""

    def __init__(
        self,
        machine_id: str,
        equipment_type: str = "turbofan",
        degradation_rate: float = 0.002,
        initial_cycle: int = 1,
    ) -> None:
        self.machine_id = machine_id
        self.equipment_type = equipment_type
        self.degradation_rate = degradation_rate
        self.current_cycle = initial_cycle
        self.fault_active: str | None = None
        self.fault_severity: float = 0.0

    def inject_fault(self, fault_type: str, severity: float = 1.0) -> None:
        """Inject a physical fault condition into this machine."""
        self.fault_active = fault_type
        self.fault_severity = min(1.0, max(0.1, severity))
        logger.warning(
            "Injected fault '%s' (severity %.2f) into %s", fault_type, severity, self.machine_id
        )

    def clear_fault(self) -> None:
        """Clear active fault condition."""
        self.fault_active = None
        self.fault_severity = 0.0

    def generate_reading(self) -> dict[str, Any]:
        """Generate the next timestamped sensor reading."""
        self.current_cycle += 1
        now = datetime.datetime.now(datetime.UTC).isoformat()
        cycle = self.current_cycle
        deg = 1.0 + (cycle * self.degradation_rate)

        # Baseline sinusoidal variation
        t_osc = math.sin(cycle * 0.1) * 2.0
        p_osc = math.cos(cycle * 0.08) * 5.0
        v_osc = abs(math.sin(cycle * 0.15)) * 0.3

        if self.equipment_type == "turbofan":
            temp = 600.0 + (deg * 15.0) + t_osc + random.gauss(0, 1.5)
            pressure = 518.0 - (deg * 8.0) + p_osc + random.gauss(0, 2.0)
            rpm = 2380.0 + random.gauss(0, 10.0)
            vibration = 1.8 * deg + v_osc + random.gauss(0, 0.1)
            current = 45.0 + (deg * 2.5) + random.gauss(0, 0.5)

            if self.fault_active == "thermal_runaway":
                temp += 85.0 * self.fault_severity
                pressure -= 40.0 * self.fault_severity
            elif self.fault_active == "compressor_stall":
                vibration += 4.5 * self.fault_severity
                pressure -= 60.0 * self.fault_severity

        elif self.equipment_type == "bearing":
            temp = 70.0 + (deg * 5.0) + random.gauss(0, 0.8)
            pressure = 101.3 + random.gauss(0, 0.1)
            rpm = 2000.0 + random.gauss(0, 5.0)
            vibration = 1.2 * (deg**1.8) + random.gauss(0, 0.15)
            current = 30.0 + (deg * 1.5) + random.gauss(0, 0.3)

            if self.fault_active == "bearing_spall":
                vibration += 6.2 * self.fault_severity
                temp += 25.0 * self.fault_severity

        else:  # CNC Mill
            temp = 300.0 + (deg * 3.0) + random.gauss(0, 1.0)
            pressure = 101.3 + random.gauss(0, 0.1)
            rpm = 1500.0 - (deg * 12.0) + random.gauss(0, 8.0)
            vibration = 40.0 * deg + random.gauss(0, 2.0)  # Torque
            current = 25.0 + (deg * 3.0) + random.gauss(0, 0.4)

            if self.fault_active == "heat_dissipation":
                temp += 35.0 * self.fault_severity
            elif self.fault_active == "tool_wear":
                vibration += 30.0 * self.fault_severity

        return {
            "machine_id": self.machine_id,
            "equipment_type": self.equipment_type,
            "timestamp": now,
            "operating_cycle": cycle,
            "temperature": round(temp, 2),
            "pressure": round(pressure, 2),
            "rpm": round(rpm, 2),
            "vibration": round(vibration, 3),
            "current": round(current, 2),
            "fault_injected": self.fault_active,
        }


class FleetSimulator:
    """Simulates a fleet of industrial machines generating concurrent telemetry streams."""

    def __init__(self, num_machines: int = 10) -> None:
        self.machines: dict[str, MachineSimulator] = {}
        types = ["turbofan", "bearing", "cnc_mill"]
        for i in range(1, num_machines + 1):
            eq_type = types[(i - 1) % len(types)]
            m_id = f"{eq_type}-{i:03d}"
            # Random starting cycle
            init_cycle = random.randint(10, 120)
            self.machines[m_id] = MachineSimulator(
                m_id, equipment_type=eq_type, initial_cycle=init_cycle
            )

    async def stream_telemetry(
        self, interval_sec: float = 1.0
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Yield a batch of readings from all machines on each tick."""
        while True:
            batch = [m.generate_reading() for m in self.machines.values()]
            yield batch
            await asyncio.sleep(interval_sec)
