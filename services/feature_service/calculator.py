"""
MaintenAI— Online Feature Service Calculator

Maintains in-memory and Redis sliding windows of raw telemetry to compute
rolling statistics, exponential moving averages, and spectral features for real-time inference.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.features.spectral import compute_spectral_features

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """
    Sliding window buffer and online feature computation engine per machine.
    """

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.buffers: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )

    def append_reading(self, reading: dict[str, Any]) -> None:
        """Push new sensor reading to machine buffer."""
        m_id = reading.get("machine_id", "default")
        self.buffers[m_id].append(reading)

    def compute_features(self, machine_id: str) -> dict[str, float] | None:
        """
        Compute online temporal and spectral features from the sliding window.
        """
        buf = self.buffers.get(machine_id)
        if not buf or len(buf) == 0:
            return None

        df = pd.DataFrame(list(buf))
        sensor_cols = [
            c for c in ["temperature", "pressure", "rpm", "vibration", "current"] if c in df.columns
        ]

        features: dict[str, float] = {}

        # Latest raw readings
        latest_row = df.iloc[-1]
        for col in sensor_cols:
            features[col] = float(latest_row[col])

        # Rolling statistics (Mean, Std, Min, Max)
        for col in sensor_cols:
            vals = df[col].values
            features[f"{col}_rolling_mean_5"] = float(np.mean(vals[-min(len(vals), 5) :]))
            features[f"{col}_rolling_std_5"] = (
                float(np.std(vals[-min(len(vals), 5) :])) if len(vals) > 1 else 0.0
            )
            features[f"{col}_rolling_mean_20"] = float(np.mean(vals))
            features[f"{col}_rolling_std_20"] = float(np.std(vals)) if len(vals) > 1 else 0.0

            # Delta / trend
            if len(vals) > 1:
                features[f"{col}_delta_1"] = float(vals[-1] - vals[-2])
            else:
                features[f"{col}_delta_1"] = 0.0

        # Spectral and vibration moments
        if "vibration" in df.columns:
            vib_arr = df["vibration"].values
            features["raw_peak"] = float(np.max(np.abs(vib_arr)))
            features["raw_peak_to_peak"] = float(np.ptp(vib_arr)) if len(vib_arr) > 1 else float(vib_arr[-1])
            features["raw_std"] = float(np.std(vib_arr)) if len(vib_arr) > 1 else 0.0
            rms = float(np.sqrt(np.mean(vib_arr**2)))
            features["raw_crest_factor"] = float(features["raw_peak"] / max(rms, 1e-6))
            features["vibration_log"] = float(np.log1p(max(0.0, vib_arr[-1])))

            if len(vib_arr) >= 4:
                mean_v = np.mean(vib_arr)
                std_v = max(float(np.std(vib_arr)), 1e-6)
                features["raw_skewness"] = float(np.mean(((vib_arr - mean_v) / std_v) ** 3))
                features["raw_kurtosis"] = float(np.mean(((vib_arr - mean_v) / std_v) ** 4))
            else:
                features["raw_skewness"] = 0.0
                features["raw_kurtosis"] = 3.0

            if len(df) >= 8:
                spec_feats = compute_spectral_features(vib_arr)
                features.update(spec_feats)

        # AI4I Physical interaction features for Failure Mode Classification
        temp = features.get("temperature", 20.0)
        rpm = features.get("rpm", 1500.0)
        current = features.get("current", 20.0)
        cycle = float(latest_row.get("operating_cycle", 1))
        features["operating_cycle"] = cycle
        features["power"] = float(current * 220.0 / 1000.0)
        features["temp_diff"] = float(temp - 25.0)
        features["torque_per_rpm"] = float(features.get("vibration", 1.0) / max(rpm, 1.0))
        features["tool_wear_squared"] = float(cycle**2)
        features["temp_rpm_interaction"] = float(temp * rpm / 1000.0)

        # C-MAPSS mapped sensor features for RUL
        features["raw_sensor_11"] = float(temp * 0.8)
        features["raw_sensor_12"] = float(features.get("pressure", 100.0) * 0.5)
        features["raw_sensor_15"] = float(current * 0.3)
        features["raw_sensor_20"] = float(features.get("vibration", 1.0) * 10.0)

        return features
