"""
MaintenAI— Feature Registry & Versioning

Centralized registry of all engineered features with metadata for
auditing, reproducibility, and documentation generation.

Every feature has:
  - name: the output column name
  - source: the source column(s) it derives from
  - transform: the transformation applied
  - params: transformation parameters (window, span, etc.)
  - reason: why this feature was created (EDA-backed rationale)
  - dataset: which dataset(s) this feature applies to
  - leakage_notes: how temporal/future leakage is prevented
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# Feature Version
# ============================================================

FEATURE_VERSION = "v1"

# ============================================================
# Feature Specification
# ============================================================


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for a single engineered feature."""

    name: str
    source: str
    transform: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    dataset: str = "all"
    leakage_notes: str = "trailing window / past-only"


# ============================================================
# Sensor Column Definitions
# ============================================================

# C-MAPSS: EDA-identified priority sensors (|corr with RUL| > 0.5)
CMAPSS_PRIORITY_SENSORS = [
    "raw_sensor_15",  # Bypass ratio        — corr -0.651
    "raw_sensor_11",  # Physical fan speed   — corr +0.651
    "raw_sensor_20",  # Bleed enthalpy       — corr +0.598
    "raw_sensor_12",  # HPC outlet pressure  — corr -0.532
    "raw_sensor_9",  # Physical core speed  — corr +0.506
    "raw_sensor_17",  # Fuel flow ratio      — corr -0.500
]

# C-MAPSS: Secondary sensors (moderate correlation, kept for completeness)
CMAPSS_SECONDARY_SENSORS = [
    "raw_sensor_4",  # LPT outlet temp      — corr -0.433
    "raw_sensor_14",  # Corrected fan speed   — corr +0.432
    "raw_sensor_3",  # HPC outlet temp       — corr +0.419
    "raw_sensor_2",  # LPC outlet temp       — corr -0.374
    "raw_sensor_7",  # HPC outlet total temp — corr -0.352
    "raw_sensor_8",  # HPT coolant bleed     — corr +0.319
    "raw_sensor_13",  # Corrected core speed  — corr +0.291
    "raw_sensor_21",  # BPR (bleed)           — corr -0.214
]

# C-MAPSS: Canonical sensor names (mapped from raw sensors)
CMAPSS_CANONICAL_SENSORS = ["temperature", "pressure", "rpm", "vibration", "current"]

# C-MAPSS: Operational settings
CMAPSS_OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# IMS: Usable vibration features (EDA-validated)
# NOTE: raw_rms is EXCLUDED — it is identical to vibration (corr=1.0)
IMS_VIBRATION_FEATURES = [
    "vibration",  # RMS of accelerometer signal
    "raw_peak",  # Peak amplitude
    "raw_peak_to_peak",  # Peak-to-peak amplitude
    "raw_std",  # Standard deviation
    "raw_kurtosis",  # Kurtosis (shape statistic)
    "raw_skewness",  # Skewness (shape statistic)
    "raw_crest_factor",  # Crest factor (peak/RMS)
    "raw_spectral_energy",  # Spectral energy
    "raw_dominant_freq_idx",  # Dominant frequency index
]

# IMS: Columns to DROP (documented reasons)
IMS_DROP_COLUMNS = {
    "temperature": "100% null — IMS is accelerometer-only, no temperature sensor",
    "pressure": "100% null — IMS is accelerometer-only, no pressure sensor",
    "rpm": "100% null — IMS is accelerometer-only, no RPM sensor",
    "current": "100% null — IMS is accelerometer-only, no current sensor",
    "raw_rms": "Identical to vibration column (correlation=1.0) — redundant",
}

# IMS: Primary features for rolling/temporal analysis (amplitude-based, EDA top correlations)
IMS_PRIMARY_FEATURES = [
    "vibration",  # RMS — corr +0.662 with anomaly
    "raw_peak",  # Peak — corr +0.652
    "raw_spectral_energy",  # Spectral energy — corr +0.668
    "raw_crest_factor",  # Crest factor — corr -0.089 (weak but physically meaningful)
]

# AI4I: Sensor columns for feature engineering
AI4I_SENSORS = ["temperature", "pressure", "rpm", "vibration", "operating_cycle"]


# ============================================================
# Feature Registry Generation
# ============================================================


def _generate_cmapss_registry(
    window_sizes: list[int],
    ema_spans: list[int],
    lag_steps: list[int],
) -> list[FeatureSpec]:
    """Generate the full C-MAPSS feature registry."""
    specs = []

    # Passthrough columns (preserved, not engineered)
    for col in CMAPSS_CANONICAL_SENSORS + CMAPSS_PRIORITY_SENSORS + CMAPSS_SECONDARY_SENSORS:
        specs.append(
            FeatureSpec(
                name=col,
                source=col,
                transform="passthrough",
                reason="Original sensor value",
                dataset="cmapss",
            )
        )

    # Rolling features on priority sensors
    for col in CMAPSS_PRIORITY_SENSORS:
        for w in window_sizes:
            for stat in ["mean", "std", "min", "max"]:
                specs.append(
                    FeatureSpec(
                        name=f"{col}_roll{w}_{stat}",
                        source=col,
                        transform=f"rolling_{stat}",
                        params={"window": w},
                        reason=f"Captures {stat} trend over {w} cycles. EDA: |corr|>0.5 with RUL",
                        dataset="cmapss",
                        leakage_notes=f"Trailing window of {w}, grouped by machine_id",
                    )
                )

    # EMA on priority sensors
    for col in CMAPSS_PRIORITY_SENSORS:
        for span in ema_spans:
            specs.append(
                FeatureSpec(
                    name=f"{col}_ema{span}",
                    source=col,
                    transform="exponential_moving_average",
                    params={"span": span},
                    reason=f"Smoothed trend with span={span}. Captures degradation direction",
                    dataset="cmapss",
                    leakage_notes="EMA is causal (uses only past values)",
                )
            )

    # Lag features on priority sensors
    for col in CMAPSS_PRIORITY_SENSORS:
        for lag in lag_steps:
            specs.append(
                FeatureSpec(
                    name=f"{col}_lag{lag}",
                    source=col,
                    transform="lag",
                    params={"steps": lag},
                    reason=f"Value {lag} cycles ago. Provides explicit history access",
                    dataset="cmapss",
                    leakage_notes=f"Shift by +{lag} (past only), grouped by machine_id",
                )
            )

    # Delta (rate of change) on all informative sensors
    for col in CMAPSS_PRIORITY_SENSORS + CMAPSS_SECONDARY_SENSORS:
        specs.append(
            FeatureSpec(
                name=f"{col}_delta1",
                source=col,
                transform="first_difference",
                params={"period": 1},
                reason="Rate of change — detects onset of degradation",
                dataset="cmapss",
                leakage_notes="diff(t) = x(t) - x(t-1), past only",
            )
        )

    # Operational setting features
    for col in CMAPSS_OP_SETTINGS:
        specs.append(
            FeatureSpec(
                name=col,
                source=col,
                transform="passthrough",
                reason="Operational regime indicator",
                dataset="cmapss",
            )
        )

    # Lifecycle position feature
    specs.append(
        FeatureSpec(
            name="lifecycle_pct",
            source="operating_cycle",
            transform="normalize_by_max_in_group",
            reason="Normalized position in lifecycle (0=start, 1=end). "
            "Provides context for sensor readings",
            dataset="cmapss",
            leakage_notes="Uses only cycles seen so far (cumulative max), not future max",
        )
    )

    return specs


def _generate_ims_registry(
    window_sizes: list[int],
    ema_spans: list[int],
) -> list[FeatureSpec]:
    """Generate the full IMS feature registry."""
    specs = []

    # Usable vibration features
    for col in IMS_VIBRATION_FEATURES:
        specs.append(
            FeatureSpec(
                name=col,
                source=col,
                transform="passthrough",
                reason="Original vibration feature",
                dataset="ims",
            )
        )

    # Log-transforms for skewed features
    for col in ["vibration", "raw_peak", "raw_spectral_energy"]:
        specs.append(
            FeatureSpec(
                name=f"{col}_log",
                source=col,
                transform="log1p",
                reason="Log-transform to reduce right-skewness (skew={'vibration':3.84,'raw_peak':3.92,'raw_spectral_energy':4.66}[col]). "
                "Improves normality for distance-based models",
                dataset="ims",
            )
        )

    # Rolling features on primary vibration features
    for col in IMS_PRIMARY_FEATURES:
        for w in window_sizes:
            for stat in ["mean", "std"]:
                specs.append(
                    FeatureSpec(
                        name=f"{col}_roll{w}_{stat}",
                        source=col,
                        transform=f"rolling_{stat}",
                        params={"window": w},
                        reason=f"Captures vibration {stat} trend over {w} snapshots",
                        dataset="ims",
                        leakage_notes=f"Trailing window of {w}, grouped by machine_id (bearing)",
                    )
                )

    # EMA on primary features
    for col in IMS_PRIMARY_FEATURES:
        for span in ema_spans:
            specs.append(
                FeatureSpec(
                    name=f"{col}_ema{span}",
                    source=col,
                    transform="exponential_moving_average",
                    params={"span": span},
                    reason="Smoothed vibration trend for degradation detection",
                    dataset="ims",
                    leakage_notes="EMA is causal, grouped by machine_id",
                )
            )

    # Delta features for degradation rate
    for col in IMS_PRIMARY_FEATURES:
        specs.append(
            FeatureSpec(
                name=f"{col}_delta1",
                source=col,
                transform="first_difference",
                params={"period": 1},
                reason="Rate of vibration change — sudden increases indicate bearing degradation",
                dataset="ims",
                leakage_notes="diff(t) = x(t) - x(t-1), past only, grouped by machine_id",
            )
        )

    # Spectral ratio features
    specs.append(
        FeatureSpec(
            name="spectral_to_rms_ratio",
            source="raw_spectral_energy,vibration",
            transform="ratio",
            reason="Spectral energy to RMS ratio — frequency content relative to overall amplitude",
            dataset="ims",
        )
    )
    specs.append(
        FeatureSpec(
            name="peak_to_energy_ratio",
            source="raw_peak,raw_spectral_energy",
            transform="ratio",
            reason="Peak to spectral energy ratio — impulsive events indicator for bearing impacts",
            dataset="ims",
        )
    )

    # Spectral trend features
    for col in ["raw_spectral_energy", "vibration", "raw_peak"]:
        specs.append(
            FeatureSpec(
                name=f"{col}_trend_ratio5",
                source=col,
                transform="rolling_trend_ratio",
                params={"window": 5},
                reason=f"Deviation from recent 5-snapshot moving average for {col}",
                dataset="ims",
                leakage_notes="Trailing window of 5, grouped by machine_id",
            )
        )

    return specs


def _generate_ai4i_registry() -> list[FeatureSpec]:
    """Generate the full AI4I feature registry."""
    specs = []

    # Passthrough sensor features
    for col in AI4I_SENSORS:
        specs.append(
            FeatureSpec(
                name=col,
                source=col,
                transform="passthrough",
                reason="Original sensor/process value",
                dataset="ai4i",
            )
        )

    # Interaction features
    specs.append(
        FeatureSpec(
            name="power",
            source="vibration,rpm",
            transform="product",
            reason="Power = torque × rotational speed. Physically meaningful composite. "
            "Overstrain and power failures relate to power output",
            dataset="ai4i",
        )
    )

    specs.append(
        FeatureSpec(
            name="temp_diff",
            source="pressure,temperature",
            transform="difference",
            reason="Process temp - Air temp differential. Heat dissipation failures occur "
            "when this differential is too low for the operating conditions",
            dataset="ai4i",
        )
    )

    specs.append(
        FeatureSpec(
            name="torque_per_rpm",
            source="vibration,rpm",
            transform="ratio",
            reason="Torque-to-speed ratio. High ratio at low RPM indicates overstrain conditions",
            dataset="ai4i",
        )
    )

    specs.append(
        FeatureSpec(
            name="tool_wear_squared",
            source="operating_cycle",
            transform="square",
            reason="Non-linear tool wear effect. Failure probability accelerates with wear",
            dataset="ai4i",
        )
    )

    specs.append(
        FeatureSpec(
            name="temp_rpm_interaction",
            source="temperature,rpm",
            transform="product",
            reason="Temperature-speed interaction. High temp at high RPM increases failure risk",
            dataset="ai4i",
        )
    )

    # Product type one-hot
    for pt in ["L", "M", "H"]:
        specs.append(
            FeatureSpec(
                name=f"product_type_{pt}",
                source="product_type",
                transform="one_hot_encode",
                reason="Product quality type indicator. Failure rates differ by type",
                dataset="ai4i",
            )
        )

    return specs


def get_feature_manifest(
    dataset: str,
    window_sizes: list[int] | None = None,
    ema_spans: list[int] | None = None,
    lag_steps: list[int] | None = None,
) -> list[dict]:
    """
    Get the full feature manifest for a dataset.

    Returns list of dicts suitable for JSON serialization.
    """
    if dataset == "cmapss":
        specs = _generate_cmapss_registry(
            window_sizes or [5, 10, 20],
            ema_spans or [10, 30],
            lag_steps or [1, 5],
        )
    elif dataset == "ims":
        specs = _generate_ims_registry(
            window_sizes or [3, 5, 10],
            ema_spans or [10, 30],
        )
    elif dataset == "ai4i":
        specs = _generate_ai4i_registry()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return [asdict(s) for s in specs]


def save_feature_manifest(
    dataset: str,
    output_dir: Path,
    **kwargs,
) -> Path:
    """Save feature manifest as JSON."""
    manifest = {
        "feature_version": FEATURE_VERSION,
        "dataset": dataset,
        "features": get_feature_manifest(dataset, **kwargs),
        "n_features": len(get_feature_manifest(dataset, **kwargs)),
    }
    path = output_dir / "feature_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Saved feature manifest: {path} ({manifest['n_features']} features)")
    return path
