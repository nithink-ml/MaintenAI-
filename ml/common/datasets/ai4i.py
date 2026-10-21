"""
MaintenAI— AI4I 2020 Dataset Processor

Loads the AI4I 2020 Predictive Maintenance Dataset from UCI and maps it
to the MaintenAIcanonical schema for failure-mode classification.

AI4I Data Format:
  - UDI: Unique identifier (1-10000)
  - Product ID: Product quality variant (L/M/H + serial)
  - Type: Quality type (L=Low, M=Medium, H=High)
  - Air temperature [K]: Air temp in Kelvin
  - Process temperature [K]: Process temp in Kelvin
  - Rotational speed [rpm]: Spindle speed
  - Torque [Nm]: Torque in Newton-meters
  - Tool wear [min]: Tool wear time in minutes
  - Machine failure: Binary (0/1)
  - TWF: Tool Wear Failure
  - HDF: Heat Dissipation Failure
  - PWF: Power Failure
  - OSF: Overstrain Failure
  - RNF: Random Failure

Canonical Schema Mapping:
  - UDI → machine_id (prefixed "AI4I_")
  - Air temperature [K] → temperature (converted K → °C)
  - Rotational speed [rpm] → rpm
  - Torque [Nm] → vibration (proxy — torque correlates with mechanical stress)
  - Process temperature [K] → pressure (proxy mapping)
  - Tool wear [min] → operating_cycle (cumulative wear as cycle proxy)
  - Failure columns → failure_label, failure_mode
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Failure mode column mapping
FAILURE_COLUMNS = {
    "TWF": "tool_wear",
    "HDF": "heat_dissipation",
    "PWF": "power_failure",
    "OSF": "overstrain",
    "RNF": "random",
}


def load_raw_ai4i(raw_dir: Path) -> pd.DataFrame:
    """
    Load raw AI4I 2020 dataset.

    Searches for the CSV file in common naming patterns.
    """
    possible_names = [
        "ai4i2020.csv",
        "ai4i_2020.csv",
        "predictive_maintenance.csv",
    ]

    csv_path = None
    for name in possible_names:
        candidate = raw_dir / name
        if candidate.exists():
            csv_path = candidate
            break

    # Also check subdirectories (zip extraction may nest)
    if csv_path is None:
        for candidate in raw_dir.rglob("*.csv"):
            csv_path = candidate
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"AI4I CSV not found in {raw_dir}. Expected one of: {possible_names}"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded AI4I: {len(df)} rows from {csv_path}")
    return df


def map_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map AI4I columns to MaintenAIcanonical schema.

    Temperature conversion: Kelvin → Celsius (K - 273.15)
    """
    canonical = pd.DataFrame()

    # Machine ID
    canonical["machine_id"] = df.apply(lambda row: f"AI4I_{row.get('UDI', row.name):05d}", axis=1)

    # Synthetic timestamp (each row is a separate measurement)
    base_time = pd.Timestamp("2020-06-01", tz="UTC")
    canonical["timestamp"] = [base_time + pd.Timedelta(minutes=i * 5) for i in range(len(df))]

    # Temperature: Kelvin → Celsius
    if "Air temperature [K]" in df.columns:
        canonical["temperature"] = (df["Air temperature [K]"] - 273.15).round(2)
    elif "Air temperature [k]" in df.columns:
        canonical["temperature"] = (df["Air temperature [k]"] - 273.15).round(2)

    # Process temperature as pressure proxy
    if "Process temperature [K]" in df.columns:
        canonical["pressure"] = (df["Process temperature [K]"] - 273.15).round(2)
    elif "Process temperature [k]" in df.columns:
        canonical["pressure"] = (df["Process temperature [k]"] - 273.15).round(2)

    # RPM
    rpm_col = next((c for c in df.columns if "rotational" in c.lower()), None)
    if rpm_col:
        canonical["rpm"] = df[rpm_col].astype(float)

    # Torque as vibration proxy (mechanical stress indicator)
    torque_col = next((c for c in df.columns if "torque" in c.lower()), None)
    if torque_col:
        canonical["vibration"] = df[torque_col].astype(float)

    # Tool wear as operating cycle
    wear_col = next((c for c in df.columns if "tool wear" in c.lower()), None)
    if wear_col:
        canonical["operating_cycle"] = df[wear_col].astype(int)

    # Failure label (binary)
    failure_col = next((c for c in df.columns if "machine failure" in c.lower()), None)
    if failure_col:
        canonical["failure_label"] = df[failure_col].astype(int)

    # Individual failure modes
    for src_col, mode_name in FAILURE_COLUMNS.items():
        if src_col in df.columns:
            canonical[f"failure_{mode_name}"] = df[src_col].astype(int)

    # Determine primary failure mode
    canonical["failure_mode"] = _determine_failure_mode(df)

    # Keep product type for stratification
    if "Type" in df.columns:
        canonical["product_type"] = df["Type"]

    return canonical


def _determine_failure_mode(df: pd.DataFrame) -> pd.Series:
    """
    Determine the primary failure mode for each row.

    If multiple failure modes are active, priority is:
    HDF > PWF > OSF > TWF > RNF > none
    (Based on typical industrial severity ordering)
    """
    modes = pd.Series("none", index=df.index)

    priority_order = [
        ("RNF", "random"),
        ("TWF", "tool_wear"),
        ("OSF", "overstrain"),
        ("PWF", "power_failure"),
        ("HDF", "heat_dissipation"),
    ]

    for col, mode_name in priority_order:
        if col in df.columns:
            modes = modes.where(df[col] == 0, mode_name)

    return modes


def analyze_class_distribution(df: pd.DataFrame) -> dict[str, int]:
    """
    Analyze failure mode class distribution for imbalance assessment.

    Returns dict of {mode: count}.
    """
    if "failure_mode" not in df.columns:
        raise ValueError("DataFrame must have 'failure_mode' column")

    dist = df["failure_mode"].value_counts().to_dict()
    total = len(df)

    logger.info("=== AI4I Class Distribution ===")
    for mode, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        logger.info(f"  {mode:20s}: {count:5d} ({pct:.1f}%)")

    # Check imbalance ratio
    if "none" in dist:
        failure_count = total - dist["none"]
        imbalance_ratio = dist["none"] / max(failure_count, 1)
        logger.info(f"  Imbalance ratio (normal:failure): {imbalance_ratio:.1f}:1")

    return dist


def process_ai4i(
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Full AI4I processing pipeline.

    1. Load raw CSV
    2. Map to canonical schema
    3. Analyze class distribution
    4. Validate
    5. Save processed parquet

    Returns:
        Dict mapping split names to output file paths.
    """
    from ml.common.validation import run_full_validation

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    raw_df = load_raw_ai4i(raw_dir)

    # Map to canonical
    canonical = map_to_canonical(raw_df)

    # Analyze class distribution
    analyze_class_distribution(canonical)

    # Validate
    required = ["machine_id", "timestamp", "failure_label", "failure_mode"]
    report = run_full_validation(
        canonical, required, "ai4i_2020", timestamp_col="timestamp", machine_id_col="machine_id"
    )
    logger.info(report.summary())

    # Save
    output_path = output_dir / "ai4i_full.parquet"
    canonical.to_parquet(output_path, index=False)

    logger.info(f"AI4I processed: {len(canonical)} rows → {output_path}")
    return {"full": output_path}
