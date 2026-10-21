"""
MaintenAI— C-MAPSS Dataset Processor

Loads NASA C-MAPSS Turbofan Engine Degradation Simulation data and maps it
to the MaintenAIcanonical schema for RUL prediction.

C-MAPSS Data Format (FD001):
  - train_FD001.txt: Training data (unit_id, cycle, op1-3, sensor1-21)
  - test_FD001.txt: Test data (truncated sequences)
  - RUL_FD001.txt: True remaining useful life for test units

Canonical Schema Mapping:
  - unit_id → machine_id (prefixed "CMAPSS_")
  - cycle → operating_cycle
  - sensor_2 (LPC outlet temp) → temperature
  - sensor_7 (HPC outlet pressure) → pressure (actually total temp, see docs)
  - sensor_11 (physical fan speed) → rpm
  - sensor_15 (bypass ratio) → vibration (proxy — no direct vibration in C-MAPSS)
  - sensor_4 (LPT outlet temp) → current (proxy sensor mapping)

RUL Target Construction:
  - For training: RUL = max_cycle(unit) - current_cycle
  - Clipped to max_rul_clip (default 125) for piecewise-linear target
  - This prevents the model from trying to predict very large RUL values
    where degradation has not yet started (flat region).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# C-MAPSS column definitions
CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensor-to-canonical mapping
# We map representative sensors that correspond to our canonical schema
SENSOR_MAP = {
    "sensor_2": "temperature",  # Total temp at LPC outlet (Rankine → mapped)
    "sensor_7": "pressure",  # Total temp at HPC outlet (used as pressure proxy)
    "sensor_11": "rpm",  # Physical fan speed (rpm)
    "sensor_15": "vibration",  # Bypass ratio (used as vibration proxy)
    "sensor_4": "current",  # LPT outlet temp (used as current proxy)
}

# Sensors with near-constant values that carry no degradation signal
# These are excluded from features to reduce noise
CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]


def load_raw_cmapss(
    raw_dir: Path,
    dataset_id: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load raw C-MAPSS data files.

    Returns:
        (train_df, test_df, rul_series)
    """
    train_path = raw_dir / f"train_{dataset_id}.txt"
    test_path = raw_dir / f"test_{dataset_id}.txt"
    rul_path = raw_dir / f"RUL_{dataset_id}.txt"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    train_df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )
    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )
    rul_series = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"]).squeeze()

    logger.info(
        f"Loaded C-MAPSS {dataset_id}: "
        f"train={len(train_df)} rows ({train_df['unit_id'].nunique()} units), "
        f"test={len(test_df)} rows ({test_df['unit_id'].nunique()} units)"
    )
    return train_df, test_df, rul_series


def add_rul_labels(
    df: pd.DataFrame,
    max_rul_clip: int = 125,
) -> pd.DataFrame:
    """
    Add RUL labels to C-MAPSS training data.

    RUL(t) = max_cycle(unit) - cycle(t)
    Clipped to max_rul_clip for piecewise-linear target.

    Why clip?
    ---------
    Early in a unit's life, RUL is very large and the machine shows no
    degradation. Predicting exact large RUL values adds noise without
    value. Clipping creates a "flat then declining" target that focuses
    the model on the degradation region.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = max_cycles - df["cycle"]
    df["rul"] = df["rul"].clip(upper=max_rul_clip)
    return df


def add_test_rul_labels(
    test_df: pd.DataFrame,
    rul_true: pd.Series,
) -> pd.DataFrame:
    """
    Add RUL labels to C-MAPSS test data using ground truth.

    For test data: RUL(t) = remaining_rul + (max_cycle_in_test - cycle(t))
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit_id")["cycle"].transform("max")
    rul_map = dict(enumerate(rul_true, start=1))
    test_df["rul_at_end"] = test_df["unit_id"].map(rul_map)
    test_df["rul"] = test_df["rul_at_end"] + (max_cycles - test_df["cycle"])
    test_df.drop(columns=["rul_at_end"], inplace=True)
    return test_df


def map_to_canonical(
    df: pd.DataFrame,
    dataset_id: str = "FD001",
    split: str = "train",
) -> pd.DataFrame:
    """
    Map C-MAPSS columns to MaintenAIcanonical schema.

    Adds:
      - machine_id: "CMAPSS_{dataset_id}_{split}_{unit_id:03d}"
      - timestamp: Synthetic timestamp (cycle-based, 1 cycle = 1 hour)
      - Mapped sensor columns

    The split prefix is critical: C-MAPSS train and test datasets
    have independently-numbered units (both start at unit 1).
    Without the prefix, unit 1 in training and unit 1 in testing
    would appear to be the same machine, causing false leakage alerts.
    """
    canonical = pd.DataFrame()

    # Machine ID — prefixed with split to prevent false ID collisions
    canonical["machine_id"] = df["unit_id"].apply(
        lambda uid: f"CMAPSS_{dataset_id}_{split}_{uid:03d}"
    )

    # Synthetic timestamp (1 cycle = 1 hour from epoch)
    base_time = pd.Timestamp("2020-01-01", tz="UTC")
    canonical["timestamp"] = df["cycle"].apply(lambda c: base_time + pd.Timedelta(hours=int(c)))

    # Map sensors to canonical names
    for source_col, canonical_name in SENSOR_MAP.items():
        if source_col in df.columns:
            canonical[canonical_name] = df[source_col].astype(float)

    canonical["operating_cycle"] = df["cycle"].astype(int)

    # Carry over RUL if present
    if "rul" in df.columns:
        canonical["rul"] = df["rul"].astype(int)

    # Keep all original sensor columns for feature engineering
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    informative_sensors = [c for c in sensor_cols if c not in CONSTANT_SENSORS]
    for col in informative_sensors:
        canonical[f"raw_{col}"] = df[col].astype(float)

    # Keep operational settings
    for col in ["op_setting_1", "op_setting_2", "op_setting_3"]:
        if col in df.columns:
            canonical[col] = df[col].astype(float)

    return canonical


def process_cmapss(
    raw_dir: Path,
    output_dir: Path,
    dataset_id: str = "FD001",
    max_rul_clip: int = 125,
) -> dict[str, Path]:
    """
    Full C-MAPSS processing pipeline.

    1. Load raw data
    2. Add RUL labels
    3. Map to canonical schema
    4. Remove constant sensors
    5. Validate
    6. Save processed parquet files

    Returns:
        Dict mapping split names to output file paths.
    """
    from ml.common.validation import run_full_validation

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    train_df, test_df, rul_series = load_raw_cmapss(raw_dir, dataset_id)

    # Add RUL labels
    train_df = add_rul_labels(train_df, max_rul_clip)
    test_df = add_test_rul_labels(test_df, rul_series)

    # Map to canonical — use split prefix to avoid ID collision
    train_canonical = map_to_canonical(train_df, dataset_id, split="train")
    test_canonical = map_to_canonical(test_df, dataset_id, split="test")

    # Validate
    required = ["machine_id", "timestamp", "operating_cycle", "rul"]
    train_report = run_full_validation(
        train_canonical,
        required,
        f"cmapss_{dataset_id}_train",
        timestamp_col="timestamp",
        machine_id_col="machine_id",
    )
    logger.info(train_report.summary())

    test_report = run_full_validation(
        test_canonical,
        required,
        f"cmapss_{dataset_id}_test",
        timestamp_col="timestamp",
        machine_id_col="machine_id",
    )
    logger.info(test_report.summary())

    # Save
    train_path = output_dir / f"cmapss_{dataset_id}_train.parquet"
    test_path = output_dir / f"cmapss_{dataset_id}_test.parquet"

    train_canonical.to_parquet(train_path, index=False)
    test_canonical.to_parquet(test_path, index=False)

    logger.info(
        f"C-MAPSS {dataset_id} processed: "
        f"train={len(train_canonical)} rows → {train_path}, "
        f"test={len(test_canonical)} rows → {test_path}"
    )

    return {"train": train_path, "test": test_path}
