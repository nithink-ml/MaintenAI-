"""
MaintenAI— Train/Validation/Test Splitting with Leakage Prevention

Provides point-in-time correct splitting strategies for each dataset type.
Prevents temporal data leakage — the most common and dangerous mistake
in time-series ML.

Data Leakage in Time-Series Context:
====================================
Standard random splitting (sklearn's train_test_split) BREAKS temporal
ordering. If cycle 150 is in training and cycle 100 is in test, the model
has seen the future. This inflates metrics and produces unreliable models.

Correct approaches:
  1. C-MAPSS (RUL): Split by unit_id, not by row. Each unit's full
     lifecycle stays together in one split.
  2. IMS (Anomaly): Temporal split — early snapshots for training,
     later snapshots for testing. No shuffle.
  3. AI4I (Classification): Stratified split by failure mode,
     maintaining class distribution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def split_by_unit(
    df: pd.DataFrame,
    unit_col: str = "machine_id",
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by unit/machine ID for RUL prediction.

    Why split by unit?
    ------------------
    Each unit's sensor sequence represents one complete run-to-failure
    trajectory. Splitting by row would leak future degradation patterns
    into training. By splitting by unit, the model never sees any part
    of a test unit's lifecycle during training.

    Args:
        df: Full dataset with machine_id column.
        unit_col: Column containing unit identifiers.
        test_fraction: Fraction of units for test set.
        val_fraction: Fraction of units for validation set.
        random_seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) — each containing full unit lifecycles.
    """
    rng = np.random.default_rng(random_seed)
    units = df[unit_col].unique()
    n_units = len(units)

    # Shuffle units (not rows)
    shuffled_units = rng.permutation(units)

    n_test = max(1, int(n_units * test_fraction))
    n_val = max(1, int(n_units * val_fraction))
    n_train = n_units - n_test - n_val

    if n_train < 1:
        raise ValueError(
            f"Not enough units ({n_units}) for the requested split fractions. "
            f"Would result in {n_train} training units."
        )

    test_units = set(shuffled_units[:n_test])
    val_units = set(shuffled_units[n_test : n_test + n_val])
    train_units = set(shuffled_units[n_test + n_val :])

    train_df = df[df[unit_col].isin(train_units)].copy()
    val_df = df[df[unit_col].isin(val_units)].copy()
    test_df = df[df[unit_col].isin(test_units)].copy()

    # Verify no leakage
    assert len(train_units & test_units) == 0, "Unit leakage: train ∩ test"
    assert len(train_units & val_units) == 0, "Unit leakage: train ∩ val"
    assert len(val_units & test_units) == 0, "Unit leakage: val ∩ test"

    logger.info(
        f"Unit-based split: train={len(train_units)} units ({len(train_df)} rows), "
        f"val={len(val_units)} units ({len(val_df)} rows), "
        f"test={len(test_units)} units ({len(test_df)} rows)"
    )

    return train_df, val_df, test_df


def split_temporal(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by temporal ordering for anomaly detection.

    Why temporal split?
    -------------------
    For anomaly detection, we train on "normal" early data and test on
    later data that may contain anomalies. Random splitting would leak
    anomalous patterns into training, defeating the purpose.

    The split uses operating_cycle if available, falling back to
    timestamp ordering.

    Args:
        df: Dataset sorted by time.
        timestamp_col: Column to use for ordering.
        test_fraction: Fraction of time for test.
        val_fraction: Fraction of time for validation.

    Returns:
        (train_df, val_df, test_df)
    """
    # Sort by time
    if "operating_cycle" in df.columns:
        df = df.sort_values(["machine_id", "operating_cycle"]).copy()
    elif timestamp_col in df.columns:
        df = df.sort_values(["machine_id", timestamp_col]).copy()

    # Split per machine to handle multi-machine datasets
    train_parts, val_parts, test_parts = [], [], []

    for _machine_id, group in df.groupby("machine_id"):
        n = len(group)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))

        # Chronological split: train = early, val = middle, test = late
        train_end = n - n_test - n_val
        val_end = n - n_test

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    logger.info(
        f"Temporal split: train={len(train_df)} rows, "
        f"val={len(val_df)} rows, test={len(test_df)} rows"
    )

    return train_df, val_df, test_df


def split_stratified(
    df: pd.DataFrame,
    target_col: str = "failure_mode",
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split for classification, preserving class distribution.

    Why stratified?
    ---------------
    AI4I has severe class imbalance (>96% normal, <4% failures).
    Random splitting could produce test sets with zero samples of
    rare failure modes. Stratified splitting ensures each split
    has representative proportions of all classes.

    Args:
        df: Dataset with classification target.
        target_col: Column to stratify on.
        test_fraction: Fraction for test set.
        val_fraction: Fraction for validation set.
        random_seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    # First split: separate test
    splitter_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=random_seed
    )
    trainval_idx, test_idx = next(splitter_test.split(df, df[target_col]))

    trainval_df = df.iloc[trainval_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Second split: separate val from trainval
    val_fraction_adjusted = val_fraction / (1 - test_fraction)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction_adjusted, random_state=random_seed
    )
    train_idx, val_idx = next(splitter_val.split(trainval_df, trainval_df[target_col]))

    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()

    # Log class distributions per split
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df[target_col].value_counts()
        logger.info(f"Stratified split — {name}: {len(split_df)} rows, classes: {dist.to_dict()}")

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Save train/val/test splits as parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{prefix}_{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
        logger.info(f"Saved {name}: {len(df)} rows → {path}")

    return paths


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unit_col: str = "machine_id",
) -> bool:
    """
    Verify no unit/machine ID leakage between splits.

    Returns True if no leakage detected.
    Raises AssertionError if leakage found.
    """
    if unit_col not in train_df.columns:
        logger.warning(f"Column '{unit_col}' not found — skipping leakage check")
        return True

    train_ids = set(train_df[unit_col].unique())
    val_ids = set(val_df[unit_col].unique())
    test_ids = set(test_df[unit_col].unique())

    train_val_leak = train_ids & val_ids
    train_test_leak = train_ids & test_ids
    val_test_leak = val_ids & test_ids

    if train_val_leak or train_test_leak or val_test_leak:
        msg = (
            f"DATA LEAKAGE DETECTED!\n"
            f"  train ∩ val: {train_val_leak}\n"
            f"  train ∩ test: {train_test_leak}\n"
            f"  val ∩ test: {val_test_leak}"
        )
        logger.error(msg)
        raise AssertionError(msg)

    logger.info("✅ No data leakage detected between splits")
    return True
