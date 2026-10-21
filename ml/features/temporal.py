"""
MaintenAI— Temporal Feature Extractor

Computes time-domain features from sensor time-series data using sliding windows.
This module is the workhorse of the feature engineering pipeline, producing the
majority of features used by all three model families.

Features computed:
  1. Rolling Window Statistics: mean, std, min, max over configurable windows
  2. Exponential Moving Averages (EMA): smoothed trend indicators
  3. Lag Features: shifted values capturing recent history
  4. Rate of Change (delta): first derivatives capturing dynamics

Design Principle:
  This module uses only pandas vectorized operations and is designed to be
  used identically at training time (batch) and inference time (streaming window).
  No future-leaking operations (e.g., centered rolling) are used — all windows
  are trailing (right-aligned).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    window_sizes: list[int],
    stats: list[str] | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute rolling window statistics for specified columns.

    For each column and window size, computes requested stats:
      - rolling mean
      - rolling std
      - rolling min (optional)
      - rolling max (optional)

    All windows are trailing (no future leakage). NaN rows from insufficient
    history are filled via backfill within each group.

    Args:
        df: Input DataFrame sorted by time within each group.
        columns: Columns to compute rolling features on.
        window_sizes: List of window sizes (in rows/cycles).
        stats: List of statistics to compute (default: ['mean', 'std', 'min', 'max']).
        group_col: Column to group by (e.g., 'machine_id'). If None, treats
                   entire DataFrame as one group.

    Returns:
        DataFrame with original columns plus new rolling feature columns.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]

    result = df.copy()
    available_cols = [c for c in columns if c in df.columns]

    if not available_cols:
        logger.warning(f"No matching columns found from {columns}")
        return result

    for col in available_cols:
        for window in window_sizes:
            prefix = f"{col}_roll{window}"

            if group_col and group_col in df.columns:
                grouped = df.groupby(group_col)[col]
                rolling = grouped.rolling(window=window, min_periods=1)
            else:
                rolling = df[col].rolling(window=window, min_periods=1)

            if "mean" in stats:
                result[f"{prefix}_mean"] = (
                    rolling.mean().reset_index(level=0, drop=True) if group_col else rolling.mean()
                )
            if "std" in stats:
                result[f"{prefix}_std"] = (
                    rolling.std().reset_index(level=0, drop=True) if group_col else rolling.std()
                )
            if "min" in stats:
                result[f"{prefix}_min"] = (
                    rolling.min().reset_index(level=0, drop=True) if group_col else rolling.min()
                )
            if "max" in stats:
                result[f"{prefix}_max"] = (
                    rolling.max().reset_index(level=0, drop=True) if group_col else rolling.max()
                )

    # Fill NaN from initial window positions with backfill within groups
    new_cols = [c for c in result.columns if c not in df.columns]
    if group_col and group_col in df.columns:
        result[new_cols] = result.groupby(group_col)[new_cols].transform(
            lambda x: x.bfill().ffill()
        )
    else:
        result[new_cols] = result[new_cols].bfill().ffill()

    # Fill remaining NaN with 0 (edge case: constant columns yield NaN std)
    result[new_cols] = result[new_cols].fillna(0.0)

    n_new = len(new_cols)
    logger.debug(
        f"Rolling features: {n_new} new columns ({len(available_cols)} cols × {len(window_sizes)} windows × 4 stats)"
    )
    return result


def compute_ema_features(
    df: pd.DataFrame,
    columns: list[str],
    spans: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute Exponential Moving Average (EMA) features.

    EMA gives more weight to recent observations, useful for capturing
    short-term trends and smoothing noisy sensor data.

    Args:
        df: Input DataFrame.
        columns: Columns to compute EMA on.
        spans: EMA span values (higher = smoother).
        group_col: Grouping column.

    Returns:
        DataFrame with EMA feature columns added.
    """
    result = df.copy()
    available_cols = [c for c in columns if c in df.columns]

    for col in available_cols:
        for span in spans:
            col_name = f"{col}_ema{span}"
            if group_col and group_col in df.columns:
                result[col_name] = df.groupby(group_col)[col].transform(
                    lambda x, s=span: x.ewm(span=s, min_periods=1, adjust=False).mean()
                )
            else:
                result[col_name] = df[col].ewm(span=span, min_periods=1, adjust=False).mean()

    n_new = len(available_cols) * len(spans)
    logger.debug(f"EMA features: {n_new} new columns")
    return result


def compute_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lag_steps: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute lagged feature values.

    Lag features provide the model with explicit access to recent historical
    values, useful for capturing temporal patterns without requiring
    recurrent or attention mechanisms.

    Args:
        df: Input DataFrame.
        columns: Columns to create lags for.
        lag_steps: Number of steps to lag (e.g., [1, 5, 10]).
        group_col: Grouping column.

    Returns:
        DataFrame with lag feature columns added.
    """
    result = df.copy()
    available_cols = [c for c in columns if c in df.columns]

    for col in available_cols:
        for lag in lag_steps:
            col_name = f"{col}_lag{lag}"
            if group_col and group_col in df.columns:
                result[col_name] = df.groupby(group_col)[col].shift(lag)
            else:
                result[col_name] = df[col].shift(lag)

    # Fill NaN from lag with backfill within groups
    new_cols = [c for c in result.columns if c not in df.columns]
    if group_col and group_col in df.columns:
        result[new_cols] = result.groupby(group_col)[new_cols].transform(
            lambda x: x.bfill().ffill()
        )
    else:
        result[new_cols] = result[new_cols].bfill().ffill()

    result[new_cols] = result[new_cols].fillna(0.0)

    n_new = len(new_cols)
    logger.debug(f"Lag features: {n_new} new columns")
    return result


def compute_delta_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: list[int] | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute rate-of-change (delta/derivative) features.

    Delta features capture how quickly sensor values are changing,
    which is critical for detecting the onset of degradation.
    A sudden increase in rate-of-change often precedes failure.

    Computes:
      - delta (first difference): x[t] - x[t-period]
      - delta_pct (percent change): (x[t] - x[t-period]) / |x[t-period]|

    Args:
        df: Input DataFrame.
        columns: Columns to compute deltas for.
        periods: Periods for differencing (default: [1]).
        group_col: Grouping column.

    Returns:
        DataFrame with delta feature columns added.
    """
    if periods is None:
        periods = [1]

    result = df.copy()
    available_cols = [c for c in columns if c in df.columns]

    for col in available_cols:
        for period in periods:
            if group_col and group_col in df.columns:
                shifted = df.groupby(group_col)[col].shift(period)
            else:
                shifted = df[col].shift(period)

            delta = df[col] - shifted
            result[f"{col}_delta{period}"] = delta

            # Percent change (guarded against division by zero)
            denom = shifted.abs().replace(0, np.nan)
            result[f"{col}_delta{period}_pct"] = (delta / denom).fillna(0.0)

    # Fill NaN from initial positions
    new_cols = [c for c in result.columns if c not in df.columns]
    if group_col and group_col in df.columns:
        result[new_cols] = result.groupby(group_col)[new_cols].transform(
            lambda x: x.bfill().ffill()
        )
    else:
        result[new_cols] = result[new_cols].bfill().ffill()

    result[new_cols] = result[new_cols].fillna(0.0)

    n_new = len(new_cols)
    logger.debug(f"Delta features: {n_new} new columns")
    return result


def compute_all_temporal_features(
    df: pd.DataFrame,
    columns: list[str],
    window_sizes: list[int] | None = None,
    ema_spans: list[int] | None = None,
    lag_steps: list[int] | None = None,
    delta_periods: list[int] | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute all temporal features in a single pass.

    This is the primary entrypoint for the temporal feature module.
    Applies rolling stats, EMA, lags, and deltas sequentially.

    Args:
        df: Input DataFrame with sensor columns.
        columns: Sensor columns to featurize.
        window_sizes: Rolling window sizes (default: [5, 10, 20, 50]).
        ema_spans: EMA span values (default: [10, 30]).
        lag_steps: Lag steps (default: [1, 5, 10]).
        delta_periods: Delta periods (default: [1]).
        group_col: Column for per-machine grouping.

    Returns:
        DataFrame with all temporal features added.
    """
    if window_sizes is None:
        window_sizes = [5, 10, 20, 50]
    if ema_spans is None:
        ema_spans = [10, 30]
    if lag_steps is None:
        lag_steps = [1, 5, 10]
    if delta_periods is None:
        delta_periods = [1]

    original_cols = len(df.columns)

    result = compute_rolling_features(df, columns, window_sizes, group_col)
    result = compute_ema_features(result, columns, ema_spans, group_col)
    result = compute_lag_features(result, columns, lag_steps, group_col)
    result = compute_delta_features(result, columns, delta_periods, group_col)

    new_cols = len(result.columns) - original_cols
    logger.info(
        f"Temporal features: {new_cols} new columns from {len(columns)} sensors "
        f"(windows={window_sizes}, ema={ema_spans}, lags={lag_steps}, deltas={delta_periods})"
    )
    return result
