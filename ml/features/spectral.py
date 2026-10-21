"""
MaintenAI— Spectral Feature Extractor

Computes frequency-domain and spectral-ratio features from vibration data.
Designed primarily for IMS bearing anomaly detection but reusable for any
vibration-based monitoring pipeline.

Note on IMS data:
  The processed IMS data already contains per-snapshot summary statistics
  (RMS, peak, spectral_energy, etc.) computed from the raw 20,480-sample
  accelerometer snapshots. This module operates on those summary statistics
  to create higher-level spectral ratio and trend features — it does NOT
  re-compute FFT on raw waveform data.

Design:
  All operations are vectorized and use trailing windows only (no future
  leakage). Functions are stateless and deterministic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_spectral_ratios(
    df: pd.DataFrame,
    energy_col: str = "raw_spectral_energy",
    rms_col: str = "vibration",
    peak_col: str = "raw_peak",
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute spectral ratio features from existing vibration summary stats.

    Features:
      - spectral_to_rms_ratio: spectral_energy / RMS — indicates frequency
        content relative to overall amplitude. Increases suggest spectral
        broadening from developing faults.
      - peak_to_energy_ratio: peak / spectral_energy — high ratio suggests
        impulsive events (bearing impacts).

    Args:
        df: DataFrame with vibration summary columns.
        energy_col: Spectral energy column name.
        rms_col: RMS vibration column name.
        peak_col: Peak amplitude column name.
        group_col: Grouping column (for per-bearing computation).

    Returns:
        DataFrame with spectral ratio columns added.
    """
    result = df.copy()

    if energy_col in df.columns and rms_col in df.columns:
        denom = df[rms_col].replace(0, np.nan)
        result["spectral_to_rms_ratio"] = (df[energy_col] / denom).fillna(0.0)

    if peak_col in df.columns and energy_col in df.columns:
        denom = df[energy_col].replace(0, np.nan)
        result["peak_to_energy_ratio"] = (df[peak_col] / denom).fillna(0.0)

    n_new = len(result.columns) - len(df.columns)
    logger.debug(f"Spectral ratios: {n_new} new columns")
    return result


def compute_spectral_trends(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 5,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute rolling trend features for spectral/vibration metrics.

    For each column, computes:
      - Rolling ratio: current / rolling_mean — values > 1 indicate
        the metric is above its recent average (potential anomaly).

    This is a simple but effective anomaly indicator: a sudden increase
    in the ratio signals deviation from recent baseline.

    Args:
        df: Input DataFrame.
        columns: Columns to compute trends for.
        window: Rolling window size.
        group_col: Grouping column.

    Returns:
        DataFrame with trend columns added.
    """
    result = df.copy()
    available = [c for c in columns if c in df.columns]

    for col in available:
        if group_col and group_col in df.columns:
            rolling_mean = df.groupby(group_col)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        else:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()

        denom = rolling_mean.replace(0, np.nan)
        result[f"{col}_trend_ratio{window}"] = (df[col] / denom).fillna(1.0)

    n_new = len(result.columns) - len(df.columns)
    logger.debug(f"Spectral trends: {n_new} new columns (window={window})")
    return result


def compute_all_spectral_features(
    df: pd.DataFrame,
    energy_col: str = "raw_spectral_energy",
    rms_col: str = "vibration",
    peak_col: str = "raw_peak",
    trend_columns: list[str] | None = None,
    trend_window: int = 5,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute all spectral features in a single call.

    This is the primary entrypoint for the spectral feature module.

    Args:
        df: Input DataFrame with vibration summary columns.
        energy_col: Spectral energy column.
        rms_col: RMS vibration column.
        peak_col: Peak amplitude column.
        trend_columns: Columns for trend computation.
        trend_window: Window size for trend ratios.
        group_col: Grouping column.

    Returns:
        DataFrame with all spectral features added.
    """
    if trend_columns is None:
        trend_columns = [energy_col, rms_col, peak_col]

    original_n = len(df.columns)

    result = compute_spectral_ratios(df, energy_col, rms_col, peak_col, group_col)
    result = compute_spectral_trends(result, trend_columns, trend_window, group_col)

    n_new = len(result.columns) - original_n
    logger.info(f"Spectral features: {n_new} new columns")
    return result


def compute_spectral_features(vib_series: np.ndarray | list[float]) -> dict[str, float]:
    """
    Compute online 1D FFT spectral energy and frequency band statistics.
    """
    arr = np.asarray(vib_series, dtype=float)
    if len(arr) < 4:
        return {}

    fft_vals = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(len(arr))
    total_energy = float(np.sum(fft_vals**2))
    peak_freq = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) > 0 else 0.0

    return {
        "raw_spectral_energy": total_energy,
        "spectral_peak_freq": peak_freq,
        "spectral_mean_amplitude": float(np.mean(fft_vals)),
    }
