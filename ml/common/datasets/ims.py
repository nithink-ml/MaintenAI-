"""
MaintenAI— NASA IMS Bearing Dataset Processor

Loads the NASA IMS Bearing Dataset and maps it to the FactoryPulse
canonical schema for anomaly detection.

IMS Data Format:
  - 3 test directories (1st_test, 2nd_test, 3rd_test)
  - Each test: accelerometer data files with 4-8 channels
  - Each file: tab-delimited, one column per bearing channel
  - Sampling rate: 20,480 Hz
  - Each file captures 1-second of data (20,480 data points per channel)
  - Files recorded at periodic intervals (every 10 minutes)

Test 1 (used here):
  - 4 bearings with 2 accelerometer channels each → 8 channels total
  - Bearing 3 develops outer race failure
  - Bearing 4 develops rolling element failure

Canonical Schema Mapping:
  - file_index → operating_cycle
  - filename timestamp → timestamp
  - Vibration RMS per bearing → vibration
  - Spectral energy → frequency-domain features
  - machine_id: "IMS_bearing_{bearing_num}"

Processing Approach:
  We compute per-file statistical summaries (RMS, kurtosis, peak-to-peak,
  spectral energy) rather than storing all 20K samples per file. This creates
  a manageable time-series of summary statistics that captures bearing
  degradation patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _compute_vibration_features(signal: np.ndarray) -> dict[str, float]:
    """
    Compute time-domain and basic frequency-domain features from
    a single vibration signal snapshot.

    Args:
        signal: 1D array of accelerometer readings.

    Returns:
        Dict of feature name → value.
    """
    if len(signal) == 0:
        return {}

    # Time-domain features
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    peak_to_peak = float(np.max(signal) - np.min(signal))
    std = float(np.std(signal))
    kurtosis = float(scipy_stats.kurtosis(signal, fisher=True))
    skewness = float(scipy_stats.skew(signal))
    crest_factor = peak / rms if rms > 0 else 0.0

    # Frequency-domain features (FFT)
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_vals) / n
    spectral_energy = float(np.sum(fft_magnitude**2))

    # Dominant frequency (index of max magnitude, excluding DC)
    dominant_freq_idx = int(np.argmax(fft_magnitude[1:])) + 1 if len(fft_magnitude) > 1 else 0

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "std": std,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "spectral_energy": spectral_energy,
        "dominant_freq_idx": float(dominant_freq_idx),
    }


def _parse_ims_filename_timestamp(filename: str) -> pd.Timestamp | None:
    """
    Parse timestamp from IMS filename format: YYYY.MM.DD.HH.MM.SS

    Returns None if parsing fails.
    """
    parts = filename.split(".")
    if len(parts) >= 6:
        try:
            return pd.Timestamp(
                year=int(parts[0]),
                month=int(parts[1]),
                day=int(parts[2]),
                hour=int(parts[3]),
                minute=int(parts[4]),
                second=int(parts[5]),
                tz="UTC",
            )
        except (ValueError, IndexError):
            pass
    return None


def load_and_process_ims_test(
    test_dir: Path,
    test_name: str = "test_1",
    n_bearings: int = 4,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Load and process one IMS test directory into time-series features.

    For each data file (snapshot), compute vibration summary statistics
    per bearing channel. This produces a time-series of feature vectors
    that captures bearing degradation over time.

    Args:
        test_dir: Path to test directory (e.g., data/raw/ims/1st_test/)
        test_name: Name identifier for this test
        n_bearings: Number of bearing channels to process (4 for test 1)
        max_files: Limit number of files processed (for development)

    Returns:
        DataFrame with one row per file, columns for each bearing's features.
    """
    if not test_dir.exists():
        raise FileNotFoundError(f"IMS test directory not found: {test_dir}")

    # List and sort data files
    data_files = sorted(
        [
            f
            for f in test_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.name != "description.txt"
        ]
    )

    if not data_files:
        raise FileNotFoundError(f"No data files found in {test_dir}")

    if max_files:
        data_files = data_files[:max_files]

    logger.info(f"Processing {len(data_files)} files from {test_dir}")

    records = []
    for file_idx, filepath in enumerate(data_files):
        try:
            # Load tab-delimited data
            data = np.loadtxt(filepath, delimiter="\t")
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            # Parse timestamp from filename
            ts = _parse_ims_filename_timestamp(filepath.name)
            if ts is None:
                # Fallback: synthetic timestamp based on file index
                ts = pd.Timestamp("2003-10-22", tz="UTC") + pd.Timedelta(minutes=file_idx * 10)

            record = {
                "file_index": file_idx,
                "filename": filepath.name,
                "timestamp": ts,
            }

            # Compute features for each bearing channel
            actual_channels = min(n_bearings, data.shape[1])
            for bearing_idx in range(actual_channels):
                signal = data[:, bearing_idx]
                features = _compute_vibration_features(signal)
                for feat_name, feat_val in features.items():
                    record[f"bearing_{bearing_idx + 1}_{feat_name}"] = feat_val

            records.append(record)

        except Exception as e:
            logger.warning(f"Error processing {filepath.name}: {e}")
            continue

    df = pd.DataFrame(records)
    logger.info(f"IMS {test_name}: {len(df)} snapshots, {len(df.columns)} columns")
    return df


def map_to_canonical(
    df: pd.DataFrame,
    test_name: str = "test_1",
) -> pd.DataFrame:
    """
    Map IMS bearing features to canonical schema format.

    Creates one row per bearing per timestamp, with the bearing's
    vibration RMS as the primary vibration metric.
    """
    records = []
    bearing_cols = [c for c in df.columns if c.startswith("bearing_")]
    bearing_nums = sorted(set(int(c.split("_")[1]) for c in bearing_cols))

    for _, row in df.iterrows():
        for bearing_num in bearing_nums:
            prefix = f"bearing_{bearing_num}_"
            feature_cols = [c for c in bearing_cols if c.startswith(prefix)]

            if not feature_cols:
                continue

            record = {
                "machine_id": f"IMS_{test_name}_bearing_{bearing_num}",
                "timestamp": row["timestamp"],
                "vibration": row.get(f"{prefix}rms", None),
                "operating_cycle": int(row.get("file_index", 0)),
                # Additional features for ML
                "temperature": None,  # IMS has no temperature
                "pressure": None,  # IMS has no pressure
                "rpm": None,  # IMS has no RPM
                "current": None,  # IMS has no current
            }

            # Carry all bearing-specific features
            for col in feature_cols:
                feat_name = col.replace(prefix, "")
                record[f"raw_{feat_name}"] = row[col]

            records.append(record)

    canonical = pd.DataFrame(records)
    return canonical


def add_anomaly_labels(
    df: pd.DataFrame,
    test_name: str = "test_1",
    failure_bearings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add anomaly labels to IMS data based on known failure points.

    For IMS Test 1:
      - Bearing 3: outer race failure (last ~20% of files show degradation)
      - Bearing 4: rolling element failure (last ~15%)

    Labels:
      0 = normal
      1 = anomalous (degradation/failure region)

    The threshold is set conservatively — the last fraction of the
    test is labeled anomalous based on documented failure reports.
    """
    df = df.copy()

    if failure_bearings is None:
        failure_bearings = {
            f"IMS_{test_name}_bearing_3": 0.80,  # Failure starts ~80% through
            f"IMS_{test_name}_bearing_4": 0.85,
        }

    # Get max cycle per machine
    max_cycles = df.groupby("machine_id")["operating_cycle"].transform("max")

    df["anomaly_label"] = 0
    for machine_id, threshold_pct in failure_bearings.items():
        mask = (df["machine_id"] == machine_id) & (
            df["operating_cycle"] > max_cycles * threshold_pct
        )
        df.loc[mask, "anomaly_label"] = 1

    label_dist = df.groupby("machine_id")["anomaly_label"].value_counts().unstack(fill_value=0)
    logger.info(f"IMS anomaly labels:\n{label_dist}")

    return df


def process_ims(
    raw_dir: Path,
    output_dir: Path,
    max_files: int | None = None,
) -> dict[str, Path]:
    """
    Full IMS processing pipeline.

    1. Load raw accelerometer data files
    2. Compute per-file vibration statistics
    3. Map to canonical schema (one row per bearing per snapshot)
    4. Add anomaly labels based on known failures
    5. Validate
    6. Save processed parquet

    Returns:
        Dict of output file paths.
    """
    from ml.common.validation import run_full_validation

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find test directory
    test_dir = None
    for candidate_name in ["1st_test", "1st_Test", "test1"]:
        candidate = raw_dir / candidate_name
        if candidate.exists() and candidate.is_dir():
            test_dir = candidate
            break

    if test_dir is None:
        # Maybe the files are directly in raw_dir
        data_files = [f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        if data_files:
            test_dir = raw_dir
        else:
            raise FileNotFoundError(
                f"IMS test directory not found in {raw_dir}. "
                f"Expected subdirectory named '1st_test'."
            )

    # Load and compute features
    features_df = load_and_process_ims_test(test_dir, test_name="test_1", max_files=max_files)

    # Map to canonical
    canonical = map_to_canonical(features_df, test_name="test_1")

    # Add anomaly labels
    canonical = add_anomaly_labels(canonical, test_name="test_1")

    # Validate
    required = ["machine_id", "timestamp", "vibration", "operating_cycle"]
    report = run_full_validation(
        canonical, required, "ims_test_1", timestamp_col="timestamp", machine_id_col="machine_id"
    )
    logger.info(report.summary())

    # Save
    output_path = output_dir / "ims_test_1.parquet"
    canonical.to_parquet(output_path, index=False)

    # Also save the per-file feature summary (useful for EDA)
    features_path = output_dir / "ims_test_1_features.parquet"
    features_df.to_parquet(features_path, index=False)

    logger.info(
        f"IMS processed: {len(canonical)} rows → {output_path}, "
        f"features: {len(features_df)} rows → {features_path}"
    )

    return {"canonical": output_path, "features": features_path}
