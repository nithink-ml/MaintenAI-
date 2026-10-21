"""
MaintenAI— Feature Engineering Pipeline (DVC Entrypoint)

Orchestrates feature engineering for all three datasets:
  - C-MAPSS (FD001) → RUL prediction features
  - IMS Bearing → Anomaly detection features
  - AI4I 2020 → Failure classification features

Usage:
    python -m ml.features.build_features --dataset cmapss
    python -m ml.features.build_features --dataset ims
    python -m ml.features.build_features --dataset ai4i
    python -m ml.features.build_features --dataset all

Design Principles:
    1. Fit-then-transform: Scalers fit ONLY on training data, then applied
       to val/test identically (prevents training-serving skew).
    2. No cross-boundary operations: All rolling/lag/EMA features are
       grouped by machine_id (never computed across different engines/bearings).
    3. Trailing windows only: No centered or forward-looking operations
       (prevents temporal/future leakage).
    4. Deterministic: Same input always produces same output.
    5. Reusable: The same functions work for train, val, test, and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd

from ml.features.registry import (
    CMAPSS_PRIORITY_SENSORS,
    CMAPSS_SECONDARY_SENSORS,
    FEATURE_VERSION,
    IMS_DROP_COLUMNS,
    IMS_PRIMARY_FEATURES,
    save_feature_manifest,
)
from ml.features.spectral import compute_all_spectral_features
from ml.features.temporal import (
    compute_delta_features,
    compute_ema_features,
    compute_lag_features,
    compute_rolling_features,
)

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PARAMS_PATH = Path("params.yaml")


# ============================================================
# Parameter Loading
# ============================================================


def load_feature_params(params_path: Path = DEFAULT_PARAMS_PATH) -> dict:
    """
    Load feature engineering parameters from params.yaml.

    Returns dict with fallback defaults if params file not found.
    """
    defaults = {
        "cmapss": {
            "rolling_window_sizes": [5, 10, 20],
            "ema_spans": [10, 30],
            "lag_steps": [1, 5],
        },
        "ims": {
            "rolling_window_sizes": [3, 5, 10],
            "ema_spans": [5, 10],
        },
        "normalize": True,
        "scaler": "standard",
    }

    if not params_path.exists():
        return defaults

    try:
        import yaml

        with open(params_path) as f:
            cfg = yaml.safe_load(f) or {}
        feat_cfg = cfg.get("features", {})

        # Parse C-MAPSS params
        cmapss_cfg = feat_cfg.get("cmapss", {})
        cmapss_windows = cmapss_cfg.get(
            "rolling_window_sizes",
            feat_cfg.get("rolling_window_sizes", defaults["cmapss"]["rolling_window_sizes"]),
        )
        cmapss_ema = cmapss_cfg.get(
            "ema_spans",
            feat_cfg.get("ema_spans", defaults["cmapss"]["ema_spans"]),
        )
        cmapss_lags = cmapss_cfg.get(
            "lag_steps",
            feat_cfg.get("lag_steps", defaults["cmapss"]["lag_steps"]),
        )

        # Parse IMS params
        ims_cfg = feat_cfg.get("ims", {})
        ims_windows = ims_cfg.get("rolling_window_sizes", defaults["ims"]["rolling_window_sizes"])
        ims_ema = ims_cfg.get("ema_spans", defaults["ims"]["ema_spans"])

        return {
            "cmapss": {
                "rolling_window_sizes": cmapss_windows,
                "ema_spans": cmapss_ema,
                "lag_steps": cmapss_lags,
            },
            "ims": {
                "rolling_window_sizes": ims_windows,
                "ema_spans": ims_ema,
            },
            "normalize": feat_cfg.get("normalize", defaults["normalize"]),
            "scaler": feat_cfg.get("scaler", defaults["scaler"]),
        }
    except Exception as e:
        logger.warning(f"Failed to load {params_path}, using defaults: {e}")
        return defaults


# ============================================================
# Scaler Utilities
# ============================================================


def _get_scaler(scaler_type: str = "standard"):
    """
    Create a scaler instance based on type string.

    Supports 'standard' (v1), with extensible architecture for
    'robust' and 'minmax' without structural changes.
    """
    scaler_type_clean = scaler_type.lower()
    if scaler_type_clean == "standard":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    elif scaler_type_clean == "robust":
        from sklearn.preprocessing import RobustScaler

        return RobustScaler()
    elif scaler_type_clean == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        return MinMaxScaler()
    else:
        raise ValueError(
            f"Unknown scaler type: '{scaler_type}'. Supported: ['standard', 'robust', 'minmax']"
        )


def _fit_and_save_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    scaler_type: str = "standard",
) -> tuple[pd.DataFrame, object]:
    """
    Fit scaler on training data and save to disk.

    Returns:
        (scaled_train_df, fitted_scaler)
    """
    scaler = _get_scaler(scaler_type)
    available = [c for c in feature_cols if c in train_df.columns]

    scaler.fit(train_df[available])
    scaled = train_df.copy()
    scaled[available] = scaler.transform(train_df[available])

    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler ({scaler_type}) fit on {len(available)} features → {scaler_path}")

    return scaled, scaler


def _apply_scaler(
    df: pd.DataFrame,
    scaler: object,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a DataFrame."""
    available = [c for c in feature_cols if c in df.columns]
    result = df.copy()
    result[available] = scaler.transform(df[available])
    return result


# ============================================================
# C-MAPSS Feature Engineering
# ============================================================


def build_cmapss_features(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    window_sizes: list[int] | None = None,
    ema_spans: list[int] | None = None,
    lag_steps: list[int] | None = None,
    scaler_type: str = "standard",
    normalize: bool = True,
) -> dict[str, Path]:
    """
    Build features for C-MAPSS RUL prediction.

    Pipeline:
        1. Load train/val/test splits
        2. Compute temporal features (rolling, EMA, lag, delta) on
           priority + secondary sensors, grouped by machine_id
        3. Add lifecycle position feature
        4. Fit scaler on training data, apply to all splits
        5. Save feature parquets + manifest

    Returns:
        Dict mapping split names to output paths.
    """
    if window_sizes is None:
        window_sizes = [5, 10, 20]
    if ema_spans is None:
        ema_spans = [10, 30]
    if lag_steps is None:
        lag_steps = [1, 5]

    cmapss_dir = processed_dir / "cmapss"
    output_dir = processed_dir / "cmapss_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("C-MAPSS Feature Engineering")
    logger.info("=" * 60)

    # Load splits
    splits = {
        "train": pd.read_parquet(cmapss_dir / "cmapss_FD001_train.parquet"),
        "val": pd.read_parquet(cmapss_dir / "cmapss_FD001_val.parquet"),
        "test": pd.read_parquet(cmapss_dir / "cmapss_FD001_test.parquet"),
    }

    for name, df in splits.items():
        logger.info(f"  Loaded {name}: {len(df)} rows, {len(df.columns)} cols")

    # Define columns for temporal features
    # Priority sensors get full treatment (rolling + EMA + lag + delta)
    # Secondary sensors get delta only (rate of change)
    priority_cols = [c for c in CMAPSS_PRIORITY_SENSORS if c in splits["train"].columns]
    secondary_cols = [c for c in CMAPSS_SECONDARY_SENSORS if c in splits["train"].columns]
    all_sensor_cols = priority_cols + secondary_cols

    # Process each split identically
    result_splits = {}
    for name, df in splits.items():
        logger.info(f"  Processing {name}...")

        # Ensure sorted by machine_id + operating_cycle
        df = df.sort_values(["machine_id", "operating_cycle"]).reset_index(drop=True)

        # 1. Rolling features on priority sensors
        df = compute_rolling_features(df, priority_cols, window_sizes, group_col="machine_id")

        # 2. EMA on priority sensors
        df = compute_ema_features(df, priority_cols, ema_spans, group_col="machine_id")

        # 3. Lag features on priority sensors
        df = compute_lag_features(df, priority_cols, lag_steps, group_col="machine_id")

        # 4. Delta (rate of change) on ALL informative sensors
        df = compute_delta_features(df, all_sensor_cols, periods=[1], group_col="machine_id")

        # 5. Lifecycle position (normalized operating cycle)
        # Uses cumulative max within each unit — no future leakage
        df["lifecycle_pct"] = df.groupby("machine_id")["operating_cycle"].transform(
            lambda x: x / x.expanding().max()
        )

        result_splits[name] = df
        logger.info(f"    → {len(df.columns)} columns after feature engineering")

    # 6. Normalize (fit on train, apply to all)
    # Identify numeric feature columns (exclude IDs, timestamps, targets)
    meta_cols = ["machine_id", "timestamp", "operating_cycle", "rul"]
    if normalize:
        feature_cols = [
            c
            for c in result_splits["train"].columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(result_splits["train"][c])
        ]

        result_splits["train"], scaler = _fit_and_save_scaler(
            result_splits["train"], feature_cols, output_dir, scaler_type
        )
        for split_name in ["val", "test"]:
            result_splits[split_name] = _apply_scaler(
                result_splits[split_name], scaler, feature_cols
            )

    # 7. Save
    paths = {}
    for name, df in result_splits.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
        logger.info(f"  Saved {name}: {len(df)} rows, {len(df.columns)} cols → {path}")

    # 8. Save manifest
    save_feature_manifest(
        "cmapss",
        output_dir,
        window_sizes=window_sizes,
        ema_spans=ema_spans,
        lag_steps=lag_steps,
    )

    logger.info(f"✅ C-MAPSS feature engineering complete (v{FEATURE_VERSION})")
    return paths


# ============================================================
# IMS Feature Engineering
# ============================================================


def build_ims_features(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    window_sizes: list[int] | None = None,
    ema_spans: list[int] | None = None,
    scaler_type: str = "standard",
    normalize: bool = True,
) -> dict[str, Path]:
    """
    Build features for IMS anomaly detection.

    Pipeline:
        1. Load train/val/test splits
        2. Drop null sensor columns and redundant raw_rms
        3. Add log-transforms for skewed features
        4. Compute temporal features (rolling, EMA, delta) on primary
           vibration features, grouped by machine_id (bearing)
        5. Compute spectral ratio features
        6. Fit scaler on training data, apply to all splits
        7. Save feature parquets + manifest

    NOTE: temperature, pressure, rpm, current are DROPPED because they
    are 100% null in IMS (accelerometer-only dataset). This is documented
    in the feature registry and NOT fabricated.

    NOTE: raw_rms is DROPPED because it is identical to vibration
    (correlation=1.0). This is documented in the feature registry.

    NOTE: IMS has only 50 snapshots per bearing (2.3% of real data).
    Window sizes are reduced to [3, 5, 10] accordingly.
    """
    if window_sizes is None:
        window_sizes = [3, 5, 10]
    if ema_spans is None:
        ema_spans = [5, 10]

    ims_dir = processed_dir / "ims"
    output_dir = processed_dir / "ims_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("IMS Feature Engineering")
    logger.info("=" * 60)

    # Load splits
    splits = {
        "train": pd.read_parquet(ims_dir / "ims_train.parquet"),
        "val": pd.read_parquet(ims_dir / "ims_val.parquet"),
        "test": pd.read_parquet(ims_dir / "ims_test.parquet"),
    }

    for name, df in splits.items():
        logger.info(f"  Loaded {name}: {len(df)} rows, {len(df.columns)} cols")

    # Columns to drop (documented reasons)
    drop_cols = [c for c in IMS_DROP_COLUMNS if c in splits["train"].columns]
    logger.info(f"  Dropping {len(drop_cols)} columns: {drop_cols}")
    for reason_col, reason in IMS_DROP_COLUMNS.items():
        if reason_col in splits["train"].columns:
            logger.info(f"    {reason_col}: {reason}")

    # Primary vibration features for temporal analysis
    primary_features = [c for c in IMS_PRIMARY_FEATURES if c in splits["train"].columns]

    # Process each split identically
    result_splits = {}
    for name, df in splits.items():
        logger.info(f"  Processing {name}...")

        # Sort by machine_id + operating_cycle
        df = df.sort_values(["machine_id", "operating_cycle"]).reset_index(drop=True)

        # 1. Drop null/redundant columns
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # 2. Log-transforms for skewed features
        for col in ["vibration", "raw_peak", "raw_spectral_energy"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        # 3. Rolling features on primary vibration features
        df = compute_rolling_features(
            df, primary_features, window_sizes, stats=["mean", "std"], group_col="machine_id"
        )

        # 4. EMA on primary features
        df = compute_ema_features(df, primary_features, ema_spans, group_col="machine_id")

        # 5. Delta features
        df = compute_delta_features(df, primary_features, periods=[1], group_col="machine_id")

        # 6. Spectral features
        df = compute_all_spectral_features(
            df,
            energy_col="raw_spectral_energy",
            rms_col="vibration",
            peak_col="raw_peak",
            trend_columns=["raw_spectral_energy", "vibration", "raw_peak"],
            trend_window=min(5, window_sizes[-1] if window_sizes else 5),
            group_col="machine_id",
        )

        result_splits[name] = df
        logger.info(f"    → {len(df.columns)} columns after feature engineering")

    # 7. Normalize (fit on train, apply to all)
    meta_cols = ["machine_id", "timestamp", "operating_cycle", "anomaly_label"]
    if normalize:
        feature_cols = [
            c
            for c in result_splits["train"].columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(result_splits["train"][c])
        ]

        result_splits["train"], scaler = _fit_and_save_scaler(
            result_splits["train"], feature_cols, output_dir, scaler_type
        )
        for split_name in ["val", "test"]:
            result_splits[split_name] = _apply_scaler(
                result_splits[split_name], scaler, feature_cols
            )

    # 8. Save
    paths = {}
    for name, df in result_splits.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
        logger.info(f"  Saved {name}: {len(df)} rows, {len(df.columns)} cols → {path}")

    # 9. Save manifest
    save_feature_manifest(
        "ims",
        output_dir,
        window_sizes=window_sizes,
        ema_spans=ema_spans,
    )

    logger.info(f"✅ IMS feature engineering complete (v{FEATURE_VERSION})")
    return paths


# ============================================================
# AI4I Feature Engineering
# ============================================================


def build_ai4i_features(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    scaler_type: str = "standard",
    normalize: bool = True,
) -> dict[str, Path]:
    """
    Build features for AI4I failure classification.

    Pipeline:
        1. Load train/val/test splits
        2. Compute interaction features (power, temp_diff, torque_per_rpm)
        3. Compute polynomial features (tool_wear_squared)
        4. One-hot encode product_type
        5. Fit scaler on training data, apply to all splits
        6. Save feature parquets + manifest

    NOTE: Class imbalance is NOT addressed here — it is handled during
    model training via class_weight='balanced' (per params.yaml).
    Rebalancing during feature engineering would introduce bias.
    """
    ai4i_dir = processed_dir / "ai4i"
    output_dir = processed_dir / "ai4i_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AI4I Feature Engineering")
    logger.info("=" * 60)

    # Load splits
    splits = {
        "train": pd.read_parquet(ai4i_dir / "ai4i_train.parquet"),
        "val": pd.read_parquet(ai4i_dir / "ai4i_val.parquet"),
        "test": pd.read_parquet(ai4i_dir / "ai4i_test.parquet"),
    }

    for name, df in splits.items():
        logger.info(f"  Loaded {name}: {len(df)} rows, {len(df.columns)} cols")

    # Process each split identically
    result_splits = {}
    for name, df in splits.items():
        logger.info(f"  Processing {name}...")

        # 1. Interaction features
        if "vibration" in df.columns and "rpm" in df.columns:
            # Power = torque × rotational speed
            df["power"] = df["vibration"] * df["rpm"]

            # Torque-to-speed ratio (guarded against div-by-zero)
            denom = df["rpm"].replace(0, np.nan)
            df["torque_per_rpm"] = (df["vibration"] / denom).fillna(0.0)

        if "temperature" in df.columns and "pressure" in df.columns:
            # Temperature differential (process_temp - air_temp)
            df["temp_diff"] = df["pressure"] - df["temperature"]

        if "temperature" in df.columns and "rpm" in df.columns:
            # Temperature-speed interaction
            df["temp_rpm_interaction"] = df["temperature"] * df["rpm"]

        # 2. Polynomial features
        if "operating_cycle" in df.columns:
            df["tool_wear_squared"] = df["operating_cycle"] ** 2

        # 3. One-hot encode product_type
        if "product_type" in df.columns:
            for pt in ["L", "M", "H"]:
                df[f"product_type_{pt}"] = (df["product_type"] == pt).astype(int)

        result_splits[name] = df
        logger.info(f"    → {len(df.columns)} columns after feature engineering")

    # 4. Normalize (fit on train, apply to all)
    meta_cols = [
        "machine_id",
        "timestamp",
        "failure_label",
        "failure_mode",
        "product_type",
        # Individual failure flags (binary targets, not features for scaling)
        "failure_tool_wear",
        "failure_heat_dissipation",
        "failure_power_failure",
        "failure_overstrain",
        "failure_random",
    ]
    # Also exclude one-hot columns from scaling (they're already 0/1)
    onehot_cols = [c for c in result_splits["train"].columns if c.startswith("product_type_")]

    if normalize:
        feature_cols = [
            c
            for c in result_splits["train"].columns
            if c not in meta_cols
            and c not in onehot_cols
            and pd.api.types.is_numeric_dtype(result_splits["train"][c])
        ]

        result_splits["train"], scaler = _fit_and_save_scaler(
            result_splits["train"], feature_cols, output_dir, scaler_type
        )
        for split_name in ["val", "test"]:
            result_splits[split_name] = _apply_scaler(
                result_splits[split_name], scaler, feature_cols
            )

    # 5. Save
    paths = {}
    for name, df in result_splits.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
        logger.info(f"  Saved {name}: {len(df)} rows, {len(df.columns)} cols → {path}")

    # 6. Save manifest
    save_feature_manifest("ai4i", output_dir)

    logger.info(f"✅ AI4I feature engineering complete (v{FEATURE_VERSION})")
    return paths


# ============================================================
# CLI Entry Point
# ============================================================


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["cmapss", "ims", "ai4i", "all"]),
    default="all",
    help="Dataset to build features for.",
)
@click.option(
    "--processed-dir",
    type=click.Path(),
    default="data/processed",
    help="Path to processed data directory.",
)
@click.option(
    "--params-file",
    type=click.Path(),
    default="params.yaml",
    help="Path to params.yaml configuration file.",
)
@click.option(
    "--scaler",
    type=click.Choice(["standard", "robust", "minmax"]),
    default=None,
    help="Scaler type for normalization (overrides params.yaml).",
)
@click.option(
    "--no-normalize",
    is_flag=True,
    default=False,
    help="Skip normalization step.",
)
def main(
    dataset: str,
    processed_dir: str,
    params_file: str,
    scaler: str | None,
    no_normalize: bool,
) -> None:
    """Build MaintenAIfeatures for ML training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    proc = Path(processed_dir)
    params = load_feature_params(Path(params_file))

    scaler_type = scaler or params.get("scaler", "standard")
    normalize = not no_normalize if no_normalize else params.get("normalize", True)

    cmapss_cfg = params.get("cmapss", {})
    ims_cfg = params.get("ims", {})

    if dataset in ("cmapss", "all"):
        build_cmapss_features(
            proc,
            window_sizes=cmapss_cfg.get("rolling_window_sizes"),
            ema_spans=cmapss_cfg.get("ema_spans"),
            lag_steps=cmapss_cfg.get("lag_steps"),
            scaler_type=scaler_type,
            normalize=normalize,
        )

    if dataset in ("ims", "all"):
        build_ims_features(
            proc,
            window_sizes=ims_cfg.get("rolling_window_sizes"),
            ema_spans=ims_cfg.get("ema_spans"),
            scaler_type=scaler_type,
            normalize=normalize,
        )

    if dataset in ("ai4i", "all"):
        build_ai4i_features(
            proc,
            scaler_type=scaler_type,
            normalize=normalize,
        )


if __name__ == "__main__":
    main()
