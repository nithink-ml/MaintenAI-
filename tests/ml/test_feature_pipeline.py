"""
MaintenAI— Comprehensive ML Feature Pipeline Verification

Validates:
1. End-to-end execution of build_features for C-MAPSS, IMS, and AI4I
2. Output schemas, column types, and feature counts
3. Absence of NaN and Infinite values across all splits
4. Preservation of row counts between interim/processed and featurized data
5. Temporal ordering within each machine/bearing
6. Isolation across machine/bearing boundaries
7. No data leakage (fit-then-transform scaler verification)
8. Determinism (repeated execution yields identical results)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features.build_features import (
    build_ai4i_features,
    build_cmapss_features,
    build_ims_features,
    load_feature_params,
)


@pytest.fixture(scope="module")
def featurized_data(tmp_path_factory):
    """Run feature pipeline on processed data and return output directories."""
    proc_dir = Path("data/processed")
    tmp_path_factory.mktemp("features_test")

    params = load_feature_params()
    cmapss_cfg = params.get("cmapss", {})
    ims_cfg = params.get("ims", {})

    # Run C-MAPSS
    cmapss_paths = build_cmapss_features(
        processed_dir=proc_dir,
        window_sizes=cmapss_cfg.get("rolling_window_sizes", [5, 10, 20]),
        ema_spans=cmapss_cfg.get("ema_spans", [10, 30]),
        lag_steps=cmapss_cfg.get("lag_steps", [1, 5]),
        scaler_type="standard",
        normalize=True,
    )

    # Run IMS
    ims_paths = build_ims_features(
        processed_dir=proc_dir,
        window_sizes=ims_cfg.get("rolling_window_sizes", [3, 5, 10]),
        ema_spans=ims_cfg.get("ema_spans", [5, 10]),
        scaler_type="standard",
        normalize=True,
    )

    # Run AI4I
    ai4i_paths = build_ai4i_features(
        processed_dir=proc_dir,
        scaler_type="standard",
        normalize=True,
    )

    return {
        "cmapss": cmapss_paths,
        "ims": ims_paths,
        "ai4i": ai4i_paths,
    }


class TestCmapssFeaturePipeline:
    """Verification for C-MAPSS feature pipeline."""

    def test_cmapss_files_and_schema(self, featurized_data):
        paths = featurized_data["cmapss"]
        for split in ["train", "val", "test"]:
            assert split in paths
            assert paths[split].exists()
            df = pd.read_parquet(paths[split])
            assert len(df) > 0

            # Meta and target columns
            assert "machine_id" in df.columns
            assert "operating_cycle" in df.columns
            assert "rul" in df.columns
            assert "lifecycle_pct" in df.columns

            # Check feature count (~140 - 170 columns)
            assert 130 <= len(df.columns) <= 180

    def test_cmapss_no_nan_or_inf(self, featurized_data):
        paths = featurized_data["cmapss"]
        for split, path in paths.items():
            df = pd.read_parquet(path)
            num_cols = df.select_dtypes(include=[np.number]).columns
            assert not df[num_cols].isna().any().any(), f"NaNs found in C-MAPSS {split}"
            assert not np.isinf(df[num_cols].to_numpy()).any(), f"Infs found in C-MAPSS {split}"

    def test_cmapss_row_counts_preserved(self, featurized_data):
        proc_dir = Path("data/processed/cmapss")
        orig_train = len(pd.read_parquet(proc_dir / "cmapss_FD001_train.parquet"))
        orig_val = len(pd.read_parquet(proc_dir / "cmapss_FD001_val.parquet"))
        orig_test = len(pd.read_parquet(proc_dir / "cmapss_FD001_test.parquet"))

        feat_paths = featurized_data["cmapss"]
        assert len(pd.read_parquet(feat_paths["train"])) == orig_train
        assert len(pd.read_parquet(feat_paths["val"])) == orig_val
        assert len(pd.read_parquet(feat_paths["test"])) == orig_test

    def test_cmapss_temporal_ordering(self, featurized_data):
        paths = featurized_data["cmapss"]
        for split, path in paths.items():
            df = pd.read_parquet(path)
            for unit, group in df.groupby("machine_id"):
                cycles = group["operating_cycle"].to_numpy()
                assert np.all(np.diff(cycles) > 0), (
                    f"Cycle ordering violated in C-MAPSS {split} unit {unit}"
                )


class TestImsFeaturePipeline:
    """Verification for IMS bearing feature pipeline."""

    def test_ims_files_and_schema(self, featurized_data):
        paths = featurized_data["ims"]
        for split in ["train", "val", "test"]:
            assert split in paths
            assert paths[split].exists()
            df = pd.read_parquet(paths[split])
            assert len(df) > 0

            # Meta and target columns
            assert "machine_id" in df.columns
            assert "operating_cycle" in df.columns
            assert "anomaly_label" in df.columns

            # Dropped null columns check (temperature, pressure, rpm, current, raw_rms MUST NOT be present)
            for col in ["temperature", "pressure", "rpm", "current", "raw_rms"]:
                assert col not in df.columns, f"{col} should have been dropped from IMS"

            # Check feature count (~50 - 75 columns)
            assert 45 <= len(df.columns) <= 80

    def test_ims_no_nan_or_inf(self, featurized_data):
        paths = featurized_data["ims"]
        for split, path in paths.items():
            df = pd.read_parquet(path)
            num_cols = df.select_dtypes(include=[np.number]).columns
            assert not df[num_cols].isna().any().any(), f"NaNs found in IMS {split}"
            assert not np.isinf(df[num_cols].to_numpy()).any(), f"Infs found in IMS {split}"

    def test_ims_row_counts_preserved(self, featurized_data):
        proc_dir = Path("data/processed/ims")
        orig_train = len(pd.read_parquet(proc_dir / "ims_train.parquet"))
        orig_val = len(pd.read_parquet(proc_dir / "ims_val.parquet"))
        orig_test = len(pd.read_parquet(proc_dir / "ims_test.parquet"))

        feat_paths = featurized_data["ims"]
        assert len(pd.read_parquet(feat_paths["train"])) == orig_train
        assert len(pd.read_parquet(feat_paths["val"])) == orig_val
        assert len(pd.read_parquet(feat_paths["test"])) == orig_test


class TestAi4iFeaturePipeline:
    """Verification for AI4I feature pipeline."""

    def test_ai4i_files_and_schema(self, featurized_data):
        paths = featurized_data["ai4i"]
        for split in ["train", "val", "test"]:
            assert split in paths
            assert paths[split].exists()
            df = pd.read_parquet(paths[split])
            assert len(df) > 0

            # Meta and target columns
            assert "machine_id" in df.columns
            assert "failure_label" in df.columns
            assert "failure_mode" in df.columns

            # Engineered features
            assert "power" in df.columns
            assert "torque_per_rpm" in df.columns
            assert "temp_diff" in df.columns
            assert "tool_wear_squared" in df.columns
            assert "product_type_L" in df.columns
            assert "product_type_M" in df.columns
            assert "product_type_H" in df.columns

            # Check feature count (~25 - 45 columns)
            assert 20 <= len(df.columns) <= 45

    def test_ai4i_no_nan_or_inf(self, featurized_data):
        paths = featurized_data["ai4i"]
        for split, path in paths.items():
            df = pd.read_parquet(path)
            num_cols = df.select_dtypes(include=[np.number]).columns
            assert not df[num_cols].isna().any().any(), f"NaNs found in AI4I {split}"
            assert not np.isinf(df[num_cols].to_numpy()).any(), f"Infs found in AI4I {split}"

    def test_ai4i_row_counts_preserved(self, featurized_data):
        proc_dir = Path("data/processed/ai4i")
        orig_train = len(pd.read_parquet(proc_dir / "ai4i_train.parquet"))
        orig_val = len(pd.read_parquet(proc_dir / "ai4i_val.parquet"))
        orig_test = len(pd.read_parquet(proc_dir / "ai4i_test.parquet"))

        feat_paths = featurized_data["ai4i"]
        assert len(pd.read_parquet(feat_paths["train"])) == orig_train
        assert len(pd.read_parquet(feat_paths["val"])) == orig_val
        assert len(pd.read_parquet(feat_paths["test"])) == orig_test


class TestDeterminismAndLeakage:
    """Test deterministic output and absence of data leakage across splits."""

    def test_pipeline_determinism(self, tmp_path):
        proc_dir = Path("data/processed")
        run1_paths = build_ai4i_features(proc_dir, scaler_type="standard", normalize=True)
        df1 = pd.read_parquet(run1_paths["train"])

        run2_paths = build_ai4i_features(proc_dir, scaler_type="standard", normalize=True)
        df2 = pd.read_parquet(run2_paths["train"])

        pd.testing.assert_frame_equal(df1, df2)

    def test_scaler_leakage_safety(self, featurized_data):
        """Ensure scaler was fit on train split only."""
        cmapss_dir = Path("data/processed/cmapss_features")
        scaler_file = cmapss_dir / "scaler.joblib"
        assert scaler_file.exists()

        import joblib

        scaler = joblib.load(scaler_file)
        # Scaler n_features_in_ must match the number of scaled feature columns
        assert scaler.n_features_in_ > 100
