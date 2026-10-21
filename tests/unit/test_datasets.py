"""
MaintenAI— Unit Tests for Dataset Processors (C-MAPSS, AI4I, IMS)

Tests schema mapping, RUL target construction, and data transformations.
Uses synthetic data to avoid external download dependencies in CI.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Create synthetic datasets in a temp directory for testing."""
    from ml.common.data_download import create_synthetic_fallback

    for name in ["cmapss", "ims", "ai4i"]:
        create_synthetic_fallback(name, tmp_path)
    return tmp_path


class TestCMAPSSProcessor:
    """Test C-MAPSS dataset processing."""

    def test_load_raw(self, synthetic_data_dir):
        from ml.common.datasets.cmapss import load_raw_cmapss

        train, test, rul = load_raw_cmapss(synthetic_data_dir / "cmapss")
        assert len(train) > 0
        assert len(test) > 0
        assert len(rul) > 0
        assert "unit_id" in train.columns
        assert "cycle" in train.columns

    def test_rul_label_construction(self, synthetic_data_dir):
        from ml.common.datasets.cmapss import add_rul_labels, load_raw_cmapss

        train, _, _ = load_raw_cmapss(synthetic_data_dir / "cmapss")
        labeled = add_rul_labels(train, max_rul_clip=125)

        assert "rul" in labeled.columns
        # RUL at last cycle of each unit should be 0
        for uid, group in labeled.groupby("unit_id"):
            last_row = group.loc[group["cycle"].idxmax()]
            assert last_row["rul"] == 0, f"Unit {uid} last cycle RUL should be 0"

    def test_rul_clipping(self, synthetic_data_dir):
        from ml.common.datasets.cmapss import add_rul_labels, load_raw_cmapss

        train, _, _ = load_raw_cmapss(synthetic_data_dir / "cmapss")
        labeled = add_rul_labels(train, max_rul_clip=50)

        assert labeled["rul"].max() <= 50, "RUL should be clipped to max_rul_clip"

    def test_canonical_mapping(self, synthetic_data_dir):
        from ml.common.datasets.cmapss import load_raw_cmapss, map_to_canonical

        train, _, _ = load_raw_cmapss(synthetic_data_dir / "cmapss")
        canonical = map_to_canonical(train)

        assert "machine_id" in canonical.columns
        assert "timestamp" in canonical.columns
        assert "temperature" in canonical.columns
        assert "operating_cycle" in canonical.columns

        # Machine IDs should be prefixed with dataset and split
        assert all(mid.startswith("CMAPSS_FD001_train_") for mid in canonical["machine_id"])

    def test_full_processing(self, synthetic_data_dir):
        from ml.common.datasets.cmapss import process_cmapss

        output_dir = synthetic_data_dir / "processed" / "cmapss"
        paths = process_cmapss(
            synthetic_data_dir / "cmapss",
            output_dir,
        )

        assert "train" in paths
        assert "test" in paths
        assert paths["train"].exists()

        train_df = pd.read_parquet(paths["train"])
        assert len(train_df) > 0
        assert "rul" in train_df.columns


class TestAI4IProcessor:
    """Test AI4I dataset processing."""

    def test_load_raw(self, synthetic_data_dir):
        from ml.common.datasets.ai4i import load_raw_ai4i

        df = load_raw_ai4i(synthetic_data_dir / "ai4i")
        assert len(df) > 0

    def test_canonical_mapping(self, synthetic_data_dir):
        from ml.common.datasets.ai4i import load_raw_ai4i, map_to_canonical

        raw = load_raw_ai4i(synthetic_data_dir / "ai4i")
        canonical = map_to_canonical(raw)

        assert "machine_id" in canonical.columns
        assert "failure_mode" in canonical.columns
        assert "failure_label" in canonical.columns

    def test_failure_mode_assignment(self, synthetic_data_dir):
        from ml.common.datasets.ai4i import load_raw_ai4i, map_to_canonical

        raw = load_raw_ai4i(synthetic_data_dir / "ai4i")
        canonical = map_to_canonical(raw)

        # Valid failure modes
        valid_modes = {
            "none",
            "tool_wear",
            "heat_dissipation",
            "power_failure",
            "overstrain",
            "random",
        }
        assert set(canonical["failure_mode"].unique()).issubset(valid_modes)

    def test_temperature_conversion(self, synthetic_data_dir):
        from ml.common.datasets.ai4i import load_raw_ai4i, map_to_canonical

        raw = load_raw_ai4i(synthetic_data_dir / "ai4i")
        canonical = map_to_canonical(raw)

        if "temperature" in canonical.columns:
            # AI4I temperatures are ~300K = ~27°C
            # After conversion, should be reasonable Celsius
            assert canonical["temperature"].mean() < 100, (
                "Temperature should be in Celsius, not Kelvin"
            )


class TestIMSProcessor:
    """Test IMS bearing dataset processing."""

    def test_vibration_feature_computation(self):
        from ml.common.datasets.ims import _compute_vibration_features

        signal = np.sin(np.linspace(0, 10 * np.pi, 2048))
        features = _compute_vibration_features(signal)

        assert "rms" in features
        assert "kurtosis" in features
        assert "spectral_energy" in features
        assert features["rms"] > 0
        assert features["peak"] > 0

    def test_empty_signal(self):
        from ml.common.datasets.ims import _compute_vibration_features

        features = _compute_vibration_features(np.array([]))
        assert features == {}

    def test_full_processing(self, synthetic_data_dir):
        from ml.common.datasets.ims import process_ims

        output_dir = synthetic_data_dir / "processed" / "ims"
        paths = process_ims(synthetic_data_dir / "ims", output_dir, max_files=10)

        assert "canonical" in paths
        assert paths["canonical"].exists()

        df = pd.read_parquet(paths["canonical"])
        assert "machine_id" in df.columns
        assert "vibration" in df.columns

    def test_anomaly_labels(self, synthetic_data_dir):
        from ml.common.datasets.ims import process_ims

        output_dir = synthetic_data_dir / "processed" / "ims"
        paths = process_ims(synthetic_data_dir / "ims", output_dir, max_files=50)

        df = pd.read_parquet(paths["canonical"])
        assert "anomaly_label" in df.columns
        assert set(df["anomaly_label"].unique()).issubset({0, 1})
