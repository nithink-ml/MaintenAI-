"""
MaintenAI— Unit Tests for Feature Engineering Modules
"""

import numpy as np
import pandas as pd
import pytest

from ml.features.build_features import (
    _apply_scaler,
    _fit_and_save_scaler,
    _get_scaler,
)
from ml.features.registry import (
    get_feature_manifest,
    save_feature_manifest,
)
from ml.features.spectral import (
    compute_all_spectral_features,
    compute_spectral_ratios,
    compute_spectral_trends,
)
from ml.features.temporal import (
    compute_delta_features,
    compute_ema_features,
    compute_lag_features,
    compute_rolling_features,
)

# ============================================================
# Temporal Feature Extractor Tests
# ============================================================


class TestTemporalFeatures:
    """Test temporal feature transformations and grouping safety."""

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series for 2 machines with 10 steps each."""
        np.random.seed(42)
        n = 10
        m1 = pd.DataFrame(
            {
                "machine_id": ["M1"] * n,
                "operating_cycle": list(range(1, n + 1)),
                "sensor_a": np.arange(1.0, n + 1.0),
                "sensor_b": np.sin(np.linspace(0, 3.14, n)),
            }
        )
        m2 = pd.DataFrame(
            {
                "machine_id": ["M2"] * n,
                "operating_cycle": list(range(1, n + 1)),
                "sensor_a": np.arange(100.0, 100.0 + n),
                "sensor_b": np.cos(np.linspace(0, 3.14, n)),
            }
        )
        return pd.concat([m1, m2], ignore_index=True)

    def test_rolling_features_group_isolation(self, sample_time_series):
        """Verify rolling calculations never cross machine/bearing boundaries."""
        res = compute_rolling_features(
            sample_time_series,
            columns=["sensor_a"],
            window_sizes=[3],
            group_col="machine_id",
        )

        # For M2 at step 1 (index 10), sensor_a is 100.0.
        # Rolling mean (window 3, min_periods=1) should be 100.0, NOT influenced by M1 (which ended at 10.0).
        m2_first_mean = res.loc[res["machine_id"] == "M2", "sensor_a_roll3_mean"].iloc[0]
        assert m2_first_mean == 100.0

        # M1 last step mean should be (8+9+10)/3 = 9.0
        m1_last_mean = res.loc[res["machine_id"] == "M1", "sensor_a_roll3_mean"].iloc[-1]
        assert np.isclose(m1_last_mean, 9.0)

    def test_rolling_features_trailing_only(self):
        """Verify rolling features are strictly trailing (causal, no future leakage)."""
        df = pd.DataFrame(
            {
                "machine_id": ["M1"] * 5,
                "val": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        res = compute_rolling_features(df, ["val"], [3], group_col="machine_id")

        # Step 0: [10] -> mean 10.0
        assert res["val_roll3_mean"].iloc[0] == 10.0
        # Step 1: [10, 20] -> mean 15.0
        assert res["val_roll3_mean"].iloc[1] == 15.0
        # Step 2: [10, 20, 30] -> mean 20.0
        assert res["val_roll3_mean"].iloc[2] == 20.0
        # Step 3: [20, 30, 40] -> mean 30.0
        assert res["val_roll3_mean"].iloc[3] == 30.0

    def test_ema_features(self, sample_time_series):
        """Test exponential moving average calculation and causality."""
        res = compute_ema_features(
            sample_time_series,
            columns=["sensor_a"],
            spans=[5],
            group_col="machine_id",
        )
        assert "sensor_a_ema5" in res.columns
        assert not res["sensor_a_ema5"].isna().any()
        # M2 first row must be isolated
        assert res.loc[res["machine_id"] == "M2", "sensor_a_ema5"].iloc[0] == 100.0

    def test_lag_features_no_future_leakage(self, sample_time_series):
        """Test lag features provide past values only and respect group boundaries."""
        res = compute_lag_features(
            sample_time_series,
            columns=["sensor_a"],
            lag_steps=[1, 2],
            group_col="machine_id",
        )
        # For M1, step 2 (sensor_a=3) lag1 should be 2.0, lag2 should be 1.0
        m1_row2 = res[(res["machine_id"] == "M1") & (res["operating_cycle"] == 3)].iloc[0]
        assert m1_row2["sensor_a_lag1"] == 2.0
        assert m1_row2["sensor_a_lag2"] == 1.0

        # For M2, step 1 lag1 must NOT leak M1's last value
        m2_row0 = res[res["machine_id"] == "M2"].iloc[0]
        # Backfilled within group -> 100.0
        assert m2_row0["sensor_a_lag1"] == 100.0

    def test_delta_features(self, sample_time_series):
        """Test rate of change (first differences)."""
        res = compute_delta_features(
            sample_time_series,
            columns=["sensor_a"],
            periods=[1],
            group_col="machine_id",
        )
        assert "sensor_a_delta1" in res.columns
        assert "sensor_a_delta1_pct" in res.columns
        # For sensor_a (arithmetic progression step 1.0), delta1 is 1.0
        m1_vals = res.loc[res["machine_id"] == "M1", "sensor_a_delta1"]
        assert (m1_vals == 1.0).all()


# ============================================================
# Spectral Feature Extractor Tests
# ============================================================


class TestSpectralFeatures:
    """Test spectral ratio and trend computations."""

    @pytest.fixture
    def sample_vib_df(self):
        return pd.DataFrame(
            {
                "machine_id": ["bearing_1"] * 5,
                "vibration": [0.2, 0.4, 0.5, 0.8, 1.2],
                "raw_peak": [0.6, 1.2, 1.5, 2.4, 3.6],
                "raw_spectral_energy": [0.04, 0.16, 0.25, 0.64, 1.44],
            }
        )

    def test_spectral_ratios(self, sample_vib_df):
        res = compute_spectral_ratios(sample_vib_df)
        assert "spectral_to_rms_ratio" in res.columns
        assert "peak_to_energy_ratio" in res.columns
        assert not res["spectral_to_rms_ratio"].isna().any()
        assert not res["peak_to_energy_ratio"].isna().any()
        assert not np.isinf(res["spectral_to_rms_ratio"]).any()

    def test_spectral_trends(self, sample_vib_df):
        res = compute_spectral_trends(
            sample_vib_df,
            columns=["vibration"],
            window=3,
            group_col="machine_id",
        )
        assert "vibration_trend_ratio3" in res.columns
        assert not res["vibration_trend_ratio3"].isna().any()

    def test_compute_all_spectral_features(self, sample_vib_df):
        res = compute_all_spectral_features(sample_vib_df, group_col="machine_id")
        assert "spectral_to_rms_ratio" in res.columns
        assert "peak_to_energy_ratio" in res.columns
        assert "raw_spectral_energy_trend_ratio5" in res.columns


# ============================================================
# Scaler & Configuration Tests
# ============================================================


class TestScalerUtilities:
    """Test scaler factory and fit-then-transform integrity."""

    def test_scaler_factory(self):
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        assert isinstance(_get_scaler("standard"), StandardScaler)
        assert isinstance(_get_scaler("robust"), RobustScaler)
        assert isinstance(_get_scaler("minmax"), MinMaxScaler)

        with pytest.raises(ValueError, match="Unknown scaler type"):
            _get_scaler("invalid_scaler")

    def test_fit_and_save_scaler_train_only(self, tmp_path):
        train_df = pd.DataFrame({"feat1": [10.0, 20.0, 30.0], "feat2": [100.0, 200.0, 300.0]})
        test_df = pd.DataFrame({"feat1": [40.0, 50.0], "feat2": [400.0, 500.0]})

        scaled_train, scaler = _fit_and_save_scaler(
            train_df, feature_cols=["feat1", "feat2"], output_dir=tmp_path, scaler_type="standard"
        )
        scaled_test = _apply_scaler(test_df, scaler, feature_cols=["feat1", "feat2"])

        # Train mean should be 0 and std 1
        assert np.isclose(scaled_train["feat1"].mean(), 0.0, atol=1e-7)
        assert np.isclose(scaled_train["feat1"].std(ddof=0), 1.0, atol=1e-7)

        # Test set is scaled using TRAIN statistics (train mean=20, std=8.1649658)
        expected_test_feat1_0 = (40.0 - 20.0) / (np.std([10.0, 20.0, 30.0]))
        assert np.isclose(scaled_test["feat1"].iloc[0], expected_test_feat1_0)


# ============================================================
# Feature Registry & Manifest Tests
# ============================================================


class TestFeatureRegistry:
    """Test feature registry generation and serialization."""

    def test_manifest_generation(self):
        cmapss_manifest = get_feature_manifest("cmapss")
        assert len(cmapss_manifest) > 50
        for item in cmapss_manifest:
            assert "name" in item
            assert "reason" in item
            assert len(item["reason"]) > 0

        ims_manifest = get_feature_manifest("ims", window_sizes=[3, 5, 10], ema_spans=[5, 10])
        assert len(ims_manifest) > 30

        ai4i_manifest = get_feature_manifest("ai4i")
        assert len(ai4i_manifest) >= 10

    def test_save_feature_manifest(self, tmp_path):
        manifest_file = save_feature_manifest("cmapss", tmp_path)
        assert manifest_file.exists()
        import json

        data = json.loads(manifest_file.read_text())
        assert data["dataset"] == "cmapss"
        assert len(data["features"]) == data["n_features"]
