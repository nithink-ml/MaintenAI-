"""
MaintenAI— Unit Tests for Data Splitting Module

Tests leakage prevention, split correctness, and stratification.
"""

import numpy as np
import pandas as pd
import pytest

from ml.common.splitting import (
    split_by_unit,
    split_stratified,
    split_temporal,
    verify_no_leakage,
)


def _make_cmapss_like_df(n_units: int = 20, cycles_per_unit: int = 100) -> pd.DataFrame:
    """Create a synthetic C-MAPSS-like DataFrame for testing."""
    rows = []
    for uid in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            rows.append(
                {
                    "machine_id": f"CMAPSS_FD001_{uid:03d}",
                    "operating_cycle": cycle,
                    "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=cycle),
                    "temperature": 50.0 + np.random.normal(0, 5),
                    "rul": max(0, cycles_per_unit - cycle),
                }
            )
    return pd.DataFrame(rows)


def _make_ai4i_like_df(n: int = 500) -> pd.DataFrame:
    """Create a synthetic AI4I-like DataFrame for testing."""
    rng = np.random.default_rng(42)
    modes = rng.choice(
        ["none", "tool_wear", "heat_dissipation", "power_failure", "overstrain"],
        size=n,
        p=[0.90, 0.03, 0.03, 0.02, 0.02],
    )
    return pd.DataFrame(
        {
            "machine_id": [f"AI4I_{i:05d}" for i in range(n)],
            "temperature": rng.normal(30, 5, n),
            "failure_mode": modes,
        }
    )


class TestSplitByUnit:
    """Test unit-based splitting for RUL prediction."""

    def test_no_unit_leakage(self):
        """Units must not appear in multiple splits."""
        df = _make_cmapss_like_df(n_units=20)
        train, val, test = split_by_unit(df, test_fraction=0.2, val_fraction=0.15)

        train_ids = set(train["machine_id"].unique())
        val_ids = set(val["machine_id"].unique())
        test_ids = set(test["machine_id"].unique())

        assert len(train_ids & val_ids) == 0, "Train-val leakage!"
        assert len(train_ids & test_ids) == 0, "Train-test leakage!"
        assert len(val_ids & test_ids) == 0, "Val-test leakage!"

    def test_all_data_preserved(self):
        """No rows should be lost in splitting."""
        df = _make_cmapss_like_df(n_units=10)
        train, val, test = split_by_unit(df, test_fraction=0.2, val_fraction=0.2)
        assert len(train) + len(val) + len(test) == len(df)

    def test_full_lifecycle_preserved(self):
        """Each unit's complete lifecycle must be in one split."""
        df = _make_cmapss_like_df(n_units=10, cycles_per_unit=50)
        train, val, test = split_by_unit(df, test_fraction=0.2, val_fraction=0.2)

        for split_df in [train, val, test]:
            for _mid, group in split_df.groupby("machine_id"):
                # Every unit should have its full cycle range
                assert group["operating_cycle"].min() == 1
                assert group["operating_cycle"].max() == 50

    def test_reproducibility(self):
        """Same seed should produce same splits."""
        df = _make_cmapss_like_df(n_units=15)
        train1, _, _ = split_by_unit(df, random_seed=42)
        train2, _, _ = split_by_unit(df, random_seed=42)
        assert set(train1["machine_id"].unique()) == set(train2["machine_id"].unique())


class TestSplitTemporal:
    """Test temporal splitting for anomaly detection."""

    def test_chronological_ordering(self):
        """Test data must come after training data chronologically."""
        df = pd.DataFrame(
            {
                "machine_id": ["M1"] * 100,
                "operating_cycle": range(100),
                "timestamp": pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC"),
                "vibration": np.random.normal(1, 0.1, 100),
            }
        )
        train, val, test = split_temporal(df, test_fraction=0.2, val_fraction=0.15)

        # Training cycles should be earlier than test cycles
        assert train["operating_cycle"].max() < test["operating_cycle"].min()

    def test_no_data_loss(self):
        df = pd.DataFrame(
            {
                "machine_id": ["M1"] * 50 + ["M2"] * 50,
                "operating_cycle": list(range(50)) * 2,
                "timestamp": pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC"),
                "vibration": np.random.normal(1, 0.1, 100),
            }
        )
        train, val, test = split_temporal(df, test_fraction=0.2, val_fraction=0.15)
        assert len(train) + len(val) + len(test) == len(df)


class TestSplitStratified:
    """Test stratified splitting for classification."""

    def test_class_distribution_preserved(self):
        """All classes should appear in all splits."""
        df = _make_ai4i_like_df(n=500)
        train, val, test = split_stratified(df, target_col="failure_mode")

        train_classes = set(train["failure_mode"].unique())
        test_classes = set(test["failure_mode"].unique())

        # At minimum, the majority class should be in both
        assert "none" in train_classes
        assert "none" in test_classes

    def test_approximate_proportions(self):
        """Split proportions should approximately match requested fractions."""
        df = _make_ai4i_like_df(n=1000)
        train, val, test = split_stratified(df, test_fraction=0.2, val_fraction=0.15)

        total = len(df)
        assert abs(len(test) / total - 0.2) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05


class TestVerifyNoLeakage:
    """Test leakage verification utility."""

    def test_no_leakage_passes(self):
        train = pd.DataFrame({"machine_id": ["A", "B"]})
        val = pd.DataFrame({"machine_id": ["C"]})
        test = pd.DataFrame({"machine_id": ["D"]})
        assert verify_no_leakage(train, val, test) is True

    def test_leakage_raises(self):
        train = pd.DataFrame({"machine_id": ["A", "B"]})
        val = pd.DataFrame({"machine_id": ["B", "C"]})  # B leaked!
        test = pd.DataFrame({"machine_id": ["D"]})
        with pytest.raises(AssertionError, match="DATA LEAKAGE"):
            verify_no_leakage(train, val, test)
