"""
MaintenAI— Unit Tests for EDA Modules

Tests the EDA generation scripts for all three datasets:
  - C-MAPSS EDA (degradation curves, sensor distributions, correlation, RUL, lifecycle)
  - IMS EDA (vibration trends, spectral energy, kurtosis, normal vs anomalous)
  - AI4I EDA (failure distribution, feature comparison, class imbalance)
  - Runner (orchestration, index generation)

Tests use synthetic DataFrames matching the processed data schemas to avoid
dependency on real downloaded datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────
# Synthetic Data Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for EDA artifacts."""
    out = tmp_path / "eda_output"
    out.mkdir()
    return out


@pytest.fixture
def cmapss_df() -> pd.DataFrame:
    """
    Synthetic C-MAPSS-like DataFrame matching canonical schema.
    2 units, 50 cycles each = 100 rows.
    """
    rng = np.random.default_rng(42)
    records = []
    for unit_num in range(1, 3):
        max_cycle = 50
        for cycle in range(1, max_cycle + 1):
            rul = min(max_cycle - cycle, 125)
            records.append(
                {
                    "machine_id": f"CMAPSS_FD001_train_{unit_num:03d}",
                    "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=cycle),
                    "temperature": 644 + rng.normal(0, 3),
                    "pressure": 555 + rng.normal(0, 2),
                    "rpm": 46.5 + rng.normal(0, 0.7),
                    "vibration": 8.5 + rng.normal(0, 0.07),
                    "current": 1415 + rng.normal(0, 15),
                    "operating_cycle": cycle,
                    "rul": rul,
                    "raw_sensor_2": 644 + rng.normal(0, 3),
                    "raw_sensor_3": 1580 + rng.normal(0, 10),
                    "raw_sensor_4": 1415 + rng.normal(0, 15),
                    "raw_sensor_7": 555 + rng.normal(0, 2),
                    "raw_sensor_8": 2388 + rng.normal(0, 1),
                    "raw_sensor_9": 9020 + rng.normal(0, 20),
                    "raw_sensor_11": 46.5 + rng.normal(0, 0.7),
                    "raw_sensor_12": 525 + rng.normal(0, 3),
                    "raw_sensor_13": 2388 + rng.normal(0, 1),
                    "raw_sensor_14": 8120 + rng.normal(0, 15),
                    "raw_sensor_15": 8.5 + rng.normal(0, 0.07),
                    "raw_sensor_17": 394 + rng.normal(0, 2),
                    "raw_sensor_20": 38.5 + rng.normal(0, 0.4),
                    "raw_sensor_21": 23.4 + rng.normal(0, 0.1),
                    "op_setting_1": rng.normal(0, 0.01),
                    "op_setting_2": rng.normal(0, 0.0005),
                    "op_setting_3": 100 + rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def ims_df() -> pd.DataFrame:
    """
    Synthetic IMS-like DataFrame matching canonical schema.
    4 bearings, 50 snapshots each = 200 rows.
    """
    rng = np.random.default_rng(42)
    records = []
    for bearing in range(1, 5):
        for snap in range(50):
            # Bearings 3,4 get anomalous labels in last 20%/15%
            is_anomalous = 0
            if bearing == 3 and snap >= 40 or bearing == 4 and snap >= 43:
                is_anomalous = 1

            rms = 0.05 + rng.normal(0, 0.01) + (0.3 if is_anomalous else 0)
            records.append(
                {
                    "machine_id": f"IMS_test_1_bearing_{bearing}",
                    "timestamp": pd.Timestamp("2003-10-22", tz="UTC")
                    + pd.Timedelta(minutes=snap * 10),
                    "vibration": rms,
                    "operating_cycle": snap,
                    "temperature": None,
                    "pressure": None,
                    "rpm": None,
                    "current": None,
                    "raw_rms": rms,
                    "raw_peak": rms * 3 + rng.normal(0, 0.01),
                    "raw_peak_to_peak": rms * 5 + rng.normal(0, 0.02),
                    "raw_std": rms * 0.7 + rng.normal(0, 0.005),
                    "raw_kurtosis": 3.0 + rng.normal(0, 0.5) + (5.0 if is_anomalous else 0),
                    "raw_skewness": rng.normal(0, 0.3),
                    "raw_crest_factor": 3.5 + rng.normal(0, 0.2),
                    "raw_spectral_energy": 0.002
                    + rng.normal(0, 0.0005)
                    + (0.02 if is_anomalous else 0),
                    "raw_dominant_freq_idx": float(rng.integers(10, 100)),
                    "anomaly_label": is_anomalous,
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def ai4i_df() -> pd.DataFrame:
    """
    Synthetic AI4I-like DataFrame matching canonical schema.
    200 samples: ~175 normal, ~25 failures.
    """
    rng = np.random.default_rng(42)
    records = []
    failure_modes = (
        ["none"] * 175
        + ["tool_wear"] * 12
        + ["heat_dissipation"] * 5
        + ["power_failure"] * 4
        + ["overstrain"] * 3
        + ["random"] * 1
    )
    rng.shuffle(failure_modes)

    product_types = ["L"] * 100 + ["M"] * 60 + ["H"] * 40
    rng.shuffle(product_types)

    for i, (mode, ptype) in enumerate(zip(failure_modes, product_types, strict=False)):
        is_failure = 0 if mode == "none" else 1
        records.append(
            {
                "machine_id": f"AI4I_{i:05d}",
                "timestamp": pd.Timestamp("2020-06-01", tz="UTC") + pd.Timedelta(minutes=i * 5),
                "temperature": 25 + rng.normal(0, 3),
                "pressure": 35 + rng.normal(0, 2),
                "rpm": 1500 + rng.normal(0, 200),
                "vibration": 40 + rng.normal(0, 10),
                "operating_cycle": rng.integers(0, 250),
                "failure_label": is_failure,
                "failure_tool_wear": 1 if mode == "tool_wear" else 0,
                "failure_heat_dissipation": 1 if mode == "heat_dissipation" else 0,
                "failure_power_failure": 1 if mode == "power_failure" else 0,
                "failure_overstrain": 1 if mode == "overstrain" else 0,
                "failure_random": 1 if mode == "random" else 0,
                "failure_mode": mode,
                "product_type": ptype,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def cmapss_data_dir(tmp_path: Path, cmapss_df: pd.DataFrame) -> Path:
    """Create temp data directory with C-MAPSS parquet file."""
    data_dir = tmp_path / "data" / "cmapss"
    data_dir.mkdir(parents=True)
    cmapss_df.to_parquet(data_dir / "cmapss_FD001_train.parquet", index=False)
    return tmp_path / "data"


@pytest.fixture
def ims_data_dir(tmp_path: Path, ims_df: pd.DataFrame) -> Path:
    """Create temp data directory with IMS parquet file."""
    data_dir = tmp_path / "data" / "ims"
    data_dir.mkdir(parents=True)
    ims_df.to_parquet(data_dir / "ims_test_1.parquet", index=False)
    return tmp_path / "data"


@pytest.fixture
def ai4i_data_dir(tmp_path: Path, ai4i_df: pd.DataFrame) -> Path:
    """Create temp data directory with AI4I parquet file."""
    data_dir = tmp_path / "data" / "ai4i"
    data_dir.mkdir(parents=True)
    ai4i_df.to_parquet(data_dir / "ai4i_full.parquet", index=False)
    return tmp_path / "data"


@pytest.fixture
def all_data_dir(tmp_path: Path, cmapss_df, ims_df, ai4i_df) -> Path:
    """Create temp data directory with all three datasets."""
    data_dir = tmp_path / "all_data"

    for name, df, filename in [
        ("cmapss", cmapss_df, "cmapss_FD001_train.parquet"),
        ("ims", ims_df, "ims_test_1.parquet"),
        ("ai4i", ai4i_df, "ai4i_full.parquet"),
    ]:
        subdir = data_dir / name
        subdir.mkdir(parents=True)
        df.to_parquet(subdir / filename, index=False)

    return data_dir


# ──────────────────────────────────────────────────────────────
# C-MAPSS EDA Tests
# ──────────────────────────────────────────────────────────────


class TestCmapssEDA:
    """Tests for the C-MAPSS EDA module."""

    def test_plot_sensor_degradation_curves(self, cmapss_df, tmp_output_dir):
        from ml.common.eda.cmapss_eda import plot_sensor_degradation_curves

        path = plot_sensor_degradation_curves(cmapss_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_sensor_distributions(self, cmapss_df, tmp_output_dir):
        from ml.common.eda.cmapss_eda import plot_sensor_distributions

        path = plot_sensor_distributions(cmapss_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_plot_correlation_heatmap(self, cmapss_df, tmp_output_dir):
        from ml.common.eda.cmapss_eda import plot_correlation_heatmap

        path = plot_correlation_heatmap(cmapss_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_rul_distribution(self, cmapss_df, tmp_output_dir):
        from ml.common.eda.cmapss_eda import plot_rul_distribution

        path = plot_rul_distribution(cmapss_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_unit_lifecycle_lengths(self, cmapss_df, tmp_output_dir):
        from ml.common.eda.cmapss_eda import plot_unit_lifecycle_lengths

        path = plot_unit_lifecycle_lengths(cmapss_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_generate_summary_stats(self, cmapss_df):
        from ml.common.eda.cmapss_eda import generate_summary_stats

        stats = generate_summary_stats(cmapss_df)

        assert stats["total_rows"] == 100
        assert stats["total_units"] == 2
        assert "rul_min" in stats
        assert "rul_max" in stats
        assert "sensor_stats" in stats
        assert "temperature" in stats["sensor_stats"]
        assert "lifecycle_mean" in stats

    def test_generate_markdown_report(self, cmapss_df, tmp_output_dir):
        from ml.common.eda.cmapss_eda import (
            generate_markdown_report,
            generate_summary_stats,
            plot_sensor_degradation_curves,
        )

        stats = generate_summary_stats(cmapss_df)
        deg_path = plot_sensor_degradation_curves(cmapss_df, tmp_output_dir)
        plot_paths = {"degradation": deg_path}
        report = generate_markdown_report(stats, plot_paths, tmp_output_dir)

        assert report.exists()
        content = report.read_text()
        assert "# C-MAPSS (FD001)" in content
        assert "Dataset Overview" in content
        assert "Sensor Statistics" in content

    def test_run_cmapss_eda(self, cmapss_data_dir, tmp_output_dir):
        from ml.common.eda.cmapss_eda import run_cmapss_eda

        results = run_cmapss_eda(cmapss_data_dir, tmp_output_dir)

        assert len(results) >= 6  # 5 plots + 1 report
        assert "report" in results
        assert results["report"].exists()
        assert "degradation" in results
        assert "distributions" in results


# ──────────────────────────────────────────────────────────────
# IMS EDA Tests
# ──────────────────────────────────────────────────────────────


class TestImsEDA:
    """Tests for the IMS Bearing EDA module."""

    def test_plot_vibration_rms_trends(self, ims_df, tmp_output_dir):
        from ml.common.eda.ims_eda import plot_vibration_rms_trends

        path = plot_vibration_rms_trends(ims_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_spectral_energy_progression(self, ims_df, tmp_output_dir):
        from ml.common.eda.ims_eda import plot_spectral_energy_progression

        path = plot_spectral_energy_progression(ims_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_kurtosis_evolution(self, ims_df, tmp_output_dir):
        from ml.common.eda.ims_eda import plot_kurtosis_evolution

        path = plot_kurtosis_evolution(ims_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_normal_vs_anomalous(self, ims_df, tmp_output_dir):
        from ml.common.eda.ims_eda import plot_normal_vs_anomalous

        path = plot_normal_vs_anomalous(ims_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_feature_correlation(self, ims_df, tmp_output_dir):
        from ml.common.eda.ims_eda import plot_feature_correlation

        path = plot_feature_correlation(ims_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_generate_summary_stats(self, ims_df):
        from ml.common.eda.ims_eda import generate_summary_stats

        stats = generate_summary_stats(ims_df)

        assert stats["total_rows"] == 200
        assert stats["total_bearings"] == 4
        assert "normal_count" in stats
        assert "anomalous_count" in stats
        assert stats["anomalous_count"] > 0
        assert "per_bearing_anomalies" in stats

    def test_generate_markdown_report(self, ims_df, tmp_output_dir):
        from ml.common.eda.ims_eda import (
            generate_markdown_report,
            generate_summary_stats,
        )

        stats = generate_summary_stats(ims_df)
        report = generate_markdown_report(stats, {}, tmp_output_dir)

        assert report.exists()
        content = report.read_text()
        assert "# IMS Bearing" in content
        assert "Per-Bearing Anomaly Breakdown" in content

    def test_run_ims_eda(self, ims_data_dir, tmp_output_dir):
        from ml.common.eda.ims_eda import run_ims_eda

        results = run_ims_eda(ims_data_dir, tmp_output_dir)

        assert len(results) >= 6  # 5 plots + 1 report
        assert "report" in results


# ──────────────────────────────────────────────────────────────
# AI4I EDA Tests
# ──────────────────────────────────────────────────────────────


class TestAi4iEDA:
    """Tests for the AI4I 2020 EDA module."""

    def test_plot_failure_mode_distribution(self, ai4i_df, tmp_output_dir):
        from ml.common.eda.ai4i_eda import plot_failure_mode_distribution

        path = plot_failure_mode_distribution(ai4i_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_feature_distributions_by_type(self, ai4i_df, tmp_output_dir):
        from ml.common.eda.ai4i_eda import plot_feature_distributions_by_type

        path = plot_feature_distributions_by_type(ai4i_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_failure_mode_feature_comparison(self, ai4i_df, tmp_output_dir):
        from ml.common.eda.ai4i_eda import plot_failure_mode_feature_comparison

        path = plot_failure_mode_feature_comparison(ai4i_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_correlation_heatmap(self, ai4i_df, tmp_output_dir):
        from ml.common.eda.ai4i_eda import plot_correlation_heatmap

        path = plot_correlation_heatmap(ai4i_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_plot_class_imbalance_analysis(self, ai4i_df, tmp_output_dir):
        from ml.common.eda.ai4i_eda import plot_class_imbalance_analysis

        path = plot_class_imbalance_analysis(ai4i_df, tmp_output_dir)
        assert isinstance(path, Path)
        assert path.exists()

    def test_generate_summary_stats(self, ai4i_df):
        from ml.common.eda.ai4i_eda import generate_summary_stats

        stats = generate_summary_stats(ai4i_df)

        assert stats["total_rows"] == 200
        assert "failure_mode_dist" in stats
        assert "none" in stats["failure_mode_dist"]
        assert stats["failure_count"] > 0
        assert stats["imbalance_ratio"] > 1
        assert "sensor_stats" in stats

    def test_generate_summary_stats_multi_label(self, ai4i_df):
        from ml.common.eda.ai4i_eda import generate_summary_stats

        stats = generate_summary_stats(ai4i_df)
        # Should track multi-label samples (even if 0 in synthetic data)
        assert "multi_label_samples" in stats

    def test_generate_markdown_report(self, ai4i_df, tmp_output_dir):
        from ml.common.eda.ai4i_eda import (
            generate_markdown_report,
            generate_summary_stats,
        )

        stats = generate_summary_stats(ai4i_df)
        report = generate_markdown_report(stats, {}, tmp_output_dir)

        assert report.exists()
        content = report.read_text()
        assert "# AI4I 2020" in content
        assert "Failure Mode Distribution" in content
        assert "class imbalance" in content.lower()

    def test_run_ai4i_eda(self, ai4i_data_dir, tmp_output_dir):
        from ml.common.eda.ai4i_eda import run_ai4i_eda

        results = run_ai4i_eda(ai4i_data_dir, tmp_output_dir)

        assert len(results) >= 6  # 5 plots + 1 report
        assert "report" in results


# ──────────────────────────────────────────────────────────────
# Runner Tests
# ──────────────────────────────────────────────────────────────


class TestEDARunner:
    """Tests for the EDA runner orchestrator."""

    def test_run_all_eda(self, all_data_dir, tmp_output_dir):
        from ml.common.eda.runner import run_all_eda

        results = run_all_eda(all_data_dir, tmp_output_dir)

        assert "cmapss" in results
        assert "ims" in results
        assert "ai4i" in results
        # Each dataset should have produced artifacts
        for ds in ["cmapss", "ims", "ai4i"]:
            assert len(results[ds]) > 0

        # Index page should be generated
        index_path = tmp_output_dir / "README.md"
        assert index_path.exists()
        index_content = index_path.read_text()
        assert "C-MAPSS" in index_content or "cmapss" in index_content

    def test_run_single_dataset(self, all_data_dir, tmp_output_dir):
        from ml.common.eda.runner import run_all_eda

        results = run_all_eda(all_data_dir, tmp_output_dir, datasets=["cmapss"])

        assert "cmapss" in results
        assert len(results["cmapss"]) > 0
        # Other datasets should not be present
        assert "ims" not in results
        assert "ai4i" not in results

    def test_invalid_dataset_name(self, all_data_dir, tmp_output_dir):
        from ml.common.eda.runner import run_all_eda

        with pytest.raises(ValueError, match="Unknown dataset"):
            run_all_eda(all_data_dir, tmp_output_dir, datasets=["nonexistent"])

    def test_missing_data_graceful(self, tmp_path, tmp_output_dir):
        """Runner should handle missing data files gracefully."""
        from ml.common.eda.runner import run_all_eda

        empty_data = tmp_path / "empty_data"
        empty_data.mkdir()
        results = run_all_eda(empty_data, tmp_output_dir)

        # All datasets should have empty results (not crash)
        for ds in ["cmapss", "ims", "ai4i"]:
            assert len(results[ds]) == 0


# ──────────────────────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_cmapss_missing_rul(self, tmp_output_dir):
        """Should handle DataFrame without RUL column gracefully."""
        from ml.common.eda.cmapss_eda import plot_rul_distribution

        df = pd.DataFrame(
            {
                "machine_id": ["A", "B"],
                "temperature": [1.0, 2.0],
            }
        )
        plot_rul_distribution(df, tmp_output_dir)
        # Should return without error

    def test_ims_missing_spectral_energy(self, tmp_output_dir):
        """Should handle missing spectral energy column gracefully."""
        from ml.common.eda.ims_eda import plot_spectral_energy_progression

        df = pd.DataFrame(
            {
                "machine_id": ["IMS_test_1_bearing_1"] * 10,
                "vibration": np.random.randn(10),
                "operating_cycle": range(10),
                "anomaly_label": [0] * 10,
            }
        )
        plot_spectral_energy_progression(df, tmp_output_dir)
        # Should return without crash

    def test_ai4i_no_failures(self, tmp_output_dir):
        """Should handle dataset with no failures gracefully."""
        from ml.common.eda.ai4i_eda import generate_summary_stats

        df = pd.DataFrame(
            {
                "machine_id": [f"AI4I_{i:05d}" for i in range(10)],
                "failure_mode": ["none"] * 10,
                "failure_label": [0] * 10,
                "temperature": np.random.randn(10),
            }
        )
        stats = generate_summary_stats(df)
        assert stats["failure_count"] == 0
        assert stats["imbalance_ratio"] > 0
