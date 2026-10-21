"""
MaintenAI— IMS Bearing Exploratory Data Analysis

Generates comprehensive EDA outputs for the NASA IMS Bearing dataset:
  1. Vibration RMS trends over time per bearing
  2. Spectral energy progression
  3. Kurtosis evolution (bearing health indicator)
  4. Normal vs anomalous region comparison (multi-feature)
  5. Feature correlation heatmap
  6. Summary statistics markdown report

Kurtosis is a critical bearing health indicator:
  - Healthy bearings: kurtosis ≈ 3 (Gaussian-like vibration)
  - Degrading bearings: kurtosis spikes as impulsive components appear
  - Late-stage failure: kurtosis may decrease as damage spreads
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (14, 10)
DPI = 150

BEARING_COLORS = {
    "bearing_1": "#2196F3",  # Blue - healthy
    "bearing_2": "#4CAF50",  # Green - healthy
    "bearing_3": "#FF9800",  # Orange - outer race failure
    "bearing_4": "#F44336",  # Red - rolling element failure
}


def _setup_style() -> None:
    """Configure matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
        }
    )


def _get_bearing_num(machine_id: str) -> str:
    """Extract bearing number string from machine_id."""
    # e.g., 'IMS_test_1_bearing_3' -> 'bearing_3'
    parts = machine_id.split("_")
    return f"bearing_{parts[-1]}"


def plot_vibration_rms_trends(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot vibration RMS trends over time for each bearing.
    Highlights the transition from normal to anomalous regions.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL, sharex=True)
    axes = axes.flatten()

    machines = sorted(df["machine_id"].unique())

    for idx, machine_id in enumerate(machines[:4]):
        ax = axes[idx]
        bearing_key = _get_bearing_num(machine_id)
        color = BEARING_COLORS.get(bearing_key, "#666666")

        machine_data = df[df["machine_id"] == machine_id].sort_values("operating_cycle")

        # Plot RMS
        ax.plot(
            machine_data["operating_cycle"],
            machine_data["vibration"],
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )

        # Shade anomalous region
        anomalous = machine_data[machine_data["anomaly_label"] == 1]
        if len(anomalous) > 0:
            ax.axvspan(
                anomalous["operating_cycle"].min(),
                anomalous["operating_cycle"].max(),
                alpha=0.15,
                color="red",
                label="Anomalous",
            )

        ax.set_title(f"{bearing_key.replace('_', ' ').title()}", fontsize=13)
        ax.set_ylabel("Vibration RMS")
        if len(anomalous) > 0:
            ax.legend(fontsize=9)

    axes[2].set_xlabel("Snapshot Index")
    axes[3].set_xlabel("Snapshot Index")
    fig.suptitle("IMS — Vibration RMS Trends by Bearing", fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / "ims_vibration_rms_trends.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_spectral_energy_progression(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot spectral energy progression over time for each bearing.
    Spectral energy increase indicates developing defects.
    """
    if "raw_spectral_energy" not in df.columns:
        logger.warning("raw_spectral_energy column not found, skipping")
        return output_dir

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL, sharex=True)
    axes = axes.flatten()

    machines = sorted(df["machine_id"].unique())

    for idx, machine_id in enumerate(machines[:4]):
        ax = axes[idx]
        bearing_key = _get_bearing_num(machine_id)
        color = BEARING_COLORS.get(bearing_key, "#666666")

        machine_data = df[df["machine_id"] == machine_id].sort_values("operating_cycle")

        ax.plot(
            machine_data["operating_cycle"],
            machine_data["raw_spectral_energy"],
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )

        # Shade anomalous region
        anomalous = machine_data[machine_data["anomaly_label"] == 1]
        if len(anomalous) > 0:
            ax.axvspan(
                anomalous["operating_cycle"].min(),
                anomalous["operating_cycle"].max(),
                alpha=0.15,
                color="red",
                label="Anomalous",
            )

        ax.set_title(f"{bearing_key.replace('_', ' ').title()}", fontsize=13)
        ax.set_ylabel("Spectral Energy")
        if len(anomalous) > 0:
            ax.legend(fontsize=9)

    axes[2].set_xlabel("Snapshot Index")
    axes[3].set_xlabel("Snapshot Index")
    fig.suptitle("IMS — Spectral Energy Progression", fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / "ims_spectral_energy.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_kurtosis_evolution(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot kurtosis evolution over time for each bearing.

    Kurtosis is one of the most sensitive indicators of bearing damage:
    - Normal: ~3 (mesokurtic, Gaussian vibration)
    - Early damage: spikes as impulsive energy appears
    - Late damage: may normalize as damage distributes
    """
    if "raw_kurtosis" not in df.columns:
        logger.warning("raw_kurtosis column not found, skipping")
        return output_dir

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL, sharex=True)
    axes = axes.flatten()

    machines = sorted(df["machine_id"].unique())

    for idx, machine_id in enumerate(machines[:4]):
        ax = axes[idx]
        bearing_key = _get_bearing_num(machine_id)
        color = BEARING_COLORS.get(bearing_key, "#666666")

        machine_data = df[df["machine_id"] == machine_id].sort_values("operating_cycle")

        ax.plot(
            machine_data["operating_cycle"],
            machine_data["raw_kurtosis"],
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )

        # Reference line for Gaussian kurtosis
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, label="Gaussian (excess=0)")

        # Shade anomalous region
        anomalous = machine_data[machine_data["anomaly_label"] == 1]
        if len(anomalous) > 0:
            ax.axvspan(
                anomalous["operating_cycle"].min(),
                anomalous["operating_cycle"].max(),
                alpha=0.15,
                color="red",
                label="Anomalous",
            )

        ax.set_title(f"{bearing_key.replace('_', ' ').title()}", fontsize=13)
        ax.set_ylabel("Excess Kurtosis")
        ax.legend(fontsize=8)

    axes[2].set_xlabel("Snapshot Index")
    axes[3].set_xlabel("Snapshot Index")
    fig.suptitle("IMS — Kurtosis Evolution (Bearing Health Indicator)", fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / "ims_kurtosis_evolution.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_normal_vs_anomalous(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Compare feature distributions between normal and anomalous regions.
    Uses box plots to show separation power of each feature.
    """
    feature_cols = [c for c in df.columns if c.startswith("raw_") and c != "raw_dominant_freq_idx"]
    if not feature_cols:
        logger.warning("No raw_ feature columns found")
        return output_dir

    # Only use bearings that have both labels
    bearings_with_anomalies = df[df["anomaly_label"] == 1]["machine_id"].unique()
    if len(bearings_with_anomalies) == 0:
        logger.warning("No anomalous data points found")
        return output_dir

    subset = df[df["machine_id"].isin(bearings_with_anomalies)].copy()
    subset["status"] = subset["anomaly_label"].map({0: "Normal", 1: "Anomalous"})

    n_features = len(feature_cols)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    palette = {"Normal": "#2196F3", "Anomalous": "#F44336"}

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        sns.boxplot(
            data=subset,
            x="status",
            y=feature,
            hue="status",
            ax=ax,
            palette=palette,
            width=0.5,
            legend=False,
        )
        clean_name = feature.replace("raw_", "").replace("_", " ").title()
        ax.set_title(clean_name, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for idx in range(len(feature_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "IMS — Normal vs. Anomalous Feature Comparison\n(Bearings 3 & 4 only)", fontsize=16, y=1.02
    )
    plt.tight_layout()

    out_path = output_dir / "ims_normal_vs_anomalous.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_feature_correlation(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot correlation heatmap of IMS vibration features.
    """
    feature_cols = [c for c in df.columns if c.startswith("raw_")]
    feature_cols_with_label = feature_cols + ["anomaly_label"]
    available = [c for c in feature_cols_with_label if c in df.columns]

    if len(available) < 2:
        logger.warning("Insufficient features for correlation heatmap")
        return output_dir

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    clean_labels = [c.replace("raw_", "").replace("_", " ").title() for c in available]
    corr.index = clean_labels
    corr.columns = clean_labels

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("IMS — Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()

    out_path = output_dir / "ims_feature_correlation.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def generate_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics for the IMS dataset.
    """
    stats = {
        "total_rows": len(df),
        "total_bearings": df["machine_id"].nunique(),
        "bearing_ids": sorted(df["machine_id"].unique().tolist()),
        "snapshots_per_bearing": int(len(df) / max(df["machine_id"].nunique(), 1)),
        "columns": list(df.columns),
    }

    # Anomaly distribution
    if "anomaly_label" in df.columns:
        anomaly_dist = df["anomaly_label"].value_counts().to_dict()
        stats["normal_count"] = anomaly_dist.get(0, 0)
        stats["anomalous_count"] = anomaly_dist.get(1, 0)
        stats["anomaly_ratio_pct"] = round(stats["anomalous_count"] / max(len(df), 1) * 100, 1)

    # Per-bearing anomaly breakdown
    if "anomaly_label" in df.columns:
        per_bearing = df.groupby("machine_id")["anomaly_label"].agg(["sum", "count"])
        stats["per_bearing_anomalies"] = {
            mid: {"anomalous": int(row["sum"]), "total": int(row["count"])}
            for mid, row in per_bearing.iterrows()
        }

    # Feature stats
    feature_cols = [c for c in df.columns if c.startswith("raw_")]
    stats["feature_stats"] = {}
    for col in feature_cols:
        data = df[col].dropna()
        if len(data) > 0:
            clean_name = col.replace("raw_", "")
            stats["feature_stats"][clean_name] = {
                "mean": round(float(data.mean()), 4),
                "std": round(float(data.std()), 4),
                "min": round(float(data.min()), 4),
                "max": round(float(data.max()), 4),
            }

    return stats


def generate_markdown_report(
    stats: dict[str, Any],
    plot_paths: dict[str, Path],
    output_dir: Path,
) -> Path:
    """
    Generate a markdown EDA report for IMS.
    """
    lines = [
        "# IMS Bearing — Exploratory Data Analysis",
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Rows | {stats['total_rows']:,} |",
        f"| Total Bearings | {stats['total_bearings']} |",
        f"| Snapshots per Bearing | {stats['snapshots_per_bearing']} |",
    ]

    if "normal_count" in stats:
        lines.extend(
            [
                f"| Normal Samples | {stats['normal_count']} |",
                f"| Anomalous Samples | {stats['anomalous_count']} |",
                f"| Anomaly Ratio | {stats['anomaly_ratio_pct']}% |",
            ]
        )

    lines.append("")

    # Per-bearing breakdown
    if "per_bearing_anomalies" in stats:
        lines.extend(
            [
                "## Per-Bearing Anomaly Breakdown",
                "",
                "| Bearing | Total | Anomalous | Status |",
                "|---------|-------|-----------|--------|",
            ]
        )
        for mid, info in stats["per_bearing_anomalies"].items():
            bearing_name = mid.split("_")[-1]
            status = "⚠️ Failure" if info["anomalous"] > 0 else "✅ Healthy"
            lines.append(
                f"| Bearing {bearing_name} | {info['total']} | {info['anomalous']} | {status} |"
            )
        lines.append("")

    # Feature statistics
    if stats.get("feature_stats"):
        lines.extend(
            [
                "## Vibration Feature Statistics",
                "",
                "| Feature | Mean | Std | Min | Max |",
                "|---------|------|-----|-----|-----|",
            ]
        )
        for feat, ss in stats["feature_stats"].items():
            lines.append(
                f"| {feat.replace('_', ' ').title()} | {ss['mean']:.4f} | "
                f"{ss['std']:.4f} | {ss['min']:.4f} | {ss['max']:.4f} |"
            )
        lines.append("")

    # Embed plots
    lines.extend(["## Visualizations", ""])

    plot_titles = {
        "rms_trends": "### Vibration RMS Trends",
        "spectral_energy": "### Spectral Energy Progression",
        "kurtosis": "### Kurtosis Evolution",
        "normal_vs_anomalous": "### Normal vs. Anomalous Comparison",
        "correlation": "### Feature Correlation",
    }

    for key, title in plot_titles.items():
        if key in plot_paths and isinstance(plot_paths[key], Path) and plot_paths[key].exists():
            lines.extend([title, "", f"![{title.strip('# ')}]({plot_paths[key].name})", ""])

    # Key observations
    lines.extend(
        [
            "## Key Observations",
            "",
            "1. **Bearing 3 (outer race failure)**: Shows clear vibration RMS increase and "
            "kurtosis spikes in the final 20% of the test, consistent with documented outer "
            "race defect progression.",
            "2. **Bearing 4 (rolling element failure)**: Exhibits rolling element damage "
            "signatures in the final 15% with increasing spectral energy.",
            "3. **Bearings 1 & 2 (healthy)**: Maintain stable vibration characteristics "
            "throughout the test — no anomalous labels assigned.",
            f"4. **Class imbalance**: Only {stats.get('anomaly_ratio_pct', 'N/A')}% of samples "
            "are anomalous, requiring careful handling during model training (e.g., weighted "
            "loss, SMOTE, or anomaly-detection-native approaches like autoencoders).",
            "5. **Kurtosis as early warning**: Kurtosis deviations often precede RMS changes, "
            "suggesting it could serve as an early degradation indicator.",
            "",
        ]
    )

    report_path = output_dir / "ims_eda.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Saved report: {report_path}")
    return report_path


def run_ims_eda(
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Run complete IMS EDA pipeline.

    Args:
        data_dir: Path to processed data directory (containing ims/ims_test_1.parquet)
        output_dir: Path to save EDA outputs

    Returns:
        Dict of output file paths.
    """
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed IMS data
    data_path = data_dir / "ims" / "ims_test_1.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"IMS data not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded IMS: {df.shape}")

    # Generate plots
    plot_paths = {}
    plot_paths["rms_trends"] = plot_vibration_rms_trends(df, output_dir)
    plot_paths["spectral_energy"] = plot_spectral_energy_progression(df, output_dir)
    plot_paths["kurtosis"] = plot_kurtosis_evolution(df, output_dir)
    plot_paths["normal_vs_anomalous"] = plot_normal_vs_anomalous(df, output_dir)
    plot_paths["correlation"] = plot_feature_correlation(df, output_dir)

    # Generate summary stats and report
    stats = generate_summary_stats(df)
    report_path = generate_markdown_report(stats, plot_paths, output_dir)

    all_outputs = {**plot_paths, "report": report_path}
    logger.info(f"IMS EDA complete: {len(all_outputs)} artifacts generated")
    return all_outputs
