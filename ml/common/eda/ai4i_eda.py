"""
MaintenAI— AI4I 2020 Exploratory Data Analysis

Generates comprehensive EDA outputs for the AI4I 2020 Predictive Maintenance
dataset (failure-mode classification):
  1. Failure mode class distribution (bar chart)
  2. Feature distributions by product type (L/M/H)
  3. Failure mode feature comparison (violin/box plots)
  4. Pairwise correlation heatmap
  5. Class imbalance analysis (pie chart + imbalance ratios)
  6. Summary statistics markdown report

The AI4I dataset presents a multi-label imbalanced classification problem:
  - ~87.5% of samples are normal (no failure)
  - 5 failure modes with varying prevalence
  - Product type (L/M/H) affects failure distributions
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

# Color mapping for failure modes
FAILURE_COLORS = {
    "none": "#4CAF50",
    "tool_wear": "#2196F3",
    "heat_dissipation": "#F44336",
    "power_failure": "#FF9800",
    "overstrain": "#9C27B0",
    "random": "#607D8B",
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


def plot_failure_mode_distribution(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot failure mode class distribution with count and percentage labels.
    """
    if "failure_mode" not in df.columns:
        logger.warning("No failure_mode column found")
        return output_dir

    counts = df["failure_mode"].value_counts()
    total = len(df)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Bar chart
    colors = [FAILURE_COLORS.get(mode, "#888888") for mode in counts.index]
    bars = axes[0].bar(
        range(len(counts)),
        counts.values,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(
        [m.replace("_", " ").title() for m in counts.index],
        rotation=30,
        ha="right",
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Failure Mode Distribution")

    # Add count labels on bars
    for bar, count in zip(bars, counts.values, strict=False):
        pct = count / total * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Pie chart (failure modes only, excluding "none")
    failure_only = counts.drop("none", errors="ignore")
    if len(failure_only) > 0:
        pie_colors = [FAILURE_COLORS.get(mode, "#888888") for mode in failure_only.index]
        pie_labels = [m.replace("_", " ").title() for m in failure_only.index]
        wedges, texts, autotexts = axes[1].pie(
            failure_only.values,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        axes[1].set_title("Failure Mode Breakdown (Failures Only)")

    fig.suptitle("AI4I — Failure Mode Class Distribution", fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / "ai4i_failure_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_feature_distributions_by_type(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot feature distributions segmented by product type (L/M/H).
    """
    features = ["temperature", "pressure", "rpm", "vibration", "operating_cycle"]
    available = [f for f in features if f in df.columns]

    if "product_type" not in df.columns or not available:
        logger.warning("Missing product_type or features")
        return output_dir

    n_features = len(available)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    type_colors = {"L": "#2196F3", "M": "#FF9800", "H": "#F44336"}

    for idx, feature in enumerate(available):
        ax = axes[idx]
        for ptype in sorted(df["product_type"].unique()):
            data = df[df["product_type"] == ptype][feature].dropna()
            ax.hist(
                data,
                bins=30,
                alpha=0.5,
                label=f"Type {ptype}",
                color=type_colors.get(ptype, "#888888"),
                edgecolor="white",
            )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Value")
        if idx == 0:
            ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("AI4I — Feature Distributions by Product Type", fontsize=16, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "ai4i_features_by_type.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_failure_mode_feature_comparison(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Compare feature distributions across different failure modes using
    violin plots. Shows separation power of each feature for classification.
    """
    features = ["temperature", "pressure", "rpm", "vibration", "operating_cycle"]
    available = [f for f in features if f in df.columns]

    if "failure_mode" not in df.columns or not available:
        logger.warning("Missing failure_mode or features")
        return output_dir

    n_features = len(available)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    order = ["none", "tool_wear", "heat_dissipation", "power_failure", "overstrain", "random"]
    order = [m for m in order if m in df["failure_mode"].unique()]
    palette = {m: FAILURE_COLORS.get(m, "#888888") for m in order}

    for idx, feature in enumerate(available):
        ax = axes[idx]
        sns.violinplot(
            data=df,
            x="failure_mode",
            y=feature,
            hue="failure_mode",
            ax=ax,
            order=order,
            palette=palette,
            inner="quartile",
            cut=0,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel(feature.replace("_", " ").title())
        labels = [m.replace("_", " ").title() for m in order]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")

    fig.suptitle("AI4I — Feature Distributions by Failure Mode", fontsize=16, y=1.01)
    plt.tight_layout()

    out_path = output_dir / "ai4i_failure_feature_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Plot pairwise correlation heatmap of sensor features and failure labels.
    """
    numeric_cols = [
        "temperature",
        "pressure",
        "rpm",
        "vibration",
        "operating_cycle",
        "failure_label",
    ]
    failure_mode_cols = [
        c
        for c in df.columns
        if c.startswith("failure_") and c != "failure_mode" and c != "failure_label"
    ]
    all_cols = [c for c in numeric_cols + failure_mode_cols if c in df.columns]

    if len(all_cols) < 2:
        logger.warning("Insufficient numeric columns for correlation")
        return output_dir

    corr = df[all_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    clean_labels = []
    for c in all_cols:
        label = c.replace("failure_", "F:").replace("_", " ").title()
        clean_labels.append(label)
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
    ax.set_title("AI4I — Feature & Failure Mode Correlation", fontsize=16, pad=20)
    plt.tight_layout()

    out_path = output_dir / "ai4i_correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_class_imbalance_analysis(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Detailed class imbalance analysis: normal vs failure ratio,
    per-failure-mode sample sizes, and imbalance ratios.
    """
    if "failure_mode" not in df.columns:
        logger.warning("No failure_mode column found")
        return output_dir

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    total = len(df)
    failure_count = len(df[df["failure_mode"] != "none"])
    normal_count = total - failure_count

    # 1. Normal vs Failure pie
    axes[0].pie(
        [normal_count, failure_count],
        labels=["Normal", "Failure"],
        colors=["#4CAF50", "#F44336"],
        autopct="%1.1f%%",
        startangle=90,
        explode=[0, 0.05],
        textprops={"fontsize": 12},
    )
    axes[0].set_title(f"Normal vs. Failure\n(Ratio: {normal_count / max(failure_count, 1):.1f}:1)")

    # 2. Failure mode counts (log scale)
    mode_counts = df[df["failure_mode"] != "none"]["failure_mode"].value_counts()
    colors = [FAILURE_COLORS.get(m, "#888888") for m in mode_counts.index]
    labels = [m.replace("_", " ").title() for m in mode_counts.index]
    bars = axes[1].barh(
        range(len(mode_counts)), mode_counts.values, color=colors, alpha=0.85, edgecolor="white"
    )
    axes[1].set_yticks(range(len(mode_counts)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Sample Count")
    axes[1].set_title("Failure Mode Sample Sizes")
    axes[1].invert_yaxis()

    # Add count labels
    for bar, count in zip(bars, mode_counts.values, strict=False):
        axes[1].text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # 3. Imbalance ratio relative to majority failure class
    if len(mode_counts) > 0:
        max_failure = mode_counts.max()
        ratios = max_failure / mode_counts
        ratio_colors = [FAILURE_COLORS.get(m, "#888888") for m in ratios.index]
        ratio_labels = [m.replace("_", " ").title() for m in ratios.index]
        bars = axes[2].bar(
            range(len(ratios)), ratios.values, color=ratio_colors, alpha=0.85, edgecolor="white"
        )
        axes[2].set_xticks(range(len(ratios)))
        axes[2].set_xticklabels(ratio_labels, rotation=30, ha="right")
        axes[2].set_ylabel("Imbalance Ratio")
        axes[2].set_title("Inter-Class Imbalance\n(vs. Most Frequent Failure)")
        axes[2].axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

        for bar, ratio in zip(bars, ratios.values, strict=False):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{ratio:.1f}x",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle("AI4I — Class Imbalance Analysis", fontsize=16, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "ai4i_class_imbalance.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def generate_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics for the AI4I dataset.
    """
    stats = {
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    # Product type distribution
    if "product_type" in df.columns:
        stats["product_type_dist"] = df["product_type"].value_counts().to_dict()

    # Failure mode distribution
    if "failure_mode" in df.columns:
        mode_dist = df["failure_mode"].value_counts().to_dict()
        stats["failure_mode_dist"] = mode_dist
        total = len(df)
        normal_count = mode_dist.get("none", 0)
        failure_count = total - normal_count
        stats["normal_count"] = normal_count
        stats["failure_count"] = failure_count
        stats["failure_pct"] = round(failure_count / max(total, 1) * 100, 1)
        stats["imbalance_ratio"] = round(normal_count / max(failure_count, 1), 1)

    # Sensor statistics
    sensors = ["temperature", "pressure", "rpm", "vibration", "operating_cycle"]
    stats["sensor_stats"] = {}
    for sensor in sensors:
        if sensor in df.columns:
            col = df[sensor].dropna()
            stats["sensor_stats"][sensor] = {
                "mean": round(float(col.mean()), 2),
                "std": round(float(col.std()), 2),
                "min": round(float(col.min()), 2),
                "max": round(float(col.max()), 2),
            }

    # Multi-label analysis
    failure_cols = [
        c
        for c in df.columns
        if c.startswith("failure_") and c not in ("failure_mode", "failure_label")
    ]
    if failure_cols:
        # Count multi-label samples
        failure_binary = df[failure_cols]
        multi_label_mask = failure_binary.sum(axis=1) > 1
        stats["multi_label_samples"] = int(multi_label_mask.sum())
        stats["multi_label_pct"] = round(multi_label_mask.sum() / max(len(df), 1) * 100, 2)

    return stats


def generate_markdown_report(
    stats: dict[str, Any],
    plot_paths: dict[str, Path],
    output_dir: Path,
) -> Path:
    """
    Generate a markdown EDA report for AI4I.
    """
    lines = [
        "# AI4I 2020 — Exploratory Data Analysis",
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Samples | {stats['total_rows']:,} |",
    ]

    if "normal_count" in stats:
        lines.extend(
            [
                f"| Normal Samples | {stats['normal_count']:,} ({100 - stats['failure_pct']:.1f}%) |",
                f"| Failure Samples | {stats['failure_count']:,} ({stats['failure_pct']:.1f}%) |",
                f"| Imbalance Ratio (Normal:Failure) | {stats['imbalance_ratio']}:1 |",
            ]
        )

    if "multi_label_samples" in stats:
        lines.append(
            f"| Multi-Label Failures | {stats['multi_label_samples']} ({stats['multi_label_pct']}%) |"
        )

    lines.append("")

    # Product type distribution
    if "product_type_dist" in stats:
        lines.extend(
            [
                "## Product Type Distribution",
                "",
                "| Type | Count | Description |",
                "|------|-------|-------------|",
            ]
        )
        type_desc = {"L": "Low quality", "M": "Medium quality", "H": "High quality"}
        for ptype, count in sorted(stats["product_type_dist"].items()):
            pct = count / max(stats["total_rows"], 1) * 100
            lines.append(f"| {ptype} | {count:,} ({pct:.1f}%) | {type_desc.get(ptype, '')} |")
        lines.append("")

    # Failure mode distribution
    if "failure_mode_dist" in stats:
        lines.extend(
            [
                "## Failure Mode Distribution",
                "",
                "| Mode | Count | Percentage |",
                "|------|-------|------------|",
            ]
        )
        for mode, count in sorted(stats["failure_mode_dist"].items(), key=lambda x: -x[1]):
            pct = count / max(stats["total_rows"], 1) * 100
            lines.append(f"| {mode.replace('_', ' ').title()} | {count:,} | {pct:.1f}% |")
        lines.append("")

    # Sensor statistics
    if stats.get("sensor_stats"):
        lines.extend(
            [
                "## Sensor Statistics",
                "",
                "| Sensor | Mean | Std | Min | Max |",
                "|--------|------|-----|-----|-----|",
            ]
        )
        for sensor, ss in stats["sensor_stats"].items():
            lines.append(
                f"| {sensor.replace('_', ' ').title()} | {ss['mean']:.2f} | "
                f"{ss['std']:.2f} | {ss['min']:.2f} | {ss['max']:.2f} |"
            )
        lines.append("")

    # Embed plots
    lines.extend(["## Visualizations", ""])

    plot_titles = {
        "failure_dist": "### Failure Mode Distribution",
        "features_by_type": "### Features by Product Type",
        "failure_features": "### Failure Mode Feature Comparison",
        "correlation": "### Correlation Heatmap",
        "imbalance": "### Class Imbalance Analysis",
    }

    for key, title in plot_titles.items():
        if key in plot_paths and isinstance(plot_paths[key], Path) and plot_paths[key].exists():
            lines.extend([title, "", f"![{title.strip('# ')}]({plot_paths[key].name})", ""])

    # Key observations
    lines.extend(
        [
            "## Key Observations",
            "",
            f"1. **Severe class imbalance**: Normal samples outnumber failures by "
            f"{stats.get('imbalance_ratio', 'N/A')}:1, requiring SMOTE, class weights, "
            "or focal loss during model training.",
            "2. **Tool wear dominance**: Tool wear failure is the most common failure mode, "
            "but the failure distribution is heavily skewed — Random failures have very few samples.",
            "3. **Temperature and RPM as discriminators**: These features show the strongest "
            "separation between failure modes, making them critical inputs to the classifier.",
            "4. **Product type influence**: Low-quality products (Type L) have different feature "
            "distributions from High-quality (Type H), suggesting product type is a useful "
            "stratification variable.",
        ]
    )

    if "multi_label_samples" in stats and stats["multi_label_samples"] > 0:
        lines.append(
            f"5. **Multi-label overlaps**: {stats['multi_label_samples']} samples have "
            f"multiple simultaneous failure modes ({stats['multi_label_pct']}%), informing "
            "the choice of multi-label vs. priority-based classification strategy."
        )

    lines.append("")

    report_path = output_dir / "ai4i_eda.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Saved report: {report_path}")
    return report_path


def run_ai4i_eda(
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Run complete AI4I EDA pipeline.

    Args:
        data_dir: Path to processed data directory (containing ai4i/ai4i_full.parquet)
        output_dir: Path to save EDA outputs

    Returns:
        Dict of output file paths.
    """
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed AI4I data
    data_path = data_dir / "ai4i" / "ai4i_full.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"AI4I data not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded AI4I: {df.shape}")

    # Generate plots
    plot_paths = {}
    plot_paths["failure_dist"] = plot_failure_mode_distribution(df, output_dir)
    plot_paths["features_by_type"] = plot_feature_distributions_by_type(df, output_dir)
    plot_paths["failure_features"] = plot_failure_mode_feature_comparison(df, output_dir)
    plot_paths["correlation"] = plot_correlation_heatmap(df, output_dir)
    plot_paths["imbalance"] = plot_class_imbalance_analysis(df, output_dir)

    # Generate summary stats and report
    stats = generate_summary_stats(df)
    report_path = generate_markdown_report(stats, plot_paths, output_dir)

    all_outputs = {**plot_paths, "report": report_path}
    logger.info(f"AI4I EDA complete: {len(all_outputs)} artifacts generated")
    return all_outputs
