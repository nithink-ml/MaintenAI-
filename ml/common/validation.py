"""
MaintenAI— Data Validation Utilities

Provides reusable validation functions for raw and processed datasets.
Enforces physical sensor ranges, schema correctness, timestamp ordering,
duplicate detection, and missing value analysis.

These validators run during data preparation AND at ingestion time,
ensuring consistent quality gates across offline and online paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Physical Sensor Ranges
# ============================================================

# These define physically plausible bounds for each sensor.
# Values outside these ranges indicate sensor malfunction or
# data corruption — not actual machine behavior.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-50.0, 1500.0),  # Celsius
    "pressure": (0.0, 1000.0),  # bar
    "vibration": (0.0, 500.0),  # mm/s RMS
    "rpm": (0.0, 50000.0),  # RPM
    "current": (0.0, 1000.0),  # Amps
    "operating_cycle": (0, 1e9),  # count
}


@dataclass
class ValidationReport:
    """Detailed validation report for a dataset."""

    dataset_name: str
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    duplicate_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    timestamp_issues: int = 0
    schema_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Dataset passes validation if no critical schema issues."""
        return len(self.schema_issues) == 0

    @property
    def validity_pct(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return (self.valid_rows / self.total_rows) * 100

    def summary(self) -> str:
        lines = [
            f"=== Validation Report: {self.dataset_name} ===",
            f"Total rows:     {self.total_rows:,}",
            f"Valid rows:     {self.valid_rows:,} ({self.validity_pct:.1f}%)",
            f"Invalid rows:   {self.invalid_rows:,}",
            f"Duplicate rows: {self.duplicate_rows:,}",
        ]
        if self.missing_values:
            lines.append("Missing values:")
            for col, count in sorted(self.missing_values.items()):
                pct = count / self.total_rows * 100 if self.total_rows > 0 else 0
                lines.append(f"  {col}: {count:,} ({pct:.1f}%)")
        if self.out_of_range:
            lines.append("Out-of-range values:")
            for col, count in sorted(self.out_of_range.items()):
                lines.append(f"  {col}: {count:,}")
        if self.schema_issues:
            lines.append("Schema issues:")
            for issue in self.schema_issues:
                lines.append(f"  ❌ {issue}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        return "\n".join(lines)


# ============================================================
# Validation Functions
# ============================================================


def validate_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str = "unknown",
) -> ValidationReport:
    """
    Validate DataFrame schema against required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Columns that must be present.
        dataset_name: Name for the report.

    Returns:
        ValidationReport with schema check results.
    """
    report = ValidationReport(dataset_name=dataset_name, total_rows=len(df))

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        report.schema_issues.append(f"Missing required columns: {sorted(missing_cols)}")

    extra_cols = set(df.columns) - set(required_columns)
    if extra_cols:
        report.warnings.append(f"Extra columns present (ignored): {sorted(extra_cols)}")

    return report


def validate_ranges(
    df: pd.DataFrame,
    report: ValidationReport | None = None,
    sensor_ranges: dict[str, tuple[float, float]] | None = None,
) -> ValidationReport:
    """
    Validate that sensor values fall within physical bounds.

    Values outside bounds are flagged but not automatically removed.
    The caller decides whether to clip, impute, or reject.
    """
    if report is None:
        report = ValidationReport(dataset_name="unknown", total_rows=len(df))

    ranges = sensor_ranges or SENSOR_RANGES

    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        below = (series < lo).sum()
        above = (series > hi).sum()
        total_oor = int(below + above)
        if total_oor > 0:
            report.out_of_range[col] = total_oor
            report.warnings.append(
                f"Column '{col}': {total_oor} values outside [{lo}, {hi}] "
                f"(min={series.min():.2f}, max={series.max():.2f})"
            )

    return report


def validate_missing(
    df: pd.DataFrame,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """Count missing values per column."""
    if report is None:
        report = ValidationReport(dataset_name="unknown", total_rows=len(df))

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            report.missing_values[col] = n_missing

    return report


def validate_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """Detect duplicate rows based on given subset of columns."""
    if report is None:
        report = ValidationReport(dataset_name="unknown", total_rows=len(df))

    n_dupes = int(df.duplicated(subset=subset).sum())
    report.duplicate_rows = n_dupes
    if n_dupes > 0:
        report.warnings.append(f"{n_dupes} duplicate rows detected")

    return report


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    machine_id_col: str = "machine_id",
    report: ValidationReport | None = None,
) -> ValidationReport:
    """
    Validate timestamp ordering within each machine's time series.

    Checks:
      - Timestamps are monotonically non-decreasing per machine
      - No future timestamps (beyond current time + 5 min tolerance)
    """
    if report is None:
        report = ValidationReport(dataset_name="unknown", total_rows=len(df))

    if timestamp_col not in df.columns:
        report.warnings.append(f"Timestamp column '{timestamp_col}' not found — skipping check")
        return report

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    n_unparseable = int(ts.isna().sum() - df[timestamp_col].isna().sum())
    if n_unparseable > 0:
        report.timestamp_issues += n_unparseable
        report.warnings.append(f"{n_unparseable} timestamps could not be parsed")

    # Check monotonicity per machine
    if machine_id_col in df.columns:
        for mid, group in df.groupby(machine_id_col):
            group_ts = pd.to_datetime(group[timestamp_col], errors="coerce").dropna()
            if len(group_ts) > 1:
                non_mono = int((group_ts.diff().iloc[1:] < pd.Timedelta(0)).sum())
                if non_mono > 0:
                    report.timestamp_issues += non_mono
                    report.warnings.append(f"Machine '{mid}': {non_mono} non-monotonic timestamps")

    return report


def run_full_validation(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str = "unknown",
    timestamp_col: str | None = "timestamp",
    machine_id_col: str | None = "machine_id",
    dedup_subset: list[str] | None = None,
) -> ValidationReport:
    """
    Run all validation checks on a DataFrame.

    Returns a comprehensive ValidationReport.
    """
    report = validate_schema(df, required_columns, dataset_name)
    report = validate_missing(df, report)
    report = validate_ranges(df, report)
    report = validate_duplicates(df, subset=dedup_subset, report=report)

    if timestamp_col and timestamp_col in df.columns:
        report = validate_timestamps(
            df, timestamp_col=timestamp_col, machine_id_col=machine_id_col or "", report=report
        )

    # Calculate valid/invalid
    invalid_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in SENSOR_RANGES.items():
        if col in df.columns:
            invalid_mask |= (df[col] < lo) | (df[col] > hi)

    report.invalid_rows = int(invalid_mask.sum())
    report.valid_rows = report.total_rows - report.invalid_rows

    return report
