"""
MaintenAI— Unit Tests for Data Validation Module
"""

import pandas as pd

from ml.common.validation import (
    ValidationReport,
    run_full_validation,
    validate_duplicates,
    validate_missing,
    validate_ranges,
    validate_schema,
)


class TestValidationReport:
    """Test ValidationReport data class."""

    def test_validity_percentage(self):
        report = ValidationReport(
            dataset_name="test", total_rows=100, valid_rows=95, invalid_rows=5
        )
        assert report.validity_pct == 95.0

    def test_empty_report_is_valid(self):
        report = ValidationReport(dataset_name="test")
        assert report.is_valid

    def test_schema_issues_make_invalid(self):
        report = ValidationReport(dataset_name="test")
        report.schema_issues.append("Missing column: machine_id")
        assert not report.is_valid

    def test_summary_output(self):
        report = ValidationReport(dataset_name="test", total_rows=10, valid_rows=8, invalid_rows=2)
        report.missing_values = {"temperature": 3}
        summary = report.summary()
        assert "test" in summary
        assert "10" in summary
        assert "temperature" in summary


class TestValidateSchema:
    """Test schema validation."""

    def test_valid_schema(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        report = validate_schema(df, ["a", "b"], "test")
        assert report.is_valid
        assert len(report.schema_issues) == 0

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        report = validate_schema(df, ["a", "b", "c"], "test")
        assert not report.is_valid
        assert any("b" in issue for issue in report.schema_issues)

    def test_extra_columns_warning(self):
        df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
        report = validate_schema(df, ["a", "b"], "test")
        assert report.is_valid  # Extra cols are warnings, not errors
        assert len(report.warnings) > 0


class TestValidateRanges:
    """Test physical range validation."""

    def test_values_within_range(self):
        df = pd.DataFrame({"temperature": [25.0, 50.0, 100.0]})
        report = validate_ranges(df)
        assert len(report.out_of_range) == 0

    def test_temperature_out_of_range(self):
        df = pd.DataFrame({"temperature": [25.0, 2000.0, -100.0]})
        report = validate_ranges(df)
        assert "temperature" in report.out_of_range
        assert report.out_of_range["temperature"] == 2

    def test_negative_vibration(self):
        df = pd.DataFrame({"vibration": [1.0, -5.0, 3.0]})
        report = validate_ranges(df)
        assert "vibration" in report.out_of_range

    def test_missing_columns_ignored(self):
        df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        report = validate_ranges(df)
        assert len(report.out_of_range) == 0


class TestValidateMissing:
    """Test missing value detection."""

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        report = validate_missing(df)
        assert len(report.missing_values) == 0

    def test_missing_detected(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
        report = validate_missing(df)
        assert report.missing_values["a"] == 1
        assert report.missing_values["b"] == 2


class TestValidateDuplicates:
    """Test duplicate detection."""

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = validate_duplicates(df)
        assert report.duplicate_rows == 0

    def test_duplicates_found(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        report = validate_duplicates(df)
        assert report.duplicate_rows == 1

    def test_subset_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 10]})
        report = validate_duplicates(df, subset=["a"])
        assert report.duplicate_rows == 1


class TestFullValidation:
    """Test the combined validation pipeline."""

    def test_full_pipeline(self):
        df = pd.DataFrame(
            {
                "machine_id": ["M001", "M001", "M002"],
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC"),
                "temperature": [25.0, 50.0, 75.0],
                "vibration": [1.0, 2.0, 3.0],
            }
        )
        report = run_full_validation(
            df,
            required_columns=["machine_id", "timestamp", "temperature"],
            dataset_name="full_test",
        )
        assert report.is_valid
        assert report.total_rows == 3
        assert report.valid_rows == 3
