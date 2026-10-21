"""
MaintenAI— Telemetry & Prediction Drift Detection

Monitors data drift and concept drift using:
  - Population Stability Index (PSI)
  - Two-sample Kolmogorov-Smirnov (KS) statistical test
  - Jensen-Shannon Divergence / Chi-Square for categorical features
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import ks_2samp

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and production samples.

    PSI < 0.10: No significant change / Stable.
    0.10 <= PSI < 0.25: Moderate drift / Monitor closely.
    PSI >= 0.25: Significant drift / Model retraining required.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicate edges

    if len(bin_edges) < 2:
        return 0.0

    # Ensure range covers all values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * len(expected_counts))
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * len(actual_counts))

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def calculate_ks_test(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """
    Calculate 2-sample Kolmogorov-Smirnov test statistic and p-value.
    Small p-value (< 0.05) indicates distributions are significantly different.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    res = ks_2samp(expected, actual)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


class DriftMonitor:
    """
    Drift Monitor for telemetry features and model predictions.
    """

    def __init__(
        self,
        psi_threshold: float = 0.10,
        ks_alpha: float = 0.05,
        min_samples: int = 100,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.min_samples = min_samples

    def check_feature_drift(
        self,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate drift across multiple continuous features.
        """
        cols = feature_cols or [
            c for c in df_reference.columns if np.issubdtype(df_reference[c].dtype, np.number)
        ]

        report: dict[str, Any] = {
            "summary": {
                "total_features": len(cols),
                "drifted_features": 0,
                "drift_detected": False,
            },
            "features": {},
        }

        drift_count = 0
        for col in cols:
            if col not in df_current.columns:
                continue

            ref_vals = df_reference[col].dropna().values
            curr_vals = df_current[col].dropna().values

            if len(ref_vals) < self.min_samples or len(curr_vals) < self.min_samples:
                continue

            psi = calculate_psi(ref_vals, curr_vals)
            ks = calculate_ks_test(ref_vals, curr_vals)

            is_drifted = (psi >= self.psi_threshold) or (
                ks["p_value"] < self.ks_alpha and ks["statistic"] > 0.15
            )
            if is_drifted:
                drift_count += 1

            report["features"][col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks["statistic"], 4),
                "ks_p_value": round(ks["p_value"], 6),
                "drifted": is_drifted,
                "severity": "high" if psi >= 0.25 else "moderate" if psi >= 0.10 else "low",
            }

        report["summary"]["drifted_features"] = drift_count
        report["summary"]["drift_detected"] = drift_count > 0
        return report
