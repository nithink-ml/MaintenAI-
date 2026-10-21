"""
MaintenAI— Unified ML Evaluation Metrics & Quality Gate Verification

Validates performance metrics against pre-defined CI/CD quality gates in params.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def verify_quality_gates(
    metrics_summary: dict[str, Any],
    gate_params: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Check if model evaluation metrics meet quality gate standards.

    Returns:
        (passed, violations): Tuple of boolean status and list of violation explanations.
    """
    violations: list[str] = []

    # 1. RUL Gates
    rul_gates = gate_params.get("rul", {})
    max_rmse = rul_gates.get("max_rmse", 20.0)
    max_mae = rul_gates.get("max_mae", 15.0)

    for rul_model in ["xgboost", "transformer"]:
        if rul_model in metrics_summary.get("rul", {}):
            rmse_val = metrics_summary["rul"][rul_model].get("rmse", 999.0)
            mae_val = metrics_summary["rul"][rul_model].get("mae", 999.0)

            if rmse_val > max_rmse:
                violations.append(
                    f"RUL {rul_model} RMSE ({rmse_val:.2f}) exceeds gate ({max_rmse:.2f})"
                )
            if mae_val > max_mae:
                violations.append(
                    f"RUL {rul_model} MAE ({mae_val:.2f}) exceeds gate ({max_mae:.2f})"
                )

    # 2. Anomaly Detection Gates
    anom_gates = gate_params.get("anomaly", {})
    min_f1 = anom_gates.get("min_f1", 0.75)
    max_fpr = anom_gates.get("max_fpr", 0.10)

    for anom_model in ["isolation_forest", "lstm_autoencoder"]:
        if anom_model in metrics_summary.get("anomaly", {}):
            f1_val = metrics_summary["anomaly"][anom_model].get("f1", 0.0)
            fpr_val = metrics_summary["anomaly"][anom_model].get("fpr", 1.0)

            if f1_val < min_f1:
                violations.append(
                    f"Anomaly {anom_model} F1 ({f1_val:.2f}) is below gate ({min_f1:.2f})"
                )
            if fpr_val > max_fpr:
                violations.append(
                    f"Anomaly {anom_model} FPR ({fpr_val:.2f}) exceeds gate ({max_fpr:.2f})"
                )

    # 3. Failure Classification Gates
    fail_gates = gate_params.get("failure", {})
    min_f1_macro = fail_gates.get("min_f1_macro", 0.80)
    min_prec_macro = fail_gates.get("min_precision_macro", 0.75)

    if "lightgbm" in metrics_summary.get("failure", {}):
        f1_m = metrics_summary["failure"]["lightgbm"].get("f1_macro", 0.0)
        prec_m = metrics_summary["failure"]["lightgbm"].get("precision_macro", 0.0)

        if f1_m < min_f1_macro:
            violations.append(
                f"Failure LightGBM F1-Macro ({f1_m:.2f}) is below gate ({min_f1_macro:.2f})"
            )
        if prec_m < min_prec_macro:
            violations.append(
                f"Failure LightGBM Precision-Macro ({prec_m:.2f}) is below gate ({min_prec_macro:.2f})"
            )

    passed = len(violations) == 0
    return passed, violations
