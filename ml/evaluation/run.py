"""
MaintenAI— Unified ML Evaluation Engine (DVC Entrypoint)

Aggregates test metrics across all models (Anomaly, RUL, Failure Mode Classification),
validates CI/CD quality gates from params.yaml, and outputs a summary JSON artifact.

Usage:
    python -m ml.evaluation.run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import yaml

from ml.evaluation.metrics import verify_quality_gates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_SUMMARY_OUTPUT = Path("ml/evaluation/summary.json")


def load_json_metrics(path: Path) -> dict[str, Any]:
    """Load JSON metrics file if exists, else empty dict."""
    if not path.exists():
        logger.warning("Metrics file not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


@click.command()
@click.option("--params-file", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml")
@click.option("--output", default=str(DEFAULT_SUMMARY_OUTPUT), help="Output path for summary.json")
def main(params_file: str, output: str) -> None:
    """Run unified model evaluation and quality gate verification."""
    params_path = Path(params_file)
    gate_params = {}
    if params_path.exists():
        with open(params_path) as f:
            cfg = yaml.safe_load(f) or {}
            gate_params = cfg.get("evaluation", {})

    summary: dict[str, Any] = {
        "anomaly": {
            "isolation_forest": load_json_metrics(Path("ml/anomaly/metrics_iforest.json")),
            "lstm_autoencoder": load_json_metrics(Path("ml/anomaly/metrics_lstm.json")),
        },
        "rul": {
            "xgboost": load_json_metrics(Path("ml/rul/metrics_xgboost.json")),
            "transformer": load_json_metrics(Path("ml/rul/metrics_transformer.json")),
        },
        "failure": {
            "lightgbm": load_json_metrics(Path("ml/failure_classification/metrics_lgbm.json")),
        },
    }

    passed, violations = verify_quality_gates(summary, gate_params)
    summary["quality_gates"] = {
        "passed": passed,
        "violations": violations,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation Summary written to %s", out_path)
    logger.info("Quality Gates Status: %s", "PASSED" if passed else f"FAILED: {violations}")

    if not passed:
        logger.warning("Quality gate violations: %s", violations)


if __name__ == "__main__":
    main()
