"""
MaintenAI— EDA Runner

Orchestrates Exploratory Data Analysis generation for all three datasets:
  - C-MAPSS (FD001) — RUL prediction
  - IMS Bearing — Anomaly detection
  - AI4I 2020 — Failure-mode classification

Usage:
    # Run all EDA
    python -m ml.common.eda.runner

    # Run specific dataset
    python -m ml.common.eda.runner --dataset cmapss

    # Custom paths
    python -m ml.common.eda.runner --data-dir data/processed --output-dir docs/eda
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_all_eda(
    data_dir: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Run EDA for specified datasets (or all if none specified).

    Args:
        data_dir: Path to processed data directory
        output_dir: Base output directory for EDA artifacts
        datasets: List of datasets to process. Options: 'cmapss', 'ims', 'ai4i'.
                  If None, processes all datasets.

    Returns:
        Nested dict: {dataset_name: {artifact_name: artifact_path}}
    """
    all_datasets = ["cmapss", "ims", "ai4i"]
    if datasets is None:
        datasets = all_datasets

    # Validate dataset names
    invalid = set(datasets) - set(all_datasets)
    if invalid:
        raise ValueError(f"Unknown dataset(s): {invalid}. Valid options: {all_datasets}")

    results = {}
    total_start = time.time()

    for dataset_name in datasets:
        dataset_output = output_dir / dataset_name
        logger.info(f"{'=' * 60}")
        logger.info(f"Running EDA: {dataset_name.upper()}")
        logger.info(f"{'=' * 60}")
        start = time.time()

        try:
            if dataset_name == "cmapss":
                from ml.common.eda.cmapss_eda import run_cmapss_eda

                results[dataset_name] = run_cmapss_eda(data_dir, dataset_output)

            elif dataset_name == "ims":
                from ml.common.eda.ims_eda import run_ims_eda

                results[dataset_name] = run_ims_eda(data_dir, dataset_output)

            elif dataset_name == "ai4i":
                from ml.common.eda.ai4i_eda import run_ai4i_eda

                results[dataset_name] = run_ai4i_eda(data_dir, dataset_output)

            elapsed = time.time() - start
            n_artifacts = len(results[dataset_name])
            logger.info(
                f"✓ {dataset_name.upper()} EDA complete: {n_artifacts} artifacts in {elapsed:.1f}s"
            )

        except FileNotFoundError as e:
            logger.error(f"✗ {dataset_name.upper()} EDA skipped: {e}")
            results[dataset_name] = {}

        except Exception as e:
            logger.error(f"✗ {dataset_name.upper()} EDA failed: {e}", exc_info=True)
            results[dataset_name] = {}

    # Generate index page
    _generate_eda_index(output_dir, results)

    total_elapsed = time.time() - total_start
    total_artifacts = sum(len(v) for v in results.values())
    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"EDA complete: {total_artifacts} total artifacts "
        f"across {len(results)} datasets in {total_elapsed:.1f}s"
    )
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'=' * 60}")

    return results


def _generate_eda_index(
    output_dir: Path,
    results: dict[str, dict[str, Path]],
) -> Path:
    """
    Generate an index markdown page linking to all EDA reports.
    """
    lines = [
        "# MaintenAI— Exploratory Data Analysis",
        "",
        "Automated EDA reports for all three canonical datasets used in FactoryPulse.",
        "",
        "## Dataset Reports",
        "",
        "| Dataset | Purpose | Report |",
        "|---------|---------|--------|",
    ]

    dataset_info = {
        "cmapss": {
            "name": "NASA C-MAPSS (FD001)",
            "purpose": "RUL Prediction",
            "report": "cmapss/cmapss_eda.md",
        },
        "ims": {
            "name": "NASA IMS Bearing",
            "purpose": "Anomaly Detection",
            "report": "ims/ims_eda.md",
        },
        "ai4i": {
            "name": "AI4I 2020",
            "purpose": "Failure-Mode Classification",
            "report": "ai4i/ai4i_eda.md",
        },
    }

    for dataset_key, info in dataset_info.items():
        if dataset_key in results and results[dataset_key]:
            lines.append(
                f"| {info['name']} | {info['purpose']} | [View Report]({info['report']}) |"
            )
        else:
            lines.append(f"| {info['name']} | {info['purpose']} | ❌ Not generated |")

    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
        ]
    )

    for dataset_key, artifacts in results.items():
        if artifacts:
            info = dataset_info.get(dataset_key, {"name": dataset_key})
            lines.append(f"### {info['name']}")
            lines.append("")
            for name, path in sorted(artifacts.items()):
                if isinstance(path, Path) and path.exists():
                    rel_path = (
                        path.relative_to(output_dir) if path.is_relative_to(output_dir) else path
                    )
                    lines.append(f"- [{name}]({rel_path})")
            lines.append("")

    index_path = output_dir / "README.md"
    index_path.write_text("\n".join(lines))
    logger.info(f"Saved EDA index: {index_path}")
    return index_path


def main():
    """CLI entrypoint for EDA generation."""
    parser = argparse.ArgumentParser(
        description="MaintenAI— Exploratory Data Analysis Generator",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed dataset directory (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/eda"),
        help="Output directory for EDA artifacts (default: docs/eda)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cmapss", "ims", "ai4i"],
        nargs="*",
        default=None,
        help="Specific dataset(s) to analyze (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run EDA
    run_all_eda(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datasets=args.dataset,
    )


if __name__ == "__main__":
    main()
