"""
MaintenAI— Data Preparation Pipeline

Orchestrates the full data preparation workflow:
  1. Load raw datasets (C-MAPSS, IMS, AI4I)
  2. Map to canonical schema
  3. Validate
  4. Split (leakage-safe)
  5. Save processed parquet files

Usage:
    python -m ml.common.data_prepare                  # Prepare all datasets
    python -m ml.common.data_prepare --dataset cmapss  # Prepare specific
    python -m ml.common.data_prepare --synthetic       # Use synthetic fallback
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def prepare_cmapss(
    raw_dir: Path,
    processed_dir: Path,
    dataset_id: str = "FD001",
    max_rul_clip: int = 125,
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict[str, Path]:
    """
    Prepare C-MAPSS dataset for RUL prediction.

    Pipeline:
      1. Process raw C-MAPSS → canonical schema
      2. Split by unit (leakage-safe)
      3. Save train/val/test parquet files
    """
    from ml.common.datasets.cmapss import process_cmapss
    from ml.common.splitting import save_splits, split_by_unit, verify_no_leakage

    output_dir = processed_dir / "cmapss"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Process raw data
    cmapss_raw = raw_dir / "cmapss"
    paths = process_cmapss(cmapss_raw, output_dir, dataset_id, max_rul_clip)

    # Step 2: Load processed training data and split by unit
    import pandas as pd

    train_full = pd.read_parquet(paths["train"])

    # C-MAPSS test set already has its own test units with ground truth RUL
    # We split the training data into train/val, keeping the original test set
    train_df, val_df, _ = split_by_unit(
        train_full,
        unit_col="machine_id",
        test_fraction=0.0,  # Don't create another test set
        val_fraction=val_fraction,
        random_seed=random_seed,
    )

    # Use original C-MAPSS test set
    test_df = pd.read_parquet(paths["test"])

    # Step 3: Verify and save
    verify_no_leakage(train_df, val_df, test_df, unit_col="machine_id")
    split_paths = save_splits(train_df, val_df, test_df, output_dir, f"cmapss_{dataset_id}")

    logger.info(f"✅ C-MAPSS {dataset_id} preparation complete")
    return split_paths


def prepare_ai4i(
    raw_dir: Path,
    processed_dir: Path,
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict[str, Path]:
    """
    Prepare AI4I dataset for failure-mode classification.

    Pipeline:
      1. Process raw AI4I → canonical schema
      2. Stratified split by failure mode
      3. Save train/val/test parquet files
    """
    import pandas as pd

    from ml.common.datasets.ai4i import process_ai4i
    from ml.common.splitting import save_splits, split_stratified

    output_dir = processed_dir / "ai4i"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Process raw data
    ai4i_raw = raw_dir / "ai4i"
    paths = process_ai4i(ai4i_raw, output_dir)

    # Step 2: Stratified split
    full_df = pd.read_parquet(paths["full"])
    train_df, val_df, test_df = split_stratified(
        full_df,
        target_col="failure_mode",
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        random_seed=random_seed,
    )

    # Step 3: Save
    split_paths = save_splits(train_df, val_df, test_df, output_dir, "ai4i")

    logger.info("✅ AI4I preparation complete")
    return split_paths


def prepare_ims(
    raw_dir: Path,
    processed_dir: Path,
    test_fraction: float = 0.20,
    val_fraction: float = 0.15,
    max_files: int | None = None,
) -> dict[str, Path]:
    """
    Prepare IMS Bearing dataset for anomaly detection.

    Pipeline:
      1. Process raw IMS → vibration features → canonical schema
      2. Temporal split (early = train, late = test)
      3. Save train/val/test parquet files
    """
    import pandas as pd

    from ml.common.datasets.ims import process_ims
    from ml.common.splitting import save_splits, split_temporal

    output_dir = processed_dir / "ims"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Process raw data
    ims_raw = raw_dir / "ims"
    paths = process_ims(ims_raw, output_dir, max_files=max_files)

    # Step 2: Temporal split
    canonical = pd.read_parquet(paths["canonical"])
    train_df, val_df, test_df = split_temporal(
        canonical,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
    )

    # Step 3: Save
    split_paths = save_splits(train_df, val_df, test_df, output_dir, "ims")

    logger.info("✅ IMS preparation complete")
    return split_paths


def prepare_all(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    synthetic: bool = False,
) -> dict[str, dict[str, Path]]:
    """Prepare all three datasets."""
    if synthetic:
        from ml.common.data_download import create_synthetic_fallback

        for name in ["cmapss", "ims", "ai4i"]:
            create_synthetic_fallback(name, raw_dir)

    results = {}

    # C-MAPSS
    try:
        results["cmapss"] = prepare_cmapss(raw_dir, processed_dir)
    except Exception as e:
        logger.error(f"C-MAPSS preparation failed: {e}")

    # AI4I
    try:
        results["ai4i"] = prepare_ai4i(raw_dir, processed_dir)
    except Exception as e:
        logger.error(f"AI4I preparation failed: {e}")

    # IMS
    try:
        results["ims"] = prepare_ims(raw_dir, processed_dir)
    except Exception as e:
        logger.error(f"IMS preparation failed: {e}")

    return results


# ============================================================
# CLI Entry Point
# ============================================================


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["cmapss", "ims", "ai4i", "all"]),
    default="all",
)
@click.option("--raw-dir", type=click.Path(), default="data/raw")
@click.option("--processed-dir", type=click.Path(), default="data/processed")
@click.option("--synthetic", is_flag=True, help="Generate synthetic data first")
def main(dataset: str, raw_dir: str, processed_dir: str, synthetic: bool) -> None:
    """Prepare MaintenAIdatasets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw = Path(raw_dir)
    processed = Path(processed_dir)

    if synthetic:
        from ml.common.data_download import create_synthetic_fallback

        datasets = [dataset] if dataset != "all" else ["cmapss", "ims", "ai4i"]
        for ds in datasets:
            create_synthetic_fallback(ds, raw)

    if dataset == "all":
        prepare_all(raw, processed)
    elif dataset == "cmapss":
        prepare_cmapss(raw, processed)
    elif dataset == "ai4i":
        prepare_ai4i(raw, processed)
    elif dataset == "ims":
        prepare_ims(raw, processed)


if __name__ == "__main__":
    main()
