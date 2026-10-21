"""
MaintenAI— Dataset Download Utilities

Downloads the three benchmark datasets used in FactoryPulse:
  1. NASA C-MAPSS (Turbofan Engine Degradation) → RUL prediction
  2. NASA IMS Bearing Dataset → Anomaly detection
  3. AI4I 2020 Predictive Maintenance → Failure-mode classification

Usage:
    python -m ml.common.data_download                # Download all
    python -m ml.common.data_download --dataset cmapss  # Download specific
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import click

logger = logging.getLogger(__name__)

# ============================================================
# Dataset Source URLs
# ============================================================

DATASET_SOURCES: dict[str, dict] = {
    "cmapss": {
        "description": "NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001-FD004)",
        "url": "https://ti.arc.nasa.gov/c/6/",
        "alt_url": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        "filename": "CMAPSSData.zip",
        "purpose": "RUL (Remaining Useful Life) prediction",
    },
    "ai4i": {
        "description": "AI4I 2020 Predictive Maintenance Dataset (UCI)",
        "url": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "filename": "ai4i2020.zip",
        "purpose": "Failure-mode classification",
    },
    "ims": {
        "description": "NASA IMS Bearing Dataset",
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
        "filename": "IMS_Bearing.zip",
        "purpose": "Anomaly detection / bearing degradation",
    },
}

# ============================================================
# Download Functions
# ============================================================


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple progress indicator for downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def download_dataset(
    name: str,
    raw_dir: Path,
    force: bool = False,
) -> Path:
    """
    Download and extract a dataset to raw_dir/<name>/.

    Args:
        name: Dataset key ('cmapss', 'ims', or 'ai4i').
        raw_dir: Root directory for raw data (e.g. data/raw/).
        force: If True, re-download even if files exist.

    Returns:
        Path to the extracted dataset directory.
    """
    if name not in DATASET_SOURCES:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(DATASET_SOURCES.keys())}")

    info = DATASET_SOURCES[name]
    dest_dir = raw_dir / name
    marker_file = dest_dir / ".download_complete"

    if marker_file.exists() and not force:
        logger.info(
            f"Dataset '{name}' already downloaded at {dest_dir}. Use --force to re-download."
        )
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / info["filename"]

    logger.info(f"Downloading {info['description']}...")
    logger.info(f"  Source: {info['url']}")
    logger.info(f"  Destination: {zip_path}")

    url = info["url"]
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Insecure or invalid URL scheme: {url}")

    try:
        urlretrieve(url, str(zip_path), reporthook=_progress_hook)  # nosec B310
        print()  # newline after progress bar
    except Exception as e:
        # Try alternative URL if available
        alt_url = info.get("alt_url")
        if alt_url:
            if not (alt_url.startswith("http://") or alt_url.startswith("https://")):
                raise ValueError(f"Insecure or invalid URL scheme: {alt_url}") from e
            logger.warning(f"Primary URL failed ({e}), trying alternative: {alt_url}")
            urlretrieve(alt_url, str(zip_path), reporthook=_progress_hook)  # nosec B310
            print()
        else:
            raise

    logger.info(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(dest_dir))
    except zipfile.BadZipFile:
        logger.error(f"Downloaded file is not a valid zip: {zip_path}")
        raise

    # Clean up zip file to save space
    zip_path.unlink(missing_ok=True)

    # Write marker
    marker_file.write_text(f"Downloaded from {info['url']}\n")

    logger.info(f"Dataset '{name}' ready at {dest_dir}")
    return dest_dir


def download_all(raw_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download all three datasets."""
    results = {}
    for name in DATASET_SOURCES:
        try:
            results[name] = download_dataset(name, raw_dir, force)
        except Exception as e:
            logger.error(f"Failed to download '{name}': {e}")
            logger.info(
                f"You can manually download '{name}' from:\n"
                f"  {DATASET_SOURCES[name]['url']}\n"
                f"  Extract to: {raw_dir / name}/"
            )
    return results


def create_synthetic_fallback(name: str, raw_dir: Path) -> Path:
    """
    Create a small synthetic dataset for development/testing when
    the real dataset cannot be downloaded.

    IMPORTANT: This is labeled as synthetic and must never be used
    to report real metrics. It exists solely to allow the pipeline
    to run during development.
    """
    import numpy as np

    dest_dir = raw_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    if name == "cmapss":
        _create_synthetic_cmapss(dest_dir, rng)
    elif name == "ai4i":
        _create_synthetic_ai4i(dest_dir, rng)
    elif name == "ims":
        _create_synthetic_ims(dest_dir, rng)
    else:
        raise ValueError(f"No synthetic generator for: {name}")

    # Write marker
    (dest_dir / ".synthetic_data").write_text(
        "WARNING: This is synthetic data for development only.\n"
        "Do NOT use for reporting metrics or evaluation.\n"
    )
    (dest_dir / ".download_complete").write_text("Synthetic fallback data\n")

    logger.info(f"Synthetic fallback for '{name}' created at {dest_dir}")
    return dest_dir


def _create_synthetic_cmapss(dest_dir: Path, rng) -> None:
    """Create synthetic C-MAPSS FD001-like data."""
    import pandas as pd

    # C-MAPSS FD001 has: unit_id, cycle, op_setting1-3, sensor1-21
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    cols = ["unit_id", "cycle"] + op_cols + sensor_cols

    rows = []
    n_units = 50
    for unit_id in range(1, n_units + 1):
        max_cycles = rng.integers(120, 350)
        for cycle in range(1, max_cycles + 1):
            # Simulate gradual degradation
            degradation = cycle / max_cycles
            op_settings = rng.normal([0.0, 0.0, 100.0], [0.01, 0.0005, 0.01])
            sensors = rng.normal(
                [
                    518 + 10 * degradation,
                    642 + 5 * degradation,
                    1590 - 20 * degradation,
                    1400 + 30 * degradation,
                    14.6 + 0.5 * degradation,
                    21.6,
                    554 + 3 * degradation,
                    2388,
                    9046 - 50 * degradation,
                    1.3,
                    47.5 - 2 * degradation,
                    521 + 8 * degradation,
                    2388,
                    8138 - 30 * degradation,
                    8.4 + 0.2 * degradation,
                    0.03,
                    392 + 5 * degradation,
                    2388,
                    100,
                    39 - degradation,
                    23.4,
                ],
                [
                    2,
                    3,
                    10,
                    15,
                    0.1,
                    0.02,
                    2,
                    1,
                    20,
                    0.01,
                    0.5,
                    3,
                    1,
                    15,
                    0.05,
                    0.001,
                    2,
                    1,
                    0.5,
                    0.3,
                    0.1,
                ],
            )
            rows.append([unit_id, cycle] + op_settings.tolist() + sensors.tolist())

    df_train = pd.DataFrame(rows, columns=cols)

    # Create RUL labels for test data (last cycle per unit)
    test_rows = []
    rul_labels = []
    n_test_units = 20
    for unit_id in range(1, n_test_units + 1):
        max_cycles = rng.integers(120, 350)
        cutoff = rng.integers(50, max_cycles - 10)
        remaining = max_cycles - cutoff
        rul_labels.append(remaining)
        for cycle in range(1, cutoff + 1):
            degradation = cycle / max_cycles
            op_settings = rng.normal([0.0, 0.0, 100.0], [0.01, 0.0005, 0.01])
            sensors = rng.normal([518 + 10 * degradation] + [500.0] * 20, [2] + [5.0] * 20)
            test_rows.append([unit_id, cycle] + op_settings.tolist() + sensors.tolist())

    df_test = pd.DataFrame(test_rows, columns=cols)
    df_rul = pd.DataFrame({"rul": rul_labels})

    df_train.to_csv(dest_dir / "train_FD001.txt", sep=" ", header=False, index=False)
    df_test.to_csv(dest_dir / "test_FD001.txt", sep=" ", header=False, index=False)
    df_rul.to_csv(dest_dir / "RUL_FD001.txt", sep=" ", header=False, index=False)


def _create_synthetic_ai4i(dest_dir: Path, rng) -> None:
    """Create synthetic AI4I 2020-like data."""
    import numpy as np
    import pandas as pd

    n = 2000
    product_ids = [f"{'LMH'[rng.integers(3)]}{rng.integers(10000, 99999)}" for _ in range(n)]
    types = [pid[0] for pid in product_ids]

    air_temp = rng.normal(300, 2, n)
    process_temp = air_temp + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1500, 70, n)
    torque = rng.normal(40, 10, n)
    tool_wear = rng.integers(0, 250, n)

    # Failure modes
    twf = ((tool_wear > 200) & (rng.random(n) > 0.5)).astype(int)
    hdf = ((process_temp - air_temp > 11) & (rpm < 1400) & (rng.random(n) > 0.5)).astype(int)
    pwf = (((torque * rpm * 2 * 3.14159 / 60) < 3500) & (rng.random(n) > 0.7)).astype(int)
    osf = ((torque > 55) & (tool_wear > 150) & (rng.random(n) > 0.5)).astype(int)
    rnf = (rng.random(n) < 0.005).astype(int)
    failure = np.clip(twf + hdf + pwf + osf + rnf, 0, 1)

    df = pd.DataFrame(
        {
            "UDI": range(1, n + 1),
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Machine failure": failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
        }
    )
    df.to_csv(dest_dir / "ai4i2020.csv", index=False)


def _create_synthetic_ims(dest_dir: Path, rng) -> None:
    """Create synthetic IMS bearing-like data (simplified vibration signals)."""
    import numpy as np

    # IMS has 3 test sets with accelerometer channel files
    # We simulate a simplified version with per-file vibration data
    test_dir = dest_dir / "1st_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    n_files = 50  # Simulate 50 snapshots (real IMS has ~2000+)
    n_samples = 2048  # Samples per snapshot per channel (real is 20480)
    n_channels = 4  # 4 bearing accelerometers

    for file_idx in range(n_files):
        # Simulate degradation in channel 3 (bearing 3 fails in real IMS test 1)
        degradation = max(0, (file_idx - 30) / 20) if file_idx > 30 else 0
        filename = f"2003.10.22.12.{file_idx:02d}.{file_idx:02d}"

        channels = []
        for ch in range(n_channels):
            base_amplitude = 0.5
            if ch == 2 and degradation > 0:  # Bearing 3 degrades
                base_amplitude += degradation * 5.0
            signal = rng.normal(0, base_amplitude, n_samples)
            channels.append(signal)

        data = np.column_stack(channels)
        np.savetxt(test_dir / filename, data, delimiter="\t", fmt="%.6f")


# ============================================================
# CLI Entry Point
# ============================================================


@click.command()
@click.option("--dataset", type=click.Choice(["cmapss", "ims", "ai4i", "all"]), default="all")
@click.option("--data-dir", type=click.Path(), default="data/raw")
@click.option("--force", is_flag=True, help="Re-download even if data exists")
@click.option("--synthetic", is_flag=True, help="Create synthetic fallback data for development")
def main(dataset: str, data_dir: str, force: bool, synthetic: bool) -> None:
    """Download MaintenAIdatasets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        datasets = [dataset] if dataset != "all" else ["cmapss", "ims", "ai4i"]
        for ds in datasets:
            create_synthetic_fallback(ds, raw_dir)
        return

    if dataset == "all":
        download_all(raw_dir, force)
    else:
        download_dataset(dataset, raw_dir, force)


if __name__ == "__main__":
    main()
