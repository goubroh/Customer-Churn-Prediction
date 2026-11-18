"""Dataset download, loading and cleaning."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)


def download_dataset(dest: Path = config.RAW_DIR) -> Path:
    """Download and unzip the Kaggle dataset into ``dest``.

    Requires Kaggle credentials: either ``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: needs credentials

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    log.info("Downloading %s -> %s", config.KAGGLE_DATASET, dest)
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(dest), unzip=True)
    return dest


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download the churn dataset from Kaggle.")
    parser.add_argument("--dest", type=Path, default=config.RAW_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Dataset saved to {download_dataset(args.dest)}")


def load_raw(data_dir: Path = config.RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official train and test CSV files."""
    data_dir = Path(data_dir)
    train_path = data_dir / config.TRAIN_FILE
    test_path = data_dir / config.TEST_FILE
    missing = [p.name for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Run `churn-download` "
            "(needs Kaggle credentials) or place the CSV files there manually."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def clean(df: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """Drop empty rows / rows without a label, remove duplicates, cast target to int."""
    out = df.dropna(how="all")
    if target in out.columns:
        out = out.dropna(subset=[target])
        out = out.assign(**{target: out[target].astype(int)})
    return out.drop_duplicates().reset_index(drop=True)
