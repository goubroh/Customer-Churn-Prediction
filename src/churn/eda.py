"""Exploratory plots: correlation matrix and churn rate by contract length."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config, data, evaluate, features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures.")
    parser.add_argument("--data-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.FIGURES_DIR)
    args = parser.parse_args()

    train, _ = data.load_raw(args.data_dir)
    train = data.clean(train)
    X, _y = features.split_xy(train)
    numerical, _ = features.infer_columns(X)

    print("Class distribution:\n", train[config.TARGET].value_counts(normalize=True).round(4))
    print("Churn rate by Contract Length:\n",
          train.groupby("Contract Length")[config.TARGET].agg(["count", "mean"]))
    print("Saved:", evaluate.plot_correlation(train, numerical, out_dir=args.out_dir))
    print("Saved:", evaluate.plot_churn_rate(train, "Contract Length", out_dir=args.out_dir))


if __name__ == "__main__":
    main()
