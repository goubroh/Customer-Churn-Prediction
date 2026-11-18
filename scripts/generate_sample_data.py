"""Write synthetic train/test CSVs (same schema as the Kaggle data) for an offline demo.

Usage:  python scripts/generate_sample_data.py --out data/sample
Then:   churn-train --data-dir data/sample --fast
"""
import argparse
from pathlib import Path

from churn import config
from churn.synthetic import make_synthetic

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=config.DATA_DIR / "sample")
parser.add_argument("--rows", type=int, default=5000)
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
make_synthetic(args.rows, seed=1).to_csv(args.out / config.TRAIN_FILE, index=False)
make_synthetic(args.rows // 5, seed=2).to_csv(args.out / config.TEST_FILE, index=False)
print(f"Synthetic data written to {args.out}")
