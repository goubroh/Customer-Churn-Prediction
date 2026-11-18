"""Score new customers with the saved model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config


def predict_frame(df: pd.DataFrame, model_path: Path = config.MODEL_PATH,
                  threshold: float = 0.5) -> pd.DataFrame:
    """Return a copy of ``df`` with ``churn_probability`` and ``churn_prediction`` columns."""
    pipeline = joblib.load(model_path)
    X = df.drop(columns=[config.TARGET], errors="ignore")
    proba = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= threshold).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn for customers in a CSV file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = predict_frame(pd.read_csv(args.input), args.model_path, args.threshold)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output} "
          f"({result['churn_prediction'].mean():.1%} flagged as churn)")


if __name__ == "__main__":
    main()
