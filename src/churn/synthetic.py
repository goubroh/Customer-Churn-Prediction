"""Synthetic data with the same schema as the Kaggle dataset (for tests / offline demos).

This is NOT the real dataset and results on it are meaningless; it only exists so the
pipeline can be exercised without Kaggle credentials.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def make_synthetic(n: int = 1000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1, dtype=float),
            "Age": rng.integers(18, 66, n).astype(float),
            "Gender": rng.choice(["Female", "Male"], n),
            "Tenure": rng.integers(1, 61, n).astype(float),
            "Usage Frequency": rng.integers(1, 31, n).astype(float),
            "Support Calls": rng.integers(0, 11, n).astype(float),
            "Payment Delay": rng.integers(0, 31, n).astype(float),
            "Subscription Type": rng.choice(["Basic", "Standard", "Premium"], n),
            "Contract Length": rng.choice(["Monthly", "Quarterly", "Annual"], n),
            "Total Spend": rng.uniform(100, 1000, n).round(2),
            "Last Interaction": rng.integers(1, 31, n).astype(float),
        }
    )
    logit = (
        0.45 * (df["Support Calls"] - 5)
        + 0.06 * (df["Payment Delay"] - 15)
        - 0.004 * (df["Total Spend"] - 550)
        + (df["Contract Length"] == "Monthly") * 1.0
        + rng.normal(0, 1, n)
    )
    df[config.TARGET] = (logit > 0).astype(float)
    return df
