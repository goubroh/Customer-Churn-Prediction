"""Central configuration: paths, dataset names and constants."""
from __future__ import annotations

import os
from pathlib import Path

# Project root (override with CHURN_HOME if the package is installed non-editable)
ROOT = Path(os.environ.get("CHURN_HOME", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_DATASET = "muhammadshahidazeem/customer-churn-dataset"
TRAIN_FILE = "customer_churn_dataset-training-master.csv"
TEST_FILE = "customer_churn_dataset-testing-master.csv"

TARGET = "Churn"
ID_COL = "CustomerID"

RANDOM_STATE = 42
VAL_SIZE = 0.2

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
