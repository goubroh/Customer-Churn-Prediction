"""Feature handling: column inference and the preprocessing transformer."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def split_xy(df: pd.DataFrame, target: str = config.TARGET) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target]


def infer_columns(
    X: pd.DataFrame, id_col: str = config.ID_COL
) -> tuple[list[str], list[str]]:
    """Return (numerical, categorical) feature columns, excluding the ID column."""
    cols = [c for c in X.columns if c != id_col]
    numerical = X[cols].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in cols if c not in numerical]
    return numerical, categorical


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    """Impute + scale numeric features, impute + one-hot encode categorical ones."""
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numerical), ("cat", categorical_pipe, categorical)],
        remainder="drop",  # drops CustomerID and anything unexpected
        verbose_feature_names_out=False,
    )
