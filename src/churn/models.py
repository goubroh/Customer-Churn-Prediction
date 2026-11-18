"""Model zoo and pipeline construction."""
from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config


def get_models(fast: bool = False) -> dict[str, ClassifierMixin]:
    """Return the candidate classifiers. ``fast=True`` shrinks them for tests/smoke runs."""
    n = 10 if fast else 100
    seed = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=n, random_state=seed, eval_metric="logloss"),
    }


def build_pipeline(preprocessor, model: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])
