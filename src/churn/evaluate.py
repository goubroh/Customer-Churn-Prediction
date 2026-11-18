"""Metrics and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from . import config


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute headline metrics (positive class = churn)."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def importance_frame(pipelines: dict[str, Pipeline]) -> pd.DataFrame:
    """Tidy table of ``feature_importances_`` for every tree-based pipeline given."""
    frame = None
    for name, pipe in pipelines.items():
        clf = pipe.named_steps["classifier"]
        if not hasattr(clf, "feature_importances_"):
            continue
        col = pd.DataFrame({"Feature": feature_names(pipe), name: clf.feature_importances_})
        frame = col if frame is None else frame.merge(col, on="Feature")
    if frame is None:
        raise ValueError("No tree-based models available for feature importance.")
    return frame


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_importance_comparison(frame: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    models = [c for c in frame.columns if c != "Feature"]
    order = frame.sort_values(models[0], ascending=False)["Feature"]
    tidy = frame.melt(id_vars="Feature", var_name="Model", value_name="Importance")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=tidy, x="Importance", y="Feature", hue="Model", order=order,
                palette="muted", ax=ax)
    ax.set_title("Feature Importance Comparison", fontsize=14, fontweight="bold")
    return _save(fig, "feature_importance_comparison.png", out_dir)


def plot_top_features(frame: pd.DataFrame, model: str, k: int = 10,
                      out_dir: Path = config.FIGURES_DIR) -> Path:
    top = frame[["Feature", model]].sort_values(model, ascending=False).head(k)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, x=model, y="Feature", hue="Feature", palette="viridis",
                legend=False, ax=ax)
    ax.set(title=f"Top {k} Churn Drivers ({model})", xlabel="Feature importance")
    return _save(fig, "top10_churn_drivers.png", out_dir)


def plot_roc_curves(pipelines: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
                    title: str, filename: str, out_dir: Path = config.FIGURES_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in pipelines.items():
        fpr, tpr, _ = roc_curve(y, pipe.predict_proba(X)[:, 1])
        auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    return _save(fig, filename, out_dir)


def plot_correlation(df: pd.DataFrame, numerical: list[str], target: str = config.TARGET,
                     out_dir: Path = config.FIGURES_DIR) -> Path:
    corr = df[numerical + [target]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
    return _save(fig, "correlation_matrix.png", out_dir)


def plot_churn_rate(df: pd.DataFrame, by: str, target: str = config.TARGET,
                    out_dir: Path = config.FIGURES_DIR) -> Path:
    rates = df.groupby(by)[target].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=rates.index, y=rates.values, hue=rates.index, palette="rocket",
                legend=False, ax=ax)
    ax.set(title=f"Churn rate by {by}", ylabel="Churn rate", xlabel=by, ylim=(0, 1))
    fname = "churn_rate_by_" + by.lower().replace(" ", "_") + ".png"
    return _save(fig, fname, out_dir)

