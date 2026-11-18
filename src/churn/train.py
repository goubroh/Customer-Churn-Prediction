"""Train, compare and persist churn models.

Protocol
--------
1. Hold out a stratified validation split from the training file.
2. Fit every candidate on the remaining training rows; score on validation AND on the
   official test file.
3. Select the best model by *validation* ROC-AUC (never by test score).
4. Refit the winner on the full training file, score it on the test file, save it.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from . import config, data, evaluate, features, models

log = logging.getLogger("churn.train")


def run(
    data_dir: Path = config.RAW_DIR,
    model_path: Path = config.MODEL_PATH,
    metrics_path: Path = config.METRICS_PATH,
    figures_dir: Path = config.FIGURES_DIR,
    fast: bool = False,
    make_plots: bool = True,
) -> dict:
    train_df, test_df = data.load_raw(data_dir)
    train_df, test_df = data.clean(train_df), data.clean(test_df)
    has_test_labels = config.TARGET in test_df.columns

    X, y = features.split_xy(train_df)
    numerical, categorical = features.infer_columns(X)
    log.info("Numerical: %s | Categorical: %s", numerical, categorical)
    log.info("Train %s | Test %s | churn rate (train) %.3f",
             train_df.shape, test_df.shape, y.mean())

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.VAL_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    if has_test_labels:
        X_test, y_test = features.split_xy(test_df)

    preprocessor = features.build_preprocessor(numerical, categorical)
    fitted: dict = {}
    results: dict = {"validation": {}, "test": {}}

    for name, model in models.get_models(fast=fast).items():
        log.info("Training %s ...", name)
        pipe = models.build_pipeline(clone(preprocessor), model).fit(X_tr, y_tr)
        fitted[name] = pipe
        results["validation"][name] = evaluate.evaluate(pipe, X_val, y_val)
        if has_test_labels:
            results["test"][name] = evaluate.evaluate(pipe, X_test, y_test)

    best = max(results["validation"], key=lambda k: results["validation"][k]["roc_auc"])
    log.info("Best model by validation ROC-AUC: %s", best)

    final = models.build_pipeline(clone(preprocessor), clone(models.get_models(fast)[best]))
    final.fit(X, y)
    results["best_model"] = best
    results["selection_metric"] = "validation roc_auc"
    if has_test_labels:
        results["best_model_refit_test"] = evaluate.evaluate(final, X_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))

    if make_plots:
        imp = evaluate.importance_frame(
            {n: p for n, p in fitted.items() if n in ("Random Forest", "XGBoost")}
        )
        evaluate.plot_importance_comparison(imp, figures_dir)
        top_model = "Random Forest" if "Random Forest" in imp.columns else imp.columns[1]
        evaluate.plot_top_features(imp, top_model, out_dir=figures_dir)
        evaluate.plot_roc_curves(fitted, X_val, y_val, "ROC curves (validation split)",
                                 "roc_validation.png", figures_dir)
        if has_test_labels:
            evaluate.plot_roc_curves(fitted, X_test, y_test, "ROC curves (official test file)",
                                     "roc_test.png", figures_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument("--data-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=config.METRICS_PATH)
    parser.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--fast", action="store_true", help="Tiny models, for smoke tests.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    res = run(args.data_dir, args.model_path, args.metrics_path, args.figures_dir,
              args.fast, not args.no_plots)
    pd.set_option("display.width", 120)
    for split in ("validation", "test"):
        if res[split]:
            print(f"\n=== {split.upper()} ===")
            print(pd.DataFrame(res[split]).T.round(4))
    print(f"\nBest model: {res['best_model']}  ->  saved to {args.model_path}")


if __name__ == "__main__":
    main()
