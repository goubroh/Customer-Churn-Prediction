import numpy as np
import pandas as pd
import pytest

from churn import config, data, features, models, predict, train
from churn.synthetic import make_synthetic


@pytest.fixture(scope="module")
def df():
    return make_synthetic(600)


def test_clean_drops_empty_and_unlabeled_rows(df):
    dirty = pd.concat([df, pd.DataFrame([{c: np.nan for c in df.columns}])], ignore_index=True)
    dirty.loc[0, config.TARGET] = np.nan
    cleaned = data.clean(dirty)
    assert len(cleaned) == len(df) - 1
    assert cleaned[config.TARGET].dtype.kind == "i"


def test_infer_columns_excludes_id_and_target(df):
    X, _ = features.split_xy(df)
    num, cat = features.infer_columns(X)
    assert config.ID_COL not in num + cat
    assert config.TARGET not in num + cat
    assert set(cat) == {"Gender", "Subscription Type", "Contract Length"}
    assert "Support Calls" in num


def test_preprocessor_output_shape(df):
    X, _ = features.split_xy(df)
    num, cat = features.infer_columns(X)
    out = features.build_preprocessor(num, cat).fit_transform(X)
    assert out.shape == (len(X), len(num) + 2 + 3 + 3)
    assert not np.isnan(out).any()


def test_pipeline_handles_unseen_category_and_missing(df):
    X, y = features.split_xy(df)
    num, cat = features.infer_columns(X)
    pipe = models.build_pipeline(
        features.build_preprocessor(num, cat), models.get_models(fast=True)["Random Forest"]
    ).fit(X, y)
    new = X.head(5).copy()
    new.loc[new.index[0], "Contract Length"] = "Biennial"
    new.loc[new.index[1], "Age"] = np.nan
    proba = pipe.predict_proba(new)[:, 1]
    assert proba.shape == (5,) and ((proba >= 0) & (proba <= 1)).all()


def test_end_to_end_train_and_predict(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_synthetic(1500, seed=1).to_csv(raw / config.TRAIN_FILE, index=False)
    make_synthetic(400, seed=2).to_csv(raw / config.TEST_FILE, index=False)

    res = train.run(
        data_dir=raw,
        model_path=tmp_path / "m.joblib",
        metrics_path=tmp_path / "metrics.json",
        figures_dir=tmp_path / "figs",
        fast=True,
    )
    assert res["best_model"] in res["validation"]
    assert res["validation"][res["best_model"]]["roc_auc"] > 0.7
    assert (tmp_path / "metrics.json").exists()
    assert len(list((tmp_path / "figs").glob("*.png"))) >= 3

    out = predict.predict_frame(make_synthetic(50, seed=3), tmp_path / "m.joblib")
    assert {"churn_probability", "churn_prediction"} <= set(out.columns)
    assert len(out) == 50


def test_load_raw_missing_files_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn-download"):
        data.load_raw(tmp_path)
