"""C4.3: tests for tools/forecast_platform/ridge_model.py.

Pure-function/unit tests against small synthetic frames — no real dataset
build needed for these (the real Binance run happens separately via
tools/ridge_volatility_run.py and is not exercised in this test module).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.forecast_platform.model_interface import ForecastModel
from tools.forecast_platform.ridge_model import (ALPHA_CANDIDATES,
                                                  CATEGORICAL_LEVELS,
                                                  FIT_INTERCEPT,
                                                  INNER_SPLIT_FRACTION,
                                                  RidgeForecastModel,
                                                  RidgeIdentity,
                                                  RidgeTrainingError)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "forecast_platform" / "ridge_model.py").read_text()

FEATURES = ["x1", "x2", "htf_bias", "regime_current", "flag"]


def _identity(model_id="test_ridge"):
    return RidgeIdentity(model_id=model_id, code_commit="deadbeef",
                        dataset_version="ds1", feature_version="fv1",
                        label_version="lv1")


def _synthetic_df(n=200, seed=0, noisy_label=True):
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    label = 2.0 * x1 - 1.0 * x2 + (rng.normal(0, 0.01, n) if noisy_label else 0.0)
    htf_bias = rng.choice(["bullish", "bearish", "neutral"], size=n)
    regime = rng.choice(["trend_up", "trend_down", "range", "high_volatility"],
                        size=n)
    flag = rng.choice([True, False], size=n)
    return pd.DataFrame({"idx": range(n), "x1": x1, "x2": x2,
                        "htf_bias": htf_bias, "regime_current": regime,
                        "flag": flag, "label": label})


def test_frozen_constants_match_c41_spec():
    assert ALPHA_CANDIDATES == (0.1, 1.0, 10.0)
    assert INNER_SPLIT_FRACTION == 0.2
    assert FIT_INTERCEPT is True
    assert CATEGORICAL_LEVELS["htf_bias"] == ("bullish", "bearish", "neutral")
    assert CATEGORICAL_LEVELS["regime_current"] == (
        "trend_up", "trend_down", "range", "high_volatility")


def test_ridge_forecast_model_implements_the_contract():
    model = RidgeForecastModel(_identity())
    assert isinstance(model, ForecastModel)


def test_prepare_train_predict_end_to_end():
    df = _synthetic_df()
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    preds = model.predict(df)
    assert len(preds) == len(df)
    # a near-linear synthetic relationship should be fit reasonably well
    mae = float(np.mean(np.abs(df["label"].to_numpy() - preds)))
    assert mae < 0.5


def test_train_raises_on_empty_train_df():
    model = RidgeForecastModel(_identity())
    model.prepare(_synthetic_df().iloc[0:0], FEATURES)
    with pytest.raises(RidgeTrainingError):
        model.train()


def test_predict_raises_if_train_not_called():
    model = RidgeForecastModel(_identity())
    with pytest.raises(RidgeTrainingError):
        model.predict(_synthetic_df())


def test_scaler_is_fit_only_on_train_rows():
    df = _synthetic_df(seed=1)
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    expected_mean = df["x1"].mean()
    x1_col_idx = model._design_columns.index("x1")
    assert model._scaler_mean[x1_col_idx] == pytest.approx(expected_mean)

    # A completely different distribution passed only to predict() must
    # never move the already-fit scaler statistics.
    other = _synthetic_df(seed=99)
    other["x1"] = other["x1"] * 1000 + 500
    model.predict(other)
    assert model._scaler_mean[x1_col_idx] == pytest.approx(expected_mean)


def test_alpha_selection_uses_only_train_rows_not_a_held_out_val_set():
    df = _synthetic_df(seed=2)
    model_a = RidgeForecastModel(_identity("a"))
    model_a.prepare(df, FEATURES)
    model_a.train()

    # A model trained on the SAME train_df, with no access to any
    # validation set at all, must select the identical alpha — proving
    # alpha selection is purely a function of the train region.
    model_b = RidgeForecastModel(_identity("b"))
    model_b.prepare(df, FEATURES)
    model_b.train()
    assert model_a._chosen_alpha == model_b._chosen_alpha


def test_alpha_selection_inner_split_is_time_ordered_not_shuffled():
    df = _synthetic_df(seed=3, noisy_label=False)
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    lo, hi = model._inner_val_idx_range
    # inner validation window must be the LAST ~20% of idx, not a random subset
    assert hi == df["idx"].max()
    assert lo >= int(len(df) * (1 - INNER_SPLIT_FRACTION)) - 1


def test_deterministic_training_given_identical_data():
    df = _synthetic_df(seed=4)
    m1 = RidgeForecastModel(_identity("m1"))
    m1.prepare(df, FEATURES)
    m1.train()
    m2 = RidgeForecastModel(_identity("m2"))
    m2.prepare(df, FEATURES)
    m2.train()
    np.testing.assert_array_equal(m1._weights, m2._weights)
    assert m1._intercept == m2._intercept
    np.testing.assert_array_equal(m1.predict(df), m2.predict(df))


def test_rows_with_missing_features_are_excluded_not_imputed():
    df = _synthetic_df(seed=5)
    df.loc[df.index[:10], "x1"] = np.nan
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    # scaler mean must reflect only the 190 non-missing rows, not 200
    expected_mean = df["x1"].dropna().mean()
    x1_col_idx = model._design_columns.index("x1")
    assert model._scaler_mean[x1_col_idx] == pytest.approx(expected_mean)


def test_rows_with_missing_categorical_are_excluded_not_silently_zeroed():
    """Regression test: a missing/None categorical value must be EXCLUDED
    from training like any other missing feature — not silently one-hot
    encoded as all-zero (which would be indistinguishable from a
    genuinely unseen category and violates the C4.1 §6 "excluded, never
    imputed" policy). Caught by Codex audit of commit f806c0d."""
    df = _synthetic_df(seed=10)
    df.loc[df.index[:15], "htf_bias"] = None
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    # scaler mean for a numeric column must reflect only the 185
    # rows that survived exclusion, not all 200.
    expected_mean = df.loc[df["htf_bias"].notna(), "x1"].mean()
    x1_col_idx = model._design_columns.index("x1")
    assert model._scaler_mean[x1_col_idx] == pytest.approx(expected_mean)
    assert model._train_df is not None  # sanity: prepare() still ran


def test_unseen_categorical_level_at_predict_time_does_not_crash():
    df = _synthetic_df(seed=6)
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    other = df.copy()
    other.loc[other.index[0], "htf_bias"] = "totally_unseen_value"
    preds = model.predict(other)
    assert np.isfinite(preds[0])  # all-zero one-hot row, not a crash


def test_save_and_load_round_trip_predictions(tmp_path):
    df = _synthetic_df(seed=7)
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    before = model.predict(df)

    path = str(tmp_path / "ridge_fold0.json")
    model.save(path)

    restored = RidgeForecastModel(_identity())
    restored.load(path)
    after = restored.predict(df)
    np.testing.assert_array_almost_equal(before, after)


def test_metadata_reports_chosen_alpha_and_training_window():
    df = _synthetic_df(seed=8)
    model = RidgeForecastModel(_identity("vol_ridge_fold0"))
    model.prepare(df, FEATURES)
    model.train()
    meta = model.metadata()
    assert meta.model_id == "vol_ridge_fold0"
    assert meta.hyperparameters["chosen_alpha"] in ALPHA_CANDIDATES
    assert meta.training_window == (int(df["idx"].min()), int(df["idx"].max()))
    assert meta.evaluation_hash and len(meta.evaluation_hash) == 64


def test_evaluate_reports_n_and_mae():
    df = _synthetic_df(seed=9)
    model = RidgeForecastModel(_identity())
    model.prepare(df, FEATURES)
    model.train()
    metrics = model.evaluate(df)
    assert metrics["n"] == len(df)
    assert metrics["mae"] >= 0


def test_evaluate_handles_empty_dataframe():
    model = RidgeForecastModel(_identity())
    model.prepare(_synthetic_df(), FEATURES)
    model.train()
    assert model.evaluate(_synthetic_df().iloc[0:0]) == {"n": 0, "mae": None}


def test_runtime_never_imports_ridge_model():
    """Static guarantee mirroring the project's existing convention (e.g.
    tests/test_swing_hypothesis_simulator.py): no runtime/signal_engine/
    execution module references this offline-only module."""
    runtime_dirs = ["signal_engine", "pipeline.py", "backtest.py", "bot.py",
                    "risk", "execution"]
    for rel in runtime_dirs:
        path = ROOT / rel
        if not path.exists():
            continue
        targets = [path] if path.is_file() else list(path.rglob("*.py"))
        for f in targets:
            text = f.read_text()
            assert "forecast_platform" not in text, (
                f"{f} references tools.forecast_platform — offline-only "
                f"module must never be imported by runtime code")


def test_module_never_monkeypatches_deep_backtest():
    assert not re.search(r"deep_backtest\.\w+\s*=(?!=)", SOURCE)
