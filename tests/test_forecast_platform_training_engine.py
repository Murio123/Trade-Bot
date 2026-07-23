"""C4.2: tests for tools/forecast_platform/model_interface.py and
training_engine.py — exercised entirely with a mock model (predicts the
train-region label mean), so the platform's plumbing is verified without
training any real ridge/LightGBM model (deferred to C4.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.deep_backtest import Fold
from tools.forecast_platform.contracts import ModelMetadata
from tools.forecast_platform.model_interface import ForecastModel
from tools.forecast_platform.training_engine import (pooled_validation_rows,
                                                      run_training)


class MockConstantModel(ForecastModel):
    """Predicts the train-region label mean — enough to prove the training
    loop calls prepare/train/predict/evaluate correctly, without any real
    model logic."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._mean = None
        self._feature_names = None

    def prepare(self, train_df, feature_names, label_col="label"):
        self._feature_names = list(feature_names)
        self._mean = float(train_df[label_col].mean()) if len(train_df) else 0.0

    def train(self):
        pass  # nothing to fit beyond prepare()'s mean

    def predict(self, df):
        return np.full(len(df), self._mean)

    def evaluate(self, df, label_col="label"):
        if len(df) == 0:
            return {"n": 0}
        preds = self.predict(df)
        mae = float(np.mean(np.abs(df[label_col].to_numpy() - preds)))
        return {"n": len(df), "mae": mae}

    def save(self, path):
        pass

    def load(self, path):
        pass

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="mock_constant", code_commit="test", dataset_version="ds",
            feature_version="fv", label_version="lv", seed=self.seed,
            hyperparameters={}, training_window=(0, 0), calibration_window=None,
            evaluation_hash="eval", created_at_utc="2026-01-01T00:00:00+00:00")


def test_forecast_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ForecastModel()  # abstract methods unimplemented


def _synthetic_rows(n=100):
    return pd.DataFrame({"idx": range(n), "label": [i * 0.1 for i in range(n)],
                        "x": range(n)})


def _two_folds():
    # horizon_bars=2 matches the gap baked into these folds below.
    fold0 = Fold(index=0, train_lo=0, train_hi=40, val_lo=42, val_hi=60,
                purge_bars=2, embargo_bars=0)
    fold1 = Fold(index=1, train_lo=20, train_hi=60, val_lo=62, val_hi=80,
                purge_bars=2, embargo_bars=0)
    return [fold0, fold1]


def test_run_training_calls_prepare_train_predict_evaluate_per_fold():
    df = _synthetic_rows()
    folds = _two_folds()
    result = run_training(df, folds, MockConstantModel, ["x"], horizon_bars=2,
                          min_val_rows=5)
    assert len(result.fold_results) == 2
    for fr in result.fold_results:
        assert fr.n_val > 0
        assert "mae" in fr.val_metrics
        assert fr.confident  # 18 val rows per fold >= min_val_rows=5


def test_run_training_uses_a_fresh_model_instance_per_fold():
    df = _synthetic_rows()
    folds = _two_folds()
    result = run_training(df, folds, MockConstantModel, ["x"], horizon_bars=2)
    models = [fr.model for fr in result.fold_results]
    assert models[0] is not models[1]
    # fold 0's train mean must differ from fold 1's (different train regions)
    assert models[0]._mean != models[1]._mean


def test_run_training_respects_min_val_rows_confidence_threshold():
    df = _synthetic_rows()
    folds = _two_folds()
    result = run_training(df, folds, MockConstantModel, ["x"], horizon_bars=2,
                          min_val_rows=1000)
    assert all(not fr.confident for fr in result.fold_results)


def test_pooled_validation_rows_only_contains_rows_inside_some_fold_val_window():
    df = _synthetic_rows()
    folds = _two_folds()
    pooled = pooled_validation_rows(df, folds, horizon_bars=2, min_val_rows=1)
    val_windows = [(f.val_lo, f.val_hi) for f in folds]
    for idx in pooled["idx"]:
        assert any(lo <= idx < hi for lo, hi in val_windows)


def test_pooled_validation_rows_excludes_non_confident_folds_by_default():
    df = _synthetic_rows()
    folds = _two_folds()
    pooled = pooled_validation_rows(df, folds, horizon_bars=2,
                                    min_val_rows=1000)
    assert len(pooled) == 0
