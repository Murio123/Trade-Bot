"""C4.2: tests for tools/forecast_platform/evaluation_engine.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.forecast_platform.evaluation_engine import (
    bucket_calibration, constant_mean_baseline, fold_summary, mae,
    persistence_baseline_value, rank_against_random,
    random_constant_baseline, rolling_mean_baseline, rmse, spearman_corr,
    year_summary)


def test_mae_and_rmse_known_values():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]
    assert mae(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(4.0 / 3.0))


def test_spearman_perfect_positive_and_negative_rank_correlation():
    y_true = [1, 2, 3, 4, 5]
    y_pred_same = [10, 20, 30, 40, 50]
    y_pred_inverse = [50, 40, 30, 20, 10]
    assert spearman_corr(y_true, y_pred_same) == pytest.approx(1.0, abs=1e-6)
    assert spearman_corr(y_true, y_pred_inverse) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_none_when_no_variance():
    assert spearman_corr([1, 1, 1], [2, 2, 2]) is None


def test_fold_summary_reports_n_mae_rmse_spearman():
    summary = fold_summary(0, [1.0, 2.0], [1.0, 2.0])
    assert summary["fold"] == 0
    assert summary["n"] == 2
    assert summary["mae"] == 0.0
    assert summary["rmse"] == 0.0
    assert summary["spearman"] == pytest.approx(1.0)


def test_year_summary_groups_by_calendar_year():
    df = pd.DataFrame({
        "prediction_timestamp_utc": ["2024-01-01T00:00:00+00:00",
                                     "2025-06-01T00:00:00+00:00"],
        "label": [1.0, 2.0], "pred": [1.5, 2.5],
    })
    summary = year_summary(df, "pred")
    assert set(summary) == {"2024", "2025"}
    assert summary["2024"]["n"] == 1


def test_bucket_calibration_reports_monotonic_predicted_means():
    rng = np.random.RandomState(0)
    y_pred = np.linspace(0, 1, 100)
    y_true = y_pred + rng.normal(0, 0.01, 100)
    buckets = bucket_calibration(y_true, y_pred, n_buckets=5)
    assert len(buckets) == 5
    means = [b["mean_predicted"] for b in buckets]
    assert means == sorted(means)


def test_persistence_baseline_value_matches_frozen_formula():
    assert persistence_baseline_value(eps=1e-6) == math.log(1.0 + 1e-6)


def test_rolling_mean_baseline_uses_only_trailing_window():
    labels = pd.Series([0.0] * 50 + [10.0] * 10)  # last 10 values are 10.0
    assert rolling_mean_baseline(labels, window=10) == 10.0


def test_constant_mean_baseline_uses_entire_train_region():
    labels = pd.Series([0.0, 10.0])
    assert constant_mean_baseline(labels) == 5.0


def test_random_constant_baseline_is_deterministic_given_seed():
    y_true = np.array([1.0, 2.0, 3.0])
    train_labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r1 = random_constant_baseline(y_true, train_labels, n_draws=50, seed=7)
    r2 = random_constant_baseline(y_true, train_labels, n_draws=50, seed=7)
    assert r1["draws"] == r2["draws"]
    assert r1["n_draws"] == 50


def test_rank_against_random_uses_percentile_rank_semantics():
    # model_mae lower than every random draw -> percentile_rank == 0.0
    assert rank_against_random(0.0, [1.0, 2.0, 3.0]) == 0.0
    # model_mae higher than every random draw -> percentile_rank == 1.0
    assert rank_against_random(10.0, [1.0, 2.0, 3.0]) == 1.0
