"""C4.3d: the point-in-time trailing-mean baseline.

The whole reason this baseline exists is that the frozen rolling_mean_60 is a
per-fold CONSTANT, so it cannot rank anything inside a fold and Ridge's
Spearman "win" over it is vacuous. These tests pin the two properties that
make the replacement a fair opponent instead: it varies row by row, and it
never sees a label that was not fully observable at the query bar.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.forecast_platform import evaluation_engine as ee


def _series(n: int = 200, start: int = 0):
    idx = np.arange(start, start + n)
    labels = idx.astype(float)  # label == idx makes the expected mean exact
    return idx, labels


def test_output_is_one_value_per_query_row_and_varies():
    idx, labels = _series()
    q = np.arange(100, 140)
    out = ee.rolling_mean_series_baseline(idx, labels, q, window=10,
                                          horizon_bars=12)
    assert out.shape == q.shape
    assert len(np.unique(out)) == len(q), "baseline collapsed to a constant"
    assert np.all(np.diff(out) > 0), "trailing mean must track a rising series"


def test_window_ends_exactly_at_t_minus_horizon():
    """label[u] needs bars u+1..u+H, so it is knowable only at u+H."""
    idx, labels = _series()
    out = ee.rolling_mean_series_baseline(idx, labels, [100], window=10,
                                          horizon_bars=12)
    # Observable: u <= 88. Last 10 of those are 79..88, mean 83.5.
    assert out[0] == pytest.approx(83.5)


def test_a_label_one_bar_too_recent_is_excluded():
    """Direct anti-lookahead check: poison the first non-observable label and
    require the result not to move."""
    idx, labels = _series()
    clean = ee.rolling_mean_series_baseline(idx, labels, [100], window=10,
                                            horizon_bars=12)
    poisoned = labels.copy()
    poisoned[89] = 10_000.0          # idx 89 == t-11, one bar too recent
    after = ee.rolling_mean_series_baseline(idx, poisoned, [100], window=10,
                                            horizon_bars=12)
    assert after[0] == pytest.approx(clean[0])

    # ...while the newest ADMISSIBLE label (idx 88 == t-12) must move it.
    boundary = labels.copy()
    boundary[88] = 10_000.0
    moved = ee.rolling_mean_series_baseline(idx, boundary, [100], window=10,
                                            horizon_bars=12)
    assert moved[0] > clean[0] * 10


def test_horizon_zero_would_admit_the_query_bar_itself():
    """Guards the boundary convention: with H=0 the query bar's own label is
    observable, so the off-by-one lives in horizon_bars and nowhere else."""
    idx, labels = _series()
    out = ee.rolling_mean_series_baseline(idx, labels, [100], window=1,
                                          horizon_bars=0)
    assert out[0] == pytest.approx(100.0)


def test_sealed_holdout_rows_are_never_admitted():
    idx, labels = _series(n=400)
    holdout_lo = 200
    q = np.array([380])  # a query deep inside the holdout region
    out = ee.rolling_mean_series_baseline(idx, labels, q, window=5,
                                          horizon_bars=12,
                                          max_source_idx=holdout_lo)
    # Only idx < 200 may contribute: last 5 are 195..199, mean 197.
    assert out[0] == pytest.approx(197.0)


def test_gaps_in_the_index_use_available_rows_not_bar_offsets():
    """The dataset skips bars, so the window is over observed ROWS."""
    idx = np.array([0, 1, 2, 50, 51, 52, 100])
    labels = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0])
    out = ee.rolling_mean_series_baseline(idx, labels, [70], window=3,
                                          horizon_bars=12)
    # Observable at 70: idx <= 58 -> [0,1,2,50,51,52]; last 3 are the 5.0s.
    assert out[0] == pytest.approx(5.0)


def test_short_history_uses_what_exists_and_nan_when_nothing_does():
    idx, labels = _series(n=50)
    out = ee.rolling_mean_series_baseline(idx, labels, [13, 20, 5], window=60,
                                          horizon_bars=12)
    assert out[0] == pytest.approx(0.5)     # idx 0..1 observable
    assert out[1] == pytest.approx(4.0)     # idx 0..8
    assert np.isnan(out[2]), "no observable history must be NaN, not 0"


def test_batching_does_not_change_results():
    idx, labels = _series(n=500)
    q = np.arange(200, 260)
    whole = ee.rolling_mean_series_baseline(idx, labels, q, window=60)
    piecewise = np.concatenate([
        ee.rolling_mean_series_baseline(idx, labels, q[:17], window=60),
        ee.rolling_mean_series_baseline(idx, labels, q[17:], window=60),
    ])
    assert np.array_equal(whole, piecewise)


def test_deterministic_across_repeated_calls():
    idx, labels = _series(n=500)
    q = np.arange(200, 260)
    first = ee.rolling_mean_series_baseline(idx, labels, q, window=60)
    for _ in range(3):
        assert np.array_equal(
            first, ee.rolling_mean_series_baseline(idx, labels, q, window=60))


def test_rejects_malformed_source():
    with pytest.raises(ValueError):
        ee.rolling_mean_series_baseline([0, 1], [1.0], [5])
    with pytest.raises(ValueError):
        ee.rolling_mean_series_baseline([0, 0, 1], [1.0, 2.0, 3.0], [5])
    with pytest.raises(ValueError):
        ee.rolling_mean_series_baseline([2, 1, 0], [1.0, 2.0, 3.0], [5])
    with pytest.raises(ValueError):
        ee.rolling_mean_series_baseline([0, 1], [1.0, 2.0], [5], window=0)


def test_frozen_scalar_baseline_is_untouched():
    """C4.3d must not disturb the frozen C4.1 §9 baselines."""
    import pandas as pd
    labels = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert ee.rolling_mean_baseline(labels, window=2) == pytest.approx(3.5)
    assert ee.constant_mean_baseline(labels) == pytest.approx(2.5)
