"""C4.2: tests for tools/forecast_platform/calibration_engine.py."""
from __future__ import annotations

import numpy as np

from tools.forecast_platform.calibration_engine import (
    apply_isotonic, apply_platt, brier_score, expected_calibration_error,
    fit_isotonic, fit_platt, reliability_curve)


def test_brier_score_zero_for_perfect_predictions():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == 0.0


def test_brier_score_matches_hand_computation():
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    expected = np.mean([(0.8 - 1.0) ** 2, (0.3 - 0.0) ** 2])
    assert brier_score(y, p) == expected


def test_reliability_curve_bins_cover_zero_to_one():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.05, 0.95, 0.15, 0.55, 0.85])
    curve = reliability_curve(y, p, n_bins=5)
    assert len(curve) == 5
    assert sum(b["n"] for b in curve) == len(y)


def test_ece_is_zero_when_predictions_are_perfectly_calibrated():
    # Every bin's mean_predicted equals mean_actual by construction.
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, p, n_bins=2) == 0.0


def test_ece_is_positive_when_miscalibrated():
    y = np.array([1.0, 1.0, 1.0, 1.0])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    assert expected_calibration_error(y, p, n_bins=2) > 0.5


def test_fit_platt_recovers_a_strong_monotonic_relationship():
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 3, 500)
    p_true = 1.0 / (1.0 + np.exp(-2.0 * x))
    y = (rng.uniform(size=500) < p_true).astype(float)
    a, b = fit_platt(x, y)
    calibrated = apply_platt(x, a, b)
    # calibrated probabilities must be strongly rank-correlated with x
    assert np.corrcoef(x, calibrated)[0, 1] > 0.8
    assert a > 0  # recovers the known positive relationship


def test_fit_isotonic_output_is_non_decreasing():
    rng = np.random.RandomState(1)
    scores = rng.uniform(0, 1, 200)
    y = (scores + rng.normal(0, 0.05, 200) > 0.5).astype(float)
    fitted_x, fitted_y = fit_isotonic(scores, y)
    assert np.all(np.diff(fitted_y) >= -1e-9)  # non-decreasing (allow fp noise)
    assert np.all(np.diff(fitted_x) >= 0)


def test_apply_isotonic_interpolates_between_fitted_points():
    fitted_x = np.array([0.0, 1.0, 2.0])
    fitted_y = np.array([0.0, 0.5, 1.0])
    out = apply_isotonic(np.array([0.5, 1.5]), fitted_x, fitted_y)
    assert out[0] == 0.25
    assert out[1] == 0.75
