"""M01 tests: the deflated Sharpe ratio.

Covers the cases G1_SPEC.md §3 and the G1 task list require: IID zero-mean
noise, a positive-signal series, skewed and high-kurtosis series, monotonicity in
the trial count, degenerate input, and tiny samples failing closed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from validation.deflated_sharpe import (DSR_VERSION, MIN_OBSERVATIONS,
                                        DeflatedSharpeError, InsufficientData,
                                        deflated_sharpe, expected_max_sharpe,
                                        min_track_record_length, norm_cdf,
                                        norm_ppf, pbo, probabilistic_sharpe,
                                        sampling_sharpe_std, sharpe_stats)


def rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- the normal distribution helpers ----------------------------------------

def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-9)
    assert norm_cdf(-1.959963984540054) == pytest.approx(0.025, abs=1e-9)


def test_norm_ppf_is_the_inverse_of_norm_cdf():
    for p in (1e-6, 0.001, 0.02, 0.1, 0.5, 0.9, 0.975, 0.999, 1 - 1e-6):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-12)


def test_norm_ppf_refuses_the_infinite_tails():
    # Returning ±inf would propagate into an infinite expected maximum Sharpe.
    for p in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(DeflatedSharpeError):
            norm_ppf(p)


# --- moments ----------------------------------------------------------------

def test_sharpe_stats_on_a_known_series():
    x = np.array([1.0, 2.0, 3.0, 4.0] * 10)
    stats = sharpe_stats(x)
    assert stats.n == 40
    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(np.std(x, ddof=1))
    assert stats.sharpe == pytest.approx(2.5 / np.std(x, ddof=1))


def test_kurtosis_convention_is_non_excess():
    """A Gaussian must come out near 3.0, not near 0.0.

    The PSR variance term uses `(g4 - 1) / 4`; the excess convention would make
    that term too small and every p-value too confident.
    """
    x = rng(1).normal(size=200_000)
    assert sharpe_stats(x).kurtosis == pytest.approx(3.0, abs=0.05)


def test_skew_sign_follows_the_distribution():
    right = rng(2).exponential(size=20_000)
    assert sharpe_stats(right).skew > 1.0
    assert sharpe_stats(-right).skew < -1.0


# --- fail-closed ------------------------------------------------------------

def test_tiny_sample_fails_closed():
    with pytest.raises(InsufficientData):
        sharpe_stats(np.arange(MIN_OBSERVATIONS - 1, dtype=float))


def test_the_floor_is_exactly_min_observations():
    ok = rng(3).normal(size=MIN_OBSERVATIONS)
    assert sharpe_stats(ok).n == MIN_OBSERVATIONS


def test_constant_input_fails_closed_rather_than_returning_zero():
    """A zero-variance series has an undefined Sharpe. Returning 0.0 would let a
    caller treat it as a measured absence of skill.

    Regression: the first implementation guarded on `std <= 0.0`, and a constant
    array's std is 1.1e-16 rather than 0.0 in floating point, so this input
    produced a Sharpe of 3.6e15 instead of raising.
    """
    for value in (0.0, 0.4, -12.5, 1e6):
        with pytest.raises(InsufficientData):
            sharpe_stats(np.full(100, value))


def test_a_series_with_real_but_tiny_variance_is_still_measured():
    """The degeneracy guard is scale-relative, so it must not swallow a series
    that is genuinely small rather than genuinely constant."""
    x = rng(20).normal(0.0, 1e-9, size=200)
    assert sharpe_stats(x).std > 0.0


def test_empty_and_non_finite_input():
    with pytest.raises(InsufficientData):
        sharpe_stats([])
    with pytest.raises(DeflatedSharpeError):
        sharpe_stats([1.0, np.nan] * 50)


# --- the deflation itself ----------------------------------------------------

def test_iid_zero_mean_noise_is_not_significant():
    x = rng(4).normal(0.0, 1.0, size=500)
    res = deflated_sharpe(x, n_trials=1)
    assert not res.significant
    assert 0.0 < res.dsr < 1.0
    assert res.p_value == pytest.approx(1.0 - res.dsr)
    assert res.version == DSR_VERSION


def test_iid_noise_dsr_is_uniform_so_the_false_positive_rate_is_nominal():
    """The calibration property the whole suite rests on: for a mean-zero series
    the DSR is uniform, so P(dsr >= 0.95) is about 0.05."""
    hits = 0
    reps = 400
    for i in range(reps):
        x = np.random.default_rng([99, i]).normal(size=250)
        hits += deflated_sharpe(x, n_trials=1).significant
    # Binomial(400, 0.05) has SE 0.0109; 4 SE is a generous envelope for a test
    # that must not be flaky, and still fails a badly miscalibrated statistic.
    assert hits / reps < 0.05 + 4 * 0.0109


def test_a_strong_positive_signal_is_significant():
    x = rng(5).normal(0.5, 1.0, size=500)
    assert deflated_sharpe(x, n_trials=1).significant


def test_skewed_series_is_penalised_relative_to_a_symmetric_one():
    """Negative skew makes a Sharpe less trustworthy, and the variance term is
    what says so. Same n and same Sharpe, worse skew, lower confidence."""
    n, target = 4000, 0.05

    def standardised_to(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std(ddof=1) + target

    sym = sharpe_stats(standardised_to(rng(6).normal(size=n)))
    left = sharpe_stats(standardised_to(-rng(7).exponential(size=n)))

    assert sym.skew == pytest.approx(0.0, abs=0.15)
    assert left.skew < -1.0
    assert left.sharpe == pytest.approx(sym.sharpe, abs=1e-9)
    assert probabilistic_sharpe(left) < probabilistic_sharpe(sym)


def test_high_kurtosis_lowers_confidence_at_an_equal_sharpe():
    n = 4000
    thin = rng(8).normal(size=n)
    fat = rng(8).standard_t(3, size=n)
    target = 0.06

    def shifted(x):
        z = (x - x.mean()) / x.std(ddof=1)
        return z + target

    thin_stats, fat_stats = sharpe_stats(shifted(thin)), sharpe_stats(shifted(fat))
    assert thin_stats.sharpe == pytest.approx(fat_stats.sharpe, abs=1e-9)
    assert fat_stats.kurtosis > thin_stats.kurtosis + 2.0
    assert probabilistic_sharpe(fat_stats) < probabilistic_sharpe(thin_stats)


def test_more_trials_lowers_confidence_monotonically():
    x = rng(9).normal(0.15, 1.0, size=800)
    dsrs = [deflated_sharpe(x, n_trials=n).dsr
            for n in (1, 2, 5, 10, 50, 200, 1000)]
    assert all(a >= b for a, b in zip(dsrs, dsrs[1:])), dsrs
    assert dsrs[0] > dsrs[-1]


def test_a_single_trial_has_no_selection_penalty():
    x = rng(10).normal(0.1, 1.0, size=300)
    res = deflated_sharpe(x, n_trials=1)
    assert res.expected_max_sharpe == 0.0
    assert res.dsr == pytest.approx(res.probabilistic_sharpe)


def test_expected_max_sharpe_grows_with_trials_and_with_dispersion():
    assert expected_max_sharpe(1, 0.1) == 0.0
    rising = [expected_max_sharpe(n, 0.1) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(rising, rising[1:]))
    assert expected_max_sharpe(100, 0.2) > expected_max_sharpe(100, 0.1)
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_expected_max_sharpe_validates_its_inputs():
    with pytest.raises(DeflatedSharpeError):
        expected_max_sharpe(0, 0.1)
    with pytest.raises(DeflatedSharpeError):
        expected_max_sharpe(10, -0.1)


def test_measured_trial_dispersion_beats_the_fallback_and_is_recorded():
    x = rng(11).normal(0.1, 1.0, size=500)
    fallback = deflated_sharpe(x, n_trials=20)
    assert fallback.sharpe_std_source == "sampling_noise_fallback"

    # A genuinely dispersed set of trials implies a higher expected maximum and
    # therefore a lower DSR -- the fallback under-deflates, as documented.
    measured = deflated_sharpe(x, n_trials=20,
                               trial_sharpes=rng(12).normal(0, 0.3, size=20))
    assert measured.sharpe_std_source == "measured_trial_dispersion"
    assert measured.sharpe_std > fallback.sharpe_std
    assert measured.dsr < fallback.dsr


def test_trial_sharpes_needs_at_least_two_trials():
    x = rng(13).normal(size=100)
    with pytest.raises(DeflatedSharpeError):
        deflated_sharpe(x, n_trials=5, trial_sharpes=[0.2])


def test_sampling_sharpe_std_matches_the_closed_form():
    x = rng(14).normal(size=600)
    stats = sharpe_stats(x)
    term = 1.0 - stats.skew * stats.sharpe + (stats.kurtosis - 1.0) / 4.0 * stats.sharpe ** 2
    assert sampling_sharpe_std(stats) == pytest.approx(
        math.sqrt(term / (stats.n - 1)))


def test_deterministic():
    x = rng(15).normal(size=300)
    a = deflated_sharpe(x, n_trials=17)
    b = deflated_sharpe(x, n_trials=17)
    assert a.as_dict() == b.as_dict()


# --- min track record length -------------------------------------------------

def test_min_track_record_length_is_infinite_without_an_excess():
    x = rng(16).normal(0.0, 1.0, size=500)
    stats = sharpe_stats(x)
    assert min_track_record_length(stats, benchmark=stats.sharpe) == math.inf
    assert min_track_record_length(stats, benchmark=stats.sharpe + 1.0) == math.inf


def test_min_track_record_length_shrinks_as_the_edge_grows():
    strong = sharpe_stats(rng(17).normal(0.3, 1.0, size=500))
    weak = sharpe_stats(rng(17).normal(0.05, 1.0, size=500))
    assert min_track_record_length(strong) < min_track_record_length(weak)
    assert min_track_record_length(strong) > 1.0


def test_min_track_record_length_validates_alpha():
    stats = sharpe_stats(rng(18).normal(0.2, 1.0, size=100))
    with pytest.raises(DeflatedSharpeError):
        min_track_record_length(stats, alpha=0.0)


# --- PBO ---------------------------------------------------------------------

def test_pbo_is_one_when_the_in_sample_winner_always_loses_out_of_sample():
    is_m = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    oos_m = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    assert pbo(is_m, oos_m) == 1.0


def test_pbo_is_zero_when_the_winner_holds_up():
    is_m = np.array([[1.0, 0.0]] * 4)
    oos_m = np.array([[1.0, 0.0]] * 4)
    assert pbo(is_m, oos_m) == 0.0


def test_pbo_on_noise_is_near_one_half():
    r = rng(19)
    is_m = r.normal(size=(400, 8))
    oos_m = r.normal(size=(400, 8))
    assert 0.35 < pbo(is_m, oos_m) < 0.65


def test_pbo_validates_shapes():
    with pytest.raises(DeflatedSharpeError):
        pbo(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(DeflatedSharpeError):
        pbo(np.zeros((3, 1)), np.zeros((3, 1)))
    with pytest.raises(DeflatedSharpeError):
        pbo(np.zeros(3), np.zeros(3))


# --- the no-annualization invariant ------------------------------------------

def test_no_annualization_constant_appears_in_the_module():
    """G1_SPEC.md §2 freezes the per-observation convention. A sqrt(252) or
    sqrt(365) creeping in later would silently rescale every published Sharpe.

    Checked against the numeric literals in the parsed AST, not against the file
    text: the docstring says "no `sqrt(252)`" out loud, and a text scan would
    either flag that prose or be loosened until it flagged nothing.
    """
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "validation"
                      / "deflated_sharpe.py").read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)}
    for forbidden in (252, 365, 8760, 12, 24):
        assert forbidden not in literals, (
            f"{forbidden} appears as a literal; if it is an annualization "
            "factor it violates the frozen per-observation convention")
