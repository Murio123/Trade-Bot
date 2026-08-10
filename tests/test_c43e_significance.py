"""C4.3e: the significance machinery for Ridge's ranking advantage.

The rows are 12-bar overlapping windows, so the danger here is a test that
treats 4062 correlated rows as 4062 independent ones and declares almost
anything significant. These tests pin the two defences: the statistic credits
the anti-correlated baseline with its invertible skill, and every resampling
scheme is block-based.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import ridge_ranking_significance as rs


def _linked(n=600, seed=0):
    """Labels with real autocorrelation, plus an informative and a useless
    predictor."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    y = np.convolve(noise, np.ones(12) / 12, mode="same")  # overlapping windows
    informative = y + rng.normal(scale=0.5, size=n)
    useless = rng.normal(size=n)
    return y, informative, useless


def test_advantage_uses_absolute_rho_so_an_inverted_baseline_gets_credit():
    """An anti-correlated baseline is trivially invertible. Scoring it on the
    signed rho would hand Ridge an advantage it has not earned."""
    y, good, _ = _linked()
    mirrored = -good  # perfectly anti-correlated: same skill, inverted

    assert rs.advantage(y, good, mirrored) == pytest.approx(0.0, abs=1e-12)

    from tools.forecast_platform.evaluation_engine import spearman_corr
    signed = spearman_corr(y, good) - spearman_corr(y, mirrored)
    assert signed > 0.5, "the signed statistic would have claimed a big edge"


def test_advantage_is_positive_only_when_the_model_ranks_better():
    y, good, useless = _linked()
    assert rs.advantage(y, good, useless) > 0
    assert rs.advantage(y, useless, good) < 0


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    y, good, useless = _linked()
    a = rs.block_bootstrap_ci(y, good, useless, block=12, n_resamples=200)
    b = rs.block_bootstrap_ci(y, good, useless, block=12, n_resamples=200)
    assert a == b


def test_bootstrap_preserves_row_pairing():
    """If the resample broke the label/prediction pairing, a genuinely
    informative predictor would stop looking informative."""
    y, good, useless = _linked()
    out = rs.block_bootstrap_ci(y, good, useless, block=24, n_resamples=300)
    assert out["ci_lo"] > 0
    assert out["ci_lo"] < out["ci_hi"]


def test_wider_blocks_do_not_shrink_the_interval():
    """Honouring more autocorrelation must not make the test MORE confident;
    a scheme that tightened with block length would be the bug this whole
    module exists to avoid."""
    y, good, useless = _linked(n=1200)
    widths = []
    for block in (12, 60, 120):
        out = rs.block_bootstrap_ci(y, good, useless, block=block, n_resamples=300)
        widths.append(out["ci_hi"] - out["ci_lo"])
    assert widths[-1] > widths[0] * 0.7, widths


def test_permutation_does_not_reject_for_a_useless_predictor():
    y, _, useless = _linked()
    other = np.random.default_rng(7).normal(size=len(y))
    out = rs.block_permutation_p(y, useless, other, block=12, n_resamples=300)
    assert out["p_value"] > 0.05


def test_permutation_p_is_never_zero():
    """Davison-Hinkley +1/+1: a p of exactly 0 is not a possible result."""
    y, good, useless = _linked()
    out = rs.block_permutation_p(y, good, useless, block=12, n_resamples=100)
    assert out["p_value"] > 0


def test_effective_sample_is_reported_far_below_the_row_count():
    y, _, _ = _linked(n=4000)
    eff = rs.effective_sample_size(y, horizon=12)
    assert eff["n_rows"] == 4000
    assert eff["naive_independent"] == 333
    assert eff["label_autocorr_lag1"] > 0.5


def test_decision_rule_requires_every_block_length():
    ok_boot = [{"ci_lo": 0.05, "block": b} for b in rs.BLOCK_LENGTHS]
    ok_perm = [{"p_value": 0.001, "block": b} for b in rs.BLOCK_LENGTHS]
    assert rs.decide_significance(ok_boot, ok_perm)[0] == "ADVANTAGE_SIGNIFICANT"

    # one block length failing the CI is enough to reject
    bad_boot = list(ok_boot)
    bad_boot[-1] = {"ci_lo": -0.01, "block": 120}
    verdict, checks = rs.decide_significance(bad_boot, ok_perm)
    assert verdict == "ADVANTAGE_NOT_SIGNIFICANT"
    assert checks["ci_lower_positive_at_every_block"] is False

    # one block length failing the permutation is enough too
    bad_perm = list(ok_perm)
    bad_perm[0] = {"p_value": 0.20, "block": 12}
    verdict, checks = rs.decide_significance(ok_boot, bad_perm)
    assert verdict == "ADVANTAGE_NOT_SIGNIFICANT"
    assert checks["worst_p_value"] == 0.20


def test_a_ci_lower_bound_of_exactly_zero_does_not_pass():
    boot = [{"ci_lo": 0.0, "block": b} for b in rs.BLOCK_LENGTHS]
    perm = [{"p_value": 0.001, "block": b} for b in rs.BLOCK_LENGTHS]
    assert rs.decide_significance(boot, perm)[0] == "ADVANTAGE_NOT_SIGNIFICANT"


def test_empty_inputs_cannot_pass():
    assert rs.decide_significance([], [])[0] == "ADVANTAGE_NOT_SIGNIFICANT"


def test_forward_requirement_scales_back_up_for_overlap():
    """The ledger records overlapping forecasts, so the row count needed is
    the effective count times the overlap factor."""
    out = rs.required_n_for_power(0.13, eff_n=338, n_rows=4062)
    assert out["applicable"]
    assert out["effective_obs_needed"] > 200
    assert out["matured_forecasts_needed"] > out["effective_obs_needed"] * 10


def test_forward_requirement_declines_a_non_positive_advantage():
    assert rs.required_n_for_power(0.0, 338, 4062)["applicable"] is False
    assert rs.required_n_for_power(-0.1, 338, 4062)["applicable"] is False


def test_rule_text_is_frozen_and_names_its_defences():
    for token in ("EVERY tested block length", "|rho_ridge|", "12-bar label horizon",
                  "BEFORE the test was run"):
        assert token in rs.SIGNIFICANCE_RULE
