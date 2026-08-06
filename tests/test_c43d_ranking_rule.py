"""C4.3d: the frozen ranking decision rule and its boundaries.

The rule is applied mechanically. These tests exist so that a later reading
of a real result cannot quietly soften a threshold: every gate is pinned at
the value it passes and the value it fails.
"""
from __future__ import annotations

import pytest

from tools.ridge_volatility_run import (RANKING_DECISION_RULE,
                                        decide_ranking_verdict)

CONFIRMED = "RIDGE_RANKING_VALUE_CONFIRMED"
REJECTED = "RIDGE_RANKING_VALUE_REJECTED"
REPRO_FAIL = "REPRODUCTION_FAILURE"


def _passing(**over):
    """A result that clears every gate; individual tests break one at a time."""
    kwargs = dict(
        fold_advantages=[0.05, 0.05, 0.05],   # 3/3 positive, evenly split
        pooled_advantage=0.04,
        year_advantages=[0.03, 0.03, 0.03],
        holdout_rows_used=0,
        reproduced=True,
    )
    kwargs.update(over)
    return decide_ranking_verdict(**kwargs)


def test_all_gates_pass_confirms():
    verdict, checks = _passing()
    assert verdict == CONFIRMED
    assert all(checks[k] for k in ("fold_majority", "pooled_higher",
                                   "year_independence",
                                   "no_single_fold_dominance",
                                   "no_holdout_rows"))


def test_reproduction_failure_overrides_everything():
    verdict, checks = _passing(reproduced=False)
    assert verdict == REPRO_FAIL
    assert checks == {"reproduced": False}


def test_fold_majority_boundary():
    # exactly 2 of 3 positive -> passes
    assert _passing(fold_advantages=[0.05, 0.05, -0.01])[0] == CONFIRMED
    # only 1 of 3 -> fails
    verdict, checks = _passing(fold_advantages=[0.05, -0.01, -0.01])
    assert verdict == REJECTED and checks["fold_majority"] is False


def test_fold_gate_fails_closed_when_the_fold_count_is_not_three():
    """Codex audit finding: a bare >=2 is not "2 of the 3". With five folds
    two positives would be a MINORITY, so the rule as written does not apply
    and must fail rather than be silently reinterpreted."""
    verdict, checks = _passing(fold_advantages=[0.05, 0.05, 0.05, -0.01, -0.01])
    assert verdict == REJECTED and checks["fold_majority"] is False
    assert checks["n_confident_folds"] == 5

    verdict, checks = _passing(fold_advantages=[0.05, 0.05])
    assert verdict == REJECTED and checks["fold_majority"] is False


def test_two_qualifying_years_require_both_positive():
    """The two-thirds ratio is STRICTER than 2-of-3 when only two years
    qualify — one positive year out of two must not pass."""
    assert _passing(year_advantages=[0.03, 0.03])[0] == CONFIRMED
    verdict, checks = _passing(year_advantages=[0.03, -0.01])
    assert verdict == REJECTED and checks["year_independence"] is False


def test_pooled_must_be_strictly_higher():
    assert _passing(pooled_advantage=1e-9)[0] == CONFIRMED
    for bad in (0.0, -0.01, None):
        verdict, checks = _passing(pooled_advantage=bad)
        assert verdict == REJECTED and checks["pooled_higher"] is False


def test_year_independence_boundary():
    # 2 of 3 qualifying years positive -> passes
    assert _passing(year_advantages=[0.03, 0.03, -0.01])[0] == CONFIRMED
    # 1 of 3 -> fails
    verdict, checks = _passing(year_advantages=[0.03, -0.01, -0.01])
    assert verdict == REJECTED and checks["year_independence"] is False
    # fewer than two qualifying years -> fails, cannot establish independence
    verdict, checks = _passing(year_advantages=[0.03])
    assert verdict == REJECTED and checks["year_independence"] is False


def test_single_fold_dominance_boundary():
    # one fold contributing exactly 50% is allowed
    verdict, checks = _passing(fold_advantages=[0.05, 0.03, 0.02])
    assert checks["max_fold_fraction"] == pytest.approx(0.5)
    assert verdict == CONFIRMED
    # just over 50% is not
    verdict, checks = _passing(fold_advantages=[0.06, 0.03, 0.02])
    assert checks["max_fold_fraction"] > 0.5
    assert verdict == REJECTED and checks["no_single_fold_dominance"] is False


def test_any_holdout_row_rejects():
    verdict, checks = _passing(holdout_rows_used=1)
    assert verdict == REJECTED and checks["no_holdout_rows"] is False


def test_no_positive_folds_at_all_rejects():
    verdict, checks = _passing(fold_advantages=[-0.01, -0.02, -0.03],
                               pooled_advantage=-0.02)
    assert verdict == REJECTED
    assert checks["no_single_fold_dominance"] is False
    assert checks["max_fold_fraction"] is None


def test_missing_fold_values_do_not_count_as_positive():
    verdict, checks = _passing(fold_advantages=[None, None, 0.05])
    assert verdict == REJECTED and checks["fold_majority"] is False


def test_rule_text_is_frozen_and_names_every_gate():
    for token in ("2 of the 3", "pooled", "2 of 3 qualifying years",
                  "50%", "sealed-holdout", "REPRODUCTION_FAILURE"):
        assert token in RANKING_DECISION_RULE


def test_verdict_vocabulary_is_exactly_three():
    seen = {
        _passing()[0],
        _passing(pooled_advantage=-1.0)[0],
        _passing(reproduced=False)[0],
    }
    assert seen == {CONFIRMED, REJECTED, REPRO_FAIL}
