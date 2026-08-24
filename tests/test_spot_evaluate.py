"""S3 — cross-sectional evaluation, dependence handling, and trial governance.

The measurement is where a discovery stage most easily flatters itself: by
pooling overlapping events into a huge fake sample, by comparing a ranking to
nothing instead of to a matched random draw, or by quietly relaxing the
go/no-go rule once the numbers are in. Each is pinned here.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from spot.evaluate import (BLOCK_DATES, DateResult, aggregate, block_bootstrap,
                           block_stability, blocks, classify, evaluate_date,
                           spearman)
from spot.trials import BY_FEATURE, SCOPE, counts, declare_all
from validation.trial_registry import (ImmutableRecordError, TrialRegistry)


def labels_for(symbols, hits, *, terminal=0.0, excess=0.0, mae=-0.2):
    return {s: {"symbol": s, "clean_2x": 1 if s in hits else 0,
                "terminal_return": terminal + (0.01 if s in hits else 0.0),
                "excess_vs_btc": excess + (0.01 if s in hits else 0.0),
                "mae": mae}
            for s in symbols}


def run_date(values, labels, *, seed=7, n_eligible=None):
    return evaluate_date(asof="2021-01-01", regime="bull", values=values,
                         labels=labels, descending=True,
                         n_eligible=n_eligible or len(values),
                         rng=np.random.default_rng(seed))


# --- the cross-section ----------------------------------------------------


def test_a_perfect_feature_selects_every_winner():
    symbols = [f"S{i:02d}USDT" for i in range(25)]
    hits = set(symbols[:5])
    values = {s: (100.0 - i) for i, s in enumerate(symbols)}
    res = run_date(values, labels_for(symbols, hits))
    assert res.k == 5
    assert res.selection_rate == 1.0
    assert res.universe_rate == pytest.approx(5 / 25)
    assert res.lift == pytest.approx(5.0)


def test_a_useless_feature_lands_on_the_base_rate():
    symbols = [f"S{i:02d}USDT" for i in range(25)]
    hits = {symbols[i] for i in (0, 7, 13, 19, 24)}
    values = {s: 1.0 for s in symbols}          # no information at all
    res = run_date(values, labels_for(symbols, hits))
    assert res.lift is not None
    assert 0.0 <= res.selection_rate <= 1.0


def test_the_random_baseline_is_matched_and_seeded():
    """Same K, same universe, same dates — and reproducible."""
    symbols = [f"S{i:02d}USDT" for i in range(25)]
    labels = labels_for(symbols, set(symbols[:5]))
    values = {s: float(i) for i, s in enumerate(symbols)}
    a = run_date(values, labels, seed=11)
    b = run_date(values, labels, seed=11)
    assert a.random_mean == b.random_mean
    assert a.random_p95 == b.random_p95
    assert a.random_mean == pytest.approx(5 / 25, abs=0.05)


def test_a_symbol_without_a_feature_value_is_excluded_not_zeroed():
    symbols = [f"S{i:02d}USDT" for i in range(25)]
    labels = labels_for(symbols, set(symbols[:5]))
    values = {s: float(i) for i, s in enumerate(symbols[:20])}
    res = run_date(values, labels, n_eligible=25)
    assert res.coverage == pytest.approx(0.8)
    assert res.n_resolved == 20


def test_an_unresolved_symbol_is_excluded_from_rates_not_from_the_ranking():
    """Label availability must not decide the shortlist.

    An earlier version filtered the ranking by `s in labels`, so a symbol whose
    outcome was censored or gap-unresolved dropped out *before* K was chosen —
    letting information from after the decision reshape the selection.
    """
    symbols = [f"S{i:02d}USDT" for i in range(25)]
    labels = labels_for(symbols, set(symbols[:5]))
    del labels[symbols[0]]                      # censored / gap: no outcome
    values = {s: (100.0 - i) for i, s in enumerate(symbols)}
    res = run_date(values, labels)
    assert res.k == 5, "K comes from the eligible ranking, not from labels"
    # The top five are S00..S04; S00 has no outcome, so the rate is over four.
    assert res.selection_rate == 1.0
    assert res.n_resolved == 24, (
        "n_resolved counts measured outcomes, not ranked symbols; conflating "
        "them overstates the sample by every unresolved row")


# --- dependence -----------------------------------------------------------


def test_blocks_are_six_consecutive_dates_and_drop_a_partial_tail():
    """S3_SPEC §7: a block is one 180-day horizon. A two-date tail is not one,
    and resampling it as a whole block inflates the block count."""
    assert blocks(list(range(14)), BLOCK_DATES) == [[0, 1, 2, 3, 4, 5],
                                                    [6, 7, 8, 9, 10, 11]]
    assert len(blocks(list(range(67)), BLOCK_DATES)) == 11


def test_the_bootstrap_resamples_blocks_not_dates():
    """Twelve dates that are really two blocks must not read as twelve
    independent observations: the interval has to stay wide."""
    per_date = [0.5] * 6 + [-0.5] * 6
    out = block_bootstrap(per_date, draws=500)
    assert out["blocks"] == 2
    assert out["low"] < 0 < out["high"], (
        "two opposing blocks cannot produce a confident non-zero mean")


def test_lift_is_a_ratio_of_means_not_a_mean_of_ratios():
    """The defect the first S3 run shipped, now pinned.

    One quiet date where the universe rate is near zero produces a per-date
    ratio of five or ten. Averaging those ratios reported lift > 1 for rules
    whose selection rate was *below* the universe rate on aggregate — a
    headline that contradicted its own inputs.
    """
    res = ([result("2021-01-01", "bull", 0.10, 0.02)]
           + [result(f"2021-{i + 2:02d}-01", "bull", 0.20, 0.40)
              for i in range(5)])
    agg = aggregate(res)
    assert agg["mean_of_per_date_lift_ratios"] > 1.0
    assert agg["lift"] < 1.0
    assert agg["selection_rate"] < agg["universe_rate"]

    by_regime = {"bull": agg}
    _, checks = classify(agg, block_stability(res), by_regime)
    assert checks["lift_above_1"] is False


def test_the_bootstrap_refuses_to_pretend_with_one_block():
    out = block_bootstrap([0.1] * 4)
    assert out["low"] is None and out["high"] is None


def test_spearman_handles_ties_without_scipy():
    assert spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


# --- the frozen go/no-go rule --------------------------------------------


def result(asof, regime, sel, uni, *, excess=0.1, rand_p95=0.0):
    return DateResult(asof=asof, regime=regime, n_eligible=25, n_resolved=25,
                      k=5, selection_rate=sel, universe_rate=uni,
                      selection_median_return=0.1, universe_median_return=0.0,
                      selection_median_excess=excess,
                      universe_median_excess=0.0,
                      selection_median_mae=-0.2, universe_median_mae=-0.3,
                      spearman=0.1, random_mean=uni, random_p95=rand_p95,
                      secondary_top_n_rate=sel, coverage=1.0)


def series(sel, uni, regimes, n=12, **kw):
    return [result(f"20{20 + i // 12}-{i % 12 + 1:02d}-01",
                   regimes[i % len(regimes)], sel, uni, **kw)
            for i in range(n)]


def test_a_strong_stable_feature_is_kept():
    res = series(0.40, 0.20, ["bull", "range", "bear"], n=18)
    agg = aggregate(res)
    by_regime = {r: aggregate([x for x in res if x.regime == r])
                 for r in ("bull", "range", "bear")}
    verdict, checks = classify(agg, block_stability(res), by_regime)
    assert verdict == "PRELIMINARY_KEEP", checks


def test_a_feature_that_only_works_in_one_regime_is_not_kept():
    """§10 condition 5, and the reason it exists: S2 established that the
    regime, not the coin, drives most of the outcome."""
    res = ([result(f"2021-{i + 1:02d}-01", "bull", 0.50, 0.20) for i in range(6)]
           + [result(f"2022-{i + 1:02d}-01", "bear", 0.10, 0.20) for i in range(6)]
           + [result(f"2023-{i + 1:02d}-01", "range", 0.10, 0.20) for i in range(6)])
    agg = aggregate(res)
    by_regime = {r: aggregate([x for x in res if x.regime == r])
                 for r in ("bull", "range", "bear")}
    verdict, checks = classify(agg, block_stability(res), by_regime)
    assert checks["positive_in_2_of_3_regimes"] is False
    assert verdict != "PRELIMINARY_KEEP"


def test_a_positive_point_estimate_with_a_straddling_interval_is_unknown():
    """The honest outcome at eleven blocks, and it must not read as success."""
    res = (series(0.40, 0.20, ["bull"], n=6)
           + series(0.05, 0.20, ["bear"], n=6))
    agg = aggregate(res)
    by_regime = {"bull": aggregate(res[:6]), "bear": aggregate(res[6:])}
    verdict, _ = classify(agg, block_stability(res), by_regime)
    assert verdict == "PRELIMINARY_UNKNOWN"


def test_a_feature_that_selects_losers_is_removed():
    res = series(0.05, 0.20, ["bull", "range", "bear"], n=18, excess=-0.5)
    agg = aggregate(res)
    by_regime = {r: aggregate([x for x in res if x.regime == r])
                 for r in ("bull", "range", "bear")}
    verdict, _ = classify(agg, block_stability(res), by_regime)
    assert verdict == "PRELIMINARY_REMOVE"


def test_beating_the_universe_but_not_matched_random_is_not_enough():
    res = series(0.40, 0.20, ["bull", "range", "bear"], n=18, rand_p95=0.99)
    agg = aggregate(res)
    by_regime = {r: aggregate([x for x in res if x.regime == r])
                 for r in ("bull", "range", "bear")}
    verdict, checks = classify(agg, block_stability(res), by_regime)
    assert checks["beats_random_p95"] is False
    assert verdict != "PRELIMINARY_KEEP"


# --- trial governance -----------------------------------------------------


def registry(tmp):
    return TrialRegistry(os.path.join(tmp, "trials.jsonl"))


def test_six_trials_are_declared_in_the_spot_family():
    with tempfile.TemporaryDirectory() as tmp:
        reg = registry(tmp)
        ids = declare_all(reg)
        assert set(ids) == {"B4", "B5", "F1", "F2", "F3", "F4"}
        c = counts(reg)
        assert c["conservative"] == 6 and c["n_records"] == 6
        assert c["scope"]["research_objective"] == "spot_2x_discovery"


def test_an_exact_rerun_does_not_create_a_seventh_trial():
    with tempfile.TemporaryDirectory() as tmp:
        reg = registry(tmp)
        declare_all(reg)
        declare_all(reg)
        assert counts(reg)["conservative"] == 6


def test_changing_a_definition_creates_a_different_trial():
    """The mechanism that makes 'frozen' checkable: identity carries the
    definition, so an edited feature cannot reuse its predecessor's look."""
    from spot.features import Feature, momentum_6_1
    from spot.trials import _record
    original = BY_FEATURE["B4"]
    tweaked = _record(Feature("B4", "momentum_6_1", "baseline_momentum",
                              "close[-21] / close[-181] - 1", 181,
                              momentum_6_1))
    assert tweaked.trial_id != original.trial_id


def test_redeclaring_one_trial_with_different_content_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        reg = registry(tmp)
        declare_all(reg)
        original = BY_FEATURE["F1"]
        conflicting = type(original)(
            identity=original.identity, created_at=original.created_at,
            origin=original.origin, status=original.status,
            target_family=original.target_family,
            notes="a different note for the same identity")
        with pytest.raises(ImmutableRecordError):
            reg.declare(conflicting)


def test_the_spot_family_does_not_inherit_the_futures_count():
    """`n_trials = 100` is another family's number; the key keeps them apart."""
    with tempfile.TemporaryDirectory() as tmp:
        reg = registry(tmp)
        declare_all(reg)
        assert counts(reg)["conservative"] == 6
        assert SCOPE.research_objective == "spot_2x_discovery"
