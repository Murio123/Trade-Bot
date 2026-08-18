"""G1 negative-control tests.

These do not re-run the 500-replication suite — that is
`tools/g1_negative_controls.py`, whose output is recorded in
`reports/c50/G1_RESULT.md`. What is tested here is that the controls are
*constructed correctly*: determinism, that the synthetic data really has no
drift, that the statistics can detect a planted effect (a control that cannot
fail proves nothing), and that the threshold evaluation applies the frozen rules.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.g1_negative_controls import (DSR_CONTROLS, EXPECTANCY_CONTROLS,
                                        T1_MAX_FPR, aggregate, run_suite,
                                        verdict)
from validation.negative_controls import (CONTROLS, MASTER_SEED,
                                          ar1_signal, barrier_bias_diagnostic,
                                          bootstrap_mean_ci, build_synthetic_run,
                                          ci_verdict, realized_sigma, rng_for,
                                          share_band, synthetic_ohlcv)


# --- determinism --------------------------------------------------------------

def test_rng_for_is_seeded_and_reproducible():
    assert list(rng_for(3).normal(size=5)) == list(rng_for(3).normal(size=5))
    assert list(rng_for(3).normal(size=5)) != list(rng_for(4).normal(size=5))


def test_every_control_is_reproducible():
    for name, (fn, _) in CONTROLS.items():
        first, second = fn(1), fn(1)
        assert first == second, name


def test_controls_differ_between_replications():
    fn, _ = CONTROLS["NC1"]
    assert fn(1) != fn(2)


def test_the_master_seed_is_the_frozen_one():
    assert MASTER_SEED == 20260817


# --- the synthetic process really is a null ----------------------------------

def test_synthetic_ohlcv_has_no_log_drift():
    r = rng_for(11)
    logret = []
    for i in range(40):
        f = synthetic_ohlcv(500, rng_for(100 + i))
        c = np.log(f["close"].to_numpy())
        logret.append((c[-1] - c[0]) / (c.size - 1))
    ci = bootstrap_mean_ci(np.array(logret), r)
    assert ci_verdict(ci) == "CONTAINS_ZERO", ci


def test_synthetic_bars_are_internally_consistent():
    f = synthetic_ohlcv(400, rng_for(12))
    hi, lo = f["high"].to_numpy(), f["low"].to_numpy()
    op, cl = f["open"].to_numpy(), f["close"].to_numpy()
    assert np.all(hi >= np.maximum(op, cl))
    assert np.all(lo <= np.minimum(op, cl))
    assert np.all(lo > 0)
    assert np.all(np.diff(f["close_time"].to_numpy()) > 0)


def test_synthetic_bars_have_real_intrabar_range():
    """A frame whose high always equalled its close would make every event
    resolve TIME, and every control would pass while testing nothing."""
    f = synthetic_ohlcv(400, rng_for(13))
    span = (f["high"] - f["low"]) / f["close"]
    assert span.median() > 0.005


def test_realized_sigma_is_strictly_backward_looking():
    """Bar i's sigma must not depend on bar i's own return: barriers scaled by a
    volatility that includes the bar being labelled are look-ahead."""
    f = synthetic_ohlcv(300, rng_for(14))
    base = realized_sigma(f)

    perturbed = f.copy()
    perturbed.loc[150, "close"] = perturbed.loc[150, "close"] * 3.0
    after = realized_sigma(perturbed)

    assert np.allclose(base[:151], after[:151]), (
        "sigma at or before the perturbed bar changed, so it reads that bar")
    assert not np.allclose(base[152:], after[152:])


def test_ar1_signal_is_persistent_and_centred():
    x = ar1_signal(20_000, rng_for(15), phi=0.85)
    lag1 = np.corrcoef(x[1:], x[:-1])[0, 1]
    assert 0.8 < lag1 < 0.9
    assert abs(x.mean()) < 0.1


# --- the statistics can detect a planted effect ------------------------------

def test_bootstrap_ci_detects_a_planted_positive_mean():
    r = rng_for(16)
    assert ci_verdict(bootstrap_mean_ci(r.normal(0.5, 1.0, size=400), r)) == "ABOVE_ZERO"
    assert ci_verdict(bootstrap_mean_ci(r.normal(-0.5, 1.0, size=400), r)) == "BELOW_ZERO"
    assert ci_verdict(bootstrap_mean_ci(r.normal(0.0, 1.0, size=400), r)) == "CONTAINS_ZERO"


def test_bootstrap_ci_needs_data():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0], rng_for(17))


def test_share_band_narrows_with_the_replication_count():
    lo_small, hi_small = share_band(0.5, 20)
    lo_big, hi_big = share_band(0.5, 500)
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert lo_big < 0.5 < hi_big


def test_a_planted_drift_is_detected_by_the_synthetic_run():
    """The control machinery must be able to see an edge that is really there,
    otherwise its silence on the null means nothing."""
    flat = build_synthetic_run(rng_for(18), n_bars=2000, mu_per_bar=0.0)
    drifting = build_synthetic_run(rng_for(18), n_bars=2000, mu_per_bar=0.004)
    assert drifting.r_multiples.mean() > flat.r_multiples.mean() + 0.2


# --- the labelling used by the controls --------------------------------------

def test_the_controls_use_symmetric_barriers():
    """With `upper == lower` a zero-drift process has an analytically zero label
    expectancy. Asymmetric barriers would confound "the apparatus invents edge"
    with "M02 is deliberately pessimistic"."""
    from validation.negative_controls import DEFAULT_BARRIERS
    assert DEFAULT_BARRIERS.upper_mult == DEFAULT_BARRIERS.lower_mult


def test_synthetic_runs_report_both_sample_sizes():
    run = build_synthetic_run(rng_for(19))
    assert run.raw_count > 0
    assert 0 < run.effective_count <= run.raw_count


def test_synthetic_runs_never_model_funding():
    """Synthetic timestamps carry no real funding bill, so P1's explicit opt-out
    must be what propagates — never a silent zero."""
    for name, (fn, _) in CONTROLS.items():
        rec = fn(0)
        if "funding_modelled" in rec:
            assert rec["funding_modelled"] is False, name


def test_the_barrier_bias_diagnostic_measures_the_tie_rate():
    """The tie convention is a deliberate bias, and a bias whose size is
    unmeasured is indistinguishable from a bug."""
    diag = barrier_bias_diagnostic()
    for key in ("symmetric", "project_2to1"):
        block = diag[key]
        assert 0.0 < block["tie_rate"] < 0.5
        assert block["n_events"] > 100
        assert block["mean_net_r"] < block["mean_gross_r"]   # costs are real


# --- threshold evaluation -----------------------------------------------------

def _fake(name: str, n: int, **fields) -> list[dict]:
    return [{"control": name, "replication": i, "raw_count": 100,
             "effective_count": 90.0, **fields} for i in range(n)]


def test_t1_fails_when_the_false_positive_rate_is_too_high():
    recs = _fake("NC1", 100, significant=True, status="OK", dsr=0.99)
    agg = aggregate("NC1", recs, rng_for(20))
    assert agg["t1"]["fpr"] == 1.0
    assert not agg["t1"]["t1_pass"]
    assert verdict([agg])["verdict"] == "CONTROLS_FAIL"


def test_t1_passes_at_the_nominal_rate():
    recs = _fake("NC1", 100, significant=False, status="OK", dsr=0.5)
    for r in recs[:5]:
        r["significant"] = True
    agg = aggregate("NC1", recs, rng_for(21))
    assert agg["t1"]["fpr"] == 0.05 <= T1_MAX_FPR
    assert agg["t1"]["t1_pass"]


def test_t1_boundary_is_inclusive():
    recs = _fake("NC1", 1000, significant=False, status="OK", dsr=0.5)
    for r in recs[:75]:
        r["significant"] = True
    assert aggregate("NC1", recs, rng_for(22))["t1"]["t1_pass"]
    recs[75]["significant"] = True
    assert not aggregate("NC1", recs, rng_for(22))["t1"]["t1_pass"]


def test_insufficient_data_is_not_counted_as_a_false_positive():
    recs = _fake("NC1", 40, significant=False, status="INSUFFICIENT_DATA")
    agg = aggregate("NC1", recs, rng_for(23))
    assert agg["t1"]["n_insufficient"] == 40
    assert agg["t1"]["fpr"] == 0.0


def test_t2_fails_only_when_the_null_expectancy_is_above_zero():
    rng = rng_for(24)
    above = [{"control": "NC3", "mean_net_r": 0.3 + 0.01 * i,
              "mean_gross_r": 0.4, "raw_count": 10, "effective_count": 9.0}
             for i in range(60)]
    assert not aggregate("NC3", above, rng)["t2"]["t2_pass"]

    below = [{"control": "NC3", "mean_net_r": -0.3 - 0.01 * i,
              "mean_gross_r": -0.2, "raw_count": 10, "effective_count": 9.0}
             for i in range(60)]
    agg = aggregate("NC3", below, rng)
    assert agg["t2"]["t2_pass"], "a random entry paying real costs must be allowed to lose"
    assert agg["t2"]["net"]["verdict"] == "BELOW_ZERO"


def test_t3_binds_to_the_long_short_gap():
    """Amendment A3: the gap is the statistic with power. A planted asymmetry
    must fail it."""
    rng = rng_for(25)
    skewed = [{"control": "NC4", "mean_net_r": -0.1, "mean_gross_r": 0.0,
               "mean_gross_r_long": -0.2 + 0.001 * i,
               "mean_gross_r_short": 0.2, "raw_count": 10,
               "effective_count": 9.0} for i in range(80)]
    agg = aggregate("NC4", skewed, rng)
    assert not agg["t3"]["t3_gap_pass"]
    assert "asymmetric" in " ".join(verdict([agg])["failures"])


def test_t3_passes_a_symmetric_apparatus():
    rng = rng_for(26)
    r = np.random.default_rng(5)
    recs = [{"control": "NC4", "mean_net_r": -0.1, "mean_gross_r": 0.0,
             "mean_gross_r_long": float(r.normal(0, 0.05)),
             "mean_gross_r_short": float(r.normal(0, 0.05)),
             "raw_count": 10, "effective_count": 9.0} for _ in range(200)]
    assert aggregate("NC4", recs, rng)["t3"]["t3_gap_pass"]


def test_t6_leakage_is_exact_and_any_amount_fails():
    rng = rng_for(27)
    clean = _fake("NC5", 30, leakage_pairs=0, significant=False, status="OK",
                  dsr=0.5, permutation_improved=False)
    assert aggregate("NC5", clean, rng)["leakage"]["t6_pass"]

    dirty = _fake("NC5", 30, leakage_pairs=0, significant=False, status="OK",
                  dsr=0.5, permutation_improved=False)
    dirty[7]["leakage_pairs"] = 1
    agg = aggregate("NC5", dirty, rng)
    assert not agg["leakage"]["t6_pass"]
    assert agg["leakage"]["total_leakage_pairs"] == 1
    assert "T6 leakage" in " ".join(verdict([agg])["failures"])


def test_t7_flags_an_effective_count_above_the_raw_count():
    rng = rng_for(28)
    recs = _fake("NC1", 20, significant=False, status="OK", dsr=0.5)
    recs[3]["effective_count"] = 500.0
    agg = aggregate("NC1", recs, rng)
    assert not agg["sample_size"]["effective_never_exceeds_raw"]
    assert "T7" in " ".join(verdict([agg])["failures"])


def test_verdict_is_pass_only_with_no_failures():
    assert verdict([])["verdict"] == "CONTROLS_PASS"
    assert verdict([])["failures"] == []


def test_a_specification_defect_is_indeterminate_not_a_failure():
    """A criterion no correct apparatus could clear is a defect in the
    criterion, and it is a different outcome from the apparatus manufacturing
    edge. Collapsing them would invite the criterion to be rewritten until it
    passed."""
    rng = rng_for(30)
    recs = [{"control": "NC4", "mean_net_r": -0.1, "mean_gross_r": -0.1,
             "mean_gross_r_long": 0.0, "mean_gross_r_short": 0.0,
             "raw_count": 10, "effective_count": 9.0} for _ in range(200)]
    agg = aggregate("NC4", recs, rng)
    assert not agg["t3"]["t3_pass"]          # as frozen: fails
    assert agg["t3"]["t3_gap_pass"]          # the A3 proposal: passes
    v = verdict([agg])
    assert v["verdict"] == "CONTROLS_INDETERMINATE"
    assert v["specification_defects"] and not v["apparatus_failures"]


def test_a_real_apparatus_failure_outranks_a_specification_defect():
    rng = rng_for(31)
    bad = _fake("NC1", 100, significant=True, status="OK", dsr=0.99)
    spec = [{"control": "NC4", "mean_net_r": -0.1, "mean_gross_r": -0.1,
             "mean_gross_r_long": 0.0, "mean_gross_r_short": 0.0,
             "raw_count": 10, "effective_count": 9.0} for _ in range(200)]
    v = verdict([aggregate("NC1", bad, rng), aggregate("NC4", spec, rng)])
    assert v["verdict"] == "CONTROLS_FAIL"


def test_t3_as_frozen_is_still_evaluated_and_reported():
    """A3 is a proposal, not a gate. The frozen statistic must keep being
    computed, or the amendment becomes a way of not looking."""
    rng = rng_for(32)
    recs = [{"control": "NC4", "mean_net_r": -0.1, "mean_gross_r": 0.05 * (-1) ** i,
             "mean_gross_r_long": 0.0, "mean_gross_r_short": 0.0,
             "raw_count": 10, "effective_count": 9.0} for i in range(200)]
    agg = aggregate("NC4", recs, rng)
    assert agg["t3"]["share_positive"] == pytest.approx(0.5)
    assert agg["t3"]["t3_pass"]


# --- conditions the frozen spec states but an earlier runner never graded -----

def test_nc2_rank_condition_is_actually_evaluated():
    """G1_SPEC.md §4 requires NC2 not to rank above its own unshuffled null. An
    earlier runner recorded the two Sharpes and graded neither, so every
    replication could have had the shuffled signal winning and NC2 would still
    have passed on T1 alone."""
    rng = rng_for(33)
    always_better = [{"control": "NC2", "raw_count": 10, "effective_count": 10.0,
                      "significant": False, "status": "OK", "dsr": 0.5,
                      "marginal_preserved": True,
                      "shuffled_sharpe": 1.0, "unshuffled_sharpe": 0.0}
                     for _ in range(200)]
    agg = aggregate("NC2", always_better, rng)
    assert agg["nc2_rank"]["shuffled_beats_unshuffled_share"] == 1.0
    assert not agg["nc2_rank"]["nc2_rank_pass"]
    assert "shuffling the signal improved it" in " ".join(
        verdict([agg])["failures"])


def test_nc2_flags_a_permutation_that_changed_the_marginal():
    rng = rng_for(34)
    recs = [{"control": "NC2", "raw_count": 10, "effective_count": 10.0,
             "significant": False, "status": "OK", "dsr": 0.5,
             "marginal_preserved": False,
             "shuffled_sharpe": 0.0, "unshuffled_sharpe": 0.0}
            for _ in range(50)]
    assert "marginal" in " ".join(verdict([aggregate("NC2", recs, rng)])["failures"])


def test_nc6_grades_the_best_path_not_the_pooled_series():
    """T4's second half asks whether the BEST path clears dsr >= 0.95. A path
    visits every group once, so its own series has one entry per event and a
    genuine per-path DSR is computable — no reinterpretation needed."""
    rec = CONTROLS["NC6"][0](0)
    assert rec["dsr"] == rec["best_path_dsr"]
    assert "dsr_pooled" in rec and rec["dsr_pooled"] != rec["best_path_dsr"]
    assert rec["n_paths_insufficient"] == 0


def test_t7_requires_every_record_to_report_both_counts():
    rng = rng_for(35)
    recs = _fake("NC1", 20, significant=False, status="OK", dsr=0.5)
    recs[4].pop("effective_count")
    agg = aggregate("NC1", recs, rng)
    assert not agg["sample_size"]["reported"]
    assert agg["sample_size"]["n_missing"] == 1
    assert "T7" in " ".join(verdict([agg])["failures"])


def test_every_control_is_bound_by_at_least_one_threshold():
    """A control nobody grades is decoration."""
    graded = set(DSR_CONTROLS) | set(EXPECTANCY_CONTROLS)
    assert set(CONTROLS) <= graded, set(CONTROLS) - graded


# --- an end-to-end smoke run --------------------------------------------------

def test_run_suite_at_a_tiny_replication_count():
    result = run_suite({name: 3 for name in CONTROLS})
    assert result["master_seed"] == MASTER_SEED
    assert result["spec"] == "reports/c50/G1_SPEC.md"
    assert len(result["aggregates"]) == len(CONTROLS)
    assert result["verdict"] in ("CONTROLS_PASS", "CONTROLS_FAIL")
    for agg in result["aggregates"]:
        leak = agg.get("leakage", {})
        if leak.get("measured"):
            assert leak["total_leakage_pairs"] == 0, agg["control"]


def test_run_suite_is_reproducible():
    a = run_suite({name: 2 for name in CONTROLS})
    b = run_suite({name: 2 for name in CONTROLS})
    assert a["records"] == b["records"]
