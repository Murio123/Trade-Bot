"""C2.2c: tests for tools/swing_hypothesis_walkforward.py.

Pure-function tests build hand-crafted trade records (the same schema
tools.swing_hypothesis_simulator produces) to test fold aggregation, year
independence, dominance, and baseline-comparison logic without needing a
real dataset. One end-to-end test reuses the swing synthetic-dataset
generator from test_deep_discovery.py to prove the CLI/report wiring works.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.deep_backtest as deep_backtest
import tools.swing_hypothesis_walkforward as swf
from tests.test_deep_discovery import SWING_BARS, _build_swing_dataset
from tools import kline_cache
from validation.trade_costs import FUNDING_NOT_MODELLED

ROOT = Path(__file__).resolve().parent.parent
SIM_SOURCE = (ROOT / "tools" / "swing_hypothesis_simulator.py").read_text()
WF_SOURCE = (ROOT / "tools" / "swing_hypothesis_walkforward.py").read_text()


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


def _rec(idx, hyp="H2", eligible=True, outcome="win", net_r=1.0, gross_r=1.1,
        cost_r=0.1, year=2024, rejection_reason=None):
    ts = f"{year}-06-15T00:00:00+00:00"
    return {
        "hypothesis": hyp, "idx": idx, "timestamp": ts, "direction": "long",
        "entry": 100.0, "stop": 95.0, "target": 110.0, "initial_risk": 5.0,
        "rr": 2.0, "regime": "range", "volatility_percentile": 50,
        "cost_r": cost_r if eligible else None,
        "gross_r": gross_r if eligible else None,
        "net_r": net_r if eligible else None,
        "outcome": outcome if eligible else None,
        "exit_idx": idx + 3 if eligible else None,
        "holding_bars": 3 if eligible else None,
        "rejection_reason": rejection_reason, "eligible": eligible,
    }


# ---------------------------------------------------------------------------
# Frozen constants unchanged.
# ---------------------------------------------------------------------------

def test_frozen_thresholds_unchanged():
    assert swf.FOLD_POSITIVE_RATIO_MIN == 2 / 3
    assert swf.YEAR_POSITIVE_RATIO_MIN == 2 / 3
    assert swf.MIN_YEAR_TRADES == 10
    assert swf.SINGLE_FOLD_DOMINANCE_MAX == 0.5
    assert swf.MAX_TIMEOUT_SHARE == 0.9
    assert swf.MAX_UNRESOLVED_FRACTION == 0.05
    assert swf.MAX_COST_TO_GROSS_RATIO == 0.5
    assert swf.RANDOM_P95 == 0.95


def test_simulator_frozen_constants_untouched_by_this_module():
    # this module must not redefine H1/H2 numeric constants — it only
    # imports the simulator's functions, never its constants, and defines
    # no eligibility logic of its own.
    for name in ("H1_STOP_ATR_BUFFER", "H1_MIN_RR", "H2_EQ_POS_EXTREME",
                "H2_STOP_ATR_BUFFER"):
        assert name not in WF_SOURCE


# ---------------------------------------------------------------------------
# Real walk-forward geometry (reused, not reimplemented).
# ---------------------------------------------------------------------------

def test_build_geometry_reuses_deep_backtest_wfconfig():
    profile = deep_backtest.get_profile("swing")
    wf, folds, span_lo, span_hi = swf.build_geometry(profile, 1000, 1300,
                                                     holdout_frac=0.1, min_folds=2)
    assert isinstance(wf, deep_backtest.WFConfig)
    assert len(folds) >= 2
    assert wf.purge_bars >= deep_backtest.max_hold_bars(profile)


def test_holdout_section_never_includes_stats():
    section = swf.holdout_section(span_hi=1000, holdout_bars=100)
    assert section == {
        "idx_lo": 900, "idx_hi": 1000, "n_bars": 100, "sealed": True,
        "note": "geometry and bar count only; never aggregated or scored by C2.2c",
    }
    assert "net_expectancy_r" not in section
    assert "trades" not in section


def test_purge_and_embargo_respected_via_partition_setups():
    # Regression against accidental reimplementation: partition_setups is
    # the SAME function tools.deep_backtest.walk_forward uses; a record
    # whose forward horizon (idx+hold_bars) reaches into validation must be
    # excluded from train.
    wf = deep_backtest.WFConfig(train_bars=100, val_bars=50, purge_bars=10,
                               embargo_bars=5, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 400, wf)
    records = [{"idx": i} for i in range(0, 400)]
    train, val = deep_backtest.partition_setups(records, folds[0], hold_bars=10)
    assert all(f["idx"] + 10 < folds[0].val_lo for f in train)
    assert all(folds[0].val_lo <= f["idx"] < folds[0].val_hi for f in val)


# ---------------------------------------------------------------------------
# H1 zero-coverage classification.
# ---------------------------------------------------------------------------

def test_evaluate_h1_zero_eligible_before_rr_is_insufficient_evidence():
    records = [_rec(i, eligible=False, rejection_reason="regime_not_trend")
              for i in range(20)]
    wf = deep_backtest.WFConfig(train_bars=10, val_bars=5, purge_bars=2,
                               embargo_bars=1, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 20, wf)
    h1 = swf.evaluate_h1(records, folds, hold_bars=2)
    assert h1["eligible_before_rr_filter"] == 0
    assert h1["verdict"] == "INSUFFICIENT_EVIDENCE_DUE_TO_ZERO_COVERAGE"


def test_evaluate_h1_raises_on_nonzero_validation_coverage():
    # Regression: evaluate_h1's two-label contract only covers zero
    # validation-region coverage. A taken trade inside the validation
    # window must raise, not silently route to REJECT_FROZEN_SPECIFICATION
    # (the original bug: summary["n"] was folded into the same
    # eligible_before_rr counter that triggers REJECT_FROZEN_SPECIFICATION,
    # so ANY taken trade — success or not — was mislabeled as a rejection).
    records = ([_rec(i, eligible=False, rejection_reason="regime_not_trend")
               for i in range(13)]
              + [_rec(13 + j, eligible=True, net_r=1.0) for j in range(5)])
    wf = deep_backtest.WFConfig(train_bars=10, val_bars=5, purge_bars=2,
                               embargo_bars=1, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 20, wf)
    with pytest.raises(ValueError, match="does not define a verdict"):
        swf.evaluate_h1(records, folds, hold_bars=2)


def test_evaluate_h1_rr_gate_reached_and_failed_is_reject_frozen_spec():
    # fold_windows(0, 20, wf) below produces exactly one fold with
    # val=[13, 18) — place the rr_below_floor records precisely inside that
    # validation window so the (now validation-only) pooled count is exact;
    # records outside [13, 18), including other rr_below_floor rejections,
    # must NOT be counted (they'd fall in train/purge/embargo, never
    # scored).
    records = ([_rec(i, eligible=False, rejection_reason="regime_not_trend")
               for i in range(13)]
              + [_rec(13 + j, eligible=False, rejection_reason="rr_below_floor")
                 for j in range(5)]
              + [_rec(18 + j, eligible=False, rejection_reason="rr_below_floor")
                 for j in range(2)])  # idx 18-19: outside the val window
    wf = deep_backtest.WFConfig(train_bars=10, val_bars=5, purge_bars=2,
                               embargo_bars=1, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 20, wf)
    assert len(folds) == 1 and folds[0].val_lo == 13 and folds[0].val_hi == 18
    h1 = swf.evaluate_h1(records, folds, hold_bars=2)
    assert h1["eligible_before_rr_filter"] == 5  # only the in-window ones
    assert h1["verdict"] == "REJECT_FROZEN_SPECIFICATION"


# ---------------------------------------------------------------------------
# Year independence.
# ---------------------------------------------------------------------------

def test_year_independence_rejects_one_of_two_years_positive():
    records = ([_rec(i, net_r=-0.1, year=2023) for i in range(15)]
              + [_rec(15 + i, net_r=0.2, year=2024) for i in range(15)])
    yearly = swf.year_stats(records)
    ok, detail = swf.year_independence_check(yearly)
    assert ok is False
    assert detail["qualifying_years"] == 2
    assert detail["positive_years"] == 1


def test_year_independence_passes_two_of_two_years_positive():
    records = ([_rec(i, net_r=0.1, year=2023) for i in range(15)]
              + [_rec(15 + i, net_r=0.2, year=2024) for i in range(15)])
    yearly = swf.year_stats(records)
    ok, detail = swf.year_independence_check(yearly)
    assert ok is True


def test_year_independence_insufficient_with_one_qualifying_year():
    records = [_rec(i, net_r=0.1, year=2023) for i in range(15)]
    yearly = swf.year_stats(records)
    ok, detail = swf.year_independence_check(yearly)
    assert ok is False
    assert detail["qualifying_years"] == 1


# ---------------------------------------------------------------------------
# No single-fold dominance.
# ---------------------------------------------------------------------------

def test_single_fold_dominance_flags_one_fold_carrying_everything():
    per_fold = [{"sum_r": 100.0}, {"sum_r": 0.5}, {"sum_r": 0.5}]
    ok, detail = swf.single_fold_dominance_ok(per_fold)
    assert ok is False
    assert detail["fraction"] > 0.5


def test_single_fold_dominance_passes_evenly_spread_folds():
    per_fold = [{"sum_r": 10.0}, {"sum_r": 9.0}, {"sum_r": 8.0}]
    ok, detail = swf.single_fold_dominance_ok(per_fold)
    assert ok is True


def test_single_fold_dominance_fails_on_nonpositive_total():
    per_fold = [{"sum_r": -1.0}, {"sum_r": -2.0}]
    ok, detail = swf.single_fold_dominance_ok(per_fold)
    assert ok is False


# ---------------------------------------------------------------------------
# Fold positive ratio.
# ---------------------------------------------------------------------------

def test_fold_positive_ratio_two_of_three():
    per_fold = [{"n": 10, "net_expectancy_r": 0.1},
               {"n": 10, "net_expectancy_r": 0.2},
               {"n": 10, "net_expectancy_r": -0.1}]
    ratio, n_confident = swf.fold_positive_ratio(per_fold)
    assert ratio == pytest.approx(2 / 3)
    assert n_confident == 3


def test_fold_positive_ratio_ignores_empty_folds():
    per_fold = [{"n": 0, "net_expectancy_r": None},
               {"n": 10, "net_expectancy_r": 0.1}]
    ratio, n_confident = swf.fold_positive_ratio(per_fold)
    assert ratio == 1.0
    assert n_confident == 1


# ---------------------------------------------------------------------------
# Random baselines: deterministic seeds + p95.
# ---------------------------------------------------------------------------

def test_matched_coverage_random_baseline_deterministic():
    pool = [_rec(i, net_r=float(i % 3 - 1)) for i in range(50)]
    a = swf.matched_coverage_random_baseline(pool, take_n=10, n_draws=20)
    b = swf.matched_coverage_random_baseline(pool, take_n=10, n_draws=20)
    assert a == b
    assert len(a) == 20


def test_matched_coverage_random_baseline_empty_pool_or_bad_take_n():
    pool = [_rec(i) for i in range(5)]
    assert swf.matched_coverage_random_baseline(pool, take_n=0) == []
    assert swf.matched_coverage_random_baseline(pool, take_n=100) == []
    assert swf.matched_coverage_random_baseline([], take_n=1) == []


def test_random_direction_distribution_deterministic(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"low": [90.0] * 200, "high": [110.0] * 200,
                      "close": [100.0] * 200})
    records = [_rec(100, net_r=1.0)]
    a = swf.random_direction_distribution(records, df, hold_bars=5,
                                         cost_pct=0.001, n_draws=10,
                                         funding=FUNDING_NOT_MODELLED)
    b = swf.random_direction_distribution(records, df, hold_bars=5,
                                         cost_pct=0.001, n_draws=10,
                                         funding=FUNDING_NOT_MODELLED)
    assert a == b


# ---------------------------------------------------------------------------
# H1/H2 never blended.
# ---------------------------------------------------------------------------

def test_no_blending_functions_exist():
    for banned in ("def combined_verdict", "def blend", "def aggregate_hypotheses",
                  "def combined_summary"):
        assert banned not in WF_SOURCE


def test_evaluate_functions_never_mix_hypothesis_records():
    h1_records = [_rec(i, hyp="H1", eligible=False,
                      rejection_reason="regime_not_trend") for i in range(10)]
    h2_records = [_rec(i, hyp="H2", net_r=1.0) for i in range(10)]
    wf = deep_backtest.WFConfig(train_bars=5, val_bars=3, purge_bars=1,
                               embargo_bars=1, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 10, wf)
    h1 = swf.evaluate_h1(h1_records, folds, hold_bars=1)
    # evaluate_h2 requires baseline_setups/entry_df/cost_pct; verify it never
    # reads h1_records at all (signature has no such parameter)
    import inspect
    params = list(inspect.signature(swf.evaluate_h2).parameters)
    assert "h1_records" not in params and "other_hypothesis" not in params
    assert h1["taken"] == 0  # h1 evaluated purely on its own records


# ---------------------------------------------------------------------------
# Simulator reused, not reimplemented.
# ---------------------------------------------------------------------------

def test_walkforward_module_defines_no_eligibility_logic_of_its_own():
    for banned in ("h1_evaluate", "h2_evaluate", "_bar_context",
                  "_rolling_extremes", "def resolve"):
        assert banned not in WF_SOURCE


def test_walkforward_imports_simulator_functions_directly():
    assert "from tools.swing_hypothesis_simulator import" in WF_SOURCE
    assert "walk_hypothesis" in WF_SOURCE
    assert "summarize" in WF_SOURCE


# ---------------------------------------------------------------------------
# Runtime never imports this tool.
# ---------------------------------------------------------------------------

def test_runtime_does_not_import_swing_hypothesis_walkforward():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))
    for path in runtime_files:
        if not path.exists():
            continue
        assert "swing_hypothesis_walkforward" not in path.read_text(), path


def test_module_never_monkeypatches_deep_backtest():
    import re
    assert "setattr(deep_backtest" not in WF_SOURCE
    assert not re.search(r"deep_backtest\.\w+\s*=(?!=)", WF_SOURCE)


def test_evaluate_h2_pooled_stats_exclude_records_outside_validation_windows():
    # Regression: pooled/baseline/random-distribution figures must be built
    # ONLY from the union of fold validation windows, never the full input
    # lists (which span the sealed holdout and non-validation train-only
    # regions too). fold_windows(0, 20, wf) below produces one fold with
    # val=[13, 18); records outside that window carry a wildly different
    # net_r (-100.0) that would massively skew pooled stats if leaked in.
    import pandas as pd
    in_window = [_rec(13 + j, net_r=1.0, gross_r=1.1) for j in range(5)]
    outside_window = ([_rec(i, net_r=-100.0, gross_r=-99.9) for i in range(13)]
                      + [_rec(18 + j, net_r=-100.0, gross_r=-99.9)
                         for j in range(2)])
    records = in_window + outside_window
    baseline_setups = [{"idx": r["idx"], "r": r["net_r"], "outcome": "win"}
                      for r in records]
    regime_records = records  # same schema, reused as its own regime pool

    wf = deep_backtest.WFConfig(train_bars=10, val_bars=5, purge_bars=2,
                               embargo_bars=1, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 20, wf)
    assert len(folds) == 1 and folds[0].val_lo == 13 and folds[0].val_hi == 18

    entry_df = pd.DataFrame({"low": [90.0] * 30, "high": [110.0] * 30,
                             "close": [100.0] * 30})
    h2 = swf.evaluate_h2(records, regime_records, baseline_setups, entry_df,
                        hold_bars=2, cost_pct=0.001, folds=folds,
                        funding=FUNDING_NOT_MODELLED)

    assert h2["pooled_summary"]["n"] == 5
    assert h2["pooled_summary"]["net_expectancy_r"] == pytest.approx(1.0)
    assert h2["baselines"]["A_current_swing_baseline"]["pooled_net_expectancy_r"] \
        == pytest.approx(1.0)
    assert h2["baselines"]["C_regime_direction"]["pooled_summary"]["n"] == 5


# ---------------------------------------------------------------------------
# End-to-end wiring on a small synthetic swing dataset.
# ---------------------------------------------------------------------------

def test_run_walkforward_end_to_end(tmp_path):
    outdir = tmp_path / "c23"
    _build_swing_dataset(outdir)
    report = swf.run_walkforward(str(outdir), "binance", "BTCUSDT", "swing",
                                SWING_BARS, holdout_frac=0.0, min_folds=1)
    assert report["stage"] == "C2.2c"
    assert report["H1"]["verdict"] in ("INSUFFICIENT_EVIDENCE_DUE_TO_ZERO_COVERAGE",
                                      "REJECT_FROZEN_SPECIFICATION")
    assert report["H2"]["verdict"] in ("PROCEED_TO_REPORT_ONLY_FORWARD_TEST",
                                      "NEEDS_MORE_EVIDENCE", "REJECT_HYPOTHESIS")
    assert report["holdout"]["sealed"] is True
    assert "net_expectancy_r" not in report["holdout"]


def test_run_walkforward_is_deterministic(tmp_path):
    outdir = tmp_path / "c23_det"
    _build_swing_dataset(outdir)
    r1 = swf.run_walkforward(str(outdir), "binance", "BTCUSDT", "swing",
                            SWING_BARS, holdout_frac=0.0, min_folds=1)
    r2 = swf.run_walkforward(str(outdir), "binance", "BTCUSDT", "swing",
                            SWING_BARS, holdout_frac=0.0, min_folds=1)
    assert r1["H1"]["verdict"] == r2["H1"]["verdict"]
    assert r1["H2"]["verdict"] == r2["H2"]["verdict"]
    assert r1["H2"]["pooled_summary"] == r2["H2"]["pooled_summary"]
    assert r1["H2"]["baselines"]["B_random_direction"] == r2["H2"]["baselines"]["B_random_direction"]


def test_cli_writes_all_required_artifacts(tmp_path):
    outdir = tmp_path / "c23_cli"
    _build_swing_dataset(outdir)
    reports_dir = tmp_path / "reports_c23"
    rc = swf.main([
        "--dataset", str(outdir), "--exchange", "binance", "--symbol",
        "BTCUSDT", "--profile", "swing", "--bars", str(SWING_BARS),
        "--holdout-frac", "0.0", "--min-folds", "1",
        "--outdir", str(reports_dir),
    ])
    assert rc == 0
    for name in ("h1_walkforward.json", "h1_walkforward.txt",
                "h2_walkforward.json", "h2_walkforward.txt",
                "baseline_comparison.json", "DECISION.md"):
        assert (reports_dir / name).exists()
    h1_payload = json.loads((reports_dir / "h1_walkforward.json").read_text())
    assert "H2" not in h1_payload
    h2_payload = json.loads((reports_dir / "h2_walkforward.json").read_text())
    assert "H1" not in h2_payload
