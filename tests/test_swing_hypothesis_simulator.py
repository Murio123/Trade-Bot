"""C2.2b: tests for tools/swing_hypothesis_simulator.py.

Pure-function tests build small entry_df fixtures + hand-built bar contexts
directly against h1_evaluate/h2_evaluate (no real dataset needed for most
cases). One end-to-end test reuses the swing synthetic-dataset generator
from test_deep_discovery.py to prove the CLI/report wiring works.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.deep_backtest as deep_backtest
import tools.swing_hypothesis_simulator as shs
from tests.test_deep_discovery import SWING_BARS, _build_swing_dataset
from tools import kline_cache
from validation.trade_costs import FUNDING_NOT_MODELLED

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "swing_hypothesis_simulator.py").read_text()


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


def _entry_df(n=200):
    # Flat low=90/high=140 band for bars [0, n-1) — asymmetric (50-wide) so
    # a pullback entry near one extreme has real RR headroom toward the
    # other. Last bar (n-1) is the "current" bar under test and is
    # deliberately made an extreme outlier so any test that leaks
    # look-ahead into the rolling window is caught.
    rows = {"low": [90.0] * (n - 1) + [1.0], "high": [140.0] * (n - 1) + [1000.0],
           "close": [115.0] * (n - 1) + [500.0]}
    return pd.DataFrame(rows)


def _ctx(**overrides):
    base = {
        "open_time": pd.Timestamp("2024-01-01", tz="UTC"), "price": 100.0,
        "atr": 2.0, "htf_bias": "bullish", "eq_zone": "discount", "eq_pos": 0.1,
        "ema_slope": 0.01, "volatility_atr_percentile": 50,
        "regime_current": "trend_up",
    }
    base.update(overrides)
    return base


COST_PCT = 0.001  # small, won't trip cost_r ceilings unless risk is tiny


# ---------------------------------------------------------------------------
# Frozen constants.
# ---------------------------------------------------------------------------

def test_frozen_constants_match_c22a_spec():
    assert shs.STRUCTURAL_WINDOW_BARS == 40
    assert shs.H1_STOP_ATR_BUFFER == 0.25
    assert shs.H1_MIN_RR == 1.5
    assert shs.H1_MAX_VOLATILITY_PERCENTILE == 90
    assert shs.H1_MAX_COST_R == 0.15
    assert shs.H2_EQ_POS_EXTREME == 0.20
    assert shs.H2_VOLATILITY_BAND == (10, 85)
    assert shs.H2_STOP_ATR_BUFFER == 0.30
    assert shs.H2_MAX_COST_R == 0.10


# ---------------------------------------------------------------------------
# H1 eligibility.
# ---------------------------------------------------------------------------

def test_h1_rejects_range_regime():
    kind, payload = shs.h1_evaluate(_ctx(regime_current="range"), _entry_df(), 100,
                                    COST_PCT)
    assert (kind, payload) == ("reject", "regime_not_trend")


def test_h1_rejects_htf_bias_disagreement():
    ctx = _ctx(regime_current="trend_up", htf_bias="bearish")
    kind, payload = shs.h1_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "htf_bias_disagreement")


def test_h1_rejects_extreme_volatility():
    ctx = _ctx(volatility_atr_percentile=95)
    kind, payload = shs.h1_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "volatility_extreme")


def test_h1_rejects_wrong_eq_zone():
    ctx = _ctx(eq_zone="premium")  # long needs discount
    kind, payload = shs.h1_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "not_pullback_zone")


def test_h1_rejects_ema_slope_mismatch():
    ctx = _ctx(ema_slope=-0.01)  # long needs positive slope
    kind, payload = shs.h1_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "ema_slope_mismatch")


def test_h1_regime_only_mode_skips_pullback_timing():
    # premium zone + wrong-sign slope would reject under full timing, but
    # regime-direction baseline (timing=False) must ignore both.
    ctx = _ctx(eq_zone="premium", ema_slope=-0.01)
    kind, _ = shs.h1_evaluate(ctx, _entry_df(), 100, COST_PCT, pullback_timing=False)
    assert kind == "candidate"


def test_h1_rejects_structure_already_broken():
    # price below the rolling low means the pullback thesis is already
    # falsified before entry.
    df = _entry_df()
    ctx = _ctx(price=80.0)  # rolling low over [60,100) is 90.0
    kind, payload = shs.h1_evaluate(ctx, df, 100, COST_PCT)
    assert (kind, payload) == ("reject", "structure_already_broken")


def test_h1_rejects_rr_below_floor():
    # rolling window low=90/high=140; price near the high end leaves almost
    # no reward toward the (already nearly-reached) target, while risk back
    # to the stop is large -> rr well below 1.5.
    df = _entry_df()
    ctx = _ctx(price=135.0, atr=2.0)  # reward=140-135=5; risk=135-89.5=45.5
    kind, payload = shs.h1_evaluate(ctx, df, 100, COST_PCT)
    assert (kind, payload) == ("reject", "rr_below_floor")


def test_h1_rejects_cost_r_ceiling():
    df = _entry_df()
    ctx = _ctx(price=95.0, atr=2.0)  # same geometry as the candidate test
    huge_cost_pct = 1.0  # forces cost_r far above 0.15
    kind, payload = shs.h1_evaluate(ctx, df, 100, huge_cost_pct)
    assert (kind, payload) == ("reject", "cost_r_exceeds_ceiling")


def test_h1_long_candidate_stop_target_and_direction():
    df = _entry_df()
    ctx = _ctx(price=95.0, atr=2.0, regime_current="trend_up",
              htf_bias="bullish", eq_zone="discount", ema_slope=0.01)
    kind, payload = shs.h1_evaluate(ctx, df, 100, COST_PCT)
    assert kind == "candidate"
    assert payload["direction"] == "long"
    pos = payload["pos"]
    assert pos["stop_loss"] == pytest.approx(90.0 - 0.25 * 2.0)
    assert pos["target_1"] == pos["target_2"] == pytest.approx(140.0)
    assert payload["rr"] == pytest.approx((140.0 - 95.0) / (95.0 - pos["stop_loss"]))
    assert payload["rr"] > shs.H1_MIN_RR


def test_h1_short_candidate_direction_mirrors_long():
    df = _entry_df()
    ctx = _ctx(price=135.0, atr=2.0, regime_current="trend_down",
              htf_bias="bearish", eq_zone="premium", ema_slope=-0.01)
    kind, payload = shs.h1_evaluate(ctx, df, 100, COST_PCT)
    assert kind == "candidate"
    assert payload["direction"] == "short"
    pos = payload["pos"]
    assert pos["stop_loss"] == pytest.approx(140.0 + 0.25 * 2.0)
    assert pos["target_1"] == pos["target_2"] == pytest.approx(90.0)
    assert payload["rr"] > shs.H1_MIN_RR


# ---------------------------------------------------------------------------
# H2 eligibility.
# ---------------------------------------------------------------------------

def test_h2_rejects_non_range_regime():
    ctx = _ctx(regime_current="trend_up")
    kind, payload = shs.h2_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "regime_not_range")


def test_h2_rejects_volatility_out_of_band():
    ctx = _ctx(regime_current="range", volatility_atr_percentile=5)
    kind, payload = shs.h2_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "volatility_out_of_band")
    ctx2 = _ctx(regime_current="range", volatility_atr_percentile=90)
    kind2, payload2 = shs.h2_evaluate(ctx2, _entry_df(), 100, COST_PCT)
    assert (kind2, payload2) == ("reject", "volatility_out_of_band")


def test_h2_rejects_eq_pos_not_at_extreme():
    ctx = _ctx(regime_current="range", volatility_atr_percentile=50, eq_pos=0.5)
    kind, payload = shs.h2_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "not_at_extreme")


def test_h2_rejects_htf_bias_opposing_long():
    ctx = _ctx(regime_current="range", volatility_atr_percentile=50, eq_pos=0.1,
              htf_bias="bearish")
    kind, payload = shs.h2_evaluate(ctx, _entry_df(), 100, COST_PCT)
    assert (kind, payload) == ("reject", "htf_bias_opposes")


def test_h2_extreme_timing_off_uses_midpoint_side_for_direction():
    df = _entry_df()  # rolling mid = (90+140)/2 = 115.0
    ctx = _ctx(regime_current="range", volatility_atr_percentile=50, price=95.0,
              htf_bias="neutral")
    kind, payload = shs.h2_evaluate(ctx, df, 100, COST_PCT, extreme_timing=False)
    assert kind == "candidate"
    assert payload["direction"] == "long"  # price below midpoint -> fade long


def test_h2_long_candidate_stop_and_two_stage_target():
    df = _entry_df()
    ctx = _ctx(regime_current="range", volatility_atr_percentile=50, price=95.0,
              atr=2.0, eq_pos=0.1, htf_bias="neutral")
    kind, payload = shs.h2_evaluate(ctx, df, 100, COST_PCT)
    assert kind == "candidate"
    assert payload["direction"] == "long"
    pos = payload["pos"]
    assert pos["stop_loss"] == pytest.approx(90.0 - 0.30 * 2.0)
    assert pos["target_1"] == pytest.approx(115.0)   # midpoint
    assert pos["target_2"] == pytest.approx(140.0)   # opposite extreme


def test_h2_rejects_cost_r_ceiling():
    df = _entry_df()
    ctx = _ctx(regime_current="range", volatility_atr_percentile=50, price=95.0,
              eq_pos=0.1, htf_bias="neutral")
    kind, payload = shs.h2_evaluate(ctx, df, 100, 1.0)
    assert (kind, payload) == ("reject", "cost_r_exceeds_ceiling")


# ---------------------------------------------------------------------------
# No look-ahead.
# ---------------------------------------------------------------------------

def test_rolling_extremes_excludes_current_bar():
    df = _entry_df()  # bar 100's own low/high are 90/140 (flat region)
    lo, hi = shs._rolling_extremes(df, 100, shs.STRUCTURAL_WINDOW_BARS)
    assert lo == 90.0 and hi == 140.0
    # mutate the CURRENT bar (index 100) to an extreme value and confirm
    # the rolling window (which only reads [60, 100)) is unaffected.
    df2 = df.copy()
    df2.loc[100, "low"] = -1e9
    df2.loc[100, "high"] = 1e9
    lo2, hi2 = shs._rolling_extremes(df2, 100, shs.STRUCTURAL_WINDOW_BARS)
    assert (lo2, hi2) == (lo, hi)


def test_rolling_extremes_insufficient_history_returns_none():
    df = _entry_df(n=10)
    lo, hi = shs._rolling_extremes(df, 5, shs.STRUCTURAL_WINDOW_BARS)
    assert (lo, hi) == (None, None)


# ---------------------------------------------------------------------------
# Cost semantics identical to the existing resolver's formula.
# ---------------------------------------------------------------------------

def test_cost_r_matches_deep_backtest_formula():
    import config
    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100
    price, risk = 123.45, 2.5
    expected = cost_pct * price / risk
    assert shs._cost_r(cost_pct, price, risk) == pytest.approx(expected)


def test_cost_r_zero_risk_is_zero_not_division_error():
    assert shs._cost_r(0.001, 100.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# H1/H2 separation + determinism + static safety checks.
# ---------------------------------------------------------------------------

def test_summarize_never_blends_hypotheses():
    h1_records = [{"hypothesis": "H1", "idx": 1, "eligible": True,
                  "net_r": 1.0, "gross_r": 1.1, "cost_r": 0.1,
                  "outcome": "win"}]
    h2_records = [{"hypothesis": "H2", "idx": 1, "eligible": True,
                  "net_r": -1.0, "gross_r": -0.9, "cost_r": 0.1,
                  "outcome": "loss"}]
    s1, s2 = shs.summarize(h1_records), shs.summarize(h2_records)
    assert s1["net_expectancy_r"] == 1.0
    assert s2["net_expectancy_r"] == -1.0
    # never combined: no function in this module accepts both lists at once
    sig_names = [name for name, _ in inspect.getmembers(shs, inspect.isfunction)]
    assert "combined_summary" not in sig_names
    assert "aggregate_hypotheses" not in sig_names


def test_random_direction_baseline_is_deterministic_given_seed():
    df = _entry_df(n=200)
    records = [{"hypothesis": "H1", "idx": 100, "eligible": True,
               "entry": 100.0, "stop": 90.0, "target": 110.0,
               "timestamp": "t", "regime": "trend_up",
               "volatility_percentile": 50}]
    a = shs.random_direction_baseline(records, df, hold_bars=24,
                                      cost_pct=COST_PCT, seed=7,
                                      funding=FUNDING_NOT_MODELLED)
    b = shs.random_direction_baseline(records, df, hold_bars=24,
                                      cost_pct=COST_PCT, seed=7,
                                      funding=FUNDING_NOT_MODELLED)
    assert a == b


def test_walk_hypothesis_rejects_unknown_hypothesis_name():
    with pytest.raises(ValueError):
        shs.walk_hypothesis({}, {}, 100, "H3",
                            funding=FUNDING_NOT_MODELLED)


def test_module_never_monkeypatches_deep_backtest():
    import re
    assert "setattr(deep_backtest" not in SOURCE
    # no `deep_backtest.<name> = ...` assignment anywhere (monkeypatch shape)
    assert not re.search(r"deep_backtest\.\w+\s*=(?!=)", SOURCE)


def test_runtime_does_not_import_swing_hypothesis_simulator():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))
    for path in runtime_files:
        if not path.exists():
            continue
        assert "swing_hypothesis_simulator" not in path.read_text(), path


# ---------------------------------------------------------------------------
# End-to-end wiring on a small synthetic swing dataset.
# ---------------------------------------------------------------------------

def test_run_simulation_end_to_end_h1_h2_separate(tmp_path):
    outdir = tmp_path / "c22"
    _build_swing_dataset(outdir)
    report = shs.run_simulation(str(outdir), "binance", "BTCUSDT", "swing",
                               SWING_BARS)
    assert report["stage"] == "C2.2b"
    assert "H1" in report and "H2" in report
    assert all(t["hypothesis"] == "H1" for t in report["H1"]["trades"])
    assert all(t["hypothesis"] == "H2" for t in report["H2"]["trades"])
    # per-trade schema completeness (spec item 7)
    required_fields = {"hypothesis", "idx", "timestamp", "direction", "entry",
                       "stop", "target", "initial_risk", "rr", "regime",
                       "volatility_percentile", "cost_r", "gross_r", "net_r",
                       "outcome", "exit_idx", "holding_bars", "rejection_reason"}
    for hyp in ("H1", "H2"):
        for t in report[hyp]["trades"]:
            assert required_fields <= set(t.keys())


def test_run_simulation_is_deterministic(tmp_path):
    outdir = tmp_path / "c22_det"
    _build_swing_dataset(outdir)
    r1 = shs.run_simulation(str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS)
    r2 = shs.run_simulation(str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS)
    for hyp in ("H1", "H2"):
        assert r1[hyp]["trades"] == r2[hyp]["trades"]
        assert r1[hyp]["summary"] == r2[hyp]["summary"]


def test_cli_writes_four_files_per_hypothesis(tmp_path):
    outdir = tmp_path / "c22_cli"
    _build_swing_dataset(outdir)
    reports_dir = tmp_path / "reports_c22"
    rc = shs.main([
        "--dataset", str(outdir), "--exchange", "binance", "--symbol",
        "BTCUSDT", "--profile", "swing", "--bars", str(SWING_BARS),
        "--outdir", str(reports_dir),
    ])
    assert rc == 0
    for fname in ("h1_trend_pullback", "h2_range_reversion"):
        assert (reports_dir / f"{fname}.json").exists()
        assert (reports_dir / f"{fname}.txt").exists()
