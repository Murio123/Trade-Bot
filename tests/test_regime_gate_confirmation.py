"""C2.1: tests for tools/regime_gate_confirmation.py.

Pure-function tests use hand-built rows (no real data dependency). One
end-to-end test reuses the swing synthetic-dataset generator from
test_deep_discovery.py (WF_DEPTHS in test_deep_backtest.py is intraday-only)
to prove the CLI/report wiring works without asserting specific numeric
verdicts on unrealistic synthetic R values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_deep_discovery import SWING_BARS, _build_swing_dataset
from tools import kline_cache
from tools import regime_gate_confirmation as rgc


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


# ---------------------------------------------------------------------------
# gate_allows: frozen rule, pre-entry fields only, no look-ahead.
# ---------------------------------------------------------------------------

def test_gate_allows_baseline_always_true():
    assert rgc.gate_allows("A_baseline", "trend_up", "bullish") is True
    assert rgc.gate_allows("A_baseline", "range", "neutral") is True


def test_gate_allows_trend_up_exclusion_only():
    assert rgc.gate_allows("C_trend_up_exclusion_only", "trend_up", "neutral") is False
    assert rgc.gate_allows("C_trend_up_exclusion_only", "range", "bullish") is True
    assert rgc.gate_allows("C_trend_up_exclusion_only", "trend_down", "bullish") is True


def test_gate_allows_range_gate_rejects_trend_up_or_bullish_htf():
    assert rgc.gate_allows("B_range_gate", "trend_up", "neutral") is False
    assert rgc.gate_allows("B_range_gate", "range", "bullish") is False
    assert rgc.gate_allows("B_range_gate", "trend_down", "bearish") is True
    assert rgc.gate_allows("B_range_gate", "range", "neutral") is True


def test_gate_allows_unknown_variant_raises():
    with pytest.raises(ValueError):
        rgc.gate_allows("nonexistent", "range", "neutral")


def test_gate_allows_only_reads_pre_entry_fields():
    # regression: gate_allows must never consult mfe_r/mae_r/outcome/r —
    # its signature physically cannot (only regime_current/htf_bias args
    # are accepted), which is the no-look-ahead guarantee for this stage.
    import inspect
    params = list(inspect.signature(rgc.gate_allows).parameters)
    assert params == ["variant", "regime_current", "htf_bias"]


# ---------------------------------------------------------------------------
# variant_stats / calendar_year_stats / _max_losing_streak — hand-built rows.
# ---------------------------------------------------------------------------

def _row(idx, regime_current, htf_bias, r, outcome, year=2024, gross_r=None,
        cost_r=0.01):
    return {"idx": idx, "regime_current": regime_current, "htf_bias": htf_bias,
           "r": r, "outcome": outcome, "year": year,
           "gross_r": gross_r if gross_r is not None else r + cost_r,
           "cost_r": cost_r}


def _sample_rows():
    rows = []
    # range-context: 10 rows, mostly winning/flat, net positive
    for i in range(10):
        r = 0.3 if i % 3 else -1.0
        rows.append(_row(i, "range", "neutral", r,
                         "win" if r > 0 else "loss", year=2023))
    # trend_up-context: 10 rows, consistently losing
    for i in range(10, 20):
        rows.append(_row(i, "trend_up", "bullish", -0.5, "loss", year=2024))
    return rows


def test_variant_stats_baseline_includes_everything():
    rows = _sample_rows()
    s = rgc.variant_stats(rows, "A_baseline")
    assert s["n"] == 20
    assert s["coverage"] == 1.0
    assert s["rejection_rate"] == 0.0


def test_variant_stats_range_gate_excludes_trend_up_context():
    rows = _sample_rows()
    s = rgc.variant_stats(rows, "B_range_gate")
    assert s["n"] == 10
    assert s["rejection_rate"] == 0.5
    # kept rows are exactly the range-context ones -> net expectancy improves
    baseline = rgc.variant_stats(rows, "A_baseline")
    assert s["net_expectancy_r"] > baseline["net_expectancy_r"]


def test_variant_stats_empty_result_is_none_safe():
    rows = [_row(0, "trend_up", "bullish", -0.5, "loss")]
    s = rgc.variant_stats(rows, "C_trend_up_exclusion_only")
    assert s["n"] == 0
    assert s["net_expectancy_r"] is None
    assert s["max_losing_streak"] is None


def test_max_losing_streak_counts_consecutive_negative_r_in_idx_order():
    rows = [_row(0, "range", "neutral", -1.0, "loss"),
           _row(1, "range", "neutral", -1.0, "loss"),
           _row(2, "range", "neutral", 0.5, "win"),
           _row(3, "range", "neutral", -1.0, "loss")]
    assert rgc._max_losing_streak(rows) == 2


def test_calendar_year_stats_groups_by_year_per_variant():
    rows = _sample_rows()
    year_stats = rgc.calendar_year_stats(rows)
    assert set(year_stats["A_baseline"].keys()) == {"2023", "2024"}
    # 2024 is entirely trend_up-context -> excluded entirely under B
    assert year_stats["B_range_gate"]["2024"]["n"] == 0
    assert year_stats["B_range_gate"]["2023"]["n"] == 10


# ---------------------------------------------------------------------------
# random_rejection_baseline / percentile_rank — reproducibility + shape.
# ---------------------------------------------------------------------------

def test_random_rejection_baseline_is_reproducible_with_fixed_seed():
    rows = _sample_rows()
    a = rgc.random_rejection_baseline(rows, rejection_rate=0.5, n_draws=50)
    b = rgc.random_rejection_baseline(rows, rejection_rate=0.5, n_draws=50)
    assert a["draws"] == b["draws"]
    assert a["n_keep"] == 10


def test_random_rejection_baseline_zero_rejection_keeps_everyone():
    rows = _sample_rows()
    rb = rgc.random_rejection_baseline(rows, rejection_rate=0.0, n_draws=10)
    assert rb["n_keep"] == len(rows)
    expected = float(np.mean([r["r"] for r in rows]))
    assert all(abs(d - expected) < 1e-9 for d in rb["draws"])


def test_percentile_rank_matches_definition():
    draws = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rgc.percentile_rank(3.0, draws) == 0.6  # 3 of 5 draws <= 3.0
    assert rgc.percentile_rank(10.0, draws) == 1.0
    assert rgc.percentile_rank(0.0, draws) == 0.0


def test_percentile_rank_empty_draws_is_nan():
    assert rgc.percentile_rank(1.0, []) != rgc.percentile_rank(1.0, [])


# ---------------------------------------------------------------------------
# evaluate_robustness / decide — synthetic full-sample + wf + year inputs.
# ---------------------------------------------------------------------------

def _full_sample(net_a, net_b, net_c, n_b=100, n_c=100):
    def s(net, n):
        return {"n": n, "net_expectancy_r": net, "coverage": 0.8,
               "rejection_rate": 0.2, "gross_expectancy_r": net,
               "median_r": net, "win_rate": 0.5, "timeout_share": 0.1,
               "sum_r": net * n if net is not None else None,
               "avg_cost_r": 0.01, "max_losing_streak": 3}
    return {"A_baseline": s(net_a, 200), "B_range_gate": s(net_b, n_b),
           "C_trend_up_exclusion_only": s(net_c, n_c)}


def test_decide_proceeds_when_all_checks_pass():
    full_sample = _full_sample(net_a=-0.05, net_b=0.10, net_c=0.08)
    wf = {"fold_stability": {
        "B_range_gate": {"positive_folds_ratio": 1.0, "n_confident_folds": 3},
        "C_trend_up_exclusion_only": {"positive_folds_ratio": 1.0,
                                      "n_confident_folds": 3}}}
    year_stats = {
        "B_range_gate": {"2023": {"net_expectancy_r": 0.1, "n": 50},
                        "2024": {"net_expectancy_r": 0.08, "n": 50}},
        "C_trend_up_exclusion_only": {"2023": {"net_expectancy_r": 0.1, "n": 50},
                                     "2024": {"net_expectancy_r": 0.08, "n": 50}},
    }
    random_baselines = {
        "B_range_gate": {"draws": [-0.05] * 100},
        "C_trend_up_exclusion_only": {"draws": [-0.05] * 100},
    }
    checks = rgc.evaluate_robustness(full_sample, wf, year_stats,
                                     random_baselines)
    assert checks["B_range_gate"]["beats_unfiltered_baseline_net_expectancy"]
    assert checks["B_range_gate"]["net_positive_after_costs"]
    assert checks["B_range_gate"]["beats_random_rejection_p95"]
    decision, _ = rgc.decide(checks)
    assert decision == "PROCEED_TO_REPORT_ONLY_FORWARD_TEST"


def test_decide_rejects_when_net_negative():
    full_sample = _full_sample(net_a=0.05, net_b=-0.10, net_c=-0.08)
    wf = {"fold_stability": {
        "B_range_gate": {"positive_folds_ratio": 0.0, "n_confident_folds": 3},
        "C_trend_up_exclusion_only": {"positive_folds_ratio": 0.0,
                                      "n_confident_folds": 3}}}
    year_stats = {"B_range_gate": {}, "C_trend_up_exclusion_only": {}}
    random_baselines = {
        "B_range_gate": {"draws": [0.05] * 100},
        "C_trend_up_exclusion_only": {"draws": [0.05] * 100},
    }
    checks = rgc.evaluate_robustness(full_sample, wf, year_stats,
                                     random_baselines)
    decision, _ = rgc.decide(checks)
    assert decision == "REJECT_REGIME_GATE"


def test_year_independence_rejects_one_of_two_years_positive():
    # Regression: round(2 * 2/3) == 1 used to let a single positive year out
    # of two qualifying years pass "not_dependent_on_one_calendar_year" —
    # exactly the single-year dependence the check exists to catch.
    full_sample = _full_sample(net_a=-0.02, net_b=0.05, net_c=0.05)
    wf = {"fold_stability": {
        "B_range_gate": {"positive_folds_ratio": 1.0, "n_confident_folds": 3},
        "C_trend_up_exclusion_only": {"positive_folds_ratio": 1.0,
                                      "n_confident_folds": 3}}}
    year_stats = {
        "B_range_gate": {"2023": {"net_expectancy_r": -0.1, "n": 100},
                        "2024": {"net_expectancy_r": 0.2, "n": 100}},
        "C_trend_up_exclusion_only": {"2023": {"net_expectancy_r": -0.1, "n": 100},
                                     "2024": {"net_expectancy_r": 0.2, "n": 100}},
    }
    random_baselines = {
        "B_range_gate": {"draws": [-0.05] * 100},
        "C_trend_up_exclusion_only": {"draws": [-0.05] * 100},
    }
    checks = rgc.evaluate_robustness(full_sample, wf, year_stats,
                                     random_baselines)
    assert checks["B_range_gate"]["not_dependent_on_one_calendar_year"] is False


def test_decide_needs_more_evidence_when_close_but_not_all_pass():
    # net positive, beats baseline, beats random, but fold stability short
    # of 2/3 and only one calendar year available -> not REJECT, not PROCEED.
    full_sample = _full_sample(net_a=-0.02, net_b=0.06, net_c=0.05)
    wf = {"fold_stability": {
        "B_range_gate": {"positive_folds_ratio": 0.5, "n_confident_folds": 2},
        "C_trend_up_exclusion_only": {"positive_folds_ratio": 0.5,
                                      "n_confident_folds": 2}}}
    year_stats = {
        "B_range_gate": {"2024": {"net_expectancy_r": 0.06, "n": 100}},
        "C_trend_up_exclusion_only": {"2024": {"net_expectancy_r": 0.05, "n": 100}},
    }
    random_baselines = {
        "B_range_gate": {"draws": [-0.02] * 100},
        "C_trend_up_exclusion_only": {"draws": [-0.02] * 100},
    }
    checks = rgc.evaluate_robustness(full_sample, wf, year_stats,
                                     random_baselines)
    decision, _ = rgc.decide(checks)
    assert decision == "NEEDS_MORE_EVIDENCE"


# ---------------------------------------------------------------------------
# End-to-end wiring: small synthetic swing dataset, no crash, files written.
# ---------------------------------------------------------------------------

def test_run_confirmation_end_to_end_on_synthetic_swing_dataset(tmp_path):
    outdir = tmp_path / "c21"
    _build_swing_dataset(outdir)
    report = rgc.run_confirmation(str(outdir), "binance", "BTCUSDT", "swing",
                                  SWING_BARS, holdout_frac=0.0, min_folds=1)
    assert report["stage"] == "C2.1"
    assert set(report["full_sample"].keys()) == set(rgc.GATE_VARIANTS)
    assert report["decision"] in ("PROCEED_TO_REPORT_ONLY_FORWARD_TEST",
                                 "NEEDS_MORE_EVIDENCE", "REJECT_REGIME_GATE")
    # holdout sealed and geometry-only, never scored
    assert report["walk_forward"]["holdout"]["sealed"] is True
    assert "net_expectancy_r" not in report["walk_forward"]["holdout"]


def test_cli_writes_three_artifacts(tmp_path):
    outdir = tmp_path / "c21_cli"
    _build_swing_dataset(outdir)
    reports_dir = tmp_path / "reports_c21"
    rc = rgc.main([
        "--dataset", str(outdir), "--exchange", "binance", "--symbol",
        "BTCUSDT", "--profile", "swing", "--bars", str(SWING_BARS),
        "--holdout-frac", "0.0", "--min-folds", "1",
        "--outdir", str(reports_dir),
    ])
    assert rc == 0
    for name in ("regime_gate_confirmation.json", "regime_gate_confirmation.txt",
                "DECISION.md"):
        assert (reports_dir / name).exists()
    payload = json.loads((reports_dir / "regime_gate_confirmation.json").read_text())
    assert payload["stage"] == "C2.1"


def test_runtime_does_not_import_regime_gate_confirmation():
    root = Path(__file__).resolve().parent.parent
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [root / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((root / d).rglob("*.py"))
    for path in runtime_files:
        if not path.exists():
            continue
        text = path.read_text()
        assert "regime_gate_confirmation" not in text, path
