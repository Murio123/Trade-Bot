"""C1.6: тесты offline-диагностики tools/deep_diagnostics.py.

Чистые агрегаты — на синтетических сетапах; контрфакт — на крафтовых свечах
через НАСТОЯЩИЙ deep_backtest.resolve; интеграция — на том же крошечном
random-walk датасете, что и WF-тесты C1.3d. Плюс статические гарантии:
runtime не импортирует тул, совета по порогу в исходнике нет.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import tools.deep_backtest as deep_backtest
import tools.deep_diagnostics as dd
from tests.test_deep_backtest import WF_BARS, WF_DEPTHS, _build_random_walk_dataset

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "deep_diagnostics.py").read_text()


def S(idx=0, score=8, direction="long", rc="bull", rp="bull",
      outcome="win", r=1.0, hit_tp1=True):
    return {"idx": idx, "score": score, "direction": direction,
            "regime_current": rc, "regime_live_parity": rp,
            "outcome": outcome, "r": r, "hit_tp1": hit_tp1,
            "exit_idx": idx + 1}


# ---------------------------------------------------------------------------
# Статические гарантии
# ---------------------------------------------------------------------------

def test_runtime_does_not_import_deep_diagnostics():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))
    for path in runtime_files:
        if not path.exists():
            continue
        assert not re.search(r"\bdeep_diagnostics\b", path.read_text()), \
            f"{path} imports deep_diagnostics"


def test_source_has_no_threshold_advice_token():
    assert "recommended_threshold" not in SOURCE


def test_deep_backtest_source_unchanged_by_c16():
    # C1.6 не трогает deep_backtest: рекордер permанентно не вживляется.
    src = (ROOT / "tools" / "deep_backtest.py").read_text()
    assert "deep_diagnostics" not in src


# ---------------------------------------------------------------------------
# Чистые агрегаты
# ---------------------------------------------------------------------------

def test_group_stats_basic_and_unresolved_excluded():
    setups = [
        S(outcome="win", r=1.2), S(outcome="loss", r=-1.05),
        S(outcome="timeout", r=0.3), S(outcome="breakeven", r=-0.05),
        S(outcome="unresolved", r=None),
    ]
    table = dd.group_stats(setups, lambda s: "all")
    assert list(table) == ["all"]
    row = table["all"]
    assert row["n"] == 4                        # unresolved не в знаменателе
    assert row["win_rate"] == 0.25
    assert row["mean_r"] == round((1.2 - 1.05 + 0.3 - 0.05) / 4, 4)
    assert row["median_r"] == round((0.3 - 0.05) / 2, 4)
    assert row["sum_r"] == 0.4
    assert row["timeout_share"] == 0.25


def test_group_stats_groups_sorted_and_disjoint():
    setups = [S(direction="short", r=-1.0, outcome="loss"),
              S(direction="long", r=1.0), S(direction="long", r=0.5)]
    table = dd.group_stats(setups, lambda s: s["direction"])
    assert list(table) == ["long", "short"]
    assert table["long"]["n"] == 2 and table["short"]["n"] == 1


def test_score_bucket_edges():
    assert dd.score_bucket(5) == "[5,6)"
    assert dd.score_bucket(9) == "[9,10)"
    assert dd.score_bucket(10) == "[10,inf)"
    assert dd.score_bucket(14) == "[10,inf)"


def test_trend_alignment_mapping():
    # Словарь режимов — реальный выход detect_regime, не bull/bear.
    assert dd.trend_alignment("long", "trend_up") == "with_trend"
    assert dd.trend_alignment("short", "trend_up") == "counter_trend"
    assert dd.trend_alignment("short", "trend_down") == "with_trend"
    assert dd.trend_alignment("long", "trend_down") == "counter_trend"
    assert dd.trend_alignment("long", "range") == "neutral"
    assert dd.trend_alignment("short", "high_volatility") == "neutral"


def _bucket_table(means, n=100):
    labels = ["[5,6)", "[6,7)", "[7,8)", "[8,9)", "[9,10)", "[10,inf)"]
    return {lab: {"n": n, "mean_r": m} for lab, m in zip(labels, means)}


def test_monotonicity_improving_worsening_flat_mixed():
    assert dd.score_monotonicity(
        _bucket_table([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]))["trend"] == "improving"
    worse = dd.score_monotonicity(
        _bucket_table([0.2, 0.1, 0.0, -0.1, -0.2, -0.3]))
    assert worse["trend"] == "worsening"
    assert worse["top_vs_bottom"] == -0.5
    assert dd.score_monotonicity(
        _bucket_table([-0.1, -0.1, -0.11, -0.09, -0.1, -0.1]))["trend"] == "flat"
    assert dd.score_monotonicity(
        _bucket_table([-0.3, 0.1, -0.3, 0.1, -0.3, 0.1]))["trend"] == "mixed"


def test_monotonicity_ignores_small_buckets_and_insufficient():
    table = _bucket_table([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    for lab in list(table)[1:]:                 # остаётся один крупный бакет
        table[lab]["n"] = 3
    assert dd.score_monotonicity(table)["trend"] == "insufficient"


def test_d13_summary_sums():
    setups = [S(rc="bull", rp="bull", r=1.0),
              S(rc="bull", rp="range", r=-2.0, outcome="loss"),
              S(rc="bear", rp="bear", r=-0.5, outcome="loss", direction="short")]
    d13 = dd.d13_summary(setups)
    assert d13["disagreement_share"] == round(1 / 3, 4)
    assert d13["sum_r_from_disagreement"] == -2.0
    assert d13["total_sum_r"] == -1.5
    assert d13["expectancy_excluding_disagreement"] == 0.25
    assert d13["groups"]["disagree"]["n"] == 1


# ---------------------------------------------------------------------------
# Disposition
# ---------------------------------------------------------------------------

def _tables_uniform(mean_r, n=100):
    row = {"n": n, "win_rate": 0.2, "mean_r": mean_r, "median_r": mean_r,
           "sum_r": mean_r * n, "timeout_share": 0.1}
    return {dim: {"g": dict(row)} for dim in
            ("direction", "regime_current", "regime_agreement", "score_bucket")}


def test_disposition_structural_negative():
    out = dd.classify_disposition(_tables_uniform(-0.15), total_resolved=500)
    assert out["label"] == "structural_negative"


def test_disposition_targeted_fix_candidate():
    tables = _tables_uniform(-0.15)
    tables["direction"]["long"] = {"n": 80, "win_rate": 0.4, "mean_r": 0.08,
                                   "median_r": 0.0, "sum_r": 6.4,
                                   "timeout_share": 0.1}
    out = dd.classify_disposition(tables, total_resolved=500)
    assert out["label"] == "targeted_fix_candidate"
    assert "direction=long" in out["reason"]


def test_disposition_needs_more_evidence_small_sample():
    out = dd.classify_disposition(_tables_uniform(-0.15), total_resolved=50)
    assert out["label"] == "needs_more_evidence"


def test_disposition_needs_more_evidence_between_zero_and_eps():
    out = dd.classify_disposition(_tables_uniform(0.01), total_resolved=500)
    assert out["label"] == "needs_more_evidence"


def test_disposition_never_auto_assigns_technical_dry_run():
    assert "technical_dry_run_only" in dd.DISPOSITIONS
    for tables, n in [(_tables_uniform(-0.2), 500),
                      (_tables_uniform(0.2), 500),
                      (_tables_uniform(-0.2), 10)]:
        assert dd.classify_disposition(tables, n)["label"] != \
            "technical_dry_run_only"


# ---------------------------------------------------------------------------
# Контрфакт таймаут-горизонта (через НАСТОЯЩИЙ resolve)
# ---------------------------------------------------------------------------

def _entry_df(rows):
    return pd.DataFrame(rows, columns=["high", "low", "close"])


POS = {"entry_price": 100.0, "stop_loss": 95.0,
       "target_1": 103.0, "target_2": 106.0}


def test_timeout_counterfactual_converts_to_win():
    # hold=2: таймаут по close=101 (r=0.2 до издержек); hold=4: TP1 на баре 3,
    # TP2 на баре 4 (low=104 не задевает брейкивен) -> win r=1.2 до издержек.
    df = _entry_df([(100.0, 100.0, 100.0),     # bar 0 = entry
                    (101.0, 99.0, 100.5),
                    (102.0, 99.0, 101.0),
                    (104.0, 101.0, 103.5),
                    (107.0, 104.0, 106.5)])
    base = deep_backtest.resolve(df, 0, "long", POS, 2)
    assert base["outcome"] == "timeout" and base["r"] == 0.2

    cost = dd._cost_r(POS)
    setup = S(idx=0, outcome="timeout", r=round(0.2 - cost, 2), hit_tp1=False)
    cf = dd.timeout_counterfactual(df, [setup], {0: POS}, hold_bars=2,
                                   multipliers=(2,))
    assert cf["n_timeouts"] == 1
    assert cf["hit_tp1_share"] == 0.0
    hz = cf["horizons"]["x2"]
    assert hz["hold_bars"] == 4
    assert hz["conversions"]["timeout_to_win"] == 1
    assert hz["n_re_resolved"] == 1
    expected_delta = round(1.2 - cost, 2) - round(0.2 - cost, 2)
    assert hz["delta_mean_r"] == round(expected_delta, 4)


def test_timeout_counterfactual_counts_unresolved_extension():
    # История длиной 3 бара: hold=2 -> timeout ровно на последнем баре;
    # hold=4 -> будущего не хватает, исход неизвестен (не в delta).
    df = _entry_df([(100.0, 100.0, 100.0),
                    (101.0, 99.0, 100.5),
                    (102.0, 99.0, 101.0)])
    base = deep_backtest.resolve(df, 0, "long", POS, 2)
    assert base["outcome"] == "timeout"

    setup = S(idx=0, outcome="timeout", r=0.2, hit_tp1=False)
    cf = dd.timeout_counterfactual(df, [setup], {0: POS}, hold_bars=2,
                                   multipliers=(2,))
    hz = cf["horizons"]["x2"]
    assert hz["conversions"]["became_unresolved"] == 1
    assert hz["n_re_resolved"] == 0
    assert hz["delta_mean_r"] is None


def test_timeout_counterfactual_empty_when_no_timeouts():
    cf = dd.timeout_counterfactual(_entry_df([(100.0, 100.0, 100.0)]),
                                   [S(outcome="win", r=1.0)], {}, hold_bars=2)
    assert cf["n_timeouts"] == 0
    assert cf["baseline_mean_r"] is None
    assert cf["horizons"]["x2"]["delta_mean_r"] is None


# ---------------------------------------------------------------------------
# Интеграция на крошечном датасете (тот же random walk, что WF-тесты C1.3d)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diag_report(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c16") / "binance"
    _build_random_walk_dataset(outdir, depths=WF_DEPTHS)
    report = dd.run_diagnostics(str(outdir), "binance", "BTCUSDT",
                                "intraday", WF_BARS)
    return report


def test_report_structure_and_counts(diag_report):
    r = diag_report
    assert r["stage"] == "C1.6" and r["kind"] == "deep_diagnostics"
    c = r["counts"]
    assert c["setups"] == c["resolved"] + c["unresolved"]
    for name in ("exit_reason", "direction", "regime_current",
                 "regime_live_parity", "regime_agreement", "score_bucket",
                 "direction_x_regime", "trend_alignment"):
        assert name in r["tables"]
        for row in r["tables"][name].values():
            for stat in ("n", "win_rate", "mean_r", "median_r", "sum_r",
                         "timeout_share"):
                assert stat in row
    # exit_reason покрывает все resolved-сетапы без пересечений
    assert sum(row["n"] for row in r["tables"]["exit_reason"].values()) == \
        c["resolved"]


def test_report_no_threshold_advice_and_disposition_enum(diag_report):
    blob = json.dumps(diag_report, default=str)
    assert "recommended_threshold" not in blob
    assert diag_report["disposition"]["label"] in dd.DISPOSITIONS
    assert diag_report["disposition"]["label"] != "technical_dry_run_only"


def test_timeout_counterfactual_swing_only(diag_report):
    assert diag_report["timeout_counterfactual"] is None  # intraday-прогон


def test_text_report_sections(diag_report):
    text = dd.format_report(diag_report)
    for token in ("deep diagnostics (C1.6)", "exit_reason:", "score_bucket:",
                  "score->R monotonicity:", "D13:", "disposition:",
                  "limitations:"):
        assert token in text
    assert "recommended_threshold" not in text


def test_walk_with_positions_covers_setups_and_restores(tmp_path):
    outdir = _build_random_walk_dataset(tmp_path / "binance",
                                        depths=WF_DEPTHS)
    original = deep_backtest.resolve
    frames, profile, _t, _cvd = deep_backtest.prepare(
        outdir, "binance", "BTCUSDT", "intraday", WF_BARS, 0.001, False)
    walk, positions = dd.walk_with_positions(frames, profile, WF_BARS)
    assert deep_backtest.resolve is original          # рекордер снят
    idxs = {s["idx"] for s in walk["setups"]}
    assert idxs and idxs <= set(positions)
    sample = positions[next(iter(idxs))]
    for key in ("entry_price", "stop_loss", "target_1", "target_2"):
        assert key in sample


def test_cli_writes_reports(tmp_path, capsys):
    dataset = _build_random_walk_dataset(tmp_path / "binance",
                                         depths=WF_DEPTHS)
    outdir = tmp_path / "out"
    rc = dd.main(["--dataset", dataset, "--exchange", "binance",
                  "--profile", "intraday", "--bars", str(WF_BARS),
                  "--outdir", str(outdir), "--json"])
    assert rc == 0
    payload = json.loads((outdir / "intraday_binance_diagnostics.json")
                         .read_text())
    assert payload["stage"] == "C1.6"
    text = (outdir / "intraday_binance_diagnostics.txt").read_text()
    assert "disposition:" in text
    assert json.loads(capsys.readouterr().out)["stage"] == "C1.6"
