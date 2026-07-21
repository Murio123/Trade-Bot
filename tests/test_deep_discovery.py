"""C1.8: тесты рекордера tools/deep_discovery.py (шаги 1-2, recorder-only).

Статические гарантии — как в C1.6; эквивалентность рекордера настоящему
deep_walk — на синтетическом random-walk датасете, собранном локально под
профиль swing (4h/1d/12h/6h), т.к. WF_DEPTHS в test_deep_backtest.py собран
под intraday (15m) и не покрывает swing-таймфреймы. Юнит-тесты чистых
хелперов — на крафтовых свечах.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.deep_backtest as deep_backtest
import tools.deep_discovery as dd
from tools import kline_cache
from tools.deep_backtest import prepare

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "deep_discovery.py").read_text()

MS = kline_cache.INTERVAL_MS
END_MS = 1_700_006_400_000  # 2023-11-15T00:00:00Z, кратно 86_400_000
assert END_MS % 86_400_000 == 0

TF_MINUTES = {"4h": 240, "6h": 360, "12h": 720, "1d": 1440}

# Глубины с запасом под ENTRY_WARMUP(300)/HTF_WARMUP(210)/ZONE_MIN_BARS(30)
# для профиля swing (entry=4h, htf=1d, zone_tfs=12h/6h).
SWING_BARS = 60
SWING_DEPTHS = {"4h": 420, "1d": 320, "12h": 260, "6h": 340}


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


def _swing_frame(timeframe: str, n: int, seed: int) -> pd.DataFrame:
    # drift=0.0/vol=1.5/wick=0.004 подобраны так, чтобы воронка swing (block_
    # counter_trend + MIN_DIVERSE_CATEGORIES) дала и resolved, и unresolved
    # сетапы (см. поиск по сиду в C1.8: seed=4 даёт 42 сетапа / 21 resolved).
    frac = TF_MINUTES[timeframe] / 1440.0
    rng = np.random.default_rng([seed, TF_MINUTES[timeframe]])
    start = 30_000.0
    closes = start * np.exp(np.cumsum(rng.normal(0.0 * frac,
                                                  1.5 * np.sqrt(frac), n)))
    opens = np.concatenate([[start], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = np.abs(rng.normal(100, 20, n))

    interval = MS[timeframe]
    open_times = [END_MS - interval * k for k in range(n, 0, -1)]
    df = pd.DataFrame({
        "open_time": pd.to_datetime(open_times, unit="ms", utc=True),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volume, "quote_volume": volume * closes,
        "taker_buy_base": volume * rng.uniform(0.3, 0.7, n),
    })
    df["close_time"] = df["open_time"] + pd.to_timedelta(interval, unit="ms")
    return df


def _build_swing_dataset(outdir, seed: int = 4,
                         depths: dict[str, int] | None = None) -> str:
    for tf, n in (depths or SWING_DEPTHS).items():
        kline_cache.write_dataset(_swing_frame(tf, n, seed), str(outdir),
                                  exchange="binance", symbol="BTCUSDT",
                                  timeframe=tf, requested_bars=n,
                                  duplicate_count_removed=0)
    return str(outdir)


@pytest.fixture(scope="module")
def swing_walk_inputs(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c18") / "binance"
    _build_swing_dataset(outdir)
    frames, profile, _table, _cvd = prepare(
        str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS, 0.001, False)
    return frames, profile


# ---------------------------------------------------------------------------
# Статические гарантии
# ---------------------------------------------------------------------------

def test_runtime_does_not_import_deep_discovery():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))
    for path in runtime_files:
        if not path.exists():
            continue
        assert not re.search(r"\bdeep_discovery\b", path.read_text()), \
            f"{path} imports deep_discovery"


def test_source_has_no_threshold_or_production_advice_token():
    for token in ("recommended_threshold", "proceed_to_confirmation=yes",
                  "SCORE_ALERT_MIN"):
        assert token not in SOURCE


def test_deep_backtest_source_unchanged_by_c18():
    src = (ROOT / "tools" / "deep_backtest.py").read_text()
    assert "deep_discovery" not in src


def test_no_new_indicator_names_in_source():
    for forbidden in ("talib", "\"rsi\"", "'rsi'", "macd_line", "stochastic",
                      "supertrend", " adx", "ADX"):
        assert forbidden not in SOURCE, forbidden


# ---------------------------------------------------------------------------
# Рекордер: эквивалентность настоящему deep_walk, восстановление оригиналов
# ---------------------------------------------------------------------------

def test_recorder_restores_all_wrapped_originals(swing_walk_inputs):
    frames, profile = swing_walk_inputs
    wrapped_names = ("calculate_confluence_score", "compute_equilibrium",
                     "_build_htf_zones", "compute_cvd_from_klines",
                     "analyze_volatility", "get_htf_bias", "resolve")
    before = {name: getattr(deep_backtest, name) for name in wrapped_names}
    dd.record_walk(frames, profile, SWING_BARS)
    after = {name: getattr(deep_backtest, name) for name in wrapped_names}
    assert before == after


def test_recorder_output_matches_unwrapped_walk_bit_for_bit(swing_walk_inputs):
    frames, profile = swing_walk_inputs
    walk_plain = deep_backtest.deep_walk(frames, profile, SWING_BARS)
    walk_rec, records = dd.record_walk(frames, profile, SWING_BARS)

    assert walk_plain["bars_walked"] == walk_rec["bars_walked"]
    assert walk_plain["bars_evaluated"] == walk_rec["bars_evaluated"]
    assert walk_plain["setups"] == walk_rec["setups"]
    assert len(walk_rec["setups"]) > 0  # проверка не проходит вхолостую


def test_recorder_captures_a_record_for_every_resolved_setup(swing_walk_inputs):
    frames, profile = swing_walk_inputs
    walk, records = dd.record_walk(frames, profile, SWING_BARS)
    resolved = [s for s in walk["setups"] if s["outcome"] != "unresolved"]
    assert resolved  # датасет должен давать хотя бы один resolved сетап
    for s in resolved:
        assert s["idx"] in records


def test_recorder_restores_originals_even_if_deep_walk_raises(monkeypatch,
                                                               swing_walk_inputs):
    frames, profile = swing_walk_inputs
    original_resolve = deep_backtest.resolve

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(deep_backtest, "deep_walk", boom)
    with pytest.raises(RuntimeError):
        dd.record_walk(frames, profile, SWING_BARS)
    assert deep_backtest.resolve is original_resolve
    assert dd._recording is False


def test_recorder_rejects_reentrant_calls(monkeypatch, swing_walk_inputs):
    frames, profile = swing_walk_inputs
    original_resolve = deep_backtest.resolve

    def nested_call(*args, **kwargs):
        with pytest.raises(RuntimeError, match="not reentrant"):
            dd.record_walk(frames, profile, SWING_BARS)
        raise RuntimeError("outer aborted after nested attempt")

    monkeypatch.setattr(deep_backtest, "deep_walk", nested_call)
    with pytest.raises(RuntimeError, match="outer aborted"):
        dd.record_walk(frames, profile, SWING_BARS)
    # guard and originals both cleaned up after the (failed) outer call
    assert dd._recording is False
    assert deep_backtest.resolve is original_resolve


# ---------------------------------------------------------------------------
# build_feature_rows
# ---------------------------------------------------------------------------

def test_build_feature_rows_only_covers_resolved_setups(swing_walk_inputs):
    frames, profile = swing_walk_inputs
    walk, records = dd.record_walk(frames, profile, SWING_BARS)
    rows = dd.build_feature_rows(walk, records)
    resolved = [s for s in walk["setups"] if s["outcome"] != "unresolved"]
    assert len(rows) == len(resolved)
    for row in rows:
        assert row["outcome"] != "unresolved"
        assert "score_components" in row
        assert "mfe_r" in row and "mae_r" in row
        assert 0 <= row["weekday"] <= 6
        assert 1 <= row["month"] <= 12
        assert row["cost_r"] >= 0


# ---------------------------------------------------------------------------
# Чистые хелперы на крафтовых данных
# ---------------------------------------------------------------------------

def _candles(prices: list[tuple[float, float]]) -> pd.DataFrame:
    """(high, low) pairs -> minimal df usable by _mfe_mae."""
    return pd.DataFrame({"high": [p[0] for p in prices],
                         "low": [p[1] for p in prices]})


def test_mfe_mae_long_tracks_running_extremes():
    # entry=100, stop=95 (risk=5). Path after entry: favorable to 112 (MFE
    # 2.4R), adverse dip to 97 (MAE 0.6R) before continuing to timeout.
    df = _candles([(100, 99),   # idx 0: entry bar (unused by scan)
                   (103, 97),   # idx 1: adverse to 97 -> 0.6R
                   (112, 108),  # idx 2: favorable to 112 -> 2.4R
                   (110, 105)]) # idx 3: horizon
    pos = {"entry_price": 100.0, "stop_loss": 95.0}
    out = dd._mfe_mae(df, entry_idx=0, direction="long", pos=pos, hold_bars=3)
    assert out["mfe_r"] == pytest.approx(2.4)
    assert out["mae_r"] == pytest.approx(0.6)
    assert out["time_to_mfe"] == 2
    assert out["time_to_mae"] == 1


def test_mfe_mae_short_mirrors_long():
    # short: favorable = price falling (entry-low), adverse = price rising
    # (high-entry). idx1 is the adverse extreme, idx2 the favorable extreme.
    df = _candles([(100, 99), (103, 98), (100, 88), (100, 95)])
    pos = {"entry_price": 100.0, "stop_loss": 105.0}
    out = dd._mfe_mae(df, entry_idx=0, direction="short", pos=pos, hold_bars=3)
    assert out["mfe_r"] == pytest.approx((100 - 88) / 5)
    assert out["mae_r"] == pytest.approx((103 - 100) / 5)


def test_mfe_mae_window_matches_resolve_horizon():
    # 5 bars total; hold_bars=2 -> scan must stop at entry_idx+2, never touch
    # bar 3 (which would otherwise dominate MFE).
    df = _candles([(100, 99), (101, 100), (102, 101), (200, 199), (100, 50)])
    pos = {"entry_price": 100.0, "stop_loss": 95.0}
    out = dd._mfe_mae(df, entry_idx=0, direction="long", pos=pos, hold_bars=2)
    assert out["mfe_r"] == pytest.approx((102 - 100) / 5)


def test_ema_slope_positive_for_uptrend():
    close = pd.Series(list(range(1, 101)), dtype=float)
    slope = dd._ema_slope(pd.DataFrame({"close": close}), price=100.0)
    assert slope is not None and slope > 0


def test_ema_slope_none_when_too_short():
    close = pd.Series([1.0, 2.0, 3.0])
    assert dd._ema_slope(pd.DataFrame({"close": close}), price=2.0) is None


def test_nearest_zone_box_prefers_ob_then_fvg():
    zones = {
        "order_blocks": {"price_in_bullish_ob": True,
                         "bullish_ob": {"low": 1, "high": 2, "tf": "12h",
                                        "index": 5}},
        "fvg": {"price_in_bullish_fvg": True,
               "bullish_fvg": {"low": 3, "high": 4, "tf": "6h", "index": 7}},
    }
    box, kind = dd._nearest_zone_box(zones, "long")
    assert kind == "ob" and box["low"] == 1


def test_nearest_zone_box_falls_back_to_fvg():
    zones = {
        "order_blocks": {"price_in_bullish_ob": False, "bullish_ob": None},
        "fvg": {"price_in_bullish_fvg": True,
               "bullish_fvg": {"low": 3, "high": 4, "tf": "6h", "index": 7}},
    }
    box, kind = dd._nearest_zone_box(zones, "long")
    assert kind == "fvg" and box["low"] == 3


def test_nearest_zone_box_none_when_absent():
    zones = {"order_blocks": {"price_in_bullish_ob": False, "bullish_ob": None},
             "fvg": {"price_in_bullish_fvg": False, "bullish_fvg": None}}
    box, kind = dd._nearest_zone_box(zones, "long")
    assert box is None and kind is None


def test_zone_age_from_formation_index():
    zdf = pd.DataFrame({"close": range(20)})
    box = {"tf": "6h", "index": 15}
    assert dd._zone_age(box, {"6h": zdf}) == 20 - 1 - 15


def test_zone_age_none_when_tf_missing():
    box = {"tf": "6h", "index": 5}
    assert dd._zone_age(box, {}) is None


def test_nearest_level_distance():
    levels = {"highs": [110.0], "lows": [90.0]}
    assert dd._nearest_level_distance(levels, price=100.0, atr=5.0) == \
        pytest.approx(2.0)


def test_nearest_level_distance_none_without_atr():
    levels = {"highs": [110.0], "lows": [90.0]}
    assert dd._nearest_level_distance(levels, price=100.0, atr=None) is None


# ---------------------------------------------------------------------------
# Part 3 — single-feature attribution, on crafted synthetic rows.
# ---------------------------------------------------------------------------

def _row(idx, *, flag=True, r=1.0, gross_r=None, outcome="win", year=2021,
        regime="range", direction="long"):
    return {
        "idx": idx, "direction": direction, "score": 8,
        "regime_current": regime, "regime_live_parity": regime,
        "outcome": outcome, "hit_tp1": True, "r": r,
        "gross_r": gross_r if gross_r is not None else r,
        "year": year, "some_flag": flag,
    }


def _make_rows(n_true, n_false, *, r_true, r_false, year_fn=None,
              regime_fn=None):
    """Interleaved by idx (even=true, odd=false), so every temporal fold and
    every year contains both buckets — mirrors real market data where a
    feature's value alternates over time rather than sitting in one block.
    """
    rows = []
    for i in range(n_true):
        idx = 2 * i
        rows.append(_row(idx, flag=True, r=r_true,
                         year=year_fn(idx) if year_fn else 2021,
                         regime=regime_fn(idx) if regime_fn else "range"))
    for i in range(n_false):
        idx = 2 * i + 1
        rows.append(_row(idx, flag=False, r=r_false,
                         year=year_fn(idx) if year_fn else 2021,
                         regime=regime_fn(idx) if regime_fn else "range"))
    return rows


# "some_flag" is deliberately NOT registered in CATEGORICAL_FEATURES/
# ALL_FEATURES: feature_bucket_table's categorical branch is the default for
# any name not in NUMERIC_FEATURES, so the generic machinery is exercised
# here without needing to touch the real recorder feature registry.

def test_feature_bucket_table_categorical_splits_by_value():
    rows = _make_rows(60, 60, r_true=0.5, r_false=-0.5)
    table = dd.feature_bucket_table(rows, "some_flag")
    assert set(table) == {"True", "False"}
    assert table["True"]["n"] == 60
    assert table["True"]["net_mean_r"] == pytest.approx(0.5)
    assert table["False"]["net_mean_r"] == pytest.approx(-0.5)


def test_stability_reuses_pooled_numeric_bucket_edges_not_resplit_quantiles(
        monkeypatch):
    # Regression for a Codex-flagged defect: numeric bucket edges (tertiles)
    # are a property of the POOLED distribution. Recomputing pd.qcut on a
    # split subset would silently redefine "low/mid/high" per split, so a
    # stability check would compare differently-defined buckets rather than
    # testing whether the pooled best-vs-worst sign replicates.
    monkeypatch.setattr(dd, "NUMERIC_FEATURES", dd.NUMERIC_FEATURES + ["metric"])
    rows = []
    for i in range(30):
        row = _row(i, r=float(i))
        row["metric"] = float(i)  # pooled tertiles: [0,9]=low [10,19]=mid [20,29]=high
        rows.append(row)

    pooled_labels = dd._bucket_labels(rows, "metric")
    subset = rows[:10]  # entirely pooled "low"

    # Recomputing quantiles locally on a pooled-single-bucket subset invents
    # fake separation (three fresh thirds where none should exist).
    local_table = dd.feature_bucket_table(subset, "metric", labels=None)
    assert set(local_table) == {"low", "mid", "high"}

    # Using the pooled labels correctly recognizes every row here shares one
    # pooled bucket — this is the path feature_bucket_table/_stability use
    # for split-level calls once labels are threaded through explicitly.
    pooled_table = dd.feature_bucket_table(subset, "metric", labels=pooled_labels)
    assert set(pooled_table) == {"low"}


def test_feature_separation_identifies_best_and_worst():
    rows = _make_rows(60, 60, r_true=0.5, r_false=-0.5)
    table = dd.feature_bucket_table(rows, "some_flag")
    sep = dd.feature_separation(table)
    assert sep["best"] == "True" and sep["worst"] == "False"
    assert sep["spread"] == pytest.approx(1.0)
    assert sep["sign"] == "positive"


def test_feature_separation_none_below_min_group_n():
    rows = _make_rows(10, 10, r_true=0.5, r_false=-0.5)
    table = dd.feature_bucket_table(rows, "some_flag")
    assert dd.feature_separation(table) is None


def test_classify_feature_keep_when_stable_across_years():
    # 4 years, each with a clean positive split -> stable sign every year.
    # year_fn receives the *idx* (even=true, odd=false per _make_rows), so it
    # must key off idx // 2 to keep both buckets co-present in every year —
    # keying off idx % 4 directly would put true (even idx) and false (odd
    # idx) rows in disjoint year sets, leaving every year bucket-incomplete.
    def year_fn(idx):
        return 2018 + ((idx // 2) % 4)
    rows = _make_rows(240, 240, r_true=0.4, r_false=-0.4, year_fn=year_fn)
    labels = dd._bucket_labels(rows, "some_flag")
    table = dd.feature_bucket_table(rows, "some_flag", labels=labels)
    sep = dd.feature_separation(table)
    fold_of = dd._fold_assignment(rows)
    fold_stab = dd._stability(rows, "some_flag", sep,
                              lambda r: fold_of.get(r["idx"]), labels)
    year_stab = dd._stability(rows, "some_flag", sep, lambda r: r["year"],
                              labels)
    result = dd.classify_feature("some_flag", table, fold_stab, year_stab,
                                 len(rows))
    assert result["label"] == "PRELIMINARY_KEEP"


def test_classify_feature_remove_when_flat():
    rows = _make_rows(120, 120, r_true=0.01, r_false=-0.01)
    labels = dd._bucket_labels(rows, "some_flag")
    table = dd.feature_bucket_table(rows, "some_flag", labels=labels)
    sep = dd.feature_separation(table)
    fold_of = dd._fold_assignment(rows)
    fold_stab = dd._stability(rows, "some_flag", sep,
                              lambda r: fold_of.get(r["idx"]), labels)
    year_stab = dd._stability(rows, "some_flag", sep, lambda r: r["year"],
                              labels)
    result = dd.classify_feature("some_flag", table, fold_stab, year_stab,
                                 len(rows))
    assert result["label"] == "PRELIMINARY_REMOVE"


def test_classify_feature_remove_when_best_bucket_not_positive():
    rows = _make_rows(120, 120, r_true=-0.1, r_false=-0.8)
    labels = dd._bucket_labels(rows, "some_flag")
    table = dd.feature_bucket_table(rows, "some_flag", labels=labels)
    sep = dd.feature_separation(table)
    fold_of = dd._fold_assignment(rows)
    fold_stab = dd._stability(rows, "some_flag", sep,
                              lambda r: fold_of.get(r["idx"]), labels)
    year_stab = dd._stability(rows, "some_flag", sep, lambda r: r["year"],
                              labels)
    result = dd.classify_feature("some_flag", table, fold_stab, year_stab,
                                 len(rows))
    assert result["label"] == "PRELIMINARY_REMOVE"


def test_classify_feature_unknown_when_sample_too_small():
    rows = _make_rows(60, 60, r_true=0.5, r_false=-0.5)
    table = dd.feature_bucket_table(rows, "some_flag")
    sep = dd.feature_separation(table)
    fold_stab = year_stab = {"eligible": False}
    result = dd.classify_feature("some_flag", table, fold_stab, year_stab,
                                 len(rows))
    assert result["label"] == "PRELIMINARY_UNKNOWN"
    assert "< 200" in result["reason"] or "resolved" in result["reason"]


def test_classify_feature_unknown_when_unstable_across_years():
    # Positive pooled spread (clears both the flat and positive-eps gates)
    # but the sign-replication check itself reports low agreement -> not
    # enough to confirm KEEP. Stability dicts are crafted directly here
    # (rather than derived from a synthetic year split) so the test isolates
    # classify_feature's own gating logic from _stability's bucket-size
    # sensitivity, which is covered separately by test_classify_feature_*
    # sample-size tests and the real _stability unit coverage above.
    rows = _make_rows(120, 120, r_true=0.4, r_false=-0.4)
    table = dd.feature_bucket_table(rows, "some_flag")
    fold_stab = {"eligible": True, "agree": 3, "evaluated": 4,
                "agreement_ratio": 0.75}
    year_stab = {"eligible": True, "agree": 1, "evaluated": 3,
                "agreement_ratio": 0.33}
    result = dd.classify_feature("some_flag", table, fold_stab, year_stab,
                                 len(rows))
    assert result["label"] == "PRELIMINARY_UNKNOWN"


def test_regime_stability_not_applicable_for_regime_current_itself():
    rows = _make_rows(120, 120, r_true=0.4, r_false=-0.4)
    features = dd.run_single_feature_attribution(rows)
    assert features["regime_current"]["regime_stability"]["eligible"] is False


def test_score_bucket_edges():
    assert dd.score_bucket(5) == "[5,6)"
    assert dd.score_bucket(9.9) == "[9,10)"
    assert dd.score_bucket(10) == "[10,inf)"
    assert dd.score_bucket(4) == "below_min"


def test_build_summary_table_matches_report_shape():
    rows = _make_rows(120, 120, r_true=0.4, r_false=-0.4)
    features = dd.run_single_feature_attribution(rows)
    summary = dd.build_summary_table(features)
    assert {r["feature"] for r in summary} == set(features)
    for row in summary:
        for key in ("feature", "sample", "gross_r", "net_r",
                   "fold_stability", "year_stability", "regime_stability",
                   "classification"):
            assert key in row


# ---------------------------------------------------------------------------
# Integration: report structure + no threshold-advice tokens
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def discovery_report(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c18run") / "binance"
    _build_swing_dataset(outdir)
    return dd.run_discovery(str(outdir), "binance", "BTCUSDT", "swing",
                            SWING_BARS)


def test_discovery_report_top_level_keys(discovery_report):
    r = discovery_report
    assert r["stage"] == "C1.8" and r["kind"] == "single_feature_attribution"
    assert set(r["counts"]) == {"setups", "resolved", "unresolved"}
    assert set(r["features"]) == set(dd.ALL_FEATURES)
    for name, f in r["features"].items():
        assert set(f) == {"kind", "buckets", "separation", "fold_stability",
                          "year_stability", "regime_stability",
                          "classification"}
        assert f["classification"]["label"] in (
            "PRELIMINARY_KEEP", "PRELIMINARY_REMOVE", "PRELIMINARY_UNKNOWN")


def test_discovery_report_no_threshold_or_final_keep_token(discovery_report):
    blob = json.dumps(discovery_report, default=str)
    for token in ("recommended_threshold", '"label": "KEEP"',
                 "SCORE_ALERT_MIN"):
        assert token not in blob


def test_discovery_text_report_sections(discovery_report):
    text = dd.format_report(discovery_report)
    for token in ("single-feature attribution (C1.8)", "feature", "limitations:"):
        assert token in text


def test_cli_writes_single_feature_reports(tmp_path, capsys):
    dataset = _build_swing_dataset(tmp_path / "binance")
    outdir = tmp_path / "out"
    rc = dd.main(["--dataset", dataset, "--exchange", "binance",
                 "--profile", "swing", "--bars", str(SWING_BARS),
                 "--outdir", str(outdir)])
    assert rc == 0
    assert (outdir / "single_feature_report.json").exists()
    assert (outdir / "single_feature_report.txt").exists()
    with open(outdir / "single_feature_report.json") as fh:
        data = json.load(fh)
    assert data["kind"] == "single_feature_attribution"


# ---------------------------------------------------------------------------
# Part 3 (C1.8, step 5) — redundancy + candidate reduction.
# ---------------------------------------------------------------------------

def test_registry_reflects_step5_reduction():
    for name in ("cvd_bullish", "cvd_bearish", "mfe_r", "mae_r",
                "time_to_mfe", "time_to_mae"):
        assert name not in dd.ALL_FEATURES
    assert "cvd_direction" in dd.CATEGORICAL_FEATURES
    assert set(dd.DIAGNOSTIC_FEATURES) == {"mfe_r", "mae_r", "time_to_mfe",
                                           "time_to_mae"}
    assert "distance_from_equilibrium" in dd.FAILED_ROBUSTNESS_OVERRIDES


def test_cvd_direction_derivation():
    assert dd._feature_value({"cvd_bearish": True, "cvd_bullish": False},
                             "cvd_direction") == "bearish"
    assert dd._feature_value({"cvd_bearish": False, "cvd_bullish": True},
                             "cvd_direction") == "bullish"
    assert dd._feature_value({"cvd_bearish": False, "cvd_bullish": False},
                             "cvd_direction") == "neutral"


def _cvd_row(idx, *, cvd_dir, cvd_delta, r, htf="neutral", regime="range",
            direction="long", score=8, year=2021):
    return {
        "idx": idx, "direction": direction, "score": score,
        "regime_current": regime, "regime_live_parity": regime,
        "outcome": "win" if r > 0 else "loss", "hit_tp1": True, "r": r,
        "gross_r": r, "year": year,
        "cvd_bearish": cvd_dir == "bearish", "cvd_bullish": cvd_dir == "bullish",
        "cvd_last_delta": cvd_delta, "htf_bias": htf,
    }


def test_diagnostic_bucket_report_uses_numeric_tertiles():
    rows = [_row(i, r=float(i)) for i in range(30)]
    for i, row in enumerate(rows):
        row["mfe_r"] = float(i)
    report = dd.diagnostic_bucket_report(rows)
    assert set(report["mfe_r"]["buckets"]) <= {"low", "mid", "high"}
    assert report["mfe_r"]["kind"] == "numeric_diagnostic_only"


def test_redundancy_check_independent_when_sign_holds_in_every_stratum():
    # cvd_direction separates R the same way (bearish > bullish) regardless
    # of htf_bias — a clean "independent" case.
    rows = []
    idx = 0
    for htf in ("bullish", "bearish"):
        for _ in range(40):
            rows.append(_cvd_row(idx, cvd_dir="bearish", cvd_delta=0.0,
                                 r=0.3, htf=htf))
            idx += 1
            rows.append(_cvd_row(idx, cvd_dir="bullish", cvd_delta=0.0,
                                 r=-0.3, htf=htf))
            idx += 1
    labels = dd._bucket_labels(rows, "cvd_direction")
    chk = dd.redundancy_check(rows, "cvd_direction", labels, "htf_bias",
                              min_n=30)
    assert chk["verdict"] == "independent"
    assert chk["evaluated"] == 2 and chk["agree"] == 2


def test_redundancy_check_entangled_when_sign_flips_across_strata():
    rows = []
    idx = 0
    # bullish htf: bearish-cvd wins; bearish htf: sign flips (bullish-cvd wins)
    for _ in range(40):
        rows.append(_cvd_row(idx, cvd_dir="bearish", cvd_delta=0.0, r=0.3,
                             htf="bullish"))
        idx += 1
        rows.append(_cvd_row(idx, cvd_dir="bullish", cvd_delta=0.0, r=-0.3,
                             htf="bullish"))
        idx += 1
    for _ in range(40):
        rows.append(_cvd_row(idx, cvd_dir="bearish", cvd_delta=0.0, r=-0.3,
                             htf="bearish"))
        idx += 1
        rows.append(_cvd_row(idx, cvd_dir="bullish", cvd_delta=0.0, r=0.3,
                             htf="bearish"))
        idx += 1
    labels = dd._bucket_labels(rows, "cvd_direction")
    chk = dd.redundancy_check(rows, "cvd_direction", labels, "htf_bias",
                              min_n=30)
    assert chk["verdict"] == "entangled_or_inconsistent"


def test_redundancy_check_insufficient_evidence_below_stratum_floor():
    rows = [_cvd_row(i, cvd_dir="bearish" if i % 2 == 0 else "bullish",
                     cvd_delta=0.0, r=0.3 if i % 2 == 0 else -0.3,
                     htf="bullish")
           for i in range(10)]
    labels = dd._bucket_labels(rows, "cvd_direction")
    chk = dd.redundancy_check(rows, "cvd_direction", labels, "htf_bias",
                              min_n=30)
    assert chk["verdict"] == "insufficient_evidence"


def test_contingency_counts_cooccurrence():
    rows = [_cvd_row(0, cvd_dir="bearish", cvd_delta=0.0, r=0.1, htf="bullish"),
           _cvd_row(1, cvd_dir="bearish", cvd_delta=0.0, r=0.1, htf="bullish"),
           _cvd_row(2, cvd_dir="bullish", cvd_delta=0.0, r=0.1, htf="bearish")]
    primary_labels = dd._bucket_labels(rows, "cvd_direction")
    other_labels = dd._bucket_labels(rows, "htf_bias")
    table = dd._contingency(rows, primary_labels, other_labels)
    assert table["bearish"]["bullish"] == 2
    assert table["bullish"]["bearish"] == 1


def test_run_redundancy_analysis_structure(swing_walk_inputs):
    frames, profile = swing_walk_inputs
    walk, records = dd.record_walk(frames, profile, SWING_BARS)
    rows = dd.build_feature_rows(walk, records)
    features = dd.run_single_feature_attribution(rows)
    red = dd.run_redundancy_analysis(rows, features)
    assert set(red) == {"manual_overrides", "cvd_last_delta_added_information",
                        "cvd_direction_conditioning", "diagnostics_only",
                        "excluded_from_candidates"}
    assert set(red["cvd_direction_conditioning"]) == \
        set(dd.REDUNDANCY_CONDITIONING_FEATURES)
    assert "distance_from_equilibrium" in red["manual_overrides"]
    assert set(red["excluded_from_candidates"]) == set(dd.DIAGNOSTIC_FEATURES)


def test_build_shortlist_never_promotes_failed_robustness_feature():
    # Regression: distance_from_equilibrium independently reaches
    # PRELIMINARY_KEEP in single-feature attribution on real data, but the
    # manual override must keep it out of supporting_filters regardless.
    rows = _make_rows(240, 240, r_true=0.4, r_false=-0.4)
    features = dd.run_single_feature_attribution(rows)
    # Force distance_from_equilibrium's raw classification to KEEP to prove
    # the override wins even in the worst case, without depending on qcut
    # producing that outcome from "some_flag"-shaped synthetic rows.
    features["distance_from_equilibrium"] = {
        "classification": {"label": "PRELIMINARY_KEEP"},
    }
    features["cvd_direction"] = {
        "classification": {"label": "PRELIMINARY_REMOVE"},
    }
    red = {
        "cvd_last_delta_added_information": {"verdict": "insufficient_evidence"},
        "cvd_direction_conditioning": {f: {"verdict": "independent"}
                                       for f in dd.REDUNDANCY_CONDITIONING_FEATURES},
    }
    short = dd.build_shortlist(features, red)
    assert "distance_from_equilibrium" not in short["supporting_filters"]
    assert short["excluded"]["distance_from_equilibrium"] == \
        "failed_robustness (see manual_overrides)"


def test_build_shortlist_max_one_primary_max_two_supporting():
    rows = _make_rows(240, 240, r_true=0.4, r_false=-0.4)
    features = dd.run_single_feature_attribution(rows)
    features["cvd_direction"] = {"classification": {"label": "PRELIMINARY_KEEP"}}
    red = {
        "cvd_last_delta_added_information": {"verdict": "independent"},
        "cvd_direction_conditioning": {f: {"verdict": "independent"}
                                       for f in dd.REDUNDANCY_CONDITIONING_FEATURES},
    }
    short = dd.build_shortlist(features, red)
    assert short["primary"] == "cvd_direction"
    assert len(short["supporting_filters"]) <= 2


def test_build_shortlist_no_primary_when_cvd_direction_not_keep():
    rows = _make_rows(240, 240, r_true=0.01, r_false=-0.01)  # flat -> REMOVE
    features = dd.run_single_feature_attribution(rows)
    features["cvd_direction"] = {
        "classification": {"label": "PRELIMINARY_REMOVE"}}
    red = {
        "cvd_last_delta_added_information": {"verdict": "insufficient_evidence"},
        "cvd_direction_conditioning": {f: {"verdict": "insufficient_evidence"}
                                       for f in dd.REDUNDANCY_CONDITIONING_FEATURES},
    }
    short = dd.build_shortlist(features, red)
    assert short["primary"] is None


def test_cli_redundancy_flag_writes_extra_reports(tmp_path):
    dataset = _build_swing_dataset(tmp_path / "binance")
    outdir = tmp_path / "out"
    rc = dd.main(["--dataset", dataset, "--exchange", "binance",
                 "--profile", "swing", "--bars", str(SWING_BARS),
                 "--outdir", str(outdir), "--redundancy"])
    assert rc == 0
    assert (outdir / "redundancy_report.json").exists()
    assert (outdir / "redundancy_report.txt").exists()
    assert (outdir / "SWING_SHORTLIST.md").exists()
    with open(outdir / "redundancy_report.json") as fh:
        data = json.load(fh)
    assert "redundancy" in data and "shortlist" in data


def test_cli_without_redundancy_flag_skips_extra_reports(tmp_path):
    dataset = _build_swing_dataset(tmp_path / "binance")
    outdir = tmp_path / "out"
    rc = dd.main(["--dataset", dataset, "--exchange", "binance",
                 "--profile", "swing", "--bars", str(SWING_BARS),
                 "--outdir", str(outdir)])
    assert rc == 0
    assert not (outdir / "redundancy_report.json").exists()
    assert not (outdir / "SWING_SHORTLIST.md").exists()
