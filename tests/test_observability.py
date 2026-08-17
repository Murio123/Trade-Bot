"""Forecast observability stage: live snapshot, forecast ledger, mode
separation, outcome tracking, TP2 validation.

No test here can place a real order: the bot is signal-only (no execution
module exists) and every test runs on the in-memory Database fallback.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
import scheduler  # noqa: F401 — импорт на этапе collection: scheduler.py строит
# module-level asyncio.Lock(), которому нужен event loop; asyncio.run() в тестах
# обнуляет текущий loop (py3.9), поэтому первый импорт scheduler откладывать до
# рантайма нельзя (иначе get_event_loop() падает при выборочном прогоне).
from analyzer.outcomes import measure_outcome
from database import Database
from signal_engine.forecast_record import build_forecast_record
from signal_engine.no_trade_gate import TP2_SOURCES, invalid_tp2
from signal_engine.profiles import PROFILES, get_profile
from validation.trade_costs import FUNDING_NOT_MODELLED

UTC = timezone.utc


def _forecast(symbol="BTCUSDT", analysis_type="SWING", status="ENTER",
              candle_close=None, direction="long", **over):
    fc = {
        "symbol": symbol,
        "analysis_type": analysis_type,
        "timeframe": "4h",
        "signal_candle_close_time": candle_close or datetime(2026, 7, 1, 4, tzinfo=UTC),
        "decision_time": datetime(2026, 7, 1, 4, 1, tzinfo=UTC),
        "data_freshness_seconds": 60.0,
        "candidate_direction": direction,
        "final_bias": "LONG",
        "analysis_status": status,
        "signal_close_price": 100_000.0,
        "executable_price_at_decision": 100_050.0,
        "stop_loss": 99_000.0,
        "take_profit_levels": [101_500.0, 103_000.0],
        "tp2_source": "structural",
    }
    fc.update(over)
    return fc


# --- 1-2. Live snapshot: prices are separate, stale close never becomes
#           the executable price -------------------------------------------

def test_snapshot_prices_are_separate_fields():
    db = Database(dsn=None)
    fid = asyncio.run(db.insert_forecast(_forecast()))
    rows = db._mem.forecasts
    assert rows[0]["signal_close_price"] == 100_000.0
    assert rows[0]["executable_price_at_decision"] == 100_050.0
    assert fid == 1


def test_stale_close_is_not_used_as_executable_price():
    """When current_price() degrades, the executable price falls back to the
    close but the degradation is marked — a silent substitution is forbidden."""
    from pipeline import _decision_snapshot
    ctx = {"price": 100_000.0, "decision_time": datetime.now(UTC),
           "last_close_time": datetime.now(UTC) - timedelta(hours=3),
           "executable_price": 100_000.0, "executable_price_degraded": True}
    snap = _decision_snapshot(ctx)
    assert snap["signal_close_price"] == 100_000.0
    assert snap["executable_price_degraded"] is True
    assert snap["data_freshness_seconds"] == pytest.approx(3 * 3600, abs=5)


def test_enter_downgraded_to_wait_when_decision_late():
    """Freshness above the mode's max_decision_delay -> ENTER must not pass."""
    for name in ("swing", "position", "intraday"):
        p = get_profile(name)
        assert p["max_decision_delay_seconds"] > 0
    # The downgrade rule itself is exercised in pipeline; here we pin the
    # thresholds ordering: intraday strictest, position most lenient.
    assert (PROFILES["intraday"]["max_decision_delay_seconds"]
            < PROFILES["swing"]["max_decision_delay_seconds"]
            < PROFILES["position"]["max_decision_delay_seconds"])


# --- 2. Candle dedup ---------------------------------------------------------

def test_same_candle_not_processed_twice():
    db = Database(dsn=None)
    close = datetime(2026, 7, 1, 8, tzinfo=UTC)
    first = asyncio.run(db.insert_forecast(_forecast(candle_close=close)))
    second = asyncio.run(db.insert_forecast(_forecast(candle_close=close)))
    assert first == 1
    assert second is None  # ON CONFLICT DO NOTHING
    assert asyncio.run(db.forecast_exists("BTCUSDT", "SWING", close)) is True
    assert len(db._mem.forecasts) == 1


def test_restart_misfire_grace_prevents_stale_publication():
    """Missed cron slots must not fire late after a restart: every analysis
    job carries coalesce=True and a short misfire_grace_time."""
    from scheduler import _ANALYSIS_CRON, _analysis_trigger
    for name, profile in PROFILES.items():
        trigger, grace = _analysis_trigger(profile)
        assert grace <= 600, name
        assert profile["entry"] in _ANALYSIS_CRON, name


# --- 3. All statuses are saved ----------------------------------------------

def test_blocked_forecast_is_saved():
    profile = get_profile("swing")
    result = {"status": "blocked", "blocked_at": "htf_filter",
              "direction": "long", "long_score": 4.0, "short_score": 1.0,
              "signal_candle_close_time": datetime(2026, 7, 1, 4, tzinfo=UTC),
              "decision_time": datetime(2026, 7, 1, 4, 1, tzinfo=UTC),
              "signal_close_price": 100_000.0,
              "executable_price_at_decision": 100_010.0}
    fc = build_forecast_record(result, {}, profile)
    assert fc is not None
    assert fc["analysis_status"] == "NO_TRADE"
    assert fc["blocked_gate"] == "htf_filter"
    assert fc["candidate_direction"] == "long"
    db = Database(dsn=None)
    assert asyncio.run(db.insert_forecast(fc)) == 1


def test_no_trade_and_wait_are_saved():
    profile = get_profile("swing")
    base = {"signal_candle_close_time": datetime(2026, 7, 1, 8, tzinfo=UTC),
            "decision_time": datetime(2026, 7, 1, 8, 1, tzinfo=UTC)}
    no_trade = build_forecast_record(
        {**base, "status": "blocked", "blocked_at": "no_trade",
         "no_trade_reasons": ["risk/reward 1.10 ниже минимума 1.50"]},
        {}, profile)
    wait = build_forecast_record(
        {**base, "status": "cooldown", "analysis_status": "WAIT",
         "signal_candle_close_time": datetime(2026, 7, 1, 12, tzinfo=UTC)},
        {}, profile)
    assert no_trade["analysis_status"] == "NO_TRADE"
    assert no_trade["no_trade_reasons"] == ["risk/reward 1.10 ниже минимума 1.50"]
    assert wait["analysis_status"] == "WAIT"


def test_forecast_record_stamps_versions_and_type():
    fc = build_forecast_record(
        {"status": "blocked", "blocked_at": "stale_data",
         "signal_candle_close_time": datetime(2026, 7, 1, 4, tzinfo=UTC),
         "decision_time": datetime(2026, 7, 1, 4, 1, tzinfo=UTC)},
        {"symbol": "BTCUSDT", "timeframe": "4h", "price": 1.0},
        get_profile("position"))
    assert fc["analysis_type"] == "POSITIONAL"
    assert fc["prompt_version"] == config.PROMPT_VERSION
    assert fc["model_version"] == config.MODEL_VERSION


# --- 4. Mode separation -------------------------------------------------------

def test_swing_and_position_cooldowns_are_independent():
    db = Database(dsn=None)
    swing_alert = {"symbol": "BTCUSDT", "timeframe": "4h", "direction": "long",
                   "entry_price": 100_000.0, "score": 8, "delivered": True,
                   "analysis_type": "SWING",
                   "created_at": datetime.now(UTC) - timedelta(minutes=5)}
    asyncio.run(db.insert_signal(swing_alert))
    # POSITIONAL stream on the SAME timeframe sees no delivered signal.
    assert asyncio.run(db.last_delivered_signal(
        "BTCUSDT", "4h", analysis_type="POSITIONAL")) is None
    assert asyncio.run(db.signals_today(
        "BTCUSDT", "4h", analysis_type="POSITIONAL")) == []
    # SWING sees its own.
    assert asyncio.run(db.last_delivered_signal(
        "BTCUSDT", "4h", analysis_type="SWING")) is not None
    assert len(asyncio.run(db.signals_today(
        "BTCUSDT", "4h", analysis_type="SWING"))) == 1


def test_analysis_type_used_in_journal_stats():
    db = Database(dsn=None)
    tid = asyncio.run(db.insert_trade({
        "direction": "long", "entry_price": 100.0, "stop_loss": 90.0,
        "symbol": "BTCUSDT", "timeframe": "4h", "analysis_type": "SWING"}))
    asyncio.run(db.close_trade(tid, 120.0, "win", 2.0))
    assert asyncio.run(db.journal_stats("BTCUSDT", "SWING"))["total"] == 1
    assert asyncio.run(db.journal_stats("BTCUSDT", "POSITIONAL"))["total"] == 0
    # Legacy call without the discriminator still works (backward compat).
    assert asyncio.run(db.journal_stats("BTCUSDT"))["total"] == 1


def test_legacy_rows_without_analysis_type_still_readable():
    db = Database(dsn=None)
    old = {"symbol": "BTCUSDT", "timeframe": "1h", "direction": "long",
           "entry_price": 100.0, "score": 8, "delivered": True}
    asyncio.run(db.insert_signal(old))  # no analysis_type key at all
    assert asyncio.run(db.last_delivered_signal("BTCUSDT", "1h")) is not None
    # Strict mode filter does NOT pick up legacy NULL rows.
    assert asyncio.run(db.last_delivered_signal(
        "BTCUSDT", "1h", analysis_type="SWING")) is None


# --- 5. Outcome tracking -------------------------------------------------------

def _klines(start: datetime, hours: int, price: float, step: float = 0.0,
            spread: float = 50.0) -> pd.DataFrame:
    rows = []
    p = price
    for i in range(hours):
        ot = start + timedelta(hours=i)
        rows.append({"open_time": pd.Timestamp(ot), "open": p,
                     "high": p + spread, "low": p - spread, "close": p + step,
                     "close_time": pd.Timestamp(ot + timedelta(hours=1))})
        p += step
    return pd.DataFrame(rows)


def test_outcome_returns_mfe_mae_and_reached():
    anchor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    fc = {"id": 1, "decision_time": anchor, "candidate_direction": "long",
          "executable_price_at_decision": 100_000.0,
          "signal_close_price": 100_000.0,
          "stop_loss": 95_000.0, "take_profit_levels": [101_000.0, 106_000.0]}
    # +100/hour drift for 72h -> +7200 with wicks +-50.
    df = _klines(anchor, 80, 100_000.0, step=100.0)
    # FUNDING_NOT_MODELLED: this test fixes the pre-P1 arithmetic, so the
    # transaction-only net figure below must stay bit-identical. The
    # funding-aware and fail-closed paths are covered in test_funding.py.
    out = measure_outcome(fc, df, anchor + timedelta(hours=80),
                          taker_fee_pct=0.05, slippage_pct=0.03,
                          funding=FUNDING_NOT_MODELLED)
    assert out["return_1h"] == pytest.approx(0.1, rel=0.1)
    assert out["return_24h"] > out["return_4h"] > 0
    assert out["return_72h"] is not None
    assert out["mfe_points"] > 7000
    assert out["mae_points"] <= 50
    assert out["reached_500"] and out["reached_1500"] and out["reached_3000"]
    assert out["tp1_hit"] and out["tp2_hit"] and not out["stop_hit"]
    assert out["resolved"] is True
    assert out["net_after_costs"] == pytest.approx(
        out["return_72h"] - (2 * 0.05 + 0.03), abs=1e-6)


def test_unresolved_forecast_is_censored_not_deleted():
    anchor = datetime.now(UTC) - timedelta(hours=2)
    fc = {"id": 7, "decision_time": anchor, "candidate_direction": "long",
          "executable_price_at_decision": 100_000.0,
          "stop_loss": 90_000.0, "take_profit_levels": [150_000.0, 160_000.0]}
    df = _klines(anchor, 2, 100_000.0, step=10.0)
    out = measure_outcome(fc, df, datetime.now(UTC), 0.05, 0.03)
    assert out["return_1h"] is not None
    assert out["return_4h"] is None and out["return_72h"] is None  # censored
    assert out["resolved"] is False
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_forecast()))
    asyncio.run(db.upsert_outcome({**out, "forecast_id": 1}))
    # Unresolved outcome stays in the stats (never dropped).
    stats = asyncio.run(db.forecast_stats("BTCUSDT"))
    assert stats["total"] == 1 and stats["unresolved"] == 1
    assert asyncio.run(db.forecasts_pending_outcomes("BTCUSDT")) != []


def test_outcome_recompute_is_idempotent():
    anchor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    fc = {"id": 3, "decision_time": anchor, "candidate_direction": "short",
          "executable_price_at_decision": 100_000.0,
          "stop_loss": 103_000.0, "take_profit_levels": [99_000.0, 97_000.0]}
    df = _klines(anchor, 80, 100_000.0, step=-100.0)
    now = anchor + timedelta(hours=80)
    a = measure_outcome(fc, df, now, 0.05, 0.03)
    b = measure_outcome(fc, df, now, 0.05, 0.03)
    assert a == b
    assert a["tp2_hit"] and a["resolved"]


def test_outcome_anchor_off_candle_boundary_still_measures_1h():
    """Cron decisions land at ~HH:01 while 1H bars open on the hour: the bar
    containing the anchor must be included, else return_1h is censored forever."""
    start = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    anchor = start + timedelta(minutes=1, seconds=30)  # production alignment
    fc = {"id": 9, "decision_time": anchor, "candidate_direction": "long",
          "executable_price_at_decision": 100_000.0,
          "stop_loss": 90_000.0, "take_profit_levels": [150_000.0, 160_000.0]}
    df = _klines(start, 10, 100_000.0, step=100.0)
    out = measure_outcome(fc, df, anchor + timedelta(hours=5), 0.05, 0.03)
    assert out["return_1h"] is not None
    assert out["return_4h"] is not None
    assert out["mfe_points"] > 0


def test_measure_outcome_sets_realized_r_for_enter_resolved():
    """Stage 11: ENTER + resolved -> measure_outcome кладёт raw realized_r."""
    anchor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    fc = {"id": 1, "decision_time": anchor, "analysis_status": "ENTER",
          "candidate_direction": "long",
          "executable_price_at_decision": 100_000.0, "signal_close_price": 100_000.0,
          "stop_loss": 95_000.0, "take_profit_levels": [101_000.0, 106_000.0]}
    df = _klines(anchor, 80, 100_000.0, step=100.0)  # доходит до TP2
    out = measure_outcome(fc, df, anchor + timedelta(hours=80), 0.05, 0.03)
    assert out["tp2_hit"] and out["resolved"]
    # sign*(tp2-ref)/risk = (106000-100000)/5000 = 1.2, raw (без округления).
    assert out["realized_r"] == pytest.approx((106_000.0 - 100_000.0) / 5_000.0)


def test_measure_outcome_realized_r_none_without_enter_status():
    """Прогноз без analysis_status ('ENTER') -> realized_r=None (не считаем R
    для не-ENTER сетапов), но остальной outcome измеряется как раньше."""
    anchor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    fc = {"id": 2, "decision_time": anchor, "candidate_direction": "long",
          "executable_price_at_decision": 100_000.0, "signal_close_price": 100_000.0,
          "stop_loss": 95_000.0, "take_profit_levels": [101_000.0, 106_000.0]}
    df = _klines(anchor, 80, 100_000.0, step=100.0)
    out = measure_outcome(fc, df, anchor + timedelta(hours=80), 0.05, 0.03)
    assert out["resolved"] and out["realized_r"] is None


def test_measure_outcome_realized_r_none_when_unresolved():
    anchor = datetime.now(UTC) - timedelta(hours=2)
    fc = {"id": 3, "decision_time": anchor, "analysis_status": "ENTER",
          "candidate_direction": "long", "executable_price_at_decision": 100_000.0,
          "stop_loss": 90_000.0, "take_profit_levels": [150_000.0, 160_000.0]}
    df = _klines(anchor, 2, 100_000.0, step=10.0)
    out = measure_outcome(fc, df, datetime.now(UTC), 0.05, 0.03)
    assert out["resolved"] is False and out["realized_r"] is None


def test_measure_outcome_realized_r_idempotent_and_persists():
    anchor = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    fc = {"id": 4, "decision_time": anchor, "analysis_status": "ENTER",
          "candidate_direction": "short", "executable_price_at_decision": 100_000.0,
          "stop_loss": 103_000.0, "take_profit_levels": [99_000.0, 97_000.0]}
    df = _klines(anchor, 80, 100_000.0, step=-100.0)
    now = anchor + timedelta(hours=80)
    a = measure_outcome(fc, df, now, 0.05, 0.03)
    b = measure_outcome(fc, df, now, 0.05, 0.03)
    assert a == b and a["realized_r"] == pytest.approx((97_000.0 - 100_000.0) * -1 / 3_000.0)
    db = Database(dsn=None)
    asyncio.run(db.upsert_outcome({**a, "forecast_id": 4}))
    assert asyncio.run(db.forecast_outcome(4))["realized_r"] == a["realized_r"]


def test_run_cascade_blocked_result_carries_snapshot():
    """Any blocked path must return (not crash) and carry the decision
    snapshot — regression for the recursive blocked() wrapper."""
    from pipeline import run_cascade
    now = datetime.now(UTC)
    ctx = {"symbol": "BTCUSDT", "timeframe": "4h", "price": 100_000.0,
           "decision_time": now, "executable_price": 100_100.0,
           "last_close_time": now - timedelta(days=3),  # stale -> first gate
           "volatility": None, "inds_by_tf": {}}
    result = asyncio.run(run_cascade(ctx, [], None, interpret=False,
                                     profile_name="swing"))
    assert result["status"] == "blocked"
    assert result["blocked_at"] == "stale_data"
    assert result["signal_close_price"] == 100_000.0
    assert result["executable_price_at_decision"] == 100_100.0
    assert result["data_freshness_seconds"] == pytest.approx(3 * 86400, abs=5)


# --- 6. TP2 validation -----------------------------------------------------------

def test_tp2_wrong_side_is_invalid():
    assert invalid_tp2("long", 100_000.0, 99_000.0, 98_000.0, 101_000.0,
                       "structural") is not None
    assert invalid_tp2("short", 100_000.0, 101_000.0, 102_000.0, 99_000.0,
                       "structural") is not None
    # Correct side passes.
    assert invalid_tp2("long", 100_000.0, 99_000.0, 103_000.0, 101_500.0,
                       "structural") is None


def test_tp2_nan_inf_zero_is_invalid():
    for bad in (float("nan"), float("inf"), -float("inf"), 0.0, None):
        assert invalid_tp2("long", 100_000.0, 99_000.0, bad, 101_000.0,
                           "structural") is not None


def test_tp2_equal_to_stop_and_unknown_source_invalid():
    assert invalid_tp2("long", 100_000.0, 99_000.0, 99_000.0, 98_000.0,
                       "structural") is not None
    assert invalid_tp2("long", 100_000.0, 99_000.0, 103_000.0, 101_000.0,
                       None) is not None
    assert invalid_tp2("long", 100_000.0, 99_000.0, 103_000.0, 101_000.0,
                       "guess") is not None


def test_tp2_fallback_is_explicitly_labelled():
    """A synthetic 3R TP2 (single structural level) must be labelled
    r_multiple_fallback, not passed off as structure."""
    from pipeline import _structure_targets
    entry, risk = 100_000.0, 1000.0
    # Exactly ONE level above entry -> TP1 structural, TP2 synthetic 3R.
    ctx = {"htf_levels": {"highs": [102_000.0], "lows": []}, "volume_profile": {}}
    tp1, tp2, structural, src = _structure_targets(
        ctx, "long", entry, risk, 101_500.0, 103_000.0)
    assert tp1 == 102_000.0 and structural is True
    assert src == "r_multiple_fallback"
    assert src in TP2_SOURCES
    # No levels at all -> both fall back, same label.
    ctx2 = {"htf_levels": {"highs": [], "lows": []}, "volume_profile": {}}
    *_rest, src2 = _structure_targets(ctx2, "long", entry, risk, 101_500.0, 103_000.0)
    assert src2 == "r_multiple_fallback"
    # Two levels -> real structural TP2.
    ctx3 = {"htf_levels": {"highs": [102_000.0, 104_000.0], "lows": []},
            "volume_profile": {}}
    _, tp2_s, _, src3 = _structure_targets(ctx3, "long", entry, risk,
                                           101_500.0, 103_000.0)
    assert tp2_s == 104_000.0 and src3 == "structural"


# --- 7. No real orders possible ------------------------------------------------

def test_no_execution_module_exists():
    """The bot is signal-only: no order/execution API is importable, so no
    test (or code path) can place a real order."""
    import importlib
    for name in ("execution", "exchange_orders", "orders"):
        assert importlib.util.find_spec(name) is None
    from analyzer.binance import BinanceClient
    assert not any("order" in m.lower() and not m.startswith("_")
                   for m in dir(BinanceClient))
