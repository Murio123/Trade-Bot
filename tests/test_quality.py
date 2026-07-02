"""Signal-quality safeguards: closed-candle analysis and no-trade vetoes."""
from datetime import timedelta

import pandas as pd

from pipeline import _drop_unclosed
from signal_engine.vetoes import crowded_funding, dead_zone


def _frame(last_closed: bool) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    times = [now - timedelta(hours=3), now - timedelta(hours=2),
             now - timedelta(hours=1) if last_closed else now + timedelta(minutes=30)]
    return pd.DataFrame({"close_time": times, "close": [1.0, 2.0, 3.0]})


def test_drop_unclosed_removes_forming_candle():
    df = _frame(last_closed=False)
    out = _drop_unclosed(df)
    assert len(out) == 2 and out["close"].iloc[-1] == 2.0


def test_drop_unclosed_keeps_closed_candles():
    df = _frame(last_closed=True)
    assert len(_drop_unclosed(df)) == 3


def test_dead_zone_blocks_mid_range_without_structure():
    assert dead_zone(0, {"pos": 0.50, "zone": "equilibrium"})
    # structure backing lifts the veto
    assert not dead_zone(3, {"pos": 0.50, "zone": "equilibrium"})
    # discount/premium are not the dead zone
    assert not dead_zone(0, {"pos": 0.30, "zone": "discount"})
    assert not dead_zone(0, None)


def test_crowded_funding_blocks_joining_the_crowd():
    hot_longs = {"current": 0.0008, "zscore": 2.5}
    assert crowded_funding("long", hot_longs)       # joining crowded longs
    assert not crowded_funding("short", hot_longs)  # fading them is fine
    hot_shorts = {"current": -0.0008, "zscore": -2.5}
    assert crowded_funding("short", hot_shorts)
    assert not crowded_funding("long", hot_shorts)
    # normal funding never vetoes
    assert not crowded_funding("long", {"current": 0.0001, "zscore": 0.5})
    assert not crowded_funding("long", None)


def test_structural_stop_for_swing():
    from pipeline import _structural_stop
    prof = {"structural_stop": True, "atr_mult": 1.5}
    ctx = {"htf_levels": {"lows": [57200], "highs": [60100]},
           "order_blocks": {}}
    stop = _structural_stop(ctx, "long", 58700, 300, prof)
    assert stop == 57200 - 150  # level minus 0.5 ATR buffer
    # tighter than the ATR stop -> keep ATR (returns None)
    assert _structural_stop({"htf_levels": {"lows": [58600]},
                             "order_blocks": {}}, "long", 58700, 300, prof) is None
    # absurdly far (> 8 ATR) -> keep ATR
    assert _structural_stop({"htf_levels": {"lows": [50000]},
                             "order_blocks": {}}, "long", 58700, 300, prof) is None
    # profile without the flag (intraday) -> never structural
    assert _structural_stop(ctx, "long", 58700, 300, {"atr_mult": 1.2}) is None


def test_last_signal_note_recent_vs_stale():
    from datetime import datetime, timedelta, timezone
    from bot.formatting import format_last_signal_note
    base = {"direction": "long", "entry_price": 58100, "stop_loss": 57050,
            "target_1": 59700, "target_2": 61300}
    recent = dict(base, created_at=datetime.now(timezone.utc) - timedelta(hours=1))
    note = format_last_signal_note(recent, max_age_hours=24)
    assert note and "Действующий сетап" in note and "58 100" in note
    stale = dict(base, created_at=datetime.now(timezone.utc) - timedelta(hours=30))
    assert format_last_signal_note(stale, max_age_hours=24) is None
    assert format_last_signal_note(None) is None


def test_reversal_alert_throttle():
    from datetime import datetime, timezone
    from signal_engine.vetoes import reversal_alert_allowed
    now = datetime.now(timezone.utc)
    last = {"direction": "bull", "time": now - timedelta(hours=1), "tf_count": 2}
    # same direction, inside cooldown, same strength -> throttled (the old spam)
    assert not reversal_alert_allowed(last, "bull", 2, now, cooldown_hours=4)
    # escalation to more timeframes -> allowed
    assert reversal_alert_allowed(last, "bull", 3, now, cooldown_hours=4)
    # opposite direction -> allowed
    assert reversal_alert_allowed(last, "bear", 2, now, cooldown_hours=4)
    # cooldown expired -> allowed
    old = {"direction": "bull", "time": now - timedelta(hours=5), "tf_count": 2}
    assert reversal_alert_allowed(old, "bull", 2, now, cooldown_hours=4)
    assert reversal_alert_allowed(None, "bull", 2, now, cooldown_hours=4)


def test_reversal_new_factors_and_higher_bar():
    import numpy as np
    from analyzer.reversal import detect_reversal
    rng = np.random.default_rng(7)
    n = 40
    base = np.linspace(60000, 57000, n)  # downtrend into the low
    df = pd.DataFrame({
        "open": base + 50, "close": base,
        "high": base + 120, "low": base - 120,
        "volume": np.abs(rng.normal(1000, 100, n)),
    })
    prior_low = float(df["low"].iloc[-11:-1].min())
    # last candle: sweep of the prior low + bullish engulfing + hammer-ish
    prev_o, prev_c = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    df.loc[df.index[-1], "low"] = prior_low - 200
    df.loc[df.index[-1], "open"] = prev_c - 10
    df.loc[df.index[-1], "close"] = prev_o + 60
    df.loc[df.index[-1], "high"] = prev_o + 80
    df.loc[df.index[-1], "volume"] = 4000
    ind = {"avg_volume": 1000, "rsi": 28, "rsi_prev": 24}
    rev = detect_reversal(df, ind)
    assert "Свип минимума — стоп-хант и возврат" in rev["factors_bull"]
    assert "Бычье поглощение — подтверждающая свеча" in rev["factors_bull"]
    # 2 factors are no longer enough to flag a reversal
    assert rev["bullish_reversal"] == (rev["bull_score"] >= 3)


def test_reversal_plan_levels():
    from bot.formatting import build_reversal_plan
    ctx = {"price": 58000.0, "atr": 300.0,
           "inds_by_tf": {"1h": {"atr": 300.0}},
           "htf_levels": {"lows": [57400], "highs": [59500]},
           "volume_profile": {}, "liquidity": {}, "order_blocks": {}}
    plan = build_reversal_plan(ctx, "bull")
    assert plan["stop"] == 57400 - 150          # support minus 0.5 ATR
    assert plan["tp1"] == 59500                 # nearest resistance (>= 1R away)
    assert plan["rr"] > 1.5


def test_stop_atr_uses_profile_stop_tf():
    from pipeline import _stop_atr
    inds = {"15m": {"atr": 90}, "1h": {"atr": 420}}
    # intraday: stop anchored to 1H ATR, not the tiny 15m ATR
    assert _stop_atr({"stop_tf": "1h"}, inds, 90) == 420
    # swing (no stop_tf): entry ATR unchanged
    assert _stop_atr({}, inds, 300) == 300
    # missing stop-tf data -> fall back to entry ATR
    assert _stop_atr({"stop_tf": "4h"}, inds, 90) == 90


def test_reversal_trend_linkage():
    from signal_engine.vetoes import (reversal_alert_min_tfs,
                                      reversal_trend_alignment)
    # bottom in an uptrend = prime pullback entry
    assert reversal_trend_alignment("bull", "bullish") == "aligned"
    assert reversal_trend_alignment("bear", "bearish") == "aligned"
    # fading the trend = counter
    assert reversal_trend_alignment("bull", "bearish") == "counter"
    assert reversal_trend_alignment("bear", "bullish") == "counter"
    assert reversal_trend_alignment("bull", "neutral") == "neutral"
    # counter-trend needs one more confirming timeframe
    assert reversal_alert_min_tfs(2, "aligned") == 2
    assert reversal_alert_min_tfs(2, "neutral") == 2
    assert reversal_alert_min_tfs(2, "counter") == 3


def test_confluence_trend_aligned_reversal_bonus():
    from signal_engine.confluence import calculate_confluence_score
    flat = {"trend_aligned_bottom": True, "rsi": 28, "macd_bullish_cross": True}
    total, scores, reasons = calculate_confluence_score(flat, "long")
    assert scores["structure"] >= 3
    assert any("точка входа" in r for r in reasons)
    # the mirror direction gets nothing from it
    _, s2, r2 = calculate_confluence_score(flat, "short")
    assert s2["structure"] == 0


def test_reversal_alert_header_by_alignment():
    from bot.formatting import format_reversal_alert
    ctx = {"price": 58000.0, "atr": 300.0, "inds_by_tf": {"1h": {"atr": 300.0}},
           "htf_levels": {"lows": [57400], "highs": [59500]},
           "volume_profile": {}, "liquidity": {}, "order_blocks": {}}
    aligned = format_reversal_alert(ctx, "bull", ["x"], False, ["4h", "1d"],
                                    alignment="aligned")
    assert "ПО ТРЕНДУ — точка входа" in aligned and "самый надёжный" in aligned
    counter = format_reversal_alert(ctx, "bull", ["x"], False, ["4h", "1d", "12h"],
                                    alignment="counter")
    assert "против тренда" in counter and "контртренд" in counter
