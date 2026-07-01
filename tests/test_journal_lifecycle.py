"""Trade lifecycle: the journal must record honest outcomes."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from bot import journal


def _df(highs, lows, start=None, freq_h=1):
    start = start or (datetime.now(timezone.utc) - timedelta(hours=len(highs)))
    times = [start + timedelta(hours=i * freq_h) for i in range(len(highs))]
    return pd.DataFrame({
        "open_time": times, "high": highs, "low": lows,
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
    })


BASE = {
    "id": 1, "direction": "long", "entry_price": 100.0, "stop_loss": 95.0,
    "tp1": 105.0, "tp2": 115.0, "stage": "open",
    "opened_at": datetime.now(timezone.utc) - timedelta(hours=10),
}


def test_tp1_then_tp2_win():
    df = _df([101, 106, 108, 116], [99, 103, 104, 112],
             start=BASE["opened_at"])
    ev = journal.evaluate_trade(dict(BASE), df, 116)
    assert ev["closed"]["outcome"] == "win"
    assert ev["closed"]["pnl_r"] == 3.0  # (115-100)/5
    assert [e["type"] for e in ev["events"]] == ["tp1", "closed"]


def test_pullback_to_breakeven_after_tp1():
    df = _df([106, 107, 101, 99.5], [104, 102, 99.9, 98],
             start=BASE["opened_at"])
    ev = journal.evaluate_trade(dict(BASE, stage="tp1"), df, 100)
    assert ev["closed"]["outcome"] == "breakeven"
    assert ev["closed"]["pnl_r"] == 0.0


def test_straight_stop_loss():
    df = _df([101, 99, 96], [99, 96, 93], start=BASE["opened_at"])
    ev = journal.evaluate_trade(dict(BASE), df, 94)
    assert ev["closed"]["outcome"] == "loss"
    assert ev["closed"]["pnl_r"] == -1.0


def test_conservative_stop_first_within_candle():
    # One candle spans both the stop and TP1 -> must count as a loss.
    df = _df([106], [94], start=BASE["opened_at"])
    ev = journal.evaluate_trade(dict(BASE), df, 100)
    assert ev["closed"]["outcome"] == "loss"


def test_no_change_returns_none():
    df = _df([102, 103], [99, 100], start=BASE["opened_at"])
    assert journal.evaluate_trade(dict(BASE), df, 101) is None


def test_events_fire_once_per_stage():
    # Already at tp1 stage -> replay must NOT re-emit the tp1 event.
    df = _df([106, 116], [104, 112], start=BASE["opened_at"])
    ev = journal.evaluate_trade(dict(BASE, stage="tp1"), df, 116)
    assert [e["type"] for e in ev["events"]] == ["closed"]


def test_pick_frame_prefers_native_tf_with_coverage():
    opened = datetime.now(timezone.utc) - timedelta(hours=5)
    native = _df([1] * 10, [0] * 10, start=opened - timedelta(hours=2))
    fallback = _df([1] * 50, [0] * 50, start=opened - timedelta(hours=40))
    frames = {"15m": native, "1h": fallback}
    trade = {"timeframe": "15m", "opened_at": opened}
    assert journal.pick_frame(trade, frames) is native


def test_pick_frame_falls_back_when_history_too_short():
    opened = datetime.now(timezone.utc) - timedelta(hours=48)
    native = _df([1] * 10, [0] * 10)  # starts ~10h ago, misses the open
    fallback = _df([1] * 60, [0] * 60, start=opened - timedelta(hours=5))
    frames = {"15m": native, "1h": fallback}
    trade = {"timeframe": "15m", "opened_at": opened}
    assert journal.pick_frame(trade, frames) is fallback
