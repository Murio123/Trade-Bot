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
