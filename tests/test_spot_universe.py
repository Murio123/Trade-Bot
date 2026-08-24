"""S2 — point-in-time eligibility and the write-once snapshot.

The tests are about what must not leak and what must not disappear: a screen
that reads a bar it could not have seen, a snapshot that changes under a rerun,
and a real token deleted by a rule aimed at leveraged products.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from spot.universe import (MIN_MEDIAN_DOLLAR_VOLUME, btc_regime, build_snapshot,
                           is_leveraged_token, month_starts, screen_symbol,
                           visible, write_snapshot)

DAY = 86_400_000


def frame(bars: int, *, start_ms: int = 0, close: float = 100.0,
          quote_volume: float = 10_000_000.0) -> pd.DataFrame:
    rows = []
    for i in range(bars):
        ot = start_ms + i * DAY
        rows.append({"open_time": ot, "open": close, "high": close * 1.01,
                     "low": close * 0.99, "close": close, "volume": 1.0,
                     "close_time": ot + DAY - 1, "quote_volume": quote_volume,
                     "trades": 10})
    return pd.DataFrame(rows)


def asof_after(df: pd.DataFrame, extra_ms: int = 1) -> int:
    return int(df["close_time"].iloc[-1]) + extra_ms


KNOWN = frozenset({"ADA", "BTC", "ETH"})


def screen(df, asof_ms, base="AAA", trading_now=True):
    return screen_symbol("AAAUSDT", df, asof_ms, base_asset=base,
                         trading_now=trading_now, known_assets=KNOWN)


# --- the point-in-time rule ----------------------------------------------


def test_only_bars_that_closed_before_the_decision_are_visible():
    df = frame(10)
    cutoff = int(df["close_time"].iloc[4])
    assert len(visible(df, cutoff)) == 4
    assert len(visible(df, cutoff + 1)) == 5


def test_a_screen_cannot_see_the_bar_it_is_deciding_on():
    """The bar closing exactly at the decision instant is not yet knowable."""
    df = frame(300)
    at_close = int(df["close_time"].iloc[-1])
    assert screen(df, at_close).bars == len(df) - 1


def test_liquidity_uses_only_the_trailing_window():
    df = frame(300, quote_volume=1_000.0)
    df.loc[df.index[-30:], "quote_volume"] = MIN_MEDIAN_DOLLAR_VOLUME * 2
    assert screen(df, asof_after(df)).eligible is True


# --- the individual screens ----------------------------------------------


def test_too_few_bars_is_rejected():
    assert screen(frame(199), asof_after(frame(199))).reason == "too_few_bars"


def test_illiquid_is_rejected():
    df = frame(300, quote_volume=MIN_MEDIAN_DOLLAR_VOLUME - 1)
    assert screen(df, asof_after(df)).reason == "illiquid"


def test_a_pegged_asset_is_rejected_on_every_date():
    """UST is excluded for its design, not for how it ended."""
    df = frame(300)
    assert screen(df, asof_after(df), base="UST").reason == "pegged"


def test_a_wrapped_asset_is_rejected():
    df = frame(300)
    assert screen(df, asof_after(df), base="WBTC").reason == "wrapped"


def test_a_gap_in_the_last_30_days_is_inactive():
    df = frame(300)
    df = pd.concat([df.iloc[:280], df.iloc[285:]], ignore_index=True)
    assert screen(df, asof_after(df)).reason == "inactive"


# --- the leveraged-token rule --------------------------------------------


@pytest.mark.parametrize("base", ["ADAUP", "ADADOWN", "ETHBULL", "BTCBEAR",
                                  "BULL", "BEAR"])
def test_leveraged_tokens_are_detected(base):
    assert is_leveraged_token(base, KNOWN) is True


@pytest.mark.parametrize("base", ["JUP", "SYRUP", "ADA", "UPBIT"])
def test_real_tokens_ending_in_up_survive(base):
    """JUP and SYRUP are ordinary tokens. A suffix-only rule deletes them from
    every universe, silently and forever."""
    assert is_leveraged_token(base, KNOWN) is False


# --- snapshots ------------------------------------------------------------


def panel_meta():
    panel = {"AAAUSDT": frame(300), "BBBUSDT": frame(300, quote_volume=1.0)}
    meta = {"AAAUSDT": {"base_asset": "AAA", "trading_now": True},
            "BBBUSDT": {"base_asset": "BBB", "trading_now": False}}
    return panel, meta


def test_a_snapshot_records_every_screen_not_only_the_winners():
    panel, meta = panel_meta()
    asof = datetime.fromtimestamp(asof_after(panel["AAAUSDT"]) / 1000,
                                  tz=timezone.utc)
    snap = build_snapshot(panel, meta, asof)
    assert snap["eligible"] == ["AAAUSDT"]
    assert {r["symbol"] for r in snap["screens"]} == {"AAAUSDT", "BBBUSDT"}
    assert [r["reason"] for r in snap["screens"]
            if r["symbol"] == "BBBUSDT"] == ["illiquid"]


def test_the_size_cap_records_why_a_symbol_was_dropped():
    panel = {f"S{i}USDT": frame(300, quote_volume=1e7 + i) for i in range(5)}
    meta = {s: {"base_asset": s[:-4], "trading_now": True} for s in panel}
    asof = datetime.fromtimestamp(asof_after(panel["S0USDT"]) / 1000,
                                  tz=timezone.utc)
    snap = build_snapshot(panel, meta, asof, cap=2)
    assert len(snap["eligible"]) == 2
    dropped = [r for r in snap["screens"] if r["reason"] == "below_size_cap"]
    assert len(dropped) == 3


def test_a_snapshot_is_write_once(tmp_path):
    panel, meta = panel_meta()
    asof = datetime.fromtimestamp(asof_after(panel["AAAUSDT"]) / 1000,
                                  tz=timezone.utc)
    snap = build_snapshot(panel, meta, asof)
    path, written = write_snapshot(snap, str(tmp_path))
    assert written is True
    _, rewritten = write_snapshot(snap, str(tmp_path))
    assert rewritten is False, "an identical rerun must be a no-op"

    changed = dict(snap)
    changed["eligible"] = []
    changed["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="write-once"):
        write_snapshot(changed, str(tmp_path))
    assert json.loads(open(path).read())["eligible"] == ["AAAUSDT"]


# --- decision grid and regime --------------------------------------------


def test_decision_dates_stop_a_full_horizon_before_the_panel_ends():
    last = int(datetime(2026, 8, 24, tzinfo=timezone.utc).timestamp() * 1000)
    dates = month_starts("2026-01-01", last, horizon_days=180)
    assert dates[0].date().isoformat() == "2026-01-01"
    assert dates[-1].date().isoformat() == "2026-02-01"


def test_regime_needs_history_before_it_will_answer():
    df = frame(100)
    assert btc_regime(df, asof_after(df)) == "unknown"


def test_regime_reads_bull_when_price_leads_a_rising_average():
    rows = frame(400)
    rows["close"] = [100.0 + i for i in range(400)]
    rows["high"] = rows["close"] * 1.01
    rows["low"] = rows["close"] * 0.99
    assert btc_regime(rows, asof_after(rows)) == "bull"


def test_regime_reads_bear_when_price_trails_a_falling_average():
    rows = frame(400)
    rows["close"] = [500.0 - i for i in range(400)]
    rows["high"] = rows["close"] * 1.01
    rows["low"] = rows["close"] * 0.99
    assert btc_regime(rows, asof_after(rows)) == "bear"
