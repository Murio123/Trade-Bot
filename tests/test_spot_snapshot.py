"""S4A — the current-market snapshot builder.

The tests here are about the difference between "was investable once" and "is
investable now". S2's E8 defect is the reason this file exists: a rule that
checks a symbol's own last thirty bars is satisfied forever by a coin that
died in 2022, and the same mistake in a live scanner would put delisted assets
in front of a user as current opportunities.

Everything is built from synthetic frames. Nothing reads `data/`, which is
gitignored and regenerable, so these run anywhere.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from tools.spot_snapshot import (BTC_SYMBOL, FIELDS, MAX_LAST_BAR_AGE_MS,
                                 SnapshotBuildError, build_snapshot,
                                 coin_facts, percentiles, realized_vol,
                                 screen_now, write_snapshot)

DAY = 86_400_000
HOUR = 3_600_000
# An arbitrary but fixed "now": 2024-01-01T00:00:00Z, one millisecond after a
# daily bar closed. Fixed so nothing in this file depends on the wall clock.
NOW = 1_704_067_200_000


def frame(bars: int, *, ends_ms: int = NOW, close: float = 100.0,
          quote_volume: float = 10_000_000.0,
          drift: float = 0.0) -> pd.DataFrame:
    """`bars` daily bars whose LAST bar closes at `ends_ms - 1`.

    Anchored at the end rather than the start because every rule under test
    asks how far the last bar is from now, and building forward from an epoch
    would make each test compute that offset itself.
    """
    rows = []
    first_open = ends_ms - bars * DAY
    price = close
    for i in range(bars):
        ot = first_open + i * DAY
        price = price * (1.0 + drift)
        rows.append({"open_time": ot, "open": price, "high": price * 1.01,
                     "low": price * 0.99, "close": price, "volume": 1.0,
                     "close_time": ot + DAY - 1,
                     "quote_volume": quote_volume, "trades": 10})
    return pd.DataFrame(rows)


KNOWN = frozenset({"ADA", "BTC", "ETH", "AAA", "BBB"})


def screen(df, *, base="AAA", trading_now=True, now_ms=NOW):
    return screen_now("AAAUSDT", df, now_ms, base_asset=base,
                      trading_now=trading_now, known_assets=KNOWN)


# --- current eligibility ---------------------------------------------------


def test_a_healthy_liquid_symbol_is_eligible():
    assert screen(frame(400)) == "eligible"


def test_a_delisted_asset_cannot_appear_in_todays_scanner():
    """The S2 E8 defect, in the tense where it would hurt a user directly.

    This frame is impeccable by every historical measure: 400 consecutive
    bars, ample volume, its own last thirty bars perfectly regular. It stopped
    trading a year ago. A scanner that shows it is offering a coin nobody can
    buy.
    """
    dead = frame(400, ends_ms=NOW - 365 * DAY)
    assert screen(dead) == "stale"
    # And the venue's own word is enough on its own, before any bar is read.
    assert screen(frame(400), trading_now=False) == "not_trading"


def test_the_recency_test_is_anchored_to_now_not_to_the_symbols_own_history():
    """The exact shape of E8. `tail(30)` on a dead frame is always tidy."""
    dead = frame(400, ends_ms=NOW - 365 * DAY)
    tail = dead.tail(30)
    span = (int(tail["open_time"].iloc[-1]) - int(tail["open_time"].iloc[0]))
    assert span == 29 * DAY, "the dead frame's own last 30 bars ARE regular"
    assert screen(dead) != "eligible"


def test_a_symbol_whose_last_bar_is_just_inside_the_window_survives():
    fresh = frame(400, ends_ms=NOW - MAX_LAST_BAR_AGE_MS + HOUR)
    assert screen(fresh) == "eligible"
    just_out = frame(400, ends_ms=NOW - MAX_LAST_BAR_AGE_MS - HOUR)
    assert screen(just_out) == "stale"


def test_a_symbol_with_holes_in_the_last_month_is_inactive():
    df = frame(400)
    # Remove ten of the last thirty bars, keeping the final one so the staleness
    # test still passes and `inactive` is what actually fires.
    keep = list(range(len(df) - 30, len(df) - 20))
    df = df.drop(index=keep).reset_index(drop=True)
    assert screen(df) == "inactive"


def test_the_history_and_liquidity_floors_are_s2s_own():
    assert screen(frame(150)) == "too_few_bars"
    assert screen(frame(400, quote_volume=1_000.0)) == "illiquid"


def test_asset_class_exclusions_run_before_any_bar_is_read():
    assert screen(frame(400), base="USDC") == "pegged"
    assert screen(frame(400), base="WBTC") == "wrapped"
    assert screen(frame(400), base="ADAUP") == "leveraged"
    # And the leveraged rule still does not eat ordinary tokens.
    assert screen(frame(400), base="JUP") == "eligible"


def test_a_bar_that_has_not_closed_yet_is_not_visible():
    """Live klines include the day in progress; it must not be read."""
    df = frame(400, ends_ms=NOW + DAY)  # last bar closes in the future
    facts = coin_facts("AAAUSDT", df, df, NOW, base_asset="AAA")
    assert facts is not None
    assert facts["last_close_ms"] < NOW


# --- facts ------------------------------------------------------------------


def test_every_declared_field_is_present_and_finite():
    df = frame(400, drift=0.001)
    facts = coin_facts("AAAUSDT", df, frame(400), NOW, base_asset="AAA")
    assert facts is not None
    for name in FIELDS:
        assert math.isfinite(facts[name]), name


def test_a_symbol_missing_one_number_yields_no_card_at_all():
    """All-or-nothing: a blank where the drawdown belongs invites a guess."""
    assert coin_facts("AAAUSDT", frame(100), frame(400), NOW,
                      base_asset="AAA") is None


def test_realized_volatility_is_zero_for_a_flat_series():
    assert realized_vol(frame(200)) == pytest.approx(0.0)


def test_realized_volatility_rises_with_the_size_of_the_moves():
    calm = frame(200, drift=0.0)
    calm.loc[calm.index[-10:], "close"] *= 1.01
    wild = frame(200, drift=0.0)
    wild.loc[wild.index[-10:], "close"] *= 1.30
    assert realized_vol(wild) > realized_vol(calm)


def test_percentiles_are_midrank_and_never_divide_by_zero():
    coins = [{f: float(i) for f in FIELDS} for i in range(5)]
    percentiles(coins)
    assert coins[0]["percentiles"]["price"] == 10   # 0 below, half of 1 tie
    assert coins[-1]["percentiles"]["price"] == 90
    single = [{f: 1.0 for f in FIELDS}]
    percentiles(single)
    assert single[0]["percentiles"]["price"] == 50


def code_only(path: str) -> str:
    """Source with comments and string literals removed.

    The banned words below are ones this project writes *about* constantly —
    the whole point of the S3 record is explaining why there is no composite
    score. Scanning raw text would make an honest comment fail the test and
    teach the next person to stop writing the comment, so only executable
    code is scanned.
    """
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING,
                            tokenize.NL, tokenize.NEWLINE):
                continue
            out.append(tok.string)
    return " ".join(out)


BANNED_IN_CODE = ("composite", "total_score", "weighted", "confluence",
                  "combine_features", "probability", "confidence")


def test_percentiles_stay_per_field_and_are_never_aggregated():
    """A mean of percentiles is a composite score wearing a different hat."""
    code = code_only("tools/spot_snapshot.py")
    for banned in BANNED_IN_CODE:
        assert banned not in code, banned


# --- the assembled snapshot -------------------------------------------------


def panel(n: int = 4) -> tuple[dict, dict]:
    frames = {BTC_SYMBOL: frame(400, drift=0.0005)}
    meta = {BTC_SYMBOL: {"base_asset": "BTC", "trading_now": True}}
    for i in range(n):
        sym = f"C{i}USDT"
        frames[sym] = frame(400, drift=0.0001 * (i + 1),
                            quote_volume=10_000_000.0 * (i + 1))
        meta[sym] = {"base_asset": f"C{i}", "trading_now": True}
    return frames, meta


def test_a_snapshot_carries_the_benchmark_and_every_eligible_coin():
    frames, meta = panel()
    payload = build_snapshot(frames, meta, NOW, source="test")
    assert payload["schema"] == "spot.snapshot/1"
    assert payload["counts"]["eligible"] == 5
    assert payload["btc"]["symbol"] == BTC_SYMBOL
    assert [c["symbol"] for c in payload["coins"]] == sorted(frames)


def test_a_snapshot_without_the_benchmark_is_refused():
    frames, meta = panel()
    frames.pop(BTC_SYMBOL)
    with pytest.raises(SnapshotBuildError, match="BTCUSDT"):
        build_snapshot(frames, meta, NOW, source="test")


def test_a_snapshot_with_nothing_eligible_is_refused_rather_than_emptied():
    frames = {BTC_SYMBOL: frame(400)}
    meta = {BTC_SYMBOL: {"base_asset": "BTC", "trading_now": False}}
    with pytest.raises(SnapshotBuildError, match="current eligibility"):
        build_snapshot(frames, meta, NOW, source="test")


def test_data_asof_is_the_oldest_bar_behind_anything_shown():
    """The freshness promise has to hold for every coin on the screen."""
    frames, meta = panel()
    frames["C0USDT"] = frame(400, ends_ms=NOW - 24 * HOUR)
    payload = build_snapshot(frames, meta, NOW, source="test")
    assert payload["data_asof_ms"] == min(c["last_close_ms"]
                                          for c in payload["coins"])
    assert payload["data_asof_ms"] < payload["generated_at_ms"]


def test_the_same_inputs_build_the_same_snapshot():
    frames, meta = panel()
    a = build_snapshot(frames, meta, NOW, source="test")
    b = build_snapshot(frames, meta, NOW, source="test")
    assert a["content_sha256"] == b["content_sha256"]


def test_an_excluded_symbol_is_counted_by_reason_rather_than_vanishing():
    frames, meta = panel()
    frames["DEADUSDT"] = frame(400, ends_ms=NOW - 400 * DAY)
    meta["DEADUSDT"] = {"base_asset": "DEAD", "trading_now": False}
    payload = build_snapshot(frames, meta, NOW, source="test")
    assert "DEADUSDT" not in [c["symbol"] for c in payload["coins"]]
    assert payload["counts"]["excluded"]["not_trading"] == 1
    assert payload["counts"]["screened"] == len(frames)


def test_the_snapshot_is_replaced_atomically(tmp_path):
    """A handler may be reading the file at the moment it is rewritten."""
    frames, meta = panel()
    payload = build_snapshot(frames, meta, NOW, source="test")
    path = write_snapshot(payload, str(tmp_path))
    again = write_snapshot(build_snapshot(frames, meta, NOW + 1000,
                                          source="test"), str(tmp_path))
    assert path == again
    assert not list(tmp_path.glob("*.tmp"))
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["generated_at_ms"] == NOW + 1000


def test_the_builder_never_reads_futures_or_runtime_state():
    code = code_only("tools/spot_snapshot.py")
    for banned in ("database", "open_trades", "signal_engine", "scheduler",
                   "pipeline", "analyzer"):
        assert banned not in code, banned
