"""S2 — CLEAN_2X(180d): the label, and the two things it must never confuse.

The label is M02's; what is tested here is the configuration (fixed ratios out
of a constant sigma) and the interpretation that is ours — a coin that died
inside the horizon is a resolved failure, a coin still trading past the panel's
edge is missing data, and collapsing the two in either direction rebuilds the
bias the whole spot track exists to avoid.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spot.labels import (CENSORED, CONFIG, DEAD, GAP, HIT, HORIZON_BARS,
                         STOPPED, TIMED_OUT, bar_index_asof, label_event,
                         summarize)

DAY = 86_400_000


def series(closes: list[float], *, highs=None, lows=None,
           start_ms: int = 0) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(closes):
        ot = start_ms + i * DAY
        rows.append({
            "open_time": ot, "open": c,
            "high": (highs[i] if highs else c),
            "low": (lows[i] if lows else c),
            "close": c, "volume": 1.0,
            "close_time": ot + DAY - 1, "quote_volume": 1e7, "trades": 5,
        })
    return pd.DataFrame(rows)


def label(df, idx=0, trading_now=True, btc=None):
    return label_event(df, idx, symbol="AAAUSDT", asof="2020-01-01",
                       asof_ms=int(df["close_time"].iloc[idx]) + 1,
                       trading_now=trading_now, btc=btc)


# --- the frozen configuration --------------------------------------------


def test_the_barriers_are_the_frozen_ratios():
    assert (CONFIG.upper_mult, CONFIG.lower_mult) == (1.0, 0.4)
    assert CONFIG.vertical_bars == HORIZON_BARS == 180


def test_a_constant_sigma_turns_m02_into_fixed_percentages():
    """+100% and -40% of the entry price, not of its volatility."""
    up = series([100.0] * 181,
                highs=[100.0] + [199.9] * 5 + [100.0] * 175)
    assert label(up).outcome == TIMED_OUT, "199.9 is not yet a double"

    up2 = up.copy()
    up2.loc[3, "high"] = 200.0
    assert label(up2).outcome == HIT


def test_the_lower_barrier_is_minus_forty_percent():
    down = series([100.0] * 181, lows=[100.0] + [60.1] * 180)
    assert label(down).outcome == TIMED_OUT
    down.loc[2, "low"] = 60.0
    assert label(down).outcome == STOPPED


# --- ordering and the tie rule -------------------------------------------


def test_the_first_touch_decides_not_the_best_one():
    """-40% on day 2, +100% on day 5: the stop came first, so it is a 0."""
    df = series([100.0] * 181,
                highs=[100.0] * 5 + [250.0] + [100.0] * 175,
                lows=[100.0, 100.0, 55.0] + [100.0] * 178)
    lab = label(df)
    assert lab.outcome == STOPPED and lab.clean_2x == 0
    assert lab.mfe > 1.0, "the upside still happened and is still recorded"


def test_a_bar_spanning_both_barriers_resolves_against_the_holder():
    """M02's audited tie rule (G1 amendment A1), inherited unchanged."""
    df = series([100.0] * 181,
                highs=[100.0, 250.0] + [100.0] * 179,
                lows=[100.0, 50.0] + [100.0] * 179)
    lab = label(df)
    assert lab.tie is True
    assert lab.outcome == STOPPED and lab.clean_2x == 0


# --- death vs censoring ---------------------------------------------------


def test_a_coin_that_died_inside_the_horizon_is_a_resolved_zero():
    """The single most important line in this module.

    Dropping these as 'unknown' would delete exactly the failures and rebuild
    survivorship bias one layer up.
    """
    df = series([100.0] * 40)          # dies well inside 180 bars
    lab = label(df, trading_now=False)
    assert lab.outcome == DEAD and lab.clean_2x == 0
    assert lab.resolved is True


def test_a_living_coin_past_the_panel_edge_is_censored_not_a_zero():
    df = series([100.0] * 40)
    lab = label(df, trading_now=True)
    assert lab.outcome == CENSORED and lab.clean_2x is None
    assert lab.resolved is False


def test_a_dead_coin_that_doubled_before_dying_still_counts_as_a_hit():
    df = series([100.0] * 40, highs=[100.0] * 3 + [210.0] + [100.0] * 36)
    lab = label(df, trading_now=False)
    assert lab.outcome == HIT and lab.clean_2x == 1


# --- the record vector ----------------------------------------------------


def test_the_vector_keeps_what_the_binary_throws_away():
    df = series([100.0] * 181,
                highs=[100.0] * 100 + [250.0] + [100.0] * 80,
                lows=[100.0] * 50 + [65.0] + [100.0] * 130)
    lab = label(df)
    assert lab.outcome == HIT
    assert lab.mfe == pytest.approx(1.5)
    assert lab.mae == pytest.approx(-0.35)
    assert lab.days_to_2x == 100
    assert lab.window_bars == 180


def test_excess_vs_btc_is_measured_over_the_same_calendar_window():
    coin = series([100.0] * 181)
    coin.loc[180, "close"] = 150.0
    btc = series([100.0] * 181)
    btc.loc[180, "close"] = 120.0
    lab = label(coin, btc=btc)
    assert lab.excess_vs_btc == pytest.approx(
        pytest.importorskip("math").log(1.5) - pytest.importorskip("math").log(1.2))


# --- gaps and summaries ---------------------------------------------------


def test_an_event_spanning_a_hole_in_the_grid_has_no_answer():
    df = series([100.0] * 181)
    df = pd.concat([df.iloc[:5], df.iloc[9:]], ignore_index=True)
    assert label(df).outcome == GAP


def test_the_summary_counts_only_resolved_events_in_the_base_rate():
    hit = label(series([100.0] * 40, highs=[100.0, 250.0] + [100.0] * 38),
                trading_now=False)
    dead = label(series([100.0] * 40), trading_now=False)
    censored = label(series([100.0] * 40), trading_now=True)
    s = summarize([hit, dead, censored])
    assert s["resolved"] == 2 and s["hits"] == 1
    assert s["base_rate"] == 0.5, "the censored event must not dilute the rate"


def test_bar_index_asof_never_returns_a_bar_that_had_not_closed():
    df = series([100.0] * 10)
    close5 = int(df["close_time"].iloc[5])
    assert bar_index_asof(df, close5) == 4
    assert bar_index_asof(df, close5 + 1) == 5
    assert bar_index_asof(df, -1) is None
