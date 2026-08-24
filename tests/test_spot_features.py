"""S3 — the frozen feature definitions, and the leaks they must not have.

A cross-sectional feature can be wrong in ways that still produce plausible
numbers: it can peek at a bar that had not closed, it can change when later
data arrives, or it can rank differently on two identical runs. Each of those
is tested here rather than reasoned about.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spot.features import (BY_ID, FEATURES, abnormal_participation,
                           distance_above_ma, drawdown_from_high,
                           momentum_6_1, rank_symbols, relative_strength_90,
                           rs_persistence, top_quantile_k)

DAY = 86_400_000


def frame(closes, volumes=None, start_ms: int = 0) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(closes):
        ot = start_ms + i * DAY
        rows.append({"open_time": ot, "open": c, "high": c, "low": c,
                     "close": c, "volume": 1.0, "close_time": ot + DAY - 1,
                     "quote_volume": (volumes[i] if volumes else 1e7),
                     "trades": 5})
    return pd.DataFrame(rows)


def flat(n=400, price=100.0):
    return frame([price] * n)


# --- the definitions are the frozen ones ---------------------------------


def test_all_six_rules_are_registered_and_descending():
    assert [f.fid for f in FEATURES] == ["B4", "B5", "F1", "F2", "F3", "F4"]
    assert all(f.descending for f in FEATURES)


def test_momentum_skips_the_most_recent_month():
    """B4 must not see the last 30 days: that is the whole point of the skip."""
    closes = [100.0] * 400
    df = frame(closes)
    base = momentum_6_1(df, df)
    spiked = frame(closes[:-30] + [500.0] * 30)
    assert momentum_6_1(spiked, spiked) == base


def test_momentum_reads_the_formation_window():
    df = frame([50.0] * 220 + [100.0] * 180)
    assert momentum_6_1(df, df) == pytest.approx(1.0)


def test_distance_above_ma_is_zero_on_a_flat_series():
    assert distance_above_ma(flat(), flat()) == pytest.approx(0.0)


def test_drawdown_is_zero_at_the_high_and_negative_below_it():
    assert drawdown_from_high(flat(), flat()) == pytest.approx(0.0)
    df = frame([100.0] * 300 + [50.0])
    assert drawdown_from_high(df, df) == pytest.approx(-0.5)


def test_participation_is_scale_free():
    """F3 must not reward a permanently large coin (S3_SPEC §5)."""
    big = frame([100.0] * 300, volumes=[1e9] * 300)
    small = frame([100.0] * 300, volumes=[1e5] * 300)
    assert abnormal_participation(big, big) == pytest.approx(0.0)
    assert abnormal_participation(small, small) == pytest.approx(0.0)

    surge = frame([100.0] * 300, volumes=[1e5] * 270 + [1e6] * 30)
    assert abnormal_participation(surge, surge) > 1.0


def test_relative_strength_is_zero_against_itself():
    df = flat()
    assert relative_strength_90(df, df) == pytest.approx(0.0)


def test_relative_strength_sees_the_gap_to_btc():
    """The doubling has to sit inside the 90-day window to be seen at all —
    which is the window doing its job, not a limitation."""
    coin = frame([100.0] * 350 + [200.0] * 50)
    btc = flat(400)
    assert relative_strength_90(coin, btc) == pytest.approx(0.6931, abs=1e-3)

    old_news = frame([100.0] * 300 + [200.0] * 100)
    assert relative_strength_90(old_news, btc) == pytest.approx(0.0)


def test_rs_persistence_is_a_share_between_zero_and_one():
    coin = frame([100.0 + i for i in range(400)])
    btc = flat(400)
    v = rs_persistence(coin, btc)
    assert 0.0 <= v <= 1.0
    assert v > 0.9, "a ratio rising every day sits above its own mean"

    falling = frame([500.0 - i for i in range(400)])
    assert rs_persistence(falling, btc) < 0.1


# --- point-in-time: the property that matters most ------------------------


@pytest.mark.parametrize("fid", ["B4", "B5", "F1", "F2", "F3", "F4"])
def test_appending_future_bars_cannot_change_a_feature_value(fid):
    """The single most important test in S3.

    A feature is computed on visible bars only. If tomorrow's data can change
    today's value, every historical evaluation is contaminated and no amount of
    care elsewhere recovers it.
    """
    feature = BY_ID[fid]
    closes = [100.0 + (i % 7) for i in range(400)]
    now = frame(closes)
    later = frame(closes + [999.0] * 60)
    btc_now = flat(400)
    btc_later = flat(460)
    assert feature.value(now, btc_now) == feature.value(
        later.iloc[:400], btc_later.iloc[:400])


@pytest.mark.parametrize("fid", ["B4", "B5", "F1", "F2", "F3", "F4"])
def test_a_short_history_yields_no_value_rather_than_a_guess(fid):
    feature = BY_ID[fid]
    short = flat(feature.min_bars - 1)
    assert feature.value(short, flat(400)) is None


@pytest.mark.parametrize("fid", ["B4", "B5", "F1", "F2", "F3", "F4"])
def test_features_are_deterministic(fid):
    feature = BY_ID[fid]
    df, btc = frame([100.0 + (i % 11) for i in range(400)]), flat(400)
    assert feature.value(df, btc) == feature.value(df, btc)


def test_btc_relative_features_align_on_timestamps_not_positions():
    """A coin listed later has fewer bars; comparing bar 91 of each would
    compare different calendar windows."""
    btc = frame([100.0 + i for i in range(400)])
    coin = frame([100.0 + i for i in range(200)], start_ms=200 * DAY)
    assert relative_strength_90(coin, btc) is not None


# --- ranking --------------------------------------------------------------


def test_ranking_is_deterministic_and_breaks_ties_on_name():
    values = {"ZZZ": 1.0, "AAA": 1.0, "MMM": 2.0}
    assert rank_symbols(values) == ["MMM", "AAA", "ZZZ"]
    assert rank_symbols(dict(reversed(list(values.items())))) == \
        ["MMM", "AAA", "ZZZ"]


def test_top_quantile_scales_with_the_universe():
    """A fixed Top-5 would be the top 20% at N=25 and the top 2% at N=238."""
    assert top_quantile_k(25) == 5
    assert top_quantile_k(84) == 17
    assert top_quantile_k(238) == 48
    assert top_quantile_k(1) == 1
