"""Chart rendering smoke tests: must produce a PNG and never raise."""
import os

import numpy as np
import pandas as pd

from bot.charts import render_levels_chart, render_signal_chart


def _ctx():
    rng = np.random.default_rng(3)
    n = 60
    p = np.cumsum(rng.normal(0, 100, n)) + 60000
    df = pd.DataFrame({
        "open": p, "close": p + rng.normal(0, 30, n),
        "high": p + abs(rng.normal(50, 40, n)),
        "low": p - abs(rng.normal(50, 40, n)),
        "volume": np.abs(rng.normal(1000, 200, n)),
        "open_time": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
    })
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    price = float(df["close"].iloc[-1])
    return {
        "df_signal": df, "price": price, "timeframe": "1h",
        "order_blocks": {"bullish_ob": {"low": price - 800, "high": price - 500, "tf": "12h"}},
        "fvg": {}, "equilibrium": {"eq": price + 100},
        "volume_profile": {"poc": price + 50},
        "htf_levels": {"highs": [price + 500], "lows": [price - 400]},
        "liquidity": {},
    }, price


def test_signal_chart_renders_png():
    ctx, price = _ctx()
    sig = {"entry_price": price, "stop_loss": price - 600,
           "target_1": price + 700, "target_2": price + 1400,
           "direction": "long", "style_label": "СВИНГ", "timeframe": "1h", "score": 8}
    path = render_signal_chart(ctx, sig)
    assert os.path.exists(path) and os.path.getsize(path) > 10_000


def test_levels_chart_renders_png():
    ctx, _ = _ctx()
    path = render_levels_chart(ctx)
    assert os.path.exists(path) and os.path.getsize(path) > 10_000


def test_chart_tolerates_missing_zones():
    ctx, price = _ctx()
    ctx.update({"order_blocks": {}, "fvg": {}, "equilibrium": {},
                "volume_profile": {}, "htf_levels": {}})
    sig = {"entry_price": price, "direction": "short",
           "style_label": "ИНТРАДЕЙ", "timeframe": "15m", "score": 6}
    assert os.path.exists(render_signal_chart(ctx, sig))
