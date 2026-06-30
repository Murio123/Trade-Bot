"""Fair Value Gap (FVG) detection — 3-candle imbalance zones.

A bullish FVG forms when candle 1's high is below candle 3's low, leaving an
untraded gap (a demand imbalance) that price tends to revisit. A bearish FVG is
the mirror: candle 1's low above candle 3's high. A gap is "live" until price
trades back into it (mitigation). Price reacting inside a live FVG is a
Smart-Money structure signal.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def detect_fvg(df: pd.DataFrame, lookback: int = 60,
               min_gap_atr: float = 0.1, atr_value: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bullish_fvg": None,
        "bearish_fvg": None,
        "price_in_bullish_fvg": False,
        "price_in_bearish_fvg": False,
        "fvgs": [],
    }
    n = len(df)
    if n < 5:
        return out

    price = float(df["close"].iloc[-1])
    min_gap = (atr_value or 0.0) * min_gap_atr
    start = max(2, n - lookback)

    bullish: list[dict[str, Any]] = []
    bearish: list[dict[str, Any]] = []

    highs = df["high"].values
    lows = df["low"].values

    for i in range(start, n):
        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]

        # Bullish FVG: gap between candle1 high and candle3 low.
        if c3_low > c1_high and (c3_low - c1_high) >= min_gap:
            zone = {"low": float(c1_high), "high": float(c3_low), "index": i, "type": "bullish"}
            # live until a later candle trades back down into the gap bottom
            later_low = lows[i + 1:].min() if i + 1 < n else c3_low
            zone["filled"] = bool(later_low <= c1_high)
            bullish.append(zone)

        # Bearish FVG: gap between candle3 high and candle1 low.
        if c1_low > c3_high and (c1_low - c3_high) >= min_gap:
            zone = {"low": float(c3_high), "high": float(c1_low), "index": i, "type": "bearish"}
            later_high = highs[i + 1:].max() if i + 1 < n else c3_high
            zone["filled"] = bool(later_high >= c1_low)
            bearish.append(zone)

    live_bull = [z for z in bullish if not z["filled"]]
    live_bear = [z for z in bearish if not z["filled"]]
    out["fvgs"] = (live_bull + live_bear)[-6:]

    if live_bull:
        z = live_bull[-1]
        out["bullish_fvg"] = z
        out["price_in_bullish_fvg"] = bool(z["low"] <= price <= z["high"])
    if live_bear:
        z = live_bear[-1]
        out["bearish_fvg"] = z
        out["price_in_bearish_fvg"] = bool(z["low"] <= price <= z["high"])

    return out
