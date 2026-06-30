"""Market structure: swing points, HH/HL/LH/LL, Break of Structure (BOS) and
Change of Character (CHoCH).

BOS = price breaks the most recent swing in the direction of the prevailing
trend (continuation). CHoCH = the first break against the prevailing trend
(potential reversal / shift in order flow).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _fractals(df: pd.DataFrame, left: int = 2, right: int = 2) -> tuple[list, list]:
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    n = len(df)
    for i in range(left, n - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if h[i] == win_h.max():
            highs.append((i, float(h[i])))
        if l[i] == win_l.min():
            lows.append((i, float(l[i])))
    return highs, lows


def analyze_structure(df: pd.DataFrame, left: int = 2, right: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {
        "trend": "range",
        "last_event": None,        # BOS_up | BOS_down | CHoCH_up | CHoCH_down
        "last_swing_high": None,
        "last_swing_low": None,
        "sequence": None,          # e.g. "HH/HL" or "LH/LL"
    }
    if len(df) < (left + right + 5):
        return out

    highs, lows = _fractals(df, left, right)
    if len(highs) < 2 or len(lows) < 2:
        return out

    out["last_swing_high"] = round(highs[-1][1], 2)
    out["last_swing_low"] = round(lows[-1][1], 2)

    higher_high = highs[-1][1] > highs[-2][1]
    higher_low = lows[-1][1] > lows[-2][1]
    lower_high = highs[-1][1] < highs[-2][1]
    lower_low = lows[-1][1] < lows[-2][1]

    if higher_high and higher_low:
        out["trend"] = "bullish"
        out["sequence"] = "HH/HL"
    elif lower_high and lower_low:
        out["trend"] = "bearish"
        out["sequence"] = "LH/LL"
    else:
        out["trend"] = "range"
        out["sequence"] = "mixed"

    # Detect the most recent break relative to the prior swing levels.
    price = float(df["close"].iloc[-1])
    prev_high = highs[-2][1]
    prev_low = lows[-2][1]
    if price > prev_high:
        out["last_event"] = "BOS_up" if out["trend"] == "bullish" else "CHoCH_up"
    elif price < prev_low:
        out["last_event"] = "BOS_down" if out["trend"] == "bearish" else "CHoCH_down"

    return out
