"""Liquidity analysis: equal highs/lows (stop pools) + sweep detection.

A liquidity sweep occurs when price spikes through a cluster of equal
highs/lows (where stops rest) and then closes back inside — a classic
stop-hunt followed by reversal.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _equal_levels(values: pd.Series, tolerance: float, min_count: int = 2) -> list[float]:
    """Cluster nearly-equal price levels."""
    levels: list[float] = []
    used = [False] * len(values)
    vals = values.values
    for i in range(len(vals)):
        if used[i]:
            continue
        cluster = [vals[i]]
        used[i] = True
        for j in range(i + 1, len(vals)):
            if used[j]:
                continue
            if abs(vals[j] - vals[i]) <= tolerance:
                cluster.append(vals[j])
                used[j] = True
        if len(cluster) >= min_count:
            levels.append(sum(cluster) / len(cluster))
    return levels


def detect_liquidity(df: pd.DataFrame, lookback: int = 60,
                     atr_value: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "equal_highs": [],
        "equal_lows": [],
        "liquidity_swept_below": False,
        "liquidity_swept_above": False,
        "reversal_candle": False,
    }
    if len(df) < 10:
        return out

    window = df.iloc[-lookback:]
    price = float(df["close"].iloc[-1])
    tolerance = (atr_value or price * 0.001) * 0.5

    highs = _equal_levels(window["high"], tolerance)
    lows = _equal_levels(window["low"], tolerance)
    out["equal_highs"] = sorted(highs, reverse=True)[:5]
    out["equal_lows"] = sorted(lows)[:5]

    last = df.iloc[-1]
    body_low = min(last["open"], last["close"])
    body_high = max(last["open"], last["close"])

    # Sweep below: wick takes out an equal-low cluster but body closes above it.
    for lvl in out["equal_lows"]:
        if last["low"] < lvl and body_low > lvl:
            out["liquidity_swept_below"] = True
            break
    # Sweep above: wick takes out an equal-high cluster but body closes below.
    for lvl in out["equal_highs"]:
        if last["high"] > lvl and body_high < lvl:
            out["liquidity_swept_above"] = True
            break

    # Reversal candle confirmation relative to sweep direction.
    bullish_candle = last["close"] > last["open"]
    bearish_candle = last["close"] < last["open"]
    if out["liquidity_swept_below"] and bullish_candle:
        out["reversal_candle"] = True
    if out["liquidity_swept_above"] and bearish_candle:
        out["reversal_candle"] = True

    return out
