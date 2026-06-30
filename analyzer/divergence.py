"""Automatic RSI / MACD divergence detection.

Algorithmic replacement for the visual search: compare local price extrema
against the corresponding oscillator extrema over the last N candles.

Bullish divergence: price makes a lower low while the oscillator makes a
higher low. Bearish divergence: price makes a higher high while the
oscillator makes a lower high.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _local_minima(series: pd.Series, order: int = 3) -> list[int]:
    idx = []
    vals = series.values
    n = len(vals)
    for i in range(order, n - order):
        window = vals[i - order:i + order + 1]
        if np.isnan(window).any():
            continue
        if vals[i] == np.min(window) and vals[i] < vals[i - 1]:
            idx.append(i)
    return idx


def _local_maxima(series: pd.Series, order: int = 3) -> list[int]:
    idx = []
    vals = series.values
    n = len(vals)
    for i in range(order, n - order):
        window = vals[i - order:i + order + 1]
        if np.isnan(window).any():
            continue
        if vals[i] == np.max(window) and vals[i] > vals[i - 1]:
            idx.append(i)
    return idx


def detect_divergence(df: pd.DataFrame, oscillator: pd.Series,
                      lookback: int = 60, order: int = 3) -> dict[str, Any]:
    """Return divergence flags for the most recent two pivots."""
    result = {
        "bullish_divergence": False,
        "bearish_divergence": False,
        "detail": None,
    }
    if oscillator is None or len(df) < lookback:
        return result

    price = df["close"].iloc[-lookback:].reset_index(drop=True)
    osc = oscillator.iloc[-lookback:].reset_index(drop=True)

    # Bullish: two most recent lows
    lows = _local_minima(price, order)
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if price[b] < price[a] and osc[b] > osc[a]:
            result["bullish_divergence"] = True
            result["detail"] = "price lower-low vs oscillator higher-low"

    # Bearish: two most recent highs
    highs = _local_maxima(price, order)
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if price[b] > price[a] and osc[b] < osc[a]:
            result["bearish_divergence"] = True
            result["detail"] = "price higher-high vs oscillator lower-high"

    return result
