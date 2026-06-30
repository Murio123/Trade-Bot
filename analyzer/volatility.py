"""Volatility analysis: ATR/HV percentiles, regime, and expected move.

All values are derived from the supplied klines — nothing hard-coded.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analyzer.indicators import atr


def _percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) == 0 or value is None:
        return 0.0
    return float((s <= value).mean() * 100)


def analyze_volatility(df: pd.DataFrame, tf_per_day: float = 6.0) -> dict[str, Any]:
    """tf_per_day = how many candles make a day (4H -> 6, 12H -> 2, 1D -> 1)."""
    out: dict[str, Any] = {
        "atr": None, "atr_pct": None, "atr_percentile": None,
        "hv_percentile": None, "regime": "unknown", "expected_move": {},
    }
    if df is None or len(df) < 30:
        return out

    atr_series = atr(df, 14)
    atr_now = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None
    price = float(df["close"].iloc[-1])
    out["atr"] = round(atr_now, 2) if atr_now else None
    out["atr_pct"] = round(atr_now / price * 100, 2) if atr_now and price else None
    out["atr_percentile"] = round(_percentile_rank(atr_series, atr_now), 1) if atr_now else None

    # Historical volatility: std of log returns, annualised-ish percentile.
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv = log_ret.rolling(20).std()
    hv_now = float(hv.iloc[-1]) if not pd.isna(hv.iloc[-1]) else None
    out["hv_percentile"] = round(_percentile_rank(hv, hv_now), 1) if hv_now else None

    # Regime from ATR percentile.
    p = out["atr_percentile"] or 0
    if p >= 80:
        out["regime"] = "expansion"
    elif p <= 20:
        out["regime"] = "compression"
    else:
        out["regime"] = "normal"

    # Expected move over horizons: ATR * sqrt(candles in horizon).
    if atr_now:
        for label, days in (("24h", 1), ("3d", 3), ("7d", 7), ("14d", 14)):
            candles = max(days * tf_per_day, 1)
            move = atr_now * math.sqrt(candles)
            out["expected_move"][label] = {
                "abs": round(move, 0),
                "pct": round(move / price * 100, 2) if price else None,
            }
    return out
