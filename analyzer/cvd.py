"""Cumulative Volume Delta (CVD).

Two ways to derive buy/sell pressure:

1. ``compute_cvd_from_klines`` (preferred) — per-candle delta over the trade
   timeframe. If the exchange provides taker buy volume in klines (Binance:
   ``taker_buy_base``), the delta is exact: buy - sell = 2*taker_buy - volume.
   Otherwise it is estimated from where each candle closes within its range
   (close near the high -> net buying), which works for any OHLCV source.

2. ``compute_cvd`` (fallback) — sums signed quantity from raw aggregated
   trades. Only covers the most recent trades, so it is a short-window proxy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_cvd_from_klines(df: pd.DataFrame, lookback: int = 50) -> dict[str, Any]:
    empty = {"cvd": 0.0, "last_delta": 0.0, "cvd_bullish": False,
             "cvd_bearish": False, "method": "none"}
    if df is None or len(df) == 0:
        return empty

    window = df.iloc[-lookback:]
    has_taker = (
        "taker_buy_base" in window.columns
        and window["taker_buy_base"].notna().any()
    )

    if has_taker:
        taker_buy = window["taker_buy_base"].fillna(0)
        delta = 2 * taker_buy - window["volume"]
        method = "exact"
    else:
        rng = (window["high"] - window["low"]).replace(0, np.nan)
        # position of close within the bar, in [-1, 1]
        loc = (2 * window["close"] - window["high"] - window["low"]) / rng
        loc = loc.fillna(0).clip(-1, 1)
        delta = window["volume"] * loc
        method = "estimate"

    cvd = float(delta.sum())
    last_delta = float(delta.iloc[-1]) if len(delta) else 0.0
    return {
        "cvd": cvd,
        "last_delta": last_delta,
        "cvd_bullish": cvd > 0,
        "cvd_bearish": cvd < 0,
        "method": method,
    }


def compute_cvd(agg_trades: list[dict[str, Any]]) -> dict[str, Any]:
    cvd = 0.0
    buy_vol = 0.0
    sell_vol = 0.0
    for t in agg_trades:
        qty = float(t.get("q", 0.0))
        is_maker_buyer = bool(t.get("m", False))
        if is_maker_buyer:
            cvd -= qty
            sell_vol += qty
        else:
            cvd += qty
            buy_vol += qty
    return {
        "cvd": cvd,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "cvd_bullish": cvd > 0,
        "cvd_bearish": cvd < 0,
        "method": "trades",
    }
