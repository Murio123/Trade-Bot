"""Cumulative Volume Delta from aggregated trades.

For each aggTrade, the `m` flag marks whether the buyer is the market maker.
m == True  -> the aggressor is a seller (taker sell)  -> negative delta
m == False -> the aggressor is a buyer  (taker buy)   -> positive delta
"""
from __future__ import annotations

from typing import Any


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
    }
