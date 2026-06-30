"""Level 1: Higher-Timeframe Bias Filter (blocking).

Signals against the daily trend are blocked entirely — not sent, not recorded
as "weak". This is the protection against entering against the higher trend.
"""
from __future__ import annotations

from typing import Any, Optional


def get_htf_bias(data_1d: dict[str, Any]) -> str:
    price = data_1d.get("price")
    ema20 = data_1d.get("ema20")
    ema50 = data_1d.get("ema50")
    ema200 = data_1d.get("ema200")
    if None in (price, ema20, ema50, ema200):
        return "neutral"
    if price > ema20 > ema50 > ema200:
        return "bullish"
    if price < ema20 < ema50 < ema200:
        return "bearish"
    return "neutral"


def filter_by_htf(signal_direction: Optional[str], htf_bias: str) -> Optional[str]:
    if htf_bias == "bullish" and signal_direction == "short":
        return None
    if htf_bias == "bearish" and signal_direction == "long":
        return None
    return signal_direction
