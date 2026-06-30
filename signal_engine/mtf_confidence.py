"""Level 5: Multi-timeframe agreement (confidence modifier, non-blocking)."""
from __future__ import annotations


def trend_label(data: dict) -> str:
    if data.get("ema_aligned_bullish"):
        return "bullish"
    if data.get("ema_aligned_bearish"):
        return "bearish"
    price = data.get("price")
    ema50 = data.get("ema50")
    if price is not None and ema50 is not None:
        return "bullish" if price > ema50 else "bearish"
    return "neutral"


def apply_mtf_confidence(base_confidence: float, trend_1h: str,
                         trend_4h: str, trend_1d: str) -> float:
    confidence = base_confidence
    if trend_1h != trend_4h:
        confidence *= 0.7  # neighbouring-timeframe disagreement penalty
    if trend_1h == trend_4h == trend_1d:
        confidence *= 1.15  # full agreement bonus
    return min(confidence, 1.0)
