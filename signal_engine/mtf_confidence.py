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


def mtf_confidence_factor(trends: list[str], direction: str) -> tuple[float, dict]:
    """Direction-aware multi-timeframe agreement over N timeframes.

    Rewards the confidence when more timeframes trend WITH the trade direction
    and penalises when they conflict. Returns (factor, info).
    """
    want = "bullish" if direction == "long" else "bearish"
    rated = [t for t in trends if t in ("bullish", "bearish")]
    total = len(rated)
    if total == 0:
        return 1.0, {"agree": 0, "total": 0, "ratio": 0.0}
    agree = sum(1 for t in rated if t == want)
    ratio = agree / total
    if ratio == 1.0:
        factor = 1.2
    elif ratio >= 0.66:
        factor = 1.05
    elif ratio >= 0.5:
        factor = 0.9
    else:
        factor = 0.7
    return factor, {"agree": agree, "total": total, "ratio": round(ratio, 2)}
