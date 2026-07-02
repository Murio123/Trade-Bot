"""Levels 2 & 3: Confluence scoring + categorical diversity requirement.

Each indicator contributes points to one of five categories. The scoring is
symmetric: the same structure scores a long setup from bullish evidence and a
short setup from the mirror-image bearish evidence.

Decorrelation rules (Part A quality pass):
- one phenomenon is counted ONCE: the multi-TF trend-aligned reversal
  REPLACES the plain single-TF reversal points instead of stacking on top;
- category totals are CAPPED so a pile of correlated structure flags cannot
  single-handedly clear the alert threshold;
- direction-neutral evidence (a BB squeeze without a breakout) adds no points.
"""
from __future__ import annotations

from typing import Any

import config

CATEGORIES = ["trend", "momentum", "volume", "structure", "macro"]

# Per-category caps: correlated flags within a category saturate instead of
# stacking (structure alone could reach ~16 and dominate every other signal).
CATEGORY_CAPS = {"structure": 8, "momentum": 4, "volume": 2, "trend": 2, "macro": 3}




def calculate_confluence_score(data: dict[str, Any], direction: str) -> tuple[int, dict[str, int], list[str]]:
    """Score a setup for the given direction ('long' or 'short')."""
    scores = {cat: 0 for cat in CATEGORIES}
    reasons: list[str] = []
    bull = direction == "long"

    # --- momentum --------------------------------------------------------
    rsi = data.get("rsi")
    if rsi is not None:
        if bull and rsi < 30:
            scores["momentum"] += 2
            reasons.append("RSI перепродан")
        elif not bull and rsi > 70:
            scores["momentum"] += 2
            reasons.append("RSI перекуплен")

    if bull and data.get("macd_bullish_cross"):
        scores["momentum"] += 2
        reasons.append("MACD бычий кроссовер")
    if not bull and data.get("macd_bearish_cross"):
        scores["momentum"] += 2
        reasons.append("MACD медвежий кроссовер")

    if bull and data.get("bullish_divergence"):
        scores["momentum"] += 2
        reasons.append("Бычья дивергенция RSI/MACD")
    if not bull and data.get("bearish_divergence"):
        scores["momentum"] += 2
        reasons.append("Медвежья дивергенция RSI/MACD")

    # BBW squeeze is direction-neutral: it only counts through a directional
    # breakout (it used to add +1 to BOTH totals, inflating weak setups).
    if bull and data.get("bb_breakout_up"):
        scores["momentum"] += 1
        reasons.append("Пробой верхней полосы Боллинджера"
                       + (" из сжатия BBW" if data.get("bb_squeeze") else ""))
    if not bull and data.get("bb_breakout_down"):
        scores["momentum"] += 1
        reasons.append("Пробой нижней полосы Боллинджера"
                       + (" из сжатия BBW" if data.get("bb_squeeze") else ""))

    # --- trend -----------------------------------------------------------
    # +2 (was +1): the declared hierarchy puts the higher-TF trend on top,
    # yet it used to weigh a single point against structure's pile.
    if bull and data.get("ema_aligned_bullish"):
        scores["trend"] += 2
        reasons.append("EMA бычий порядок")
    if not bull and data.get("ema_aligned_bearish"):
        scores["trend"] += 2
        reasons.append("EMA медвежий порядок")

    # --- structure (weighted highest, capped) -----------------------------
    if bull and data.get("price_in_bullish_ob") and data.get("rejection_wick"):
        scores["structure"] += 3
        reasons.append("Реакция от бычьего Order Block")
    if not bull and data.get("price_in_bearish_ob") and data.get("rejection_wick"):
        scores["structure"] += 3
        reasons.append("Реакция от медвежьего Order Block")

    if bull and data.get("liquidity_swept_below") and data.get("reversal_candle"):
        scores["structure"] += 3
        reasons.append("Снятие ликвидности снизу + разворот")
    if not bull and data.get("liquidity_swept_above") and data.get("reversal_candle"):
        scores["structure"] += 3
        reasons.append("Снятие ликвидности сверху + разворот")

    # FVG: price reacting inside a live Fair Value Gap (imbalance).
    if bull and data.get("price_in_bullish_fvg"):
        scores["structure"] += 2
        reasons.append("Цена в бычьем FVG (имбаланс)")
    if not bull and data.get("price_in_bearish_fvg"):
        scores["structure"] += 2
        reasons.append("Цена в медвежьем FVG (имбаланс)")

    # Premium/Discount: longs are favoured in discount, shorts in premium.
    if bull and data.get("in_discount"):
        scores["structure"] += 2
        reasons.append("Вход в дисконте (ниже равновесия)")
    if not bull and data.get("in_premium"):
        scores["structure"] += 2
        reasons.append("Вход в премиуме (выше равновесия)")

    # Reversal / exhaustion at an extreme. The multi-TF trend-aligned read
    # REPLACES the single-TF one — it is the same phenomenon confirmed on
    # more timeframes, not an independent extra signal (it used to stack:
    # up to +6 for one reversal).
    if bull:
        if data.get("trend_aligned_bottom"):
            scores["structure"] += 3
            reasons.append("Дно отката по тренду (мультиТФ) — точка входа")
        elif data.get("bullish_reversal"):
            scores["structure"] += 3 if data.get("reversal_strong_bull") else 2
            reasons.append("Разворотный сетап — признаки дна")
    else:
        if data.get("trend_aligned_top"):
            scores["structure"] += 3
            reasons.append("Пик отскока по тренду (мультиТФ) — точка входа")
        elif data.get("bearish_reversal"):
            scores["structure"] += 3 if data.get("reversal_strong_bear") else 2
            reasons.append("Разворотный сетап — признаки пика")

    # --- volume ----------------------------------------------------------
    volume = data.get("volume")
    avg_volume = data.get("avg_volume")
    if volume is not None and avg_volume:
        if volume > avg_volume * 1.5:
            scores["volume"] += 1
            reasons.append("Объём подтверждает движение")

    if bull and data.get("cvd_bullish"):
        scores["volume"] += 1
        reasons.append("CVD на стороне покупателей")
    if not bull and data.get("cvd_bearish"):
        scores["volume"] += 1
        reasons.append("CVD на стороне продавцов")

    # --- macro / crypto-specific ----------------------------------------
    # Funding is contrarian evidence only when it is actually STRETCHED
    # relative to its own history (z-score), not merely non-zero: the old
    # ±0.0001 threshold fired on Binance's BASE funding rate nearly every
    # cycle and handed +3 "экстремально" points to one side for free.
    funding = data.get("funding")
    funding_z = data.get("funding_z")
    if funding is not None and funding_z is not None:
        if bull and funding < 0 and funding_z <= -config.FUNDING_Z_SIGNAL:
            scores["macro"] += 2
            reasons.append(f"Funding заметно отрицательный (z={funding_z:.1f})")
        elif not bull and funding > 0 and funding_z >= config.FUNDING_Z_SIGNAL:
            scores["macro"] += 2
            reasons.append(f"Funding заметно положительный (z={funding_z:.1f})")

    netflow = data.get("exchange_netflow")
    if netflow is not None:
        if bull and netflow < 0:
            scores["macro"] += 1
            reasons.append("Отток с бирж")
        elif not bull and netflow > 0:
            scores["macro"] += 1
            reasons.append("Приток на биржи")

    for cat, cap in CATEGORY_CAPS.items():
        scores[cat] = min(scores[cat], cap)
    total = sum(scores.values())
    return total, scores, reasons


def has_diverse_confirmation(scores_by_category: dict[str, int], min_categories: int = 3) -> bool:
    active = [cat for cat, score in scores_by_category.items() if score > 0]
    return len(active) >= min_categories
