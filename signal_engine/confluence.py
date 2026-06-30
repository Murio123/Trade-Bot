"""Levels 2 & 3: Confluence scoring + categorical diversity requirement.

Each indicator contributes points to one of five categories. The scoring is
symmetric: the same structure scores a long setup from bullish evidence and a
short setup from the mirror-image bearish evidence. Smart-Money / structure
signals carry the highest weight, as per the ТЗ.
"""
from __future__ import annotations

from typing import Any

CATEGORIES = ["trend", "momentum", "volume", "structure", "macro"]


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

    # BBW: a squeeze (coiled volatility) breaking out in the trade direction.
    if bull and data.get("bb_breakout_up"):
        scores["momentum"] += 1
        reasons.append("Пробой верхней полосы Боллинджера")
    if not bull and data.get("bb_breakout_down"):
        scores["momentum"] += 1
        reasons.append("Пробой нижней полосы Боллинджера")
    if data.get("bb_squeeze"):
        scores["momentum"] += 1
        reasons.append("Сжатие BBW — готовится импульс")

    # --- trend -----------------------------------------------------------
    if bull and data.get("ema_aligned_bullish"):
        scores["trend"] += 1
        reasons.append("EMA бычий порядок")
    if not bull and data.get("ema_aligned_bearish"):
        scores["trend"] += 1
        reasons.append("EMA медвежий порядок")

    # --- structure (weighted highest) -----------------------------------
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

    # Reversal / exhaustion at an extreme (catching a bottom / top).
    if bull and data.get("bullish_reversal"):
        scores["structure"] += 3 if data.get("reversal_strong_bull") else 2
        reasons.append("Разворотный сетап — признаки дна")
    if not bull and data.get("bearish_reversal"):
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
    funding = data.get("funding")
    if funding is not None:
        if bull and funding < -0.0001:
            scores["macro"] += 3
            reasons.append("Funding экстремально отрицательный")
        elif not bull and funding > 0.0001:
            scores["macro"] += 3
            reasons.append("Funding экстремально положительный")

    netflow = data.get("exchange_netflow")
    if netflow is not None:
        if bull and netflow < 0:
            scores["macro"] += 1
            reasons.append("Отток с бирж")
        elif not bull and netflow > 0:
            scores["macro"] += 1
            reasons.append("Приток на биржи")

    total = sum(scores.values())
    return total, scores, reasons


def has_diverse_confirmation(scores_by_category: dict[str, int], min_categories: int = 3) -> bool:
    active = [cat for cat, score in scores_by_category.items() if score > 0]
    return len(active) >= min_categories
