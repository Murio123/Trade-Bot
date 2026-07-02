"""Market regime from the 1D chart — the top of the analysis hierarchy.

One source of truth for "what market are we in"; the confluence category
weights depend on it (a trend-following signal means little in a range, a
mean-reversion signal means little in a strong trend).
"""
from __future__ import annotations

from typing import Any

REGIMES = ("trend_up", "trend_down", "range", "high_volatility")

# Multipliers applied to the confluence category scores per regime. Neutral
# 1.0 everywhere would reproduce the old flat sum; the hierarchy instead says:
# in a trend, trend/structure evidence leads; in a range, structure (S/R
# reactions) and momentum extremes lead while trend-following is discounted.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "trend_up":        {"trend": 1.5, "structure": 1.2, "momentum": 0.8,
                        "volume": 1.0, "macro": 1.0},
    "trend_down":      {"trend": 1.5, "structure": 1.2, "momentum": 0.8,
                        "volume": 1.0, "macro": 1.0},
    "range":           {"trend": 0.5, "structure": 1.3, "momentum": 1.2,
                        "volume": 1.0, "macro": 1.0},
    # Blow-off conditions: everything is discounted; the abnormal-volatility
    # gate usually blocks outright before weights matter.
    "high_volatility": {"trend": 0.8, "structure": 0.8, "momentum": 0.8,
                        "volume": 0.8, "macro": 0.8},
}


def detect_regime(ind_1d: dict[str, Any] | None,
                  vol_1d: dict[str, Any] | None = None) -> str:
    """Classify the market regime from 1D EMAs and the 1D ATR percentile."""
    if vol_1d and (vol_1d.get("atr_percentile") or 0) >= 90:
        return "high_volatility"
    ind_1d = ind_1d or {}
    price, e20 = ind_1d.get("price"), ind_1d.get("ema20")
    e50, e200 = ind_1d.get("ema50"), ind_1d.get("ema200")
    if None in (price, e20, e50, e200):
        return "range"
    if price > e20 > e50 > e200:
        return "trend_up"
    if price < e20 < e50 < e200:
        return "trend_down"
    return "range"


def weighted_total(scores: dict[str, int], regime: str) -> float:
    """Regime-weighted score replacing the flat category sum."""
    weights = REGIME_WEIGHTS.get(regime) or {}
    return round(sum(v * weights.get(cat, 1.0) for cat, v in scores.items()), 1)
