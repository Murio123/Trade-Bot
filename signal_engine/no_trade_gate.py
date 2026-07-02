"""Mandatory NO_TRADE conditions (Part B): pure, testable predicates.

Each returns a reason string when the trade must NOT be taken, else None.
run_cascade collects them; a signal cannot leave the cascade while any
reason stands.
"""
from __future__ import annotations

from typing import Any

import config


def bad_risk_reward(entry: float, stop: float, tp2: float) -> str | None:
    risk = abs(entry - stop)
    if risk <= 0:
        return "нет стоп-лосса — сделка невозможна"
    rr = abs(tp2 - entry) / risk
    if rr < config.MIN_RISK_REWARD:
        return (f"risk/reward {rr:.2f} ниже минимума "
                f"{config.MIN_RISK_REWARD:.2f}")
    return None


def missing_invalidation(stop: float | None, invalidation: float | None) -> str | None:
    if not stop:
        return "нет стоп-лосса"
    if not invalidation:
        return "нет уровня отмены сценария (invalidation)"
    return None


def low_confidence(confidence: float) -> str | None:
    if confidence < config.MIN_CONFIDENCE:
        return (f"уверенность {confidence:.2f} ниже минимума "
                f"{config.MIN_CONFIDENCE:.2f}")
    return None


def tf_conflict(mtf_info: dict[str, Any]) -> str | None:
    """All rated timeframes against the trade = hard conflict."""
    total = mtf_info.get("total", 0)
    if total >= 2 and mtf_info.get("agree", 0) == 0:
        return f"все {total} таймфрейма против направления сделки"
    return None


def position_conflict(open_trades: list[dict[str, Any]] | None,
                      direction: str, symbol: str) -> str | None:
    """An open trade in the OPPOSITE direction on the same symbol."""
    for t in open_trades or []:
        if t.get("symbol") in (None, symbol) and t.get("direction") not in (None, direction):
            return (f"открыта встречная позиция #{t.get('id')} "
                    f"({t.get('direction')}) — конфликт риска")
    return None
