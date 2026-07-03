"""Mandatory NO_TRADE conditions (Part B): pure, testable predicates.

Each returns a reason string when the trade must NOT be taken, else None.
run_cascade collects them; a signal cannot leave the cascade while any
reason stands.
"""
from __future__ import annotations

import math
from typing import Any

import config

# Every TP2 must carry a known origin; the expected-move gate refuses to
# measure a target whose provenance is unknown.
TP2_SOURCES = {"structural", "liquidity", "order_block", "fvg",
               "volume_profile", "r_multiple_fallback"}


def invalid_tp2(direction: str, entry: float, stop: float | None,
                tp2: float | None, tp1: float | None,
                tp2_source: str | None) -> str | None:
    """Reject a TP2 that must not feed expected_move: wrong side, non-finite,
    colliding with the stop, or of unknown origin. Returns the reason or None."""
    if tp2 is None or not math.isfinite(tp2) or tp2 <= 0:
        return "TP2 не задан или не является конечным положительным числом"
    if tp2_source not in TP2_SOURCES:
        return f"неизвестное происхождение TP2: {tp2_source!r}"
    beyond_entry = (tp2 > entry) if direction == "long" else (tp2 < entry)
    if not beyond_entry:
        return "TP2 не на стороне направления сделки"
    if tp1 is not None and math.isfinite(tp1):
        farther = (tp2 > tp1) if direction == "long" else (tp2 < tp1)
        if not farther:
            return "TP2 не дальше TP1"
    if stop is not None and abs(tp2 - stop) < 1e-9:
        return "TP2 совпадает со стоп-лоссом"
    return None


def effective_expected_move(entry: float, tp2: float, atr: float,
                            horizon_hours: float, tf_hours: float) -> float:
    """Expected move = the STRUCTURAL distance to TP2 (targets already come
    from market structure / liquidity / OB / FVG / volume nodes), capped by
    what volatility can plausibly deliver over the mode's forecast horizon
    (ATR * sqrt(bars)). The threshold below is a scenario-quality FILTER —
    it never feeds back into the targets themselves."""
    structural = abs(tp2 - entry)
    if atr and atr > 0 and tf_hours > 0:
        ceiling = atr * (horizon_hours / tf_hours) ** 0.5
        return round(min(structural, ceiling), 2)
    return round(structural, 2)


def insufficient_expected_move(move_points: float, min_points: float,
                               analysis_type: str) -> str | None:
    if move_points < min_points:
        return (f"ожидаемый ход {move_points:.0f} пт ниже минимума "
                f"{min_points:.0f} пт для режима {analysis_type}")
    return None


def bad_risk_reward(entry: float, stop: float, tp2: float,
                    min_rr: float | None = None) -> str | None:
    min_rr = min_rr if min_rr is not None else config.MIN_RISK_REWARD
    risk = abs(entry - stop)
    if risk <= 0:
        return "нет стоп-лосса — сделка невозможна"
    rr = abs(tp2 - entry) / risk
    if rr < min_rr:
        return f"risk/reward {rr:.2f} ниже минимума {min_rr:.2f}"
    return None


def missing_invalidation(stop: float | None, invalidation: float | None) -> str | None:
    if not stop:
        return "нет стоп-лосса"
    if not invalidation:
        return "нет уровня отмены сценария (invalidation)"
    return None


def low_confidence(confidence: float,
                   min_conf: float | None = None) -> str | None:
    min_conf = min_conf if min_conf is not None else config.MIN_CONFIDENCE
    if confidence < min_conf:
        return f"уверенность {confidence:.2f} ниже минимума {min_conf:.2f}"
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
