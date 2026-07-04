"""Per-forecast realized R — единый источник формулы (Stage 10 определение,
Stage 11 producer).

Leaf-модуль: только stdlib/typing, без side effects. НЕ импортирует database,
tools или analyzer.outcomes — чтобы его могли безопасно использовать и runtime
(analyzer/outcomes.py как producer), и offline CLI (tools/realized_r.py) без
циклов и без подтягивания аналитики в decision-path.

realized_r НЕ участвует в торговых решениях, scoring, risk или gating — это
измерение постфактум (нужен закрытый outcome). Определение R едино с
bot/journal.evaluate_trade и backtest._resolve:

    risk = |reference_price - stop_loss|
    sign = +1 (long) / -1 (short)
    R    = sign * (exit - reference_price) / risk

Формула — см. classify(). Округления НЕТ: возвращается raw float (кроме точных
-1.0 / 0.0). Округление — только на display (CLI/summary).
"""
from __future__ import annotations

from typing import Any

ENTER = "ENTER"


def _levels(forecast: dict[str, Any]) -> tuple[float | None, float | None]:
    tps = forecast.get("take_profit_levels") or []
    tp1 = float(tps[0]) if len(tps) > 0 and tps[0] else None
    tp2 = float(tps[1]) if len(tps) > 1 and tps[1] else None
    return tp1, tp2


def _ref(forecast: dict[str, Any], outcome: dict[str, Any] | None) -> float | None:
    """Якорь-«вход»: reference_price из outcome, иначе та же дефиниция, что в
    analyzer/outcomes.measure_outcome (executable_price → signal_close)."""
    if outcome is not None and outcome.get("reference_price") is not None:
        return float(outcome["reference_price"])
    ref = (forecast.get("executable_price_at_decision")
           or forecast.get("signal_close_price"))
    return float(ref) if ref else None


def _mtm(return_72h: Any, ref: float, risk: float) -> float | None:
    """Mark-to-market R из sign-adjusted %-возврата за 72h."""
    if return_72h is None:
        return None
    return (float(return_72h) / 100.0) * ref / risk


def classify(forecast: dict[str, Any],
             outcome: dict[str, Any] | None) -> tuple[float | None, str]:
    """Вернуть (realized_r | None, kind). kind описывает выбранную ветку либо
    причину недоступности (для summary/CLI-разбивки)."""
    if forecast.get("analysis_status") != ENTER:
        return None, "none_not_enter"          # WAIT / NO_TRADE / blocked / нет статуса
    direction = forecast.get("candidate_direction")
    if direction not in ("long", "short"):
        return None, "none_no_direction"
    if outcome is None:
        return None, "none_no_outcome"
    if not outcome.get("resolved"):
        return None, "none_unresolved"          # censored — окно не закрыто

    ref = _ref(forecast, outcome)
    if ref is None:
        return None, "none_missing_reference"
    stop = forecast.get("stop_loss")
    if stop is None:
        return None, "none_missing_stop"
    risk = abs(ref - float(stop))
    if risk == 0:
        return None, "none_zero_risk"

    sign = 1.0 if direction == "long" else -1.0
    tp1, tp2 = _levels(forecast)
    tp1_hit = bool(outcome.get("tp1_hit"))
    tp2_hit = bool(outcome.get("tp2_hit"))
    stop_hit = bool(outcome.get("stop_hit"))
    r72 = outcome.get("return_72h")

    if stop_hit and not tp1_hit:
        return -1.0, "loss"
    if tp2_hit and tp2 is not None:
        return sign * (tp2 - ref) / risk, "tp2_win"
    if tp1_hit and stop_hit:
        return 0.0, "breakeven"
    if tp1_hit and not tp2_hit and not stop_hit:
        if tp2 is None and tp1 is not None:
            return sign * (tp1 - ref) / risk, "tp1_single_win"
        mtm = _mtm(r72, ref, risk)
        return (mtm, "mark_to_market") if mtm is not None else (None, "none_no_return72h")
    # Ни TP, ни стоп (или tp2_hit без сохранённого tp2) — resolved -> MTM.
    mtm = _mtm(r72, ref, risk)
    return (mtm, "mark_to_market") if mtm is not None else (None, "none_no_return72h")


def realized_r(forecast: dict[str, Any],
               outcome: dict[str, Any] | None) -> float | None:
    """Честный per-forecast R (raw float) или None (см. модульный docstring)."""
    return classify(forecast, outcome)[0]
