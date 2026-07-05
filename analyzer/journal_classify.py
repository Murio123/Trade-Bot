"""Stage 13: pure classifier for Trading Journal v2.

Классифицирует УЖЕ персистируемые строки forecasts + forecast_outcomes в
человекочитаемую метку решения — БЕЗ изменения торгового поведения, scoring,
risk, gating, схемы БД или producer'ов. Чистая, детерминированная деривация
постфактум.

Leaf-модуль: только stdlib/typing. НЕ импортирует database, scheduler,
pipeline, signal_engine, tools, analyzer.outcomes или analyzer.realized_r — так
он остаётся полностью независимым от runtime/decision-path и от формулы R
(формула R не дублируется и не меняется: значение realized_r приходит на вход
как уже измеренное поле outcome).

Классификатор НЕ выдумывает исходов: если уровней/направления/outcome нет —
возвращает none / no_levels / unresolved, но никогда не трактует отсутствие
данных как успех или неудачу.

Определения гипотетического исхода для не-ENTER (WAIT / NO_TRADE / blocked)
совпадают с трактовкой хитов в analyzer.outcomes / analyzer.realized_r:

    avoided_loss       — сработал бы чистый стоп (stop_hit и не tp1_hit);
    missed_opportunity — сетап отработал бы (tp2_hit, либо tp1_hit без стопа);
    иначе (breakeven / ни то ни другое / нет resolved-outcome) — none.
"""
from __future__ import annotations

from typing import Any

ENTER = "ENTER"

# Метки классификации (стабильные строковые константы для отчёта/тестов).
GOOD_TRADE = "good_trade"
BAD_TRADE = "bad_trade"
AVOIDED_LOSS = "avoided_loss"
MISSED_OPPORTUNITY = "missed_opportunity"
UNRESOLVED = "unresolved"
NO_LEVELS = "no_levels"
NONE = "none"

CLASSES = (
    GOOD_TRADE, BAD_TRADE, AVOIDED_LOSS, MISSED_OPPORTUNITY,
    UNRESOLVED, NO_LEVELS, NONE,
)


def _usable_levels(forecast: dict[str, Any]) -> bool:
    """Есть ли достаточно уровней, чтобы судить об исходе: стоп + хотя бы TP1."""
    stop = forecast.get("stop_loss")
    tps = forecast.get("take_profit_levels") or []
    tp1 = tps[0] if len(tps) > 0 else None
    return stop is not None and tp1 is not None


def _hypothetical_favorable(outcome: dict[str, Any]) -> bool:
    """Исход в пользу прогноза: TP2, либо TP1 без выбитого стопа.

    Единая с win-ветками realized_r трактовка «сетап отработал»."""
    return bool(outcome.get("tp2_hit")
                or (outcome.get("tp1_hit") and not outcome.get("stop_hit")))


def _hypothetical_loss(outcome: dict[str, Any]) -> bool:
    """Чистый стоп: выбит стоп и TP1 ни разу не задет (та же дефиниция loss,
    что даёт realized_r == -1.0)."""
    return bool(outcome.get("stop_hit") and not outcome.get("tp1_hit"))


def classify(forecast: dict[str, Any],
             outcome: dict[str, Any] | None) -> str:
    """Вернуть одну метку из CLASSES для пары (forecast, outcome | None).

    Чистая функция: результат зависит только от переданных полей.

        ENTER:
            outcome отсутствует / не resolved -> unresolved
            resolved, realized_r > 0          -> good_trade
            resolved, realized_r < 0          -> bad_trade
            resolved, realized_r == 0 / None  -> none  (безубыток или R неизвестен;
                                                        не трактуем как успех/провал)
        WAIT / NO_TRADE / blocked:
            нет направления                   -> none
            нет достаточных уровней           -> no_levels
            outcome отсутствует / не resolved -> none
            гипотетический чистый стоп        -> avoided_loss
            гипотетически отработал бы         -> missed_opportunity
            иначе (breakeven / неоднозначно)  -> none
    """
    status = forecast.get("analysis_status")
    direction = forecast.get("candidate_direction")

    if status == ENTER:
        if outcome is None or not outcome.get("resolved"):
            return UNRESOLVED
        realized_r = outcome.get("realized_r")
        if realized_r is None:
            return NONE
        if realized_r > 0:
            return GOOD_TRADE
        if realized_r < 0:
            return BAD_TRADE
        return NONE  # безубыток (realized_r == 0): ни хорошо, ни плохо

    # --- не-ENTER: WAIT / NO_TRADE / blocked (гипотетический исход) ---------
    if direction not in ("long", "short"):
        return NONE
    if not _usable_levels(forecast):
        return NO_LEVELS
    if outcome is None or not outcome.get("resolved"):
        return NONE
    if _hypothetical_loss(outcome):
        return AVOIDED_LOSS
    if _hypothetical_favorable(outcome):
        return MISSED_OPPORTUNITY
    return NONE
