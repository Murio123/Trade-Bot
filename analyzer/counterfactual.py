"""Stage 13B: pure offline counterfactual outcome для non-ENTER прогнозов.

Отвечает на вопрос «а что было бы, если бы бот вошёл?» для решений, где входа
НЕ было — WAIT / NO_TRADE / blocked / journal-only. Полностью offline, read-only:

  * НЕ пишет в БД, НЕ ходит в сеть/Binance, НЕ трогает Railway/env;
  * НЕ меняет торговое поведение, scoring, thresholds, gating, confidence,
    realized_r или Telegram; НЕ импортируется runtime decision-кодом;
  * НЕ дублирует логику TP/stop/MFE/MAE — переиспользует уже проверенный
    producer analyzer.outcomes.measure_outcome (его НЕ меняя). Та же дефиниция
    хитов, тот же 72h-горизонт, тот же censoring по `now` → ноль расхождений и
    никакого look-ahead (читаются только свечи ПОСЛЕ decision_time и до `now`).

Честность пропусков — жёсткое правило: отсутствие направления/уровней/окна НЕ
превращается в win/loss. Такие строки помечаются no_direction / no_levels /
unresolved, а не evaluated.

realized_r для non-ENTER остаётся None (measure_outcome → analyzer.realized_r
отдаёт None вне ENTER) — формула R не трогается и не дублируется. Метка
avoided_loss / missed_opportunity выводится из хитов через тот же критерий, что
и Trading Journal v2 (analyzer.journal_classify), чтобы результат стыковался с
журналом.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

import config
from analyzer.journal_classify import AVOIDED_LOSS, MISSED_OPPORTUNITY, NONE, classify
from analyzer.outcomes import measure_outcome
from validation.trade_costs import FUNDING_NOT_MODELLED

# Модельная round-trip стоимость. На counterfactual-исход НЕ влияет —
# measure_outcome использует её только для net_after_costs, который здесь не
# выводится; параметры оставлены для parity. P1: берутся из config, а не
# копируются числами — две частные копии констант и были одним из девяти мест,
# где стоимость жила своей жизнью.
_TAKER_FEE_PCT = config.TAKER_FEE_PCT
_SLIPPAGE_PCT = config.SLIPPAGE_PCT

ENTER = "ENTER"

# Статусы результата (стабильные строковые константы для CLI/тестов).
EVALUATED = "evaluated"
NO_LEVELS = "no_levels"
NO_DIRECTION = "no_direction"
UNRESOLVED = "unresolved"
UNSUPPORTED = "unsupported"

# classification_hint: avoided_loss / missed_opportunity / unresolved /
# no_levels / none (стыкуется с Trading Journal v2).
HINT_UNRESOLVED = "unresolved"
HINT_NO_LEVELS = "no_levels"
HINT_NONE = NONE


def _usable_levels(forecast: dict[str, Any]) -> bool:
    """Достаточно ли уровней для гипотетического исхода: стоп + хотя бы TP1."""
    stop = forecast.get("stop_loss")
    tps = forecast.get("take_profit_levels") or []
    tp1 = tps[0] if len(tps) > 0 else None
    return stop is not None and tp1 is not None


def _reference_price(forecast: dict[str, Any]) -> float | None:
    ref = (forecast.get("executable_price_at_decision")
           or forecast.get("signal_close_price"))
    return float(ref) if ref else None


def _result(status: str, hint: str, reason: str,
            *, outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    """Единая форма результата. Всегда hypothetical=True; поля исхода = None,
    пока (и если) не измерены — пропуск не выдаётся за успех/неудачу."""
    o = outcome or {}
    return {
        "status": status,
        "hypothetical": True,
        "tp1_hit": o.get("tp1_hit"),
        "tp2_hit": o.get("tp2_hit"),
        "stop_hit": o.get("stop_hit"),
        "mfe_points": o.get("mfe_points"),
        "mae_points": o.get("mae_points"),
        "realized_r": o.get("realized_r"),  # None для non-ENTER by design
        "classification_hint": hint,
        "reason": reason,
    }


def counterfactual_outcome(forecast: dict[str, Any], df: pd.DataFrame,
                           now: datetime) -> dict[str, Any]:
    """Гипотетический исход для одного non-ENTER прогноза (чистая функция).

    Вход НЕ мутируется (ни forecast, ни df). Всегда возвращает dict со `status`:

        no_direction — нет candidate_direction long/short;
        unsupported  — ENTER (у него есть фактический исход, не гипотетический),
                       либо нет reference price / decision_time для якоря;
        no_levels    — нет стопа и/или TP1;
        unresolved   — окно 72h не закрыто (censored) — НЕ win/loss;
        evaluated    — исход посчитан; classification_hint = avoided_loss /
                       missed_opportunity / none.
    """
    status = forecast.get("analysis_status")
    direction = forecast.get("candidate_direction")

    if status == ENTER:
        return _result(UNSUPPORTED, HINT_NONE,
                       "ENTER несёт фактический исход; counterfactual — для non-ENTER")
    if direction not in ("long", "short"):
        return _result(NO_DIRECTION, HINT_NONE, "нет candidate_direction long/short")
    if not _usable_levels(forecast):
        return _result(NO_LEVELS, HINT_NO_LEVELS, "нет стопа и/или TP1 — исход неизмерим")
    if _reference_price(forecast) is None or forecast.get("decision_time") is None:
        return _result(UNSUPPORTED, HINT_NONE, "нет reference price / decision_time для якоря")

    # measure_outcome требует forecast['id'] — не мутируем исходный dict.
    fc = forecast if "id" in forecast else {**forecast, "id": 0}
    # FUNDING_NOT_MODELLED, explicitly: this path consumes the TOUCH outcome
    # (stop/TP/censoring), never net_after_costs, so there is no funding bill to
    # charge — and the sentinel records that rather than implying zero.
    outcome = measure_outcome(fc, df, now, _TAKER_FEE_PCT, _SLIPPAGE_PCT,
                              FUNDING_NOT_MODELLED)
    if outcome is None:
        # Нет свечей после decision_time (окно пустое) — censored, не провал.
        return _result(UNRESOLVED, HINT_UNRESOLVED,
                       "нет свечей после decision_time — окно censored")
    if not outcome.get("resolved"):
        return _result(UNRESOLVED, HINT_UNRESOLVED,
                       "горизонт 72h не закрыт — censored", outcome=outcome)

    # Resolved: та же трактовка avoided/missed, что и в Trading Journal v2.
    hint = classify(forecast, outcome)  # non-ENTER -> avoided_loss/missed_opportunity/none
    reason = {
        AVOIDED_LOSS: "гипотетически сработал бы чистый стоп — убыток избежан",
        MISSED_OPPORTUNITY: "гипотетически отработал бы TP — движение пропущено",
    }.get(hint, "resolved, но ни чистого стопа, ни чистого профита")
    return _result(EVALUATED, hint, reason, outcome=outcome)
