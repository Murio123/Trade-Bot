"""Deep Analysis Renderer v2 (Stage 14) — чистый explanation-слой.

Renderer ОБЪЯСНЯЕТ уже принятое решение ``run_cascade``/``final_gate``, но
никогда его не меняет. На вход подаётся готовый result (полный сигнал ИЛИ
blocked-результат, либо строка forecasts из БД); на выход — структурированное
объяснение по секциям, презентационно-нейтральное (Telegram/AI/тесты читают
одинаково).

Инварианты (Stage 14):
    * decision ENTER/WAIT/NO_TRADE берётся из result через тот же STATUS_MAP,
      что и persist-слой — renderer его НЕ вычисляет заново;
    * bias/scores/confidence/expected_move/stop/TP/RR только читаются;
    * renderer не изобретает числа: отсутствующее поле -> None
      (презентация показывает «н/д»);
    * никакой торговой логики, БД, сети, AI и DRY_RUN-эффектов.

Опциональные обогащения (bollinger / volatility / historical) подаются уже
вычисленными вызывающим кодом и деградируют в None при отсутствии.
"""
from __future__ import annotations

from typing import Any

from signal_engine.schema import STATUS_MAP

_BIAS = {"long": "LONG", "short": "SHORT"}


def render_deep(result: dict[str, Any], *,
                bollinger: dict[str, Any] | None = None,
                volatility: dict[str, Any] | None = None,
                historical: dict[str, Any] | None = None) -> dict[str, Any]:
    """Собрать structured explanation из готового cascade-result.

    Не пересчитывает ни одного решения: все значения — снимок того, что движок
    уже положил в ``result``. Секции соответствуют 22 пунктам Stage 14.
    """
    status = result.get("status", "blocked")
    direction = result.get("candidate_direction") or result.get("direction")

    # 1. Финальное решение — тот же маппинг, что в forecast_record/persist.
    decision = result.get("analysis_status") or STATUS_MAP.get(status, "NO_TRADE")

    # 2. Финальный bias.
    bias = result.get("final_bias") or _BIAS.get(direction, "NEUTRAL")

    # 10. blocked_gate — совпадает с логикой build_forecast_record.
    blocked_gate = result.get("blocked_gate")
    if blocked_gate is None and status == "blocked":
        blocked_gate = result.get("blocked_at") or "unknown"

    tps = result.get("take_profit_levels")
    if not tps:
        tps = [t for t in (result.get("target_1"), result.get("target_2")) if t]

    no_trade_reasons = list(result.get("no_trade_reasons") or [])

    sections = {
        "decision": decision,                                   # 1
        "bias": bias,                                            # 2
        "analysis_type": result.get("analysis_type"),           # 3
        "confidence": _confidence(result),                      # 4
        "scores": {                                             # 5
            "long": result.get("long_score"),
            "short": result.get("short_score"),
        },
        "expected_move": {                                      # 6
            "points": result.get("expected_move_points"),
            "percent": result.get("expected_move_percent"),
            "atr": result.get("expected_move_atr"),
        },
        "stop": {                                               # 7
            "stop_loss": result.get("stop_loss"),
            "invalidation_level": result.get("invalidation_level")
            or result.get("stop_loss"),
        },
        "targets": {                                            # 8
            "levels": tps or None,
            "tp2_source": result.get("tp2_source"),
            "structure": result.get("targets_structure"),
        },
        "risk_reward": result.get("risk_reward")                # 9
        or result.get("expected_risk_reward"),
        "blocked_gate": blocked_gate,                           # 10
        "no_trade_reasons": no_trade_reasons or None,           # 11
        "supporting_factors": list(                             # 12
            result.get("confirming_factors") or result.get("reasons") or []
        ) or None,
        "contradicting_factors": list(                          # 13
            result.get("contradicting_factors") or []
        ) or None,
        "mtf_structure": _mtf(result),                          # 14
        "volatility": _volatility(result, volatility),          # 15
        "bollinger": _passthrough(bollinger),                   # 16
        "regime": {                                             # 17
            "market_regime": result.get("market_regime"),
            "volatility_regime": (result.get("volatility_regime")
                                  or (volatility or {}).get("regime")),
        },
        "historical_quality": _passthrough(historical),         # 18
        "similar_setup": None,                                  # 19 (Stage 17)
        "what_must_change": _what_must_change(                  # 20
            decision, blocked_gate, no_trade_reasons),
        "what_invalidates": list(                               # 21
            result.get("cancel_conditions") or []) or None,
        "what_to_watch": list(                                  # 22
            result.get("confirmation_conditions") or []) or None,
    }
    return sections


def _confidence(result: dict[str, Any]) -> dict[str, Any]:
    """Confidence + факторы, из-за которых он такой (только чтение)."""
    raw = result.get("raw_confidence")
    if raw is None:
        raw = result.get("confidence")
    score = result.get("confidence_score")
    if score is None and raw is not None:
        score = int(round(raw * 100))
    return {
        "score": score,
        "raw": raw,
        "modifier": result.get("confidence_modifier"),
        "conflict_factor": result.get("conflict_factor"),
        "mtf_agreement": result.get("mtf_agreement"),
    }


def _mtf(result: dict[str, Any]) -> dict[str, Any] | None:
    """Мультитаймфрейм-структура из уже посчитанных полей result."""
    align = result.get("timeframe_alignment")
    trends = result.get("mtf")
    if not align and not trends:
        return None
    return {
        "trends": (align or {}).get("trends") if align else trends,
        "alignment": align,
    }


def _volatility(result: dict[str, Any],
                volatility: dict[str, Any] | None) -> dict[str, Any] | None:
    """Волатильностный контекст: подробный dict, если передан, иначе то немногое,
    что уже лежит в самом result."""
    if volatility:
        return _passthrough(volatility)
    regime = result.get("volatility_regime")
    return {"regime": regime} if regime is not None else None


def _what_must_change(decision: str, blocked_gate: Any,
                      no_trade_reasons: list[str]) -> list[str] | None:
    """Что должно измениться для ENTER. Это НЕ новое решение — только пересказ
    уже существующих причин блокировки/ожидания движка."""
    if decision == "ENTER":
        return None
    items: list[str] = []
    if blocked_gate:
        items.append(f"снятие гейта: {blocked_gate}")
    items.extend(no_trade_reasons)
    return items or None


def _passthrough(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Опциональное обогащение как есть; пустое/None -> None (никаких выдумок)."""
    if not value:
        return None
    return dict(value)
