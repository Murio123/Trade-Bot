"""Advanced swing analysis narrative via Claude.

The deterministic engine (signal_engine.quality_score) computes the /100 score
and the final decision from real data; Claude writes the institutional-style
11-section narrative around those numbers. The LLM is explicitly told NOT to
override the decision gate and NOT to invent data that wasn't provided.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import config
from ai import claude as claude_mod

log = logging.getLogger(__name__)

SYSTEM = (
    "Ты — институциональный аналитик BTC-фьючерсов (свинг, плечо 5x, ТФ "
    "1D/12H/4H). Тебе дают РЕАЛЬНЫЕ посчитанные данные и детерминированный "
    "Trade Quality Score (/100) с финальным решением. Твоя задача — написать "
    "разбор по 11 разделам (режим рынка, исторические аналоги, Smart Money, "
    "деривативы, корреляции, волатильность, истощение тренда, 3 сценария, "
    "риски, итоговый score, финальное решение).\n\n"
    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "1) НЕ выдумывай данные. Если поле помечено null/N/A — пиши «нет данных».\n"
    "2) НЕ меняй финальное решение и score — они уже посчитаны детерминированно. "
    "Если решение NO TRADE — объясни, что должно измениться для входа.\n"
    "3) Краткость и конкретика, по-русски. Никаких финансовых гарантий.\n"
    "4) Сценарии (база/альтернатива/инвалидация) с вероятностями в сумме 100%."
)


def _struct(s: dict[str, Any] | None) -> dict[str, Any] | None:
    """Clarify that a missing structure event means 'range', not missing data."""
    if not s:
        return s
    out = dict(s)
    if out.get("last_event") is None:
        out["last_event"] = "нет слома структуры (диапазон)"
    return out


def _payload(ctx: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "price": ctx.get("price"),
        "decision": quality.get("decision"),
        "overall_score": quality.get("overall"),
        "direction": quality.get("direction"),
        "htf_bias": quality.get("htf_bias"),
        "section_scores": quality.get("scores"),
        "section_notes": quality.get("notes"),
        "plan": quality.get("plan"),
        "structure": {
            "1d": _struct(ctx.get("structure_1d")),
            "12h": _struct(ctx.get("structure_12h")),
            "4h": _struct(ctx.get("structure_4h")),
        },
        "derivatives": {
            "funding": (ctx.get("funding") or {}).get("current"),
            "oi_rising": ctx.get("oi_rising"),
            "long_short_ratio": (ctx.get("long_short_ratio") or {}).get("ratio"),
            "cvd_bullish": (ctx.get("cvd") or {}).get("cvd_bullish"),
        },
        "volatility": ctx.get("volatility"),
        "historical": ctx.get("historical"),
        "correlation": ctx.get("correlation"),
        "reversal": {
            "bullish": (ctx.get("reversal") or {}).get("bullish_reversal"),
            "bearish": (ctx.get("reversal") or {}).get("bearish_reversal"),
            "factors_bull": (ctx.get("reversal") or {}).get("factors_bull"),
            "factors_bear": (ctx.get("reversal") or {}).get("factors_bear"),
        },
    }


async def generate_report(ctx: dict[str, Any], quality: dict[str, Any]) -> str:
    client = claude_mod._get_client()
    payload = _payload(ctx, quality)
    if client is None:
        return _fallback(payload)
    try:
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1600,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
        )
        return claude_mod._extract_text(resp)
    except Exception as exc:  # noqa: BLE001
        log.warning("swing report failed: %s", exc)
        return _fallback(payload)


def _fallback(payload: dict[str, Any]) -> str:
    """Deterministic text when Claude is unavailable."""
    hist = payload.get("historical") or {}
    corr = payload.get("correlation") or {}
    vol = payload.get("volatility") or {}
    lines = [
        "🧠 Глубокий анализ (без ИИ-нарратива — нет ANTHROPIC_API_KEY)",
        f"Исторические аналоги: {hist.get('matches', 0)} шт, "
        f"бычьих исходов {hist.get('bullish_pct')}%, conf {hist.get('confidence')}",
        f"Волатильность: режим {vol.get('regime')}, ATR перцентиль {vol.get('atr_percentile')}",
        f"Корреляции: {corr.get('verdict')} ({corr.get('supportive')}/{corr.get('counted')})",
    ]
    return "\n".join(lines)
