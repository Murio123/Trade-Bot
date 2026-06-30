"""Claude integration: turn a confluence result into signal text, and power
the interactive /ask chat mode with the current market context.

Uses the async Anthropic SDK. Degrades to a deterministic template if no API
key is set, so the bot still produces readable output without Anthropic.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import config

log = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic  # type: ignore
    _HAS_ANTHROPIC = True
except Exception:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore
    _HAS_ANTHROPIC = False


_client = None


def _get_client():
    global _client
    if _client is None and _HAS_ANTHROPIC and config.ANTHROPIC_API_KEY:
        _client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


SIGNAL_SYSTEM = (
    "Ты — профессиональный крипто-аналитик BTC. На основе предоставленного "
    "confluence-анализа дай краткую оценку уверенности в сигнале (0-100%) и "
    "1-2 предложения обоснования на русском языке. Будь конкретным, без "
    "финансовых обещаний. Формат ответа строго JSON: "
    '{"confidence": <int 0-100>, "comment": "<текст>"}'
)

ASK_SYSTEM = (
    "Ты — ассистент крипто-трейдера, встроенный в Telegram-бота для анализа "
    "BTC/USDT фьючерсов. Отвечай кратко, по делу, на русском. Используй "
    "предоставленный рыночный контекст. Не давай финансовых гарантий, "
    "напоминай о рисках, когда уместно."
)


async def interpret_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Return {'confidence': float 0-1, 'comment': str}."""
    client = _get_client()
    payload = {
        "direction": signal.get("direction"),
        "timeframe": signal.get("timeframe"),
        "score": signal.get("score"),
        "category_scores": signal.get("category_scores"),
        "reasons": signal.get("reasons"),
        "htf_bias": signal.get("htf_bias"),
        "mtf_confidence": signal.get("confidence"),
    }
    if client is None:
        # Deterministic fallback: derive confidence from score + mtf modifier.
        base = (signal.get("score", 0) / 10.0)
        conf = max(0.0, min(1.0, base * signal.get("confidence_modifier", 1.0)))
        return {
            "confidence": round(conf, 2),
            "comment": "; ".join(signal.get("reasons", [])[:3]) or "Confluence подтверждён.",
        }
    try:
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=400,
            system=SIGNAL_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        text = _extract_text(resp)
        data = json.loads(text)
        conf = float(data.get("confidence", 50)) / 100.0
        return {"confidence": round(conf, 2), "comment": data.get("comment", "")}
    except Exception as exc:  # noqa: BLE001
        log.warning("Claude interpret_signal failed: %s", exc)
        base = signal.get("score", 0) / 10.0
        return {"confidence": round(base, 2), "comment": "; ".join(signal.get("reasons", [])[:3])}


async def ask(question: str, market_context: dict[str, Any]) -> str:
    client = _get_client()
    if client is None:
        return (
            "AI-режим недоступен (не задан ANTHROPIC_API_KEY). "
            "Текущая цена: "
            f"{market_context.get('price', 'н/д')}."
        )
    context_str = json.dumps(market_context, ensure_ascii=False, default=str)
    try:
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=800,
            system=ASK_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Рыночный контекст:\n{context_str}\n\nВопрос: {question}",
            }],
        )
        return _extract_text(resp)
    except Exception as exc:  # noqa: BLE001
        log.warning("Claude ask failed: %s", exc)
        return f"Ошибка обращения к AI: {exc}"


def _extract_text(resp: Any) -> str:
    parts = []
    for block in getattr(resp, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()
