from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from bot import formatting, handlers


def _signal(style: str, entry: float, display: float) -> dict:
    return {
        "symbol": "BTCUSDT",
        "timeframe": "15m" if style == "intraday" else "4h",
        "timestamp": "2026-07-07 12:00 UTC",
        "direction": "long",
        "style_label": "Интрадей" if style == "intraday" else "Свинг",
        "style_emoji": "⚡" if style == "intraday" else "📊",
        "display_price": display,
        "entry_price": entry,
        "stop_loss": entry - 500,
        "target_1": entry + 800,
        "target_2": entry + 1600,
        "position_size": 0.01,
        "score": 8,
        "htf_bias": "bullish",
        "htf_tf": "1h",
    }


def test_combined_signal_sections_share_display_price_and_label_candle_prices():
    display_price = 63_094.0
    text = "\n\n".join([
        formatting.format_signal(_signal("intraday", 63_094.0, display_price)),
        formatting.format_signal(_signal("swing", 63_378.0, display_price)),
    ])

    assert text.count("💰 Текущая цена: 63 094") == 2
    assert "🕯 Цена свечи / вход: 63 094" in text
    assert "🕯 Цена свечи / вход: 63 378" in text
    assert "\nЦена: 63 094" not in text
    assert "\nЦена: 63 378" not in text


def test_signal_command_captures_one_display_price_snapshot(monkeypatch):
    calls = {"current_price": 0, "run_signal": []}

    class FakeBinance:
        async def current_price(self):
            calls["current_price"] += 1
            return 63_094.0

    async def fake_run_signal(update, context, profile_name, display_price=None):
        calls["run_signal"].append((profile_name, display_price))

    context = MagicMock()
    context.application.bot_data = {"binance": FakeBinance()}
    monkeypatch.setattr(handlers, "_run_signal", fake_run_signal)

    asyncio.run(handlers.signal_cmd(MagicMock(), context))

    assert calls["current_price"] == 1
    assert calls["run_signal"] == [("swing", 63_094.0)]


def test_format_blocked_uses_display_price_and_labels_context_price():
    text = formatting.format_blocked(
        {
            "blocked_at": "no_trade",
            "price": 63_378.0,
            "htf_bias": "neutral",
            "htf_tf": "4h",
        },
        display_price=63_094.0,
    )

    assert "💰 Текущая цена: 63 094" in text
    assert "🕯 Цена свечи / контекста: 63 378" in text
    assert "\nЦена: 63 378" not in text


def test_combined_blocked_sections_do_not_show_conflicting_generic_prices():
    display_price = 63_094.0
    text = "\n\n".join([
        formatting.format_blocked(
            {"blocked_at": "no_trade", "price": 63_094.0, "htf_tf": "1h"},
            display_price=display_price,
        ),
        formatting.format_blocked(
            {"blocked_at": "no_trade", "price": 63_378.0, "htf_tf": "4h"},
            display_price=display_price,
        ),
    ])

    assert text.count("💰 Текущая цена: 63 094") == 2
    assert "🕯 Цена свечи / контекста: 63 094" in text
    assert "🕯 Цена свечи / контекста: 63 378" in text
    assert "\nЦена: 63 094" not in text
    assert "\nЦена: 63 378" not in text
