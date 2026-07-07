from __future__ import annotations

import asyncio

import config
from bot import alerts, formatting


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
        self.messages.append({"chat_id": chat_id, "text": text})


def test_send_dry_run_alerts_default_is_false():
    assert config.SEND_DRY_RUN_ALERTS is False


def test_dry_run_without_send_flag_keeps_alerts_blocked(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", True)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", False)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    bot = FakeBot()

    asyncio.run(alerts.send_signal_alert(bot, "signal body"))

    assert bot.messages == []


def test_dry_run_with_send_flag_sends_manual_signal_alert(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", True)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", True)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    bot = FakeBot()

    asyncio.run(alerts.send_signal_alert(bot, "signal body"))

    assert len(bot.messages) == 1
    text = bot.messages[0]["text"]
    assert "🧪 DRY-RUN / MANUAL ONLY" in text
    assert "Сделки не открываются автоматически." in text
    assert text.endswith("signal body")


def test_live_alerts_do_not_get_dry_run_manual_prefix(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", False)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", False)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    bot = FakeBot()

    asyncio.run(alerts.send_signal_alert(bot, "signal body"))

    assert bot.messages == [{"chat_id": "1", "text": "signal body"}]


def test_status_text_for_dry_run_alert_modes():
    base = {
        "dry_run": True,
        "db_connected": True,
        "exchange_pref": "binance",
        "ai_ok": None,
    }

    blocked = formatting.format_status(
        {**base, "send_dry_run_alerts": False}
    )
    allowed = formatting.format_status(
        {**base, "send_dry_run_alerts": True}
    )

    assert "DRY-RUN 🧪 (уведомления НЕ шлются)" in blocked
    assert "DRY-RUN 🧪 (уведомления шлются, manual only)" in allowed
