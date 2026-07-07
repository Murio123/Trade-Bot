from __future__ import annotations

import asyncio

import config
import scheduler
from bot import alerts, formatting
from bot import handlers


class FakeBot:
    def __init__(self, fail: bool = False) -> None:
        self.messages: list[dict[str, str]] = []
        self.fail = fail

    async def send_message(self, chat_id: str, text: str) -> None:
        if self.fail:
            raise RuntimeError("telegram down")
        self.messages.append({"chat_id": chat_id, "text": text})


class FakeMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeUpdate:
    def __init__(self) -> None:
        self.effective_message = FakeMessage()


class FakeContext:
    def __init__(self, bot: FakeBot) -> None:
        self.application = type("App", (), {"bot": bot})()


class FakeDb:
    def __init__(self) -> None:
        self.inserted: list[dict] = []
        self.marked: list[int] = []

    async def signals_today(self, *args, **kwargs) -> list[dict]:
        return []

    async def last_delivered_signal(self, *args, **kwargs):
        return None

    async def open_trades(self) -> list[dict]:
        return []

    async def insert_signal(self, record: dict) -> int:
        self.inserted.append(record)
        return 42

    async def mark_delivered(self, signal_id: int) -> None:
        self.marked.append(signal_id)


async def _run_scheduler_alert(monkeypatch, send_result: bool) -> FakeDb:
    fake_db = FakeDb()
    trades: list[int] = []
    app = type("App", (), {
        "bot": FakeBot(),
        "bot_data": {"binance": object()},
    })()
    result = {"status": "alert", "score": 8, "symbol": "BTCUSDT"}

    monkeypatch.setattr(scheduler.config, "ENABLE_FORECAST_LEDGER", False)
    monkeypatch.setattr(scheduler, "db", fake_db)
    monkeypatch.setattr(scheduler, "_maybe_reversal_alert",
                        lambda *args, **kwargs: _noop())
    monkeypatch.setattr(scheduler, "gather_market_context",
                        lambda *args, **kwargs: _ctx())
    monkeypatch.setattr(scheduler, "run_cascade",
                        lambda *args, **kwargs: _result(result))
    monkeypatch.setattr(scheduler.formatting, "format_signal", lambda r: "signal body")
    monkeypatch.setattr(scheduler.formatting, "format_risk_cap_note", lambda: "risk cap")
    monkeypatch.setattr(scheduler.journal, "can_open_new_trade",
                        lambda symbol: _result(True))
    monkeypatch.setattr(scheduler.journal, "record_signal_as_trade",
                        lambda signal_id, signal: _record_trade(trades, signal_id))
    monkeypatch.setattr(scheduler.alerts, "send_signal_alert",
                        lambda bot, text: _result(send_result))
    monkeypatch.setattr(scheduler, "_send_signal_chart",
                        lambda *args, **kwargs: _noop())

    await scheduler.analysis_job(app, "swing")
    fake_db.trades = trades
    return fake_db


async def _noop():
    return None


async def _ctx():
    return {}


async def _result(value):
    return value


async def _record_trade(trades: list[int], signal_id: int):
    trades.append(signal_id)
    return signal_id


def test_send_dry_run_alerts_default_is_false():
    assert config.SEND_DRY_RUN_ALERTS is False


def test_dry_run_without_send_flag_keeps_alerts_blocked(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", True)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", False)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    bot = FakeBot()

    sent = asyncio.run(alerts.send_signal_alert(bot, "signal body"))

    assert sent is False
    assert bot.messages == []


def test_dry_run_with_send_flag_sends_manual_signal_alert(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", True)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", True)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    bot = FakeBot()

    sent = asyncio.run(alerts.send_signal_alert(bot, "signal body"))

    assert sent is True
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

    sent = asyncio.run(alerts.send_signal_alert(bot, "signal body"))

    assert sent is True
    assert bot.messages == [{"chat_id": "1", "text": "signal body"}]


def test_send_dry_run_alerts_does_not_unlock_generic_broadcast(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", True)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", True)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    bot = FakeBot()

    sent = asyncio.run(alerts.broadcast(bot, "price alert"))

    assert sent is False
    assert bot.messages == []


def test_signal_alert_telegram_failure_returns_false(monkeypatch):
    monkeypatch.setattr(alerts.config, "DRY_RUN", False)
    monkeypatch.setattr(alerts.config, "SEND_DRY_RUN_ALERTS", False)
    monkeypatch.setattr(alerts.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])

    sent = asyncio.run(alerts.send_signal_alert(FakeBot(fail=True), "signal body"))

    assert sent is False


def test_scheduler_does_not_mark_or_open_trade_when_signal_send_blocked(monkeypatch):
    fake_db = asyncio.run(_run_scheduler_alert(monkeypatch, send_result=False))

    assert fake_db.inserted[0]["delivered"] is False
    assert fake_db.marked == []
    assert fake_db.trades == []


def test_scheduler_marks_and_records_trade_after_signal_send_success(monkeypatch):
    fake_db = asyncio.run(_run_scheduler_alert(monkeypatch, send_result=True))

    assert fake_db.inserted[0]["delivered"] is False
    assert fake_db.marked == [42]
    assert fake_db.trades == [42]


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


def test_testalert_reports_dry_run_alerts_disabled(monkeypatch):
    monkeypatch.setattr(handlers.config, "DRY_RUN", True)
    monkeypatch.setattr(handlers.config, "SEND_DRY_RUN_ALERTS", False)
    monkeypatch.setattr(handlers.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])
    update = FakeUpdate()

    asyncio.run(handlers.testalert_cmd(update, FakeContext(FakeBot())))

    assert "SEND_DRY_RUN_ALERTS=false" in update.effective_message.replies[0]


def test_testalert_sends_dry_run_manual_message_when_enabled(monkeypatch):
    monkeypatch.setattr(handlers.config, "DRY_RUN", True)
    monkeypatch.setattr(handlers.config, "SEND_DRY_RUN_ALERTS", True)
    monkeypatch.setattr(handlers.config, "TELEGRAM_ALERT_CHAT_IDS", ["1"])
    bot = FakeBot()
    update = FakeUpdate()

    asyncio.run(handlers.testalert_cmd(update, FakeContext(bot)))

    assert len(bot.messages) == 1
    assert "🧪 DRY-RUN / MANUAL ONLY" in bot.messages[0]["text"]
    assert "DRY-RUN / MANUAL ONLY" in update.effective_message.replies[0]


def test_no_exchange_order_execution_tokens_in_runtime_changes():
    forbidden = (
        "create_order", "market_order", "limit_order", "api_key",
        "api_secret", "exchange.create",
    )
    files = [
        "bot/alerts.py",
        "bot/handlers.py",
        "scheduler.py",
        "config.py",
    ]
    for path in files:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        for token in forbidden:
            assert token not in src
