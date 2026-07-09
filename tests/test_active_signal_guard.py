"""Duplicate-alert guard: while a journal trade for the same symbol+profile
is open, forecasts/signals are still persisted but no Telegram alert is sent,
mark_delivered is not called and no second journal trade is opened."""
from __future__ import annotations

import asyncio

import scheduler
from bot import handlers, journal


class FakeBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
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
        self.forecasts: list[dict] = []
        self.links: list[tuple[int, int]] = []

    async def insert_forecast(self, fc: dict) -> int:
        self.forecasts.append(fc)
        return 9

    async def link_forecast_signal(self, forecast_id: int, signal_id: int) -> None:
        self.links.append((forecast_id, signal_id))

    async def signals_today(self, *args, **kwargs) -> list[dict]:
        return []

    async def last_signal(self, *args, **kwargs):
        return None

    async def last_delivered_signal(self, *args, **kwargs):
        return None

    async def open_trades(self) -> list[dict]:
        return []

    async def insert_signal(self, record: dict) -> int:
        self.inserted.append(record)
        return 42

    async def mark_delivered(self, signal_id: int) -> None:
        self.marked.append(signal_id)


async def _noop(*args, **kwargs):
    return None


async def _result(value):
    return value


# --- journal.has_active_trade matching -------------------------------------

class _TradesDb:
    def __init__(self, trades: list[dict], fail: bool = False) -> None:
        self.trades = trades
        self.fail = fail
        self.inserted_trades: list[dict] = []

    async def open_trades(self) -> list[dict]:
        if self.fail:
            raise RuntimeError("db down")
        return self.trades

    async def insert_trade(self, trade: dict) -> int:
        self.inserted_trades.append(trade)
        return 7


def _open_trade(**overrides) -> dict:
    return {"symbol": "BTCUSDT", "analysis_type": "INTRADAY",
            "timeframe": "15m", **overrides}


def test_has_active_trade_matches_same_profile(monkeypatch):
    monkeypatch.setattr(journal, "db", _TradesDb([_open_trade()]))
    assert asyncio.run(journal.has_active_trade("BTCUSDT", "INTRADAY", "15m")) is True


def test_has_active_trade_ignores_other_profile(monkeypatch):
    monkeypatch.setattr(journal, "db", _TradesDb([_open_trade()]))
    assert asyncio.run(journal.has_active_trade("BTCUSDT", "SWING", "4h")) is False


def test_has_active_trade_ignores_other_symbol(monkeypatch):
    monkeypatch.setattr(journal, "db", _TradesDb([_open_trade(symbol="ETHUSDT")]))
    assert asyncio.run(journal.has_active_trade("BTCUSDT", "INTRADAY", "15m")) is False


def test_has_active_trade_legacy_rows_match_by_timeframe(monkeypatch):
    monkeypatch.setattr(
        journal, "db", _TradesDb([_open_trade(analysis_type=None)]))
    assert asyncio.run(journal.has_active_trade("BTCUSDT", "INTRADAY", "15m")) is True
    assert asyncio.run(journal.has_active_trade("BTCUSDT", "SWING", "4h")) is False


def test_has_active_trade_fails_open_on_db_error(monkeypatch):
    monkeypatch.setattr(journal, "db", _TradesDb([], fail=True))
    assert asyncio.run(journal.has_active_trade("BTCUSDT", "INTRADAY", "15m")) is False


# --- record_signal_as_trade double-check ------------------------------------

_SIGNAL = {"symbol": "BTCUSDT", "analysis_type": "INTRADAY", "timeframe": "15m",
           "direction": "short", "entry_price": 100.0, "stop_loss": 101.0,
           "target_1": 98.0, "target_2": 96.0}


def test_record_signal_as_trade_skips_when_active_trade_appears(monkeypatch):
    # An active trade shows up between the caller's guard and the insert.
    fake = _TradesDb([_open_trade()])
    monkeypatch.setattr(journal, "db", fake)

    assert asyncio.run(journal.record_signal_as_trade(42, _SIGNAL)) is None
    assert fake.inserted_trades == []


def test_record_signal_as_trade_inserts_without_active_trade(monkeypatch):
    fake = _TradesDb([])
    monkeypatch.setattr(journal, "db", fake)

    assert asyncio.run(journal.record_signal_as_trade(42, _SIGNAL)) == 7
    assert len(fake.inserted_trades) == 1
    assert fake.inserted_trades[0]["signal_id"] == 42


# --- scheduled stream -------------------------------------------------------

async def _run_scheduler_alert(monkeypatch, active_trade: bool,
                               forecast_ledger: bool = False):
    fake_db = FakeDb()
    trades: list[int] = []
    sends: list[str] = []
    app = type("App", (), {
        "bot": FakeBot(),
        "bot_data": {"binance": object()},
    })()
    result = {"status": "alert", "score": 8, "symbol": "BTCUSDT",
              "direction": "short", "timeframe": "15m",
              "analysis_type": "INTRADAY"}

    async def _send(bot, text):
        sends.append(text)
        return True

    async def _record_trade(signal_id, signal):
        trades.append(signal_id)
        return signal_id

    monkeypatch.setattr(scheduler.config, "ENABLE_FORECAST_LEDGER", forecast_ledger)
    if forecast_ledger:
        import signal_engine.forecast_record as forecast_record
        monkeypatch.setattr(forecast_record, "build_forecast_record",
                            lambda *args: {"symbol": "BTCUSDT"})
        monkeypatch.setattr(scheduler, "enrich_forecast_with_lifecycle",
                            lambda *args, **kwargs: _result({}))
    monkeypatch.setattr(scheduler, "db", fake_db)
    monkeypatch.setattr(scheduler, "_maybe_reversal_alert", _noop)
    monkeypatch.setattr(scheduler, "gather_market_context",
                        lambda *args, **kwargs: _result({}))
    monkeypatch.setattr(scheduler, "run_cascade",
                        lambda *args, **kwargs: _result(result))
    monkeypatch.setattr(scheduler.formatting, "format_signal", lambda r: "signal body")
    monkeypatch.setattr(scheduler.journal, "has_active_trade",
                        lambda *args, **kwargs: _result(active_trade))
    monkeypatch.setattr(scheduler.journal, "can_open_new_trade",
                        lambda symbol: _result(True))
    monkeypatch.setattr(scheduler.journal, "record_signal_as_trade", _record_trade)
    monkeypatch.setattr(scheduler.alerts, "send_signal_alert", _send)
    monkeypatch.setattr(scheduler, "_send_signal_chart", _noop)

    await scheduler.analysis_job(app, "intraday")
    return fake_db, trades, sends


def test_scheduler_suppresses_alert_while_trade_open(monkeypatch):
    fake_db, trades, sends = asyncio.run(
        _run_scheduler_alert(monkeypatch, active_trade=True))

    # The signal row is still persisted for history, but nothing is delivered.
    assert len(fake_db.inserted) == 1
    assert fake_db.inserted[0]["delivered"] is False
    assert sends == []
    assert fake_db.marked == []
    assert trades == []


def test_scheduler_suppression_keeps_forecast_ledger(monkeypatch):
    fake_db, trades, sends = asyncio.run(
        _run_scheduler_alert(monkeypatch, active_trade=True, forecast_ledger=True))

    # Forecast analytics row persists even though delivery is suppressed.
    assert len(fake_db.forecasts) == 1
    assert fake_db.links == [(9, 42)]
    assert fake_db.inserted[0]["delivered"] is False
    assert sends == []
    assert fake_db.marked == []
    assert trades == []


def test_scheduler_delivers_normally_without_open_trade(monkeypatch):
    fake_db, trades, sends = asyncio.run(
        _run_scheduler_alert(monkeypatch, active_trade=False))

    assert sends == ["signal body"]
    assert fake_db.marked == [42]
    assert trades == [42]


# --- manual /signal ----------------------------------------------------------

async def _run_manual_signal(monkeypatch, active_trade: bool):
    fake_db = FakeDb()
    trades: list[int] = []
    result = {"status": "alert", "score": 8, "symbol": "BTCUSDT",
              "direction": "short", "timeframe": "15m",
              "analysis_type": "INTRADAY"}

    async def _record_trade(signal_id, signal):
        trades.append(signal_id)
        return signal_id

    monkeypatch.setattr(handlers, "db", fake_db)
    monkeypatch.setattr(handlers, "_fresh_context",
                        lambda *args, **kwargs: _result({}))
    monkeypatch.setattr(handlers, "run_cascade",
                        lambda *args, **kwargs: _result(result))
    monkeypatch.setattr(handlers.formatting, "format_signal", lambda r: "signal body")
    monkeypatch.setattr(handlers.journal, "has_active_trade",
                        lambda *args, **kwargs: _result(active_trade))
    monkeypatch.setattr(handlers.journal, "can_open_new_trade",
                        lambda symbol: _result(True))
    monkeypatch.setattr(handlers.journal, "record_signal_as_trade", _record_trade)
    monkeypatch.setattr(handlers, "_send_chart", _noop)

    update = FakeUpdate()
    await handlers._run_signal(update, FakeContext(FakeBot()), "intraday")
    return update, fake_db, trades


def test_manual_signal_duplicate_is_informational_only(monkeypatch):
    update, fake_db, trades = asyncio.run(
        _run_manual_signal(monkeypatch, active_trade=True))

    # The user still gets a reply, but marked as informational — not a normal
    # actionable alert.
    reply = update.effective_message.replies[-1]
    assert "⚠️ Активный сигнал уже открыт. Новый сигнал не отправлен " \
           "и не добавлен в журнал." in reply
    assert "Только обновление анализа, не новая торговая идея." in reply
    assert not reply.startswith("signal body")
    # The history row is kept UNDELIVERED (must not feed cooldown/daily limit)
    # and no second journal trade appears.
    assert len(fake_db.inserted) == 1
    assert fake_db.inserted[0]["delivered"] is False
    assert fake_db.marked == []
    assert trades == []


def test_manual_signal_records_trade_without_open_trade(monkeypatch):
    update, fake_db, trades = asyncio.run(
        _run_manual_signal(monkeypatch, active_trade=False))

    reply = update.effective_message.replies[-1]
    assert reply == "signal body"
    assert fake_db.inserted[0]["delivered"] is True
    assert trades == [42]
