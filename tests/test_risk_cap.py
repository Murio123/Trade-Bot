"""H3 fix: aggregate open-positions cap across profiles (MAX_OPEN_TRADES)."""
from __future__ import annotations

import asyncio

import config
from bot import journal
from database import Database


def _trade(symbol="BTCUSDT", direction="long"):
    return {"signal_id": None, "direction": direction, "entry_price": 60000.0,
            "stop_loss": 59000.0, "target": 61500.0, "tp1": 61500.0,
            "tp2": 63000.0, "timeframe": "1h", "symbol": symbol}


def test_cap_blocks_after_max_open_trades(monkeypatch):
    db = Database(dsn=None)
    monkeypatch.setattr("bot.journal.db", db)
    monkeypatch.setattr(config, "MAX_OPEN_TRADES", 3)

    for _ in range(2):
        asyncio.run(db.insert_trade(_trade()))
    assert asyncio.run(journal.can_open_new_trade("BTCUSDT"))

    asyncio.run(db.insert_trade(_trade()))
    assert not asyncio.run(journal.can_open_new_trade("BTCUSDT"))


def test_cap_frees_up_when_trade_closes(monkeypatch):
    db = Database(dsn=None)
    monkeypatch.setattr("bot.journal.db", db)
    monkeypatch.setattr(config, "MAX_OPEN_TRADES", 1)

    tid = asyncio.run(db.insert_trade(_trade()))
    assert not asyncio.run(journal.can_open_new_trade("BTCUSDT"))

    asyncio.run(db.close_trade(tid, 61500.0, "win", 1.5))
    assert asyncio.run(journal.can_open_new_trade("BTCUSDT"))


def test_legacy_rows_without_symbol_count_against_cap(monkeypatch):
    db = Database(dsn=None)
    monkeypatch.setattr("bot.journal.db", db)
    monkeypatch.setattr(config, "MAX_OPEN_TRADES", 1)

    asyncio.run(db.insert_trade(_trade(symbol=None)))
    assert not asyncio.run(journal.can_open_new_trade("BTCUSDT"))
