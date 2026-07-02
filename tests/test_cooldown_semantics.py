"""C1 fix: the cooldown must count from the last DELIVERED alert.

Journal-only records (score 5-7, never sent) and repeated evaluations of a
persisting setup must not re-arm the cooldown window.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from database import Database
from signal_engine.cooldown import should_send_signal


def _signal(direction="long", price=60000.0, timeframe="1h", delivered=False,
            created_at=None, score=8):
    return {
        "symbol": "BTCUSDT", "timeframe": timeframe, "direction": direction,
        "entry_price": price, "price": price, "score": score,
        "delivered": delivered,
        "created_at": created_at or datetime.now(timezone.utc),
    }


def test_last_delivered_signal_ignores_journal_records():
    db = Database(dsn=None)
    old_alert = _signal(delivered=True,
                        created_at=datetime.now(timezone.utc) - timedelta(hours=10))
    fresh_journal = _signal(delivered=False, score=6)

    asyncio.run(db.insert_signal(old_alert))
    asyncio.run(db.insert_signal(fresh_journal))

    last = asyncio.run(db.last_delivered_signal("BTCUSDT", "1h"))
    assert last is not None
    assert last["delivered"] is True
    assert last["created_at"] == old_alert["created_at"]


def test_last_delivered_signal_filters_by_timeframe():
    db = Database(dsn=None)
    asyncio.run(db.insert_signal(_signal(timeframe="15m", delivered=True)))
    assert asyncio.run(db.last_delivered_signal("BTCUSDT", "1h")) is None
    assert asyncio.run(db.last_delivered_signal("BTCUSDT", "15m")) is not None


def test_stale_journal_record_does_not_block_new_alert():
    """End-to-end semantics: an alert delivered 10h ago (outside the 8h swing
    cooldown) must be sendable even though a journal record exists 1h ago."""
    db = Database(dsn=None)
    asyncio.run(db.insert_signal(_signal(
        delivered=True,
        created_at=datetime.now(timezone.utc) - timedelta(hours=10))))
    asyncio.run(db.insert_signal(_signal(
        delivered=False, score=6, price=61000.0,
        created_at=datetime.now(timezone.utc) - timedelta(hours=1))))

    last = asyncio.run(db.last_delivered_signal("BTCUSDT", "1h"))
    new = _signal(price=63000.0)
    # atr=500 -> price moved far beyond the 0.5*ATR dedup radius.
    assert should_send_signal(new, last, atr=500.0, cooldown_hours=8)


def test_recent_delivered_alert_still_blocks():
    db = Database(dsn=None)
    asyncio.run(db.insert_signal(_signal(
        delivered=True,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2))))
    last = asyncio.run(db.last_delivered_signal("BTCUSDT", "1h"))
    assert not should_send_signal(_signal(price=63000.0), last,
                                  atr=500.0, cooldown_hours=8)
