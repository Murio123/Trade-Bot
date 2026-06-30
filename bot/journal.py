"""Trade journal helpers: record signals as trades and compute win-rate stats."""
from __future__ import annotations

import logging
from typing import Any

from database import db

log = logging.getLogger(__name__)


async def record_signal_as_trade(signal_id: int, signal: dict[str, Any]) -> None:
    """Open a journal entry tied to a delivered signal (outcome filled later)."""
    if not db.pool:
        return
    try:
        async with db.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trades_journal
                    (signal_id, direction, entry_price, stop_loss, target, outcome)
                VALUES ($1,$2,$3,$4,$5,'open')
                """,
                signal_id, signal.get("direction"), signal.get("entry_price"),
                signal.get("stop_loss"), signal.get("target_1"),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("record_signal_as_trade failed: %s", exc)


async def get_stats(symbol: str) -> dict[str, Any]:
    return await db.journal_stats(symbol)
