"""PostgreSQL persistence layer (asyncpg).

Tables: signals, trades_journal, price_alerts, user_settings.

The module degrades gracefully: if DATABASE_URL is not configured it falls
back to an in-memory store so the rest of the bot can still run in dry-run /
local testing without a database.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config

log = logging.getLogger(__name__)

try:  # asyncpg is optional for the in-memory fallback
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None  # type: ignore


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    stop_loss       DOUBLE PRECISION,
    target_1        DOUBLE PRECISION,
    target_2        DOUBLE PRECISION,
    position_size   DOUBLE PRECISION,
    score           INTEGER NOT NULL,
    confidence      DOUBLE PRECISION,
    category_scores JSONB,
    reasons         JSONB,
    ai_text         TEXT,
    delivered       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS trades_journal (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       BIGINT REFERENCES signals(id) ON DELETE SET NULL,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at       TIMESTAMPTZ,
    direction       TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    target          DOUBLE PRECISION,
    outcome         TEXT,              -- win | loss | breakeven | open
    pnl_r           DOUBLE PRECISION,  -- realised R multiple
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS price_alerts (
    id              BIGSERIAL PRIMARY KEY,
    chat_id         TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    level           DOUBLE PRECISION NOT NULL,
    direction       TEXT NOT NULL,     -- above | below
    note            TEXT,
    triggered       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_settings (
    chat_id         TEXT PRIMARY KEY,
    settings        JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Database:
    def __init__(self, dsn: Optional[str]):
        self.dsn = dsn
        self.pool: Optional["asyncpg.Pool"] = None
        self._mem = _MemoryStore()
        self.enabled = bool(dsn and asyncpg)

    async def connect(self, retries: int = 4) -> None:
        """Connect to PostgreSQL with a few retries.

        If the database is unreachable (bad/missing DATABASE_URL, DNS failure,
        DB still booting), the bot does NOT crash: it logs a clear warning and
        falls back to the in-memory store so Telegram + analysis keep working.
        Persistence (signals/journal/alerts) is disabled until the DB is fixed.
        """
        if not self.enabled:
            log.warning("DATABASE_URL not set or asyncpg missing -> using in-memory store")
            return

        delay = 2
        for attempt in range(1, retries + 1):
            try:
                self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
                async with self.pool.acquire() as conn:
                    await conn.execute(SCHEMA)
                log.info("PostgreSQL connected and schema ensured")
                return
            except Exception as exc:  # noqa: BLE001
                self.pool = None
                log.warning("DB connect attempt %s/%s failed: %s", attempt, retries, exc)
                if attempt < retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 16)

        log.error(
            "Could not connect to PostgreSQL after %s attempts. Falling back to "
            "in-memory store (no persistence). Check that DATABASE_URL points to a "
            "reachable database, e.g. ${{Postgres.DATABASE_URL}} on Railway.",
            retries,
        )

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    # --- signals ----------------------------------------------------------
    async def insert_signal(self, sig: dict[str, Any]) -> int:
        if not self.pool:
            return self._mem.insert_signal(sig)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals
                    (symbol, timeframe, direction, entry_price, stop_loss,
                     target_1, target_2, position_size, score, confidence,
                     category_scores, reasons, ai_text, delivered)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                RETURNING id
                """,
                sig.get("symbol"), sig.get("timeframe"), sig.get("direction"),
                sig.get("entry_price"), sig.get("stop_loss"), sig.get("target_1"),
                sig.get("target_2"), sig.get("position_size"), sig.get("score"),
                sig.get("confidence"), json.dumps(sig.get("category_scores", {})),
                json.dumps(sig.get("reasons", [])), sig.get("ai_text"),
                sig.get("delivered", False),
            )
            return int(row["id"])

    async def last_signal(self, symbol: str,
                          timeframe: Optional[str] = None) -> Optional[dict[str, Any]]:
        if not self.pool:
            return self._mem.last_signal(symbol, timeframe)
        async with self.pool.acquire() as conn:
            if timeframe:
                row = await conn.fetchrow(
                    "SELECT * FROM signals WHERE symbol=$1 AND timeframe=$2 "
                    "ORDER BY created_at DESC LIMIT 1",
                    symbol, timeframe,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT * FROM signals WHERE symbol=$1 ORDER BY created_at DESC LIMIT 1",
                    symbol,
                )
            return _row_to_signal(row) if row else None

    async def last_delivered_signal(self, symbol: str) -> Optional[dict[str, Any]]:
        if not self.pool:
            return self._mem.last_delivered_signal(symbol)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM signals WHERE symbol=$1 AND delivered=TRUE "
                "ORDER BY created_at DESC LIMIT 1",
                symbol,
            )
            return _row_to_signal(row) if row else None

    async def signals_today(self, symbol: str) -> list[dict[str, Any]]:
        if not self.pool:
            return self._mem.signals_today(symbol)
        since = utcnow() - timedelta(hours=24)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM signals WHERE symbol=$1 AND delivered=TRUE "
                "AND created_at >= $2 ORDER BY created_at DESC",
                symbol, since,
            )
            return [_row_to_signal(r) for r in rows]

    async def mark_delivered(self, signal_id: int) -> None:
        if not self.pool:
            return self._mem.mark_delivered(signal_id)
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE signals SET delivered=TRUE WHERE id=$1", signal_id)

    # --- journal ----------------------------------------------------------
    async def journal_stats(self, symbol: str) -> dict[str, Any]:
        if not self.pool:
            return self._mem.journal_stats(symbol)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT outcome, pnl_r FROM trades_journal WHERE outcome IS NOT NULL "
                "AND outcome <> 'open'"
            )
        return _aggregate_journal(rows)

    # --- price alerts -----------------------------------------------------
    async def add_price_alert(self, chat_id: str, symbol: str, level: float,
                              direction: str, note: str | None = None) -> int:
        if not self.pool:
            return self._mem.add_price_alert(chat_id, symbol, level, direction, note)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO price_alerts (chat_id, symbol, level, direction, note) "
                "VALUES ($1,$2,$3,$4,$5) RETURNING id",
                chat_id, symbol, level, direction, note,
            )
            return int(row["id"])

    async def active_price_alerts(self, symbol: str) -> list[dict[str, Any]]:
        if not self.pool:
            return self._mem.active_price_alerts(symbol)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM price_alerts WHERE symbol=$1 AND triggered=FALSE",
                symbol,
            )
            return [dict(r) for r in rows]

    async def trigger_price_alert(self, alert_id: int) -> None:
        if not self.pool:
            return self._mem.trigger_price_alert(alert_id)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE price_alerts SET triggered=TRUE WHERE id=$1", alert_id
            )


def _row_to_signal(row) -> dict[str, Any]:
    d = dict(row)
    for key in ("category_scores", "reasons"):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (TypeError, ValueError):
                pass
    return d


def _aggregate_journal(rows) -> dict[str, Any]:
    total = len(rows)
    wins = sum(1 for r in rows if (r["outcome"] == "win"))
    losses = sum(1 for r in rows if (r["outcome"] == "loss"))
    r_values = [r["pnl_r"] for r in rows if r["pnl_r"] is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    winrate = (wins / total * 100) if total else 0.0
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 1),
        "avg_r": round(avg_r, 2),
    }


class _MemoryStore:
    """Tiny in-memory fallback so the bot runs without Postgres."""

    def __init__(self) -> None:
        self.signals: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self._sid = 0
        self._aid = 0

    def insert_signal(self, sig: dict[str, Any]) -> int:
        self._sid += 1
        rec = dict(sig)
        rec["id"] = self._sid
        rec.setdefault("created_at", utcnow())
        rec.setdefault("delivered", sig.get("delivered", False))
        self.signals.append(rec)
        return self._sid

    def last_signal(self, symbol: str, timeframe: Optional[str] = None):
        items = [
            s for s in self.signals
            if s.get("symbol") == symbol
            and (timeframe is None or s.get("timeframe") == timeframe)
        ]
        return items[-1] if items else None

    def last_delivered_signal(self, symbol: str):
        items = [s for s in self.signals if s.get("symbol") == symbol and s.get("delivered")]
        return items[-1] if items else None

    def signals_today(self, symbol: str):
        since = utcnow() - timedelta(hours=24)
        return [
            s for s in self.signals
            if s.get("symbol") == symbol and s.get("delivered")
            and s.get("created_at", utcnow()) >= since
        ]

    def mark_delivered(self, signal_id: int):
        for s in self.signals:
            if s["id"] == signal_id:
                s["delivered"] = True

    def journal_stats(self, symbol: str):
        return {"total": 0, "wins": 0, "losses": 0, "winrate": 0.0, "avg_r": 0.0}

    def add_price_alert(self, chat_id, symbol, level, direction, note):
        self._aid += 1
        self.alerts.append({
            "id": self._aid, "chat_id": chat_id, "symbol": symbol,
            "level": level, "direction": direction, "note": note, "triggered": False,
        })
        return self._aid

    def active_price_alerts(self, symbol: str):
        return [a for a in self.alerts if a["symbol"] == symbol and not a["triggered"]]

    def trigger_price_alert(self, alert_id: int):
        for a in self.alerts:
            if a["id"] == alert_id:
                a["triggered"] = True


# Singleton used across the app.
db = Database(config.DATABASE_URL)
