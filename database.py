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


def _coerce_dt(value: Any) -> Optional[datetime]:
    """datetime passthrough / ISO-string parse, for query params and in-memory
    comparison. Naive datetimes are treated as UTC; unparseable input -> None."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


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
    tp1             DOUBLE PRECISION,
    tp2             DOUBLE PRECISION,
    stage           TEXT DEFAULT 'open',     -- open | tp1 | closed
    current_stop    DOUBLE PRECISION,
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

CREATE TABLE IF NOT EXISTS bot_state (
    key             TEXT PRIMARY KEY,
    value           JSONB NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Append-only forecast ledger: one row per analysis run (including blocked /
-- NO_TRADE), unlike ``signals`` which only stores journal+ entries. The
-- UNIQUE constraint is the candle-level dedup: one forecast per closed
-- entry candle per mode.
CREATE TABLE IF NOT EXISTS forecasts (
    id                          BIGSERIAL PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol                      TEXT NOT NULL,
    analysis_type               TEXT NOT NULL,
    timeframe                   TEXT NOT NULL,
    signal_candle_close_time    TIMESTAMPTZ NOT NULL,
    decision_time               TIMESTAMPTZ NOT NULL,
    data_freshness_seconds      DOUBLE PRECISION,
    decision_latency_seconds    DOUBLE PRECISION,
    candidate_direction         TEXT,
    final_bias                  TEXT,
    analysis_status             TEXT NOT NULL,
    blocked_gate                TEXT,
    long_score                  DOUBLE PRECISION,
    short_score                 DOUBLE PRECISION,
    raw_confidence              DOUBLE PRECISION,
    calibrated_confidence       DOUBLE PRECISION,
    expected_move_points        DOUBLE PRECISION,
    expected_move_percent       DOUBLE PRECISION,
    expected_move_atr           DOUBLE PRECISION,
    entry_zone                  JSONB,
    signal_close_price          DOUBLE PRECISION,
    executable_price_at_decision DOUBLE PRECISION,
    stop_loss                   DOUBLE PRECISION,
    take_profit_levels          JSONB,
    tp2_source                  TEXT,
    risk_reward                 DOUBLE PRECISION,
    no_trade_reasons            JSONB,
    prompt_version              TEXT,
    model_version               TEXT,
    -- Stage 9 analytics metadata (nullable, never used in trading decisions).
    market_regime               TEXT,
    volatility_regime           TEXT,
    strategy_version            TEXT,
    context_version             TEXT,
    signal_id                   BIGINT REFERENCES signals(id) ON DELETE SET NULL,
    UNIQUE (symbol, analysis_type, signal_candle_close_time)
);

-- Outcome measurements per forecast. Horizon returns stay NULL until the
-- horizon has elapsed (right-censoring is explicit, unresolved rows are
-- never deleted).
CREATE TABLE IF NOT EXISTS forecast_outcomes (
    forecast_id     BIGINT PRIMARY KEY REFERENCES forecasts(id) ON DELETE CASCADE,
    anchor_time     TIMESTAMPTZ NOT NULL,
    reference_price DOUBLE PRECISION NOT NULL,
    return_1h       DOUBLE PRECISION,
    return_4h       DOUBLE PRECISION,
    return_12h      DOUBLE PRECISION,
    return_24h      DOUBLE PRECISION,
    return_72h      DOUBLE PRECISION,
    mfe_points      DOUBLE PRECISION,
    mae_points      DOUBLE PRECISION,
    reached_500     BOOLEAN NOT NULL DEFAULT FALSE,
    reached_1500    BOOLEAN NOT NULL DEFAULT FALSE,
    reached_3000    BOOLEAN NOT NULL DEFAULT FALSE,
    tp1_hit         BOOLEAN NOT NULL DEFAULT FALSE,
    tp2_hit         BOOLEAN NOT NULL DEFAULT FALSE,
    stop_hit        BOOLEAN NOT NULL DEFAULT FALSE,
    net_after_costs DOUBLE PRECISION,
    realized_r      DOUBLE PRECISION,
    resolved        BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Idempotent migrations for tables created before lifecycle columns existed.
MIGRATIONS = """
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS tp1 DOUBLE PRECISION;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS tp2 DOUBLE PRECISION;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS stage TEXT DEFAULT 'open';
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS current_stop DOUBLE PRECISION;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS timeframe TEXT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS symbol TEXT;

-- Mode separation: SWING and POSITIONAL share timeframe=4h, so cooldown /
-- daily-limit / stats queries need the analysis_type discriminator. Old rows
-- keep NULL (historical, from before the modes were separated).
ALTER TABLE signals ADD COLUMN IF NOT EXISTS analysis_type TEXT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS analysis_type TEXT;

-- Stage 9: additive analytics metadata on forecasts. All nullable, no defaults,
-- no backfill — old rows stay NULL and read back as "unknown" in the tools.
-- Pure ledger/analytics; never consulted by any trading decision or score.
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS market_regime TEXT;
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS volatility_regime TEXT;
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS strategy_version TEXT;
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS context_version TEXT;

-- Stage 11: additive nullable per-forecast realized R on forecast_outcomes.
-- No backfill — historical resolved rows stay NULL; coverage grows forward as
-- new forecasts resolve. Pure analytics ledger; never read by any decision.
ALTER TABLE forecast_outcomes ADD COLUMN IF NOT EXISTS realized_r DOUBLE PRECISION;

-- Indexes matching the hot query paths (cooldown, daily limit, open trades,
-- active price alerts). IF NOT EXISTS keeps this idempotent.
CREATE INDEX IF NOT EXISTS idx_signals_symbol_tf_created
    ON signals (symbol, timeframe, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_delivered
    ON signals (symbol, delivered, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades_journal (outcome);
CREATE INDEX IF NOT EXISTS idx_price_alerts_active ON price_alerts (symbol, triggered);
CREATE INDEX IF NOT EXISTS idx_signals_type
    ON signals (symbol, timeframe, analysis_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_type_created
    ON forecasts (symbol, analysis_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_unresolved
    ON forecast_outcomes (resolved) WHERE resolved = FALSE;
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
                    await conn.execute(MIGRATIONS)
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
                     category_scores, reasons, ai_text, delivered, analysis_type)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                RETURNING id
                """,
                sig.get("symbol"), sig.get("timeframe"), sig.get("direction"),
                sig.get("entry_price"), sig.get("stop_loss"), sig.get("target_1"),
                sig.get("target_2"), sig.get("position_size"), sig.get("score"),
                sig.get("confidence"), json.dumps(sig.get("category_scores", {})),
                json.dumps(sig.get("reasons", [])), sig.get("ai_text"),
                sig.get("delivered", False), sig.get("analysis_type"),
            )
            return int(row["id"])

    async def last_signal(self, symbol: str, timeframe: Optional[str] = None,
                          analysis_type: Optional[str] = None) -> Optional[dict[str, Any]]:
        if not self.pool:
            return self._mem.last_signal(symbol, timeframe, analysis_type)
        where, args = _signal_filter(symbol, timeframe, analysis_type)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM signals WHERE {where} "
                "ORDER BY created_at DESC LIMIT 1", *args,
            )
            return _row_to_signal(row) if row else None

    async def last_delivered_signal(self, symbol: str, timeframe: Optional[str] = None,
                                    analysis_type: Optional[str] = None) -> Optional[dict[str, Any]]:
        if not self.pool:
            return self._mem.last_delivered_signal(symbol, timeframe, analysis_type)
        where, args = _signal_filter(symbol, timeframe, analysis_type)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM signals WHERE {where} AND delivered=TRUE "
                "ORDER BY created_at DESC LIMIT 1", *args,
            )
            return _row_to_signal(row) if row else None

    async def signals_today(self, symbol: str, timeframe: Optional[str] = None,
                            analysis_type: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.pool:
            return self._mem.signals_today(symbol, timeframe, analysis_type)
        since = utcnow() - timedelta(hours=24)
        where, args = _signal_filter(symbol, timeframe, analysis_type)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM signals WHERE {where} AND delivered=TRUE "
                f"AND created_at >= ${len(args) + 1} ORDER BY created_at DESC",
                *args, since,
            )
            return [_row_to_signal(r) for r in rows]

    async def mark_delivered(self, signal_id: int) -> None:
        if not self.pool:
            return self._mem.mark_delivered(signal_id)
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE signals SET delivered=TRUE WHERE id=$1", signal_id)

    # --- journal ----------------------------------------------------------
    async def insert_trade(self, trade: dict[str, Any]) -> int:
        if not self.pool:
            return self._mem.insert_trade(trade)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO trades_journal
                    (signal_id, direction, entry_price, stop_loss, target,
                     tp1, tp2, stage, current_stop, outcome, timeframe, symbol,
                     analysis_type)
                VALUES ($1,$2,$3,$4,$5,$6,$7,'open',$8,'open',$9,$10,$11)
                RETURNING id
                """,
                trade.get("signal_id"), trade.get("direction"), trade.get("entry_price"),
                trade.get("stop_loss"), trade.get("target"),
                trade.get("tp1"), trade.get("tp2"), trade.get("stop_loss"),
                trade.get("timeframe"), trade.get("symbol"),
                trade.get("analysis_type"),
            )
            return int(row["id"])

    async def advance_trade_stage(self, trade_id: int, stage: str,
                                  current_stop: float) -> None:
        if not self.pool:
            return self._mem.advance_trade_stage(trade_id, stage, current_stop)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE trades_journal SET stage=$1, current_stop=$2 WHERE id=$3",
                stage, current_stop, trade_id,
            )

    async def open_trades(self) -> list[dict[str, Any]]:
        if not self.pool:
            return self._mem.open_trades()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM trades_journal WHERE outcome = 'open' ORDER BY opened_at"
            )
            return [dict(r) for r in rows]

    async def close_trade(self, trade_id: int, exit_price: float,
                          outcome: str, pnl_r: float) -> None:
        if not self.pool:
            return self._mem.close_trade(trade_id, exit_price, outcome, pnl_r)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE trades_journal SET exit_price=$1, outcome=$2, pnl_r=$3, "
                "closed_at=now() WHERE id=$4",
                exit_price, outcome, pnl_r, trade_id,
            )

    async def journal_stats(self, symbol: str,
                            analysis_type: Optional[str] = None) -> dict[str, Any]:
        if not self.pool:
            return self._mem.journal_stats(symbol, analysis_type)
        async with self.pool.acquire() as conn:
            # symbol IS NULL keeps trades recorded before the column existed.
            if analysis_type:
                rows = await conn.fetch(
                    "SELECT outcome, pnl_r FROM trades_journal WHERE outcome IS NOT NULL "
                    "AND outcome <> 'open' AND (symbol = $1 OR symbol IS NULL) "
                    "AND analysis_type = $2",
                    symbol, analysis_type,
                )
            else:
                rows = await conn.fetch(
                    "SELECT outcome, pnl_r FROM trades_journal WHERE outcome IS NOT NULL "
                    "AND outcome <> 'open' AND (symbol = $1 OR symbol IS NULL)",
                    symbol,
                )
        return _aggregate_journal(rows)

    # --- forecast ledger ----------------------------------------------------
    async def insert_forecast(self, fc: dict[str, Any]) -> Optional[int]:
        """Insert a forecast row; returns None when the candle was already
        recorded for this mode (the UNIQUE constraint is the dedup)."""
        if not self.pool:
            return self._mem.insert_forecast(fc)
        cols = [c for c in _FORECAST_COLS if c in fc]
        params = [_forecast_param(c, fc[c]) for c in cols]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"INSERT INTO forecasts ({', '.join(cols)}) VALUES ({placeholders}) "
                "ON CONFLICT (symbol, analysis_type, signal_candle_close_time) "
                "DO NOTHING RETURNING id",
                *params,
            )
            return int(row["id"]) if row else None

    async def forecast_exists(self, symbol: str, analysis_type: str,
                              candle_close_time: datetime) -> bool:
        if not self.pool:
            return self._mem.forecast_exists(symbol, analysis_type, candle_close_time)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM forecasts WHERE symbol=$1 AND analysis_type=$2 "
                "AND signal_candle_close_time=$3",
                symbol, analysis_type, candle_close_time,
            )
            return row is not None

    async def latest_forecast(self, symbol: str,
                              analysis_type: str) -> Optional[dict[str, Any]]:
        """Newest saved forecast for a mode; read-only. None when none exist.

        Used to *explain* the engine's already-persisted decision (e.g. /deep):
        pure SELECT, never writes, never influences a trading decision. JSONB
        fields are decoded via the same _row_to_signal path as other reads."""
        if not self.pool:
            return self._mem.latest_forecast(symbol, analysis_type)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM forecasts WHERE symbol=$1 AND analysis_type=$2 "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                symbol, analysis_type,
            )
            return _row_to_signal(row) if row else None

    async def previous_forecast(self, symbol: str, analysis_type: str,
                                before: Any) -> Optional[dict[str, Any]]:
        """Newest saved forecast for a mode STRICTLY BEFORE ``before`` (compared
        on signal_candle_close_time); read-only. None when none exist.

        Unlike ``latest_forecast`` (newest overall), the explicit ``before``
        cutoff picks the previous comparable row by time — so a lifecycle read
        does not depend on call ordering versus the current row being written.
        Pure SELECT, never writes. JSONB fields decode via the same
        _row_to_signal path as other reads."""
        cutoff = _coerce_dt(before)
        if cutoff is None:
            return None
        if not self.pool:
            return self._mem.previous_forecast(symbol, analysis_type, cutoff)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM forecasts WHERE symbol=$1 AND analysis_type=$2 "
                "AND signal_candle_close_time < $3 "
                "ORDER BY signal_candle_close_time DESC, created_at DESC, id DESC "
                "LIMIT 1",
                symbol, analysis_type, cutoff,
            )
            return _row_to_signal(row) if row else None

    async def link_forecast_signal(self, forecast_id: int, signal_id: int) -> None:
        if not self.pool:
            return self._mem.link_forecast_signal(forecast_id, signal_id)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE forecasts SET signal_id=$1 WHERE id=$2",
                signal_id, forecast_id,
            )

    async def forecasts_pending_outcomes(self, symbol: str,
                                         limit: int = 200) -> list[dict[str, Any]]:
        """Forecasts with a direction whose outcome row is absent or unresolved."""
        if not self.pool:
            return self._mem.forecasts_pending_outcomes(symbol, limit)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT f.*, o.resolved
                FROM forecasts f
                LEFT JOIN forecast_outcomes o ON o.forecast_id = f.id
                WHERE f.symbol = $1 AND f.candidate_direction IS NOT NULL
                  AND f.analysis_status <> 'NO_TRADE'
                  AND (o.forecast_id IS NULL OR o.resolved = FALSE)
                ORDER BY f.decision_time
                LIMIT $2
                """,
                symbol, limit,
            )
            return [_row_to_signal(r) for r in rows]

    async def upsert_outcome(self, outcome: dict[str, Any]) -> None:
        if not self.pool:
            return self._mem.upsert_outcome(outcome)
        cols = [c for c in _OUTCOME_COLS if c in outcome]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
        updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "forecast_id")
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO forecast_outcomes ({', '.join(cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT (forecast_id) DO UPDATE SET {updates}, updated_at=now()",
                *[outcome[c] for c in cols],
            )

    async def forecast_outcome(self, forecast_id: int) -> Optional[dict[str, Any]]:
        if not self.pool:
            return self._mem.forecast_outcome(forecast_id)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM forecast_outcomes WHERE forecast_id=$1", forecast_id)
            return dict(row) if row else None

    async def forecast_stats(self, symbol: str,
                             analysis_type: Optional[str] = None) -> dict[str, Any]:
        """Per-mode hit-rate over resolved outcomes (unresolved stay counted)."""
        if not self.pool:
            return self._mem.forecast_stats(symbol, analysis_type)
        async with self.pool.acquire() as conn:
            if analysis_type:
                rows = await conn.fetch(
                    "SELECT o.* FROM forecast_outcomes o "
                    "JOIN forecasts f ON f.id = o.forecast_id "
                    "WHERE f.symbol=$1 AND f.analysis_type=$2",
                    symbol, analysis_type,
                )
            else:
                rows = await conn.fetch(
                    "SELECT o.* FROM forecast_outcomes o "
                    "JOIN forecasts f ON f.id = o.forecast_id WHERE f.symbol=$1",
                    symbol,
                )
        return _aggregate_outcomes([dict(r) for r in rows])

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

    async def delete_price_alert(self, alert_id: int, chat_id: str) -> bool:
        """Delete an alert; the chat_id guard stops deleting others' alerts."""
        if not self.pool:
            return self._mem.delete_price_alert(alert_id, chat_id)
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM price_alerts WHERE id=$1 AND chat_id=$2",
                alert_id, chat_id,
            )
            return res.endswith("1")

    # --- bot state (small key->JSON blobs that must survive restarts) ------
    async def get_state(self, key: str) -> Optional[dict[str, Any]]:
        if not self.pool:
            return self._mem.get_state(key)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT value FROM bot_state WHERE key=$1", key)
        if row is None:
            return None
        val = row["value"]
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (TypeError, ValueError):
                return None
        return val

    async def set_state(self, key: str, value: dict[str, Any]) -> None:
        if not self.pool:
            return self._mem.set_state(key, value)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO bot_state (key, value, updated_at) "
                "VALUES ($1, $2::jsonb, now()) "
                "ON CONFLICT (key) DO UPDATE SET value=$2::jsonb, updated_at=now()",
                key, json.dumps(value, default=str),
            )


def _signal_filter(symbol: str, timeframe: Optional[str],
                   analysis_type: Optional[str]) -> tuple[str, list[Any]]:
    """WHERE clause + args for the signals queries. analysis_type is strict:
    NULL rows predate mode separation and must not couple the new streams."""
    where = ["symbol=$1"]
    args: list[Any] = [symbol]
    if timeframe:
        args.append(timeframe)
        where.append(f"timeframe=${len(args)}")
    if analysis_type:
        args.append(analysis_type)
        where.append(f"analysis_type=${len(args)}")
    return " AND ".join(where), args


# Column whitelists keep the dynamic INSERTs safe (keys are never user input,
# but an explicit list also documents the record shape in one place).
_FORECAST_COLS = [
    "symbol", "analysis_type", "timeframe", "signal_candle_close_time",
    "decision_time", "data_freshness_seconds", "decision_latency_seconds",
    "candidate_direction", "final_bias", "analysis_status", "blocked_gate",
    "long_score", "short_score", "raw_confidence", "calibrated_confidence",
    "expected_move_points", "expected_move_percent", "expected_move_atr",
    "entry_zone", "signal_close_price", "executable_price_at_decision",
    "stop_loss", "take_profit_levels", "tp2_source", "risk_reward",
    "no_trade_reasons", "prompt_version", "model_version",
    "market_regime", "volatility_regime", "strategy_version", "context_version",
    "signal_id",
]
_FORECAST_JSON_COLS = {"entry_zone", "take_profit_levels", "no_trade_reasons"}
_OUTCOME_COLS = [
    "forecast_id", "anchor_time", "reference_price",
    "return_1h", "return_4h", "return_12h", "return_24h", "return_72h",
    "mfe_points", "mae_points", "reached_500", "reached_1500", "reached_3000",
    "tp1_hit", "tp2_hit", "stop_hit", "net_after_costs", "realized_r", "resolved",
]


def _forecast_param(col: str, value: Any) -> Any:
    if col in _FORECAST_JSON_COLS and value is not None:
        return json.dumps(value, default=str)
    return value


def _row_to_signal(row) -> dict[str, Any]:
    d = dict(row)
    for key in ("category_scores", "reasons", "entry_zone",
                "take_profit_levels", "no_trade_reasons"):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (TypeError, ValueError):
                pass
    return d


def _aggregate_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    resolved = [r for r in rows if r.get("resolved")]
    tp1 = sum(1 for r in rows if r.get("tp1_hit"))
    tp2 = sum(1 for r in rows if r.get("tp2_hit"))
    stops = sum(1 for r in rows if r.get("stop_hit"))
    return {
        "total": total,
        "resolved": len(resolved),
        "unresolved": total - len(resolved),
        "tp1_hits": tp1,
        "tp2_hits": tp2,
        "stop_hits": stops,
        "tp1_rate": round(tp1 / total * 100, 1) if total else 0.0,
        "tp2_rate": round(tp2 / total * 100, 1) if total else 0.0,
    }


def _aggregate_journal(rows) -> dict[str, Any]:
    total = len(rows)
    wins = sum(1 for r in rows if (r["outcome"] == "win"))
    losses = sum(1 for r in rows if (r["outcome"] == "loss"))
    r_values = [r["pnl_r"] for r in rows if r["pnl_r"] is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    total_r = sum(r_values) if r_values else 0.0
    winrate = (wins / total * 100) if total else 0.0
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 1),
        "avg_r": round(avg_r, 2),
        "total_r": round(total_r, 2),
        "best_r": round(max(r_values), 2) if r_values else 0.0,
        "worst_r": round(min(r_values), 2) if r_values else 0.0,
    }


class _MemoryStore:
    """Tiny in-memory fallback so the bot runs without Postgres."""

    def __init__(self) -> None:
        self.signals: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.forecasts: list[dict[str, Any]] = []
        self.outcomes: dict[int, dict[str, Any]] = {}
        self.state: dict[str, dict[str, Any]] = {}
        self._sid = 0
        self._aid = 0
        self._tid = 0
        self._fid = 0

    def insert_signal(self, sig: dict[str, Any]) -> int:
        self._sid += 1
        rec = dict(sig)
        rec["id"] = self._sid
        rec.setdefault("created_at", utcnow())
        rec.setdefault("delivered", sig.get("delivered", False))
        self.signals.append(rec)
        return self._sid

    def last_signal(self, symbol: str, timeframe: Optional[str] = None,
                    analysis_type: Optional[str] = None):
        items = [
            s for s in self.signals
            if s.get("symbol") == symbol
            and (timeframe is None or s.get("timeframe") == timeframe)
            and (analysis_type is None or s.get("analysis_type") == analysis_type)
        ]
        return items[-1] if items else None

    def last_delivered_signal(self, symbol: str, timeframe: Optional[str] = None,
                              analysis_type: Optional[str] = None):
        items = [
            s for s in self.signals
            if s.get("symbol") == symbol and s.get("delivered")
            and (timeframe is None or s.get("timeframe") == timeframe)
            and (analysis_type is None or s.get("analysis_type") == analysis_type)
        ]
        return items[-1] if items else None

    def signals_today(self, symbol: str, timeframe: Optional[str] = None,
                      analysis_type: Optional[str] = None):
        since = utcnow() - timedelta(hours=24)
        return [
            s for s in self.signals
            if s.get("symbol") == symbol and s.get("delivered")
            and (timeframe is None or s.get("timeframe") == timeframe)
            and (analysis_type is None or s.get("analysis_type") == analysis_type)
            and s.get("created_at", utcnow()) >= since
        ]

    def mark_delivered(self, signal_id: int):
        for s in self.signals:
            if s["id"] == signal_id:
                s["delivered"] = True

    def insert_trade(self, trade: dict[str, Any]) -> int:
        self._tid += 1
        rec = dict(trade)
        rec["id"] = self._tid
        rec["outcome"] = "open"
        rec["stage"] = "open"
        rec.setdefault("current_stop", trade.get("stop_loss"))
        rec.setdefault("opened_at", utcnow())
        self.trades.append(rec)
        return self._tid

    def open_trades(self):
        return [t for t in self.trades if t.get("outcome") == "open"]

    def advance_trade_stage(self, trade_id: int, stage: str, current_stop: float):
        for t in self.trades:
            if t["id"] == trade_id:
                t["stage"] = stage
                t["current_stop"] = current_stop

    def close_trade(self, trade_id: int, exit_price: float, outcome: str, pnl_r: float):
        for t in self.trades:
            if t["id"] == trade_id:
                t["exit_price"] = exit_price
                t["outcome"] = outcome
                t["pnl_r"] = pnl_r
                t["closed_at"] = utcnow()

    def journal_stats(self, symbol: str, analysis_type: Optional[str] = None):
        closed = [t for t in self.trades
                  if t.get("outcome") and t["outcome"] != "open"
                  and t.get("symbol") in (None, symbol)
                  and (analysis_type is None
                       or t.get("analysis_type") == analysis_type)]
        return _aggregate_journal(closed)

    # --- forecast ledger (mirrors the Postgres semantics) -------------------
    def insert_forecast(self, fc: dict[str, Any]) -> Optional[int]:
        key = (fc.get("symbol"), fc.get("analysis_type"),
               fc.get("signal_candle_close_time"))
        for f in self.forecasts:
            if (f.get("symbol"), f.get("analysis_type"),
                    f.get("signal_candle_close_time")) == key:
                return None  # ON CONFLICT DO NOTHING
        self._fid += 1
        rec = {c: fc.get(c) for c in _FORECAST_COLS}
        rec["id"] = self._fid
        rec.setdefault("created_at", utcnow())
        self.forecasts.append(rec)
        return self._fid

    def forecast_exists(self, symbol: str, analysis_type: str,
                        candle_close_time) -> bool:
        return any(
            f.get("symbol") == symbol and f.get("analysis_type") == analysis_type
            and f.get("signal_candle_close_time") == candle_close_time
            for f in self.forecasts
        )

    def latest_forecast(self, symbol: str,
                        analysis_type: str) -> Optional[dict[str, Any]]:
        matches = [
            f for f in self.forecasts
            if f.get("symbol") == symbol and f.get("analysis_type") == analysis_type
        ]
        if not matches:
            return None
        newest = max(matches,
                     key=lambda f: (f.get("created_at") or utcnow(), f.get("id") or 0))
        return dict(newest)

    def previous_forecast(self, symbol: str, analysis_type: str,
                          before: Any) -> Optional[dict[str, Any]]:
        cutoff = _coerce_dt(before)
        if cutoff is None:
            return None
        matches: list[tuple[datetime, dict[str, Any]]] = []
        for f in self.forecasts:
            if (f.get("symbol") != symbol
                    or f.get("analysis_type") != analysis_type):
                continue
            cct = _coerce_dt(f.get("signal_candle_close_time"))
            if cct is None or cct >= cutoff:   # missing/after/equal -> skip
                continue
            matches.append((cct, f))
        if not matches:
            return None
        newest = max(matches, key=lambda m: (
            m[0], m[1].get("created_at") or utcnow(), m[1].get("id") or 0))
        return dict(newest[1])

    def link_forecast_signal(self, forecast_id: int, signal_id: int) -> None:
        for f in self.forecasts:
            if f["id"] == forecast_id:
                f["signal_id"] = signal_id

    def forecasts_pending_outcomes(self, symbol: str, limit: int = 200):
        out = []
        for f in self.forecasts:
            if f.get("symbol") != symbol or not f.get("candidate_direction"):
                continue
            if f.get("analysis_status") == "NO_TRADE":
                continue
            o = self.outcomes.get(f["id"])
            if o is None or not o.get("resolved"):
                out.append(dict(f))
        out.sort(key=lambda f: f.get("decision_time") or utcnow())
        return out[:limit]

    def upsert_outcome(self, outcome: dict[str, Any]) -> None:
        fid = outcome["forecast_id"]
        rec = self.outcomes.get(fid, {})
        rec.update({c: outcome[c] for c in _OUTCOME_COLS if c in outcome})
        rec["updated_at"] = utcnow()
        self.outcomes[fid] = rec

    def forecast_outcome(self, forecast_id: int) -> Optional[dict[str, Any]]:
        rec = self.outcomes.get(forecast_id)
        return dict(rec) if rec else None

    def forecast_stats(self, symbol: str, analysis_type: Optional[str] = None):
        by_id = {f["id"]: f for f in self.forecasts}
        rows = []
        for fid, o in self.outcomes.items():
            f = by_id.get(fid)
            if not f or f.get("symbol") != symbol:
                continue
            if analysis_type and f.get("analysis_type") != analysis_type:
                continue
            rows.append(o)
        return _aggregate_outcomes(rows)

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

    def delete_price_alert(self, alert_id: int, chat_id: str) -> bool:
        before = len(self.alerts)
        self.alerts = [a for a in self.alerts
                       if not (a["id"] == alert_id and str(a["chat_id"]) == str(chat_id))]
        return len(self.alerts) < before

    def get_state(self, key: str) -> Optional[dict[str, Any]]:
        return self.state.get(key)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        # Mirror the JSON round-trip Postgres does (datetimes -> strings).
        self.state[key] = json.loads(json.dumps(value, default=str))


# Singleton used across the app.
db = Database(config.DATABASE_URL)
