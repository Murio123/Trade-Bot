"""Stage 9: forecasts schema/migration safety for analytics columns.

Доказывает: 4 новых nullable-поля присутствуют в SCHEMA и MIGRATIONS; миграции —
только идемпотентный ADD COLUMN IF NOT EXISTS без non-null defaults; нет
destructive SQL для forecasts; старые forecast-dict без новых ключей вставляются
без ошибки и читаются с NULL (backward compatibility).
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from database import MIGRATIONS, SCHEMA, _FORECAST_COLS, _OUTCOME_COLS, Database

UTC = timezone.utc
NEW_COLS = ("market_regime", "volatility_regime",
            "strategy_version", "context_version")


# --- schema / migrations ----------------------------------------------------

def test_new_columns_present_in_schema_and_cols():
    for col in NEW_COLS:
        assert f"{col} " in SCHEMA, f"{col} missing from SCHEMA CREATE TABLE"
        assert col in _FORECAST_COLS, col


def test_migrations_add_columns_idempotently():
    for col in NEW_COLS:
        stmt = f"ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS {col} TEXT"
        assert stmt in MIGRATIONS, f"missing idempotent migration for {col}"


def test_new_forecast_columns_are_nullable_no_defaults():
    # Ни одна новая forecasts-миграция не должна нести NOT NULL или DEFAULT.
    for line in MIGRATIONS.splitlines():
        if "ADD COLUMN IF NOT EXISTS" in line and any(c in line for c in NEW_COLS):
            assert "NOT NULL" not in line.upper(), line
            assert "DEFAULT" not in line.upper(), line


def test_no_destructive_sql_against_forecasts():
    forbidden = re.findall(
        r"\b(DROP\s+COLUMN|DROP\s+TABLE\s+forecasts|RENAME|ALTER\s+COLUMN|"
        r"ALTER\s+TYPE|TRUNCATE)\b", MIGRATIONS, re.IGNORECASE)
    assert not forbidden, f"destructive SQL found: {set(forbidden)}"


# --- backward compatibility -------------------------------------------------

def _legacy_forecast():
    """Forecast-dict в «старом» формате — без Stage 9 полей."""
    return {
        "symbol": "BTCUSDT", "analysis_type": "SWING", "timeframe": "4h",
        "signal_candle_close_time": datetime(2026, 6, 1, 4, tzinfo=UTC),
        "decision_time": datetime(2026, 6, 1, 4, 1, tzinfo=UTC),
        "analysis_status": "ENTER", "candidate_direction": "long",
        "signal_close_price": 100_000.0,
    }


def test_legacy_forecast_without_new_fields_inserts():
    db = Database(dsn=None)
    fid = asyncio.run(db.insert_forecast(_legacy_forecast()))
    assert fid == 1


def test_legacy_row_reads_new_fields_as_none():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_legacy_forecast()))
    row = db._mem.forecasts[0]
    # Memory store materialises every _FORECAST_COLS key; missing -> None.
    for col in NEW_COLS:
        assert row[col] is None, col


# --- Stage 11: forecast_outcomes.realized_r ---------------------------------

def test_realized_r_in_outcome_schema_migration_and_cols():
    assert "realized_r " in SCHEMA, "realized_r missing from forecast_outcomes CREATE TABLE"
    assert ("ALTER TABLE forecast_outcomes ADD COLUMN IF NOT EXISTS "
            "realized_r DOUBLE PRECISION") in MIGRATIONS
    assert "realized_r" in _OUTCOME_COLS


def test_realized_r_migration_nullable_no_default():
    for line in MIGRATIONS.splitlines():
        if "ADD COLUMN IF NOT EXISTS realized_r" in line:
            assert "NOT NULL" not in line.upper(), line
            assert "DEFAULT" not in line.upper(), line


def test_no_destructive_sql_against_forecast_outcomes():
    forbidden = re.findall(
        r"\b(DROP\s+COLUMN|DROP\s+TABLE\s+forecast_outcomes|RENAME|"
        r"ALTER\s+COLUMN|ALTER\s+TYPE|UPDATE\s+forecast_outcomes|"
        r"TRUNCATE)\b", MIGRATIONS, re.IGNORECASE)
    assert not forbidden, f"destructive/backfill SQL found: {set(forbidden)}"


def _outcome(**kw):
    base = {"forecast_id": 1, "anchor_time": datetime(2026, 6, 1, 4, tzinfo=UTC),
            "reference_price": 100_000.0, "tp1_hit": True, "tp2_hit": True,
            "stop_hit": False, "resolved": True, "realized_r": 2.5484}
    base.update(kw)
    return base


def test_memory_upsert_persists_realized_r():
    db = Database(dsn=None)
    asyncio.run(db.upsert_outcome(_outcome()))
    row = asyncio.run(db.forecast_outcome(1))
    assert row["realized_r"] == 2.5484


def test_legacy_outcome_without_realized_r_does_not_break():
    db = Database(dsn=None)
    legacy = _outcome()
    legacy.pop("realized_r")               # старый outcome-dict без нового ключа
    asyncio.run(db.upsert_outcome(legacy))
    row = asyncio.run(db.forecast_outcome(1))
    assert "realized_r" not in row or row["realized_r"] is None
