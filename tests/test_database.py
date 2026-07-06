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


# --- Stage 15B1-a: setup-lifecycle schema columns (DDL only) ----------------
# Nullable analytics metadata; NOT added to _FORECAST_COLS / insert / row decode
# in this step — schema readiness only. (col_name -> SQL type.)
LIFECYCLE_COLS = {
    "setup_lifecycle_status": "TEXT",
    "setup_lifecycle_reasons": "JSONB",
    "previous_forecast_id": "BIGINT",
    "setup_lifecycle_comparable": "BOOLEAN",
    "setup_score_delta": "DOUBLE PRECISION",
    "setup_confidence_delta": "DOUBLE PRECISION",
    "setup_thresholds_used": "JSONB",
}


def test_lifecycle_columns_present_in_schema():
    for col in LIFECYCLE_COLS:
        assert f"{col} " in SCHEMA, f"{col} missing from forecasts CREATE TABLE"


def test_lifecycle_migrations_add_columns_idempotently():
    for col, sql_type in LIFECYCLE_COLS.items():
        stmt = f"ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS {col} {sql_type};"
        assert stmt in MIGRATIONS, f"missing idempotent migration for {col}"


def test_lifecycle_migrations_are_nullable_no_defaults():
    for line in MIGRATIONS.splitlines():
        if "ADD COLUMN IF NOT EXISTS" in line and any(
                c in line for c in LIFECYCLE_COLS):
            assert "NOT NULL" not in line.upper(), line
            assert "DEFAULT" not in line.upper(), line


def test_lifecycle_migrations_no_destructive_sql():
    # Guard the whole migration file: no destructive/backfill verbs at all.
    forbidden = re.findall(
        r"\b(DROP|RENAME|DELETE|TRUNCATE|UPDATE)\b", MIGRATIONS, re.IGNORECASE)
    assert not forbidden, f"destructive/backfill SQL found: {set(forbidden)}"


def test_lifecycle_columns_not_in_forecast_cols_yet():
    # Stage 15B1-a is DDL only: insert wiring (_FORECAST_COLS) comes in 15B1-b.
    for col in LIFECYCLE_COLS:
        assert col not in _FORECAST_COLS, (
            f"{col} added to _FORECAST_COLS prematurely (Stage 15B1-a is DDL only)"
        )


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


# --- Stage 14 Option B / Step 2.1: read-only latest_forecast ----------------
# Helper ЧИТАЕТ последний сохранённый forecast (для /deep-объяснения), никогда
# не пишет и не влияет на decision-path. Проверяем фильтры, порядок и read-only.

def _fc(symbol="BTCUSDT", analysis_type="SWING", close_h=4, **kw):
    base = {
        "symbol": symbol, "analysis_type": analysis_type, "timeframe": "4h",
        "signal_candle_close_time": datetime(2026, 6, 1, close_h, tzinfo=UTC),
        "decision_time": datetime(2026, 6, 1, close_h, 1, tzinfo=UTC),
        "analysis_status": "ENTER", "candidate_direction": "long",
        "take_profit_levels": [101_000.0, 102_000.0],
        "no_trade_reasons": None,
    }
    base.update(kw)
    return base


def test_latest_forecast_none_when_empty():
    db = Database(dsn=None)
    assert asyncio.run(db.latest_forecast("BTCUSDT", "SWING")) is None


def test_latest_forecast_filters_by_symbol():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(symbol="ETHUSDT")))
    assert asyncio.run(db.latest_forecast("BTCUSDT", "SWING")) is None
    got = asyncio.run(db.latest_forecast("ETHUSDT", "SWING"))
    assert got is not None and got["symbol"] == "ETHUSDT"


def test_latest_forecast_filters_by_analysis_type():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(analysis_type="SCALP")))
    assert asyncio.run(db.latest_forecast("BTCUSDT", "SWING")) is None
    got = asyncio.run(db.latest_forecast("BTCUSDT", "SCALP"))
    assert got is not None and got["analysis_type"] == "SCALP"


def test_latest_forecast_returns_newest_by_created_at():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(close_h=4)))   # id 1
    asyncio.run(db.insert_forecast(_fc(close_h=8)))   # id 2
    # id 1 сделаем новее по created_at, несмотря на меньший id.
    db._mem.forecasts[0]["created_at"] = datetime(2026, 6, 2, tzinfo=UTC)
    db._mem.forecasts[1]["created_at"] = datetime(2026, 6, 1, tzinfo=UTC)
    got = asyncio.run(db.latest_forecast("BTCUSDT", "SWING"))
    assert got["signal_candle_close_time"] == datetime(2026, 6, 1, 4, tzinfo=UTC)


def test_latest_forecast_tie_breaks_by_id():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(close_h=4)))   # id 1
    asyncio.run(db.insert_forecast(_fc(close_h=8)))   # id 2
    ts = datetime(2026, 6, 1, 12, tzinfo=UTC)
    db._mem.forecasts[0]["created_at"] = ts
    db._mem.forecasts[1]["created_at"] = ts           # точная ничья по времени
    got = asyncio.run(db.latest_forecast("BTCUSDT", "SWING"))
    assert got["id"] == 2                              # выигрывает больший id


def test_latest_forecast_json_fields_decoded():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc()))
    got = asyncio.run(db.latest_forecast("BTCUSDT", "SWING"))
    assert got["take_profit_levels"] == [101_000.0, 102_000.0]  # list, не str


def test_latest_forecast_query_is_read_only():
    import inspect
    src = inspect.getsource(Database.latest_forecast).upper()
    # Границы слова, иначе «CREATED_AT» ложно ловится на CREATE.
    for kw in ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE", "DROP", "TRUNCATE"):
        assert not re.search(rf"\b{kw}\b", src), \
            f"latest_forecast must not contain {kw}"
    assert "SELECT" in src


# --- Guardrail 2: read-only previous_forecast(before) -----------------------
# Явно берёт ПРЕДЫДУЩИЙ сопоставимый forecast строго ДО указанного времени —
# чтобы lifecycle-чтение Stage 15B не зависело от порядка вызовов. Только SELECT.

def test_previous_forecast_none_when_empty():
    db = Database(dsn=None)
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before)) is None


def test_previous_forecast_none_when_only_current_or_after_rows():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(close_h=12)))   # == before
    asyncio.run(db.insert_forecast(_fc(close_h=16)))   # > before
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before)) is None


def test_previous_forecast_returns_row_immediately_before():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(close_h=4)))    # before
    asyncio.run(db.insert_forecast(_fc(close_h=8)))    # closest before
    asyncio.run(db.insert_forecast(_fc(close_h=16)))   # after
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    got = asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before))
    assert got is not None
    assert got["signal_candle_close_time"] == datetime(2026, 6, 1, 8, tzinfo=UTC)


def test_previous_forecast_excludes_row_equal_to_before():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(close_h=8)))    # strictly before
    asyncio.run(db.insert_forecast(_fc(close_h=12)))   # == before -> excluded
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    got = asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before))
    assert got["signal_candle_close_time"] == datetime(2026, 6, 1, 8, tzinfo=UTC)


def test_previous_forecast_filters_by_symbol():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(symbol="ETHUSDT", close_h=8)))
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before)) is None
    got = asyncio.run(db.previous_forecast("ETHUSDT", "SWING", before))
    assert got is not None and got["symbol"] == "ETHUSDT"


def test_previous_forecast_filters_by_analysis_type():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(analysis_type="SCALP", close_h=8)))
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before)) is None
    got = asyncio.run(db.previous_forecast("BTCUSDT", "SCALP", before))
    assert got is not None and got["analysis_type"] == "SCALP"


def test_previous_forecast_tie_breaks_deterministically():
    db = Database(dsn=None)
    # Две строки с ОДИНАКОВЫМ signal_candle_close_time и created_at: решает id.
    asyncio.run(db.insert_forecast(_fc(close_h=8, analysis_type="SWING")))   # id 1
    # второй ряд обходит candle-dedup через другой analysis_type, затем
    # выравниваем его обратно, чтобы получить точную ничью по времени.
    asyncio.run(db.insert_forecast(_fc(close_h=8, analysis_type="TIE")))     # id 2
    db._mem.forecasts[1]["analysis_type"] = "SWING"
    ts = datetime(2026, 6, 2, tzinfo=UTC)
    db._mem.forecasts[0]["created_at"] = ts
    db._mem.forecasts[1]["created_at"] = ts
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    got = asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before))
    assert got["id"] == 2                              # больший id выигрывает


def test_previous_forecast_json_fields_decoded():
    db = Database(dsn=None)
    asyncio.run(db.insert_forecast(_fc(close_h=8)))
    before = datetime(2026, 6, 1, 12, tzinfo=UTC)
    got = asyncio.run(db.previous_forecast("BTCUSDT", "SWING", before))
    assert got["take_profit_levels"] == [101_000.0, 102_000.0]  # list, не str


def test_previous_forecast_query_is_read_only():
    import inspect
    src = inspect.getsource(Database.previous_forecast).upper()
    for kw in ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE", "DROP", "TRUNCATE"):
        assert not re.search(rf"\b{kw}\b", src), \
            f"previous_forecast must not contain {kw}"
    assert "SELECT" in src
