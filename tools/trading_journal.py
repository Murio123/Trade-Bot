"""Stage 13: Trading Journal v2 — read-only компактная сводка решений бота.

Классифицирует и агрегирует УЖЕ персистируемые строки forecasts +
forecast_outcomes через чистый analyzer.journal_classify. Read-only:

  * НЕ пишет в БД (ни INSERT/UPDATE/DELETE/ALTER/DROP/CREATE), не мигрирует,
    не бэкфиллит;
  * НЕ ходит в Binance/сеть трейдинга, не трогает Railway/env;
  * НЕ меняет scheduler, pipeline, producer, scoring, thresholds, risk,
    final_gate, Telegram или DRY_RUN;
  * НЕ импортируется runtime-кодом и НЕ использует database.py.

Источник данных (read-only):
  * JSON-экспорт (--input FILE): полностью offline, детерминированно, для CI;
  * опционально живая БД (--database-url URL): раннер открывает СВОЙ коннект и
    выполняет ТОЛЬКО SELECT (database.py не трогается).

Ядро — чистые функции над list[dict]; тестируются без сети и без БД.

Запуск:

    python -m tools.trading_journal --input export.json
    python -m tools.trading_journal --database-url "$DATABASE_URL"
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

from analyzer.journal_classify import (
    AVOIDED_LOSS, BAD_TRADE, ENTER, GOOD_TRADE, MISSED_OPPORTUNITY,
    NO_LEVELS, NONE, UNRESOLVED, classify,
)

WAIT, NO_TRADE = "WAIT", "NO_TRADE"


# ---------------------------------------------------------------------------
# Джойн + классификация (чистые функции над list[dict])
# ---------------------------------------------------------------------------

def join_records(forecasts: list[dict[str, Any]],
                 outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Слить каждый forecast с его outcome-строкой (или None) по forecast_id."""
    by_id = {o.get("forecast_id"): o for o in outcomes}
    return [{"forecast": f, "outcome": by_id.get(f.get("id"))} for f in forecasts]


def _count(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        key = v if v is not None else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _usable_levels(forecast: dict[str, Any]) -> bool:
    stop = forecast.get("stop_loss")
    tps = forecast.get("take_profit_levels") or []
    return stop is not None and len(tps) > 0 and tps[0] is not None


def summarize(forecasts: list[dict[str, Any]],
              outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Собрать компактную сводку журнала из forecasts + outcomes."""
    joined = join_records(forecasts, outcomes)
    total = len(forecasts)

    classifications = [
        {"forecast": j["forecast"],
         "outcome": j["outcome"],
         "classification": classify(j["forecast"], j["outcome"])}
        for j in joined
    ]

    by_classification = _count([c["classification"] for c in classifications])

    enter = [c for c in classifications
             if c["forecast"].get("analysis_status") == ENTER]
    non_enter = [c for c in classifications
                 if c["forecast"].get("analysis_status") in (WAIT, NO_TRADE)]

    enter_breakdown = {
        GOOD_TRADE: sum(1 for c in enter if c["classification"] == GOOD_TRADE),
        BAD_TRADE: sum(1 for c in enter if c["classification"] == BAD_TRADE),
        UNRESOLVED: sum(1 for c in enter if c["classification"] == UNRESOLVED),
        NONE: sum(1 for c in enter if c["classification"] == NONE),
    }
    non_enter_breakdown = {
        AVOIDED_LOSS: sum(1 for c in non_enter if c["classification"] == AVOIDED_LOSS),
        MISSED_OPPORTUNITY: sum(1 for c in non_enter
                               if c["classification"] == MISSED_OPPORTUNITY),
        NO_LEVELS: sum(1 for c in non_enter if c["classification"] == NO_LEVELS),
        NONE: sum(1 for c in non_enter if c["classification"] == NONE),
    }

    # Средний realized_r для ENTER, где он посчитан (None не участвуют).
    enter_r = [c["outcome"].get("realized_r")
               for c in enter
               if c["outcome"] is not None
               and c["outcome"].get("realized_r") is not None]
    avg_enter_realized_r = (round(statistics.fmean(enter_r), 4)
                            if enter_r else None)

    return {
        "total_forecasts": total,
        "by_analysis_status": _count([f.get("analysis_status") for f in forecasts]),
        "by_analysis_type": _count([f.get("analysis_type") for f in forecasts]),
        "by_candidate_direction": _count(
            [f.get("candidate_direction") for f in forecasts]),
        "by_classification": by_classification,
        "enter": enter_breakdown,
        "non_enter": non_enter_breakdown,
        "avg_enter_realized_r": avg_enter_realized_r,
        "missing_data": _missing_inventory(joined),
    }


def _missing_inventory(joined: list[dict[str, Any]]) -> dict[str, int]:
    """Честная инвентаризация пропусков — не выдумывается, только считается."""
    missing_realized_r = 0
    missing_levels = 0
    missing_direction = 0
    unresolved_outcomes = 0
    missing_market_regime = 0
    missing_volatility_regime = 0

    for j in joined:
        f, o = j["forecast"], j["outcome"]
        direction = f.get("candidate_direction")
        has_direction = direction in ("long", "short")

        if not has_direction:
            missing_direction += 1
        if not _usable_levels(f):
            missing_levels += 1
        if f.get("market_regime") is None:
            missing_market_regime += 1
        if f.get("volatility_regime") is None:
            missing_volatility_regime += 1

        # Пропуски исхода считаем только там, где есть направление (иначе
        # исход неприменим по определению).
        if has_direction:
            if o is None or not o.get("resolved"):
                unresolved_outcomes += 1
            elif f.get("analysis_status") == ENTER and o.get("realized_r") is None:
                missing_realized_r += 1

    return {
        "missing_realized_r": missing_realized_r,
        "missing_levels": missing_levels,
        "missing_direction": missing_direction,
        "unresolved_outcomes": unresolved_outcomes,
        "missing_market_regime": missing_market_regime,
        "missing_volatility_regime": missing_volatility_regime,
    }


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def _fmt_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "    (нет данных)"
    return "\n".join(f"    {k}: {v}" for k, v in counts.items())


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "=== Trading Journal v2 (Stage 13, read-only) ===",
        f"Всего forecasts: {summary['total_forecasts']}",
        "",
        "По статусу:",
        _fmt_counts(summary["by_analysis_status"]),
        "",
        "По типу анализа:",
        _fmt_counts(summary["by_analysis_type"]),
        "",
        "По направлению-кандидату:",
        _fmt_counts(summary["by_candidate_direction"]),
        "",
        "По классификации:",
        _fmt_counts(summary["by_classification"]),
        "",
        "ENTER (good / bad / unresolved / none):",
        _fmt_counts(summary["enter"]),
        "",
        "WAIT+NO_TRADE (avoided / missed / no_levels / none):",
        _fmt_counts(summary["non_enter"]),
        "",
        f"Средний realized_r по ENTER (где посчитан): "
        f"{summary['avg_enter_realized_r']}",
        "",
        "Инвентаризация пропусков:",
        _fmt_counts(summary["missing_data"]),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Загрузчики (read-only)
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict[str, Any]:
    """Прочитать JSON-экспорт {forecasts, outcomes}."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in ("forecasts", "outcomes"):
        raw.setdefault(key, [])
    for f in raw["forecasts"]:
        tps = f.get("take_profit_levels")
        if isinstance(tps, str):
            try:
                f["take_profit_levels"] = json.loads(tps)
            except (TypeError, ValueError):
                pass
    return raw


# Только читающие (SELECT) запросы — никаких пишущих операций над БД.
_SQL_FORECASTS = (
    "SELECT id, analysis_type, analysis_status, candidate_direction, final_bias, "
    "stop_loss, take_profit_levels, blocked_gate, no_trade_reasons, "
    "market_regime, volatility_regime "
    "FROM forecasts WHERE symbol = $1"
)
_SQL_OUTCOMES = (
    "SELECT o.forecast_id, o.tp1_hit, o.tp2_hit, o.stop_hit, o.resolved, "
    "o.realized_r "
    "FROM forecast_outcomes o JOIN forecasts f ON f.id = o.forecast_id "
    "WHERE f.symbol = $1"
)


def load_db(dsn: str, symbol: str) -> dict[str, Any]:
    """Прочитать данные из БД СОБСТВЕННЫМ коннектом, только SELECT.

    Не использует database.py и не добавляет туда методов; ничего не пишет.
    """
    import asyncio

    import asyncpg  # локальный импорт: offline-путь (JSON) не требует драйвера

    async def _run() -> dict[str, Any]:
        conn = await asyncpg.connect(dsn)
        try:
            f = await conn.fetch(_SQL_FORECASTS, symbol)
            o = await conn.fetch(_SQL_OUTCOMES, symbol)
        finally:
            await conn.close()
        return {"forecasts": [_row(r) for r in f],
                "outcomes": [_row(r) for r in o]}

    return asyncio.run(_run())


def _row(record: Any) -> dict[str, Any]:
    d = dict(record)
    tps = d.get("take_profit_levels")
    if isinstance(tps, str):
        try:
            d["take_profit_levels"] = json.loads(tps)
        except (TypeError, ValueError):
            pass
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.astimezone(timezone.utc).isoformat()
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        return load_json(args.input)
    if args.database_url:
        return load_db(args.database_url, args.symbol)
    return {"forecasts": [], "outcomes": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trading Journal v2 — read-only сводка решений (Stage 13).")
    parser.add_argument("--input", help="JSON-экспорт {forecasts, outcomes}")
    parser.add_argument("--database-url", help="Postgres DSN (только SELECT)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--json", action="store_true",
                        help="вывести сводку как JSON")
    args = parser.parse_args(argv)

    data = _load(args)
    summary = summarize(data.get("forecasts", []), data.get("outcomes", []))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
