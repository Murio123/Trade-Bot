"""Stage 8: offline data-quality auditor (Option B).

Read-only measurement / reporting. НЕ меняет схему БД, production-поведение,
scheduler, pipeline, scoring или thresholds; НЕ пишет в БД; НЕ выполняет
миграций; НЕ импортируется runtime-кодом. Модуль отвечает на один вопрос:
«каких данных не хватает, чтобы в будущем (Stage 9+) честно улучшать стратегию
и аналитику», и делает это, читая ТОЛЬКО уже персистируемые данные.

Источник данных (read-only):
  * JSON-экспорт (--input FILE): полностью offline, детерминированно, для CI;
  * опционально живая БД (--database-url URL): аудитор открывает СВОЙ коннект и
    выполняет ТОЛЬКО SELECT. В database.py ничего не добавляется, ни одной
    пишущей операции не выполняется, миграции не запускаются.

Что считается:
  * per-table field coverage — доля не-NULL значений по каждому наблюдаемому
    полю (честная картина того, что реально заполнено);
  * planned Stage 9 fields — статус каждого запланированного поля:
      - present                          — уже сохраняется и заполнено;
      - available_runtime_not_persisted  — считается в runtime, но не пишется;
      - derived_offline                  — выводимо из уже сохранённых полей;
      - not_available_yet                — требует будущей работы (калибровка,
                                           определение R, live-execution);
      - missing                          — запланировано, пока нет ни колонки,
                                           ни источника.

Раздел «планируемых полей» — это НЕ обещание миграции. Это карта: какие поля
критичны, какие безопасно добавить additive-nullable в Stage 9, а какие
добавлять НЕ рекомендуется (выводимы offline или преждевременны).

Запуск:

    .venv/bin/python -m tools.data_quality_audit --input export.json
    .venv/bin/python -m tools.data_quality_audit --database-url "$DATABASE_URL"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Экспортные ключи -> человекочитаемое имя таблицы. forecast_context — будущая
# sidecar-таблица (Stage 9), сейчас источника ещё нет (export-ключ None).
# ---------------------------------------------------------------------------

EXPORT_KEY_BY_TABLE: dict[str, str | None] = {
    "forecasts": "forecasts",
    "forecast_outcomes": "outcomes",
    "trades_journal": "trades",
    "forecast_context": None,
}

# Статусы планируемых полей (стабильные строковые константы для отчёта/тестов).
PRESENT = "present"
RUNTIME_NOT_PERSISTED = "available_runtime_not_persisted"
DERIVED_OFFLINE = "derived_offline"
NOT_AVAILABLE_YET = "not_available_yet"
MISSING = "missing"

CRITICAL, NICE_TO_HAVE, NOT_RECOMMENDED = "critical", "nice_to_have", "not_recommended"


# ---------------------------------------------------------------------------
# Реестр планируемых Stage 9 полей (единый источник правды для аудита).
#
# recommended=True  -> безопасный additive-nullable кандидат на Stage 9;
# recommended=False -> НЕ добавлять (выводимо offline или преждевременно).
# default_status    -> статус, когда поле ещё не наблюдается в данных; если
#                      поле встречается заполненным в экспорте, статус
#                      динамически повышается до PRESENT.
# ---------------------------------------------------------------------------

PlannedField = dict[str, Any]

PLANNED_FIELDS: tuple[PlannedField, ...] = (
    # --- critical ---------------------------------------------------------
    {"field": "market_regime", "table": "forecasts", "tier": CRITICAL,
     "recommended": True, "default_status": RUNTIME_NOT_PERSISTED,
     "note": "detect_regime() уже считает; contracts.TechnicalContext.market_regime — нужно только протянуть в запись"},
    {"field": "volatility_regime", "table": "forecasts", "tier": CRITICAL,
     "recommended": True, "default_status": RUNTIME_NOT_PERSISTED,
     "note": "analyzer/volatility.py даёт expansion/compression/normal; не персистится"},
    {"field": "strategy_version", "table": "forecasts", "tier": CRITICAL,
     "recommended": True, "default_status": MISSING,
     "note": "нужна константа в config; без неё нельзя честно атрибутировать изменения стратегии по времени"},
    {"field": "context_version", "table": "forecasts", "tier": CRITICAL,
     "recommended": True, "default_status": MISSING,
     "note": "версия набора контекста; парная к strategy_version для честного A/B по стадиям"},
    {"field": "realized_r", "table": "forecast_outcomes", "tier": CRITICAL,
     "recommended": True, "default_status": DERIVED_OFFLINE,
     "note": "Stage 11: persisted (forecast_outcomes.realized_r, nullable) и выводим offline "
             "через analyzer.realized_r; старые resolved-строки NULL (без backfill)"},
    # --- nice-to-have -----------------------------------------------------
    {"field": "context_quality_score", "table": "forecasts", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": RUNTIME_NOT_PERSISTED,
     "note": "агрегат BlockMeta.degraded по блокам контекста; позволит фильтровать degraded-прогоны"},
    {"field": "data_quality_flags", "table": "forecasts", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": RUNTIME_NOT_PERSISTED,
     "note": "JSONB с per-block degraded-флагами; источник есть в BlockMeta"},
    {"field": "funding_regime", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "сырьё частично собирается; режим-метка не выводится/не хранится (sidecar)"},
    {"field": "oi_regime", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "open_interest_hist собирается в unified_context; режим не размечается (sidecar)"},
    {"field": "correlation_regime", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "correlations собираются; режим не размечается (sidecar)"},
    {"field": "liquidity_regime", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "ликвидность анализируется, режим-метка не хранится (sidecar)"},
    {"field": "macro_regime", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "macro собирается; режим не размечается (sidecar)"},
    {"field": "setup_id", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "contracts.Scenario.setup_type — эвристика; стабильный id требует дизайна (sidecar)"},
    {"field": "gate_trace", "table": "forecast_context", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": RUNTIME_NOT_PERSISTED,
     "note": "blocked_gate (один гейт) уже пишется; полный упорядоченный trace — JSONB (sidecar)"},
    {"field": "calibrated_confidence", "table": "forecasts", "tier": NICE_TO_HAVE,
     "recommended": True, "default_status": NOT_AVAILABLE_YET,
     "note": "колонка уже есть, всегда NULL до появления калибровочной стадии"},
    # --- not recommended (dangerous / redundant) --------------------------
    {"field": "confidence_bucket", "table": "forecasts", "tier": NOT_RECOMMENDED,
     "recommended": False, "default_status": DERIVED_OFFLINE,
     "note": "чистая производная raw_confidence; считается offline — хранить не нужно (риск drift)"},
    {"field": "tp3", "table": "forecasts", "tier": NOT_RECOMMENDED,
     "recommended": False, "default_status": NOT_AVAILABLE_YET,
     "note": "ни одна стратегия не эмитит третий тейк; схема под несуществующий выход преждевременна"},
    {"field": "expected_r", "table": "forecasts", "tier": NOT_RECOMMENDED,
     "recommended": False, "default_status": DERIVED_OFFLINE,
     "note": "выводимо из stop_loss/take_profit_levels/entry; хранить как колонку избыточно"},
    {"field": "model_prompt_version_table", "table": "forecast_context", "tier": NOT_RECOMMENDED,
     "recommended": False, "default_status": NOT_AVAILABLE_YET,
     "note": "prompt_version/model_version уже per-row в forecasts; отдельная таблица преждевременна"},
)


# ---------------------------------------------------------------------------
# Coverage (чистые функции над list[dict])
# ---------------------------------------------------------------------------

def field_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Для каждого наблюдаемого поля — сколько строк имеют не-NULL значение.

    coverage_pct считается от общего числа строк таблицы (а не только от строк,
    где ключ присутствует) — так честнее видно реальную заполненность.
    """
    total = len(rows)
    fields: dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            fields.setdefault(k, 0)
            if v is not None:
                fields[k] += 1
    out: dict[str, dict[str, Any]] = {}
    for k in sorted(fields):
        non_null = fields[k]
        out[k] = {
            "non_null": non_null,
            "total": total,
            "nulls": total - non_null,
            "coverage_pct": round(non_null / total * 100, 2) if total else None,
        }
    return out


def table_coverage(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Coverage по всем известным таблицам (по их export-ключам)."""
    out: dict[str, dict[str, Any]] = {}
    for table, key in EXPORT_KEY_BY_TABLE.items():
        if key is None:
            out[table] = {"rows": 0, "source": "not_persisted_yet", "fields": {}}
            continue
        rows = data.get(key) or []
        out[table] = {
            "rows": len(rows),
            "source": "export",
            "fields": field_coverage(rows),
        }
    return out


# ---------------------------------------------------------------------------
# Planned-field status
# ---------------------------------------------------------------------------

def _field_is_populated(data: dict[str, Any], table: str, field: str) -> bool:
    """Поле реально сохраняется и заполнено хотя бы в одной строке экспорта."""
    key = EXPORT_KEY_BY_TABLE.get(table)
    if key is None:
        return False
    for r in data.get(key) or []:
        if r.get(field) is not None:
            return True
    return False


def planned_field_status(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Статус каждого планируемого поля: PRESENT если уже заполнено, иначе
    объявленный default_status из реестра."""
    result: list[dict[str, Any]] = []
    for spec in PLANNED_FIELDS:
        populated = _field_is_populated(data, spec["table"], spec["field"])
        status = PRESENT if populated else spec["default_status"]
        result.append({
            "field": spec["field"],
            "table": spec["table"],
            "tier": spec["tier"],
            "recommended": spec["recommended"],
            "status": status,
            "note": spec["note"],
        })
    return result


def audit(data: dict[str, Any]) -> dict[str, Any]:
    """Единая точка: полный аудит из одного экспорта (чистая функция)."""
    planned = planned_field_status(data)
    missing_planned = [p for p in planned if p["status"] != PRESENT]
    critical_missing = [p for p in missing_planned if p["tier"] == CRITICAL]
    safe_candidates = [p for p in missing_planned if p["recommended"]]
    not_recommended = [p for p in planned if not p["recommended"]]
    return {
        "table_coverage": table_coverage(data),
        "planned_fields": planned,
        "missing_planned_fields": missing_planned,
        "critical_missing_fields": critical_missing,
        "safe_stage9_candidates": safe_candidates,
        "fields_not_recommended": not_recommended,
    }


# ---------------------------------------------------------------------------
# Форматирование отчёта
# ---------------------------------------------------------------------------

def _fmt_field_line(p: dict[str, Any]) -> str:
    return (f"  - {p['table']}.{p['field']} [{p['tier']}] -> {p['status']}"
            f"\n      {p['note']}")


def format_report(report: dict[str, Any]) -> str:
    L: list[str] = ["=== data-quality audit (Stage 8, read-only) ==="]

    L.append("")
    L.append("[TABLE COVERAGE]")
    for table, info in report["table_coverage"].items():
        L.append(f"  {table}: rows={info['rows']} source={info['source']}")
        for name, cov in info["fields"].items():
            L.append(f"    {name}: {cov['coverage_pct']}% "
                     f"({cov['non_null']}/{cov['total']}, nulls={cov['nulls']})")

    L.append("")
    L.append("[CRITICAL MISSING — нужно для честного улучшения стратегии]")
    if not report["critical_missing_fields"]:
        L.append("  (нет — все критичные поля заполнены)")
    for p in report["critical_missing_fields"]:
        L.append(_fmt_field_line(p))

    L.append("")
    L.append("[MISSING PLANNED FIELDS — все запланированные, пока не заполнены]")
    for p in report["missing_planned_fields"]:
        L.append(_fmt_field_line(p))

    L.append("")
    L.append("[SAFE STAGE 9 CANDIDATES — additive nullable, рекомендуется]")
    for p in report["safe_stage9_candidates"]:
        L.append(_fmt_field_line(p))

    L.append("")
    L.append("[NOT RECOMMENDED — выводимо offline или преждевременно]")
    for p in report["fields_not_recommended"]:
        L.append(_fmt_field_line(p))

    L.append("")
    L.append("NB: это карта пробелов данных, НЕ миграция. Схема БД в Stage 8 не "
             "меняется; любое поле добавляется только additive-nullable в Stage 9.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Загрузчики (read-only)
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict[str, Any]:
    """Прочитать JSON-экспорт {forecasts, outcomes, trades}."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in ("forecasts", "outcomes", "trades"):
        raw.setdefault(key, [])
    return raw


# Только читающие (SELECT) запросы. SELECT * — чтобы coverage видел ВСЕ
# колонки, включая появившиеся позже. Никаких пишущих операций над БД.
_SQL_FORECASTS = "SELECT * FROM forecasts WHERE symbol = $1"
_SQL_OUTCOMES = (
    "SELECT o.* FROM forecast_outcomes o "
    "JOIN forecasts f ON f.id = o.forecast_id WHERE f.symbol = $1"
)
_SQL_TRADES = (
    "SELECT * FROM trades_journal WHERE (symbol = $1 OR symbol IS NULL)"
)


def load_db(dsn: str, symbol: str) -> dict[str, Any]:
    """Прочитать данные из БД СОБСТВЕННЫМ коннектом, только SELECT.

    Не использует database.py и не добавляет туда методов; ничего не пишет,
    миграций не выполняет.
    """
    import asyncio

    import asyncpg  # локальный импорт: offline-путь (JSON) не требует драйвера

    async def _run() -> dict[str, Any]:
        conn = await asyncpg.connect(dsn)
        try:
            f = await conn.fetch(_SQL_FORECASTS, symbol)
            o = await conn.fetch(_SQL_OUTCOMES, symbol)
            t = await conn.fetch(_SQL_TRADES, symbol)
        finally:
            await conn.close()
        return {
            "forecasts": [_row(r) for r in f],
            "outcomes": [_row(r) for r in o],
            "trades": [_row(r) for r in t],
        }

    return asyncio.run(_run())


def _row(record: Any) -> dict[str, Any]:
    d = dict(record)
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
    return {"forecasts": [], "outcomes": [], "trades": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline read-only data-quality auditor (Stage 8).")
    parser.add_argument("--input", help="JSON export {forecasts, outcomes, trades}")
    parser.add_argument("--database-url", help="Postgres DSN (только SELECT)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--json", action="store_true", help="вывести аудит как JSON")
    args = parser.parse_args(argv)

    data = _load(args)
    report = audit(data)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
