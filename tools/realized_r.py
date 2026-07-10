"""Offline per-forecast realized R calculator (Stage 10 CLI, Option B).

Тонкий CLI/summary-слой поверх единой pure-формулы `analyzer.realized_r`
(Stage 11 вынес её в leaf-модуль, чтобы её же использовал producer
`analyzer/outcomes.py`; здесь формула НЕ дублируется).

Read-only measurement. НЕ меняет схему БД, `database.py`, outcome-tracking,
scheduler, pipeline, scoring, risk или Telegram; НЕ пишет в БД; НЕ импортируется
runtime-кодом. Считает честный per-forecast R из УЖЕ персистируемых полей
(`forecasts` + `forecast_outcomes`), без обращения к klines и без нового
lookahead — использует только то же 72h-окно, что уже измерил `measure_outcome`.

Определение R (единое с bot/journal.evaluate_trade и backtest._resolve):

    risk = |reference_price - stop_loss|
    sign = +1 (long) / -1 (short)
    R    = sign * (exit - reference_price) / risk

Формула realized_r(forecast, outcome):

    None, если forecast не ENTER, нет direction, нет outcome, unresolved,
          нет reference_price / stop_loss, либо risk == 0;
    -1.0                          если stop_hit и не tp1_hit (чистый стоп);
    sign*(tp2-ref)/risk           если tp2_hit (вин до TP2);
    0.0                           если tp1_hit и stop_hit (после TP1 — безубыток);
    sign*(tp1-ref)/risk           если tp1_hit, нет tp2 в сетапе (single-target вин);
    mark-to-market(return_72h)    если tp1_hit без TP2/стопа (mid-case, см. NOTE);
    mark-to-market(return_72h)    если ни TP, ни стоп, но resolved.

    mark-to-market = (return_72h / 100) * ref / risk  (None, если return_72h None).

Округление: pure `realized_r` возвращает ПОЛНУЮ точность float (кроме точных
-1.0 / 0.0); округление применяется только на публичном summary/CLI-выводе.

Группировка по analysis_type (Stage B3a.2): один общий пул смешивал бы режимы —
наблюдательный BOUNCE растворился бы в SWING/INTRADAY/POSITION. Отчёт по-прежнему
печатает глобальный summary (старое поведение), но добавляет разбивку по
analysis_type, а `--analysis-type TYPE` сужает популяцию до одного режима.
Фильтрация и группировка НЕ трогают формулу — только выборку строк.

Запуск:

    .venv/bin/python -m tools.realized_r --input tests/fixtures/forecast_metrics_sample.json
    .venv/bin/python -m tools.realized_r --database-url "$DATABASE_URL"
    .venv/bin/python -m tools.realized_r --input export.json --analysis-type SWING
    .venv/bin/python -m tools.realized_r --input export.json --analysis-type BOUNCE
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from typing import Any

import tools.forecast_metrics as fm
from analyzer.realized_r import ENTER, classify, realized_r  # единый источник формулы

__all__ = ["classify", "realized_r", "realized_r_summary",
           "filter_by_analysis_type", "summarize_by_analysis_type",
           "format_report", "main"]

# Прогноз без analysis_type (исторические строки) попадает в эту корзину, а не
# в чужой режим. `--analysis-type` его никогда не матчит: сентинел в нижнем
# регистре, а реальные типы нормализуются в верхний.
UNKNOWN_TYPE = "unknown"

# TP1 mid-case: цена достигла TP1, но не TP2 и не исходного стопа. Флаги
# outcome не фиксируют касание безубытка (entry) после TP1, поэтому точная
# journal-parity здесь невозможна — берём честный mark-to-market по return_72h.
MID_CASE_NOTE = (
    "TP1 mid-case (tp1_hit, без TP2 и без исходного стопа) считается через "
    "mark-to-market по return_72h: касание безубытка после TP1 не хранится в "
    "outcome-флагах, поэтому точный journal-parity R для этого случая невозможен."
)
# reference_price (executable_price_at_decision) — это proxy входа на уровне
# АНАЛИЗА для всей популяции прогнозов, а не entry_price доставленной сделки.
# Поэтому realized_r по популяции нельзя сравнивать «в лоб» с journal pnl_r.
REF_NOTE = (
    "reference_price (executable_price_at_decision) — proxy входа на уровне "
    "анализа; отличается от journal entry_price доставленной сделки. Популяция "
    "realized_r шире и не тождественна journal-подмножеству."
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_VALUE_BUCKETS = ("<=-1R", "(-1R,0R)", "0R", "(0R,1R)", "[1R,2R)", "[2R,3R)", ">=3R")


def _value_bucket(r: float) -> str:
    if r <= -1:
        return "<=-1R"
    if r < 0:
        return "(-1R,0R)"
    if r == 0:
        return "0R"
    if r < 1:
        return "(0R,1R)"
    if r < 2:
        return "[1R,2R)"
    if r < 3:
        return "[2R,3R)"
    return ">=3R"


def realized_r_summary(forecasts: list[dict[str, Any]],
                       outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Агрегировать realized_r по популяции прогнозов. None-значения не входят
    в average/median/распределение — только реально вычисленные R."""
    by_id = {o.get("forecast_id"): o for o in outcomes}
    values: list[float] = []
    kinds: Counter[str] = Counter()
    none_reasons: Counter[str] = Counter()
    dist: Counter[str] = Counter()
    eligible = 0

    for f in forecasts:
        o = by_id.get(f.get("id"))
        v, kind = classify(f, o)
        if (f.get("analysis_status") == ENTER
                and f.get("candidate_direction") in ("long", "short")):
            eligible += 1
        if v is None:
            none_reasons[kind] += 1
        else:
            values.append(v)
            kinds[kind] += 1
            dist[_value_bucket(v)] += 1

    computed = len(values)
    return {
        "total_forecasts": len(forecasts),
        "eligible_forecasts": eligible,
        "computed": computed,
        "none_unavailable": len(forecasts) - computed,
        "none_within_eligible": eligible - computed,
        "average_realized_r": round(statistics.fmean(values), 4) if values else None,
        "median_realized_r": round(statistics.median(values), 4) if values else None,
        "min_realized_r": round(min(values), 4) if values else None,
        "max_realized_r": round(max(values), 4) if values else None,
        "sum_realized_r": round(sum(values), 4) if values else None,
        "kind_counts": dict(sorted(kinds.items())),
        "none_reasons": dict(sorted(none_reasons.items())),
        "distribution": {b: dist[b] for b in _VALUE_BUCKETS if dist[b]},
    }


# ---------------------------------------------------------------------------
# analysis_type: фильтр и группировка (Stage B3a.2)
# ---------------------------------------------------------------------------

def _norm_type(value: Any) -> str:
    """analysis_type -> канонический верхний регистр, либо сентинел UNKNOWN_TYPE."""
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return UNKNOWN_TYPE


def filter_by_analysis_type(forecasts: list[dict[str, Any]],
                            analysis_type: str | None) -> list[dict[str, Any]]:
    """Сузить популяцию до одного режима. None/'' — вернуть всё (старое поведение).

    Матчинг регистронезависимый: в БД типы хранятся в верхнем регистре
    (SWING / INTRADAY / POSITION / BOUNCE), но CLI не должен об этом знать.
    Строки без analysis_type не матчатся никогда — они не принадлежат режиму.
    """
    if not analysis_type or not analysis_type.strip():
        return list(forecasts)
    want = analysis_type.strip().upper()
    return [f for f in forecasts if _norm_type(f.get("analysis_type")) == want]


def summarize_by_analysis_type(forecasts: list[dict[str, Any]],
                               outcomes: list[dict[str, Any]]
                               ) -> dict[str, dict[str, Any]]:
    """Тот же realized_r_summary, посчитанный отдельно по каждому режиму.

    Ни одна строка не попадает в два бакета, и BOUNCE не смешивается с
    трендовыми режимами. Формула не трогается — меняется только выборка.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for f in forecasts:
        buckets.setdefault(_norm_type(f.get("analysis_type")), []).append(f)
    return {name: realized_r_summary(rows, outcomes)
            for name, rows in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------

def format_report(summary: dict[str, Any],
                  by_analysis_type: dict[str, dict[str, Any]] | None = None,
                  analysis_type: str | None = None) -> str:
    lines = ["=== per-forecast realized R (Stage 10, offline read-only) ==="]
    if analysis_type:
        lines.append(f"  [FILTERED] analysis_type == {analysis_type.strip().upper()} "
                     f"— популяция сужена, глобальные числа ниже относятся "
                     f"только к этому режиму")
    for key in ("total_forecasts", "eligible_forecasts", "computed",
                "none_unavailable", "none_within_eligible",
                "average_realized_r", "median_realized_r",
                "min_realized_r", "max_realized_r", "sum_realized_r"):
        lines.append(f"  {key}: {summary[key]}")
    lines.append("  kind_counts:")
    for k, v in (summary["kind_counts"] or {"(none)": ""}).items():
        lines.append(f"    {k}: {v}")
    lines.append("  none_reasons:")
    for k, v in (summary["none_reasons"] or {"(none)": ""}).items():
        lines.append(f"    {k}: {v}")
    lines.append("  distribution:")
    for k, v in (summary["distribution"] or {"(none)": ""}).items():
        lines.append(f"    {k}: {v}")

    if by_analysis_type is not None:
        lines.append("  by_analysis_type:")
        if not by_analysis_type:
            lines.append("    (none)")
        for name, s in by_analysis_type.items():
            lines.append(
                f"    {name}: total={s['total_forecasts']} "
                f"eligible={s['eligible_forecasts']} computed={s['computed']} "
                f"avg={s['average_realized_r']} sum={s['sum_realized_r']}")

    lines.append("")
    lines.append("[LIMITATIONS]")
    lines.append(f"  - {MID_CASE_NOTE}")
    lines.append(f"  - {REF_NOTE}")
    lines.append("  - by_analysis_type группирует по полю forecasts.analysis_type; "
                 "строки без него попадают в бакет 'unknown' и никогда не "
                 "приписываются чужому режиму.")
    lines.append("  - Offline analytics metric; НЕ участвует в торговых решениях, "
                 "scoring или risk. Schema-колонка forecast_outcomes.realized_r "
                 "отложена до Stage 11 (см. docs/realized_r_stage10.md).")
    return "\n".join(lines)


def _load(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        return fm.load_json(args.input)
    if args.database_url:
        return fm.load_db(args.database_url, args.symbol)
    return {"forecasts": [], "outcomes": [], "trades": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline read-only per-forecast realized R (Stage 10).")
    parser.add_argument("--input", help="JSON export {forecasts, outcomes, trades}")
    parser.add_argument("--database-url", help="Postgres DSN (только SELECT)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--analysis-type", dest="analysis_type",
                        help="сузить отчёт до одного режима "
                             "(SWING / INTRADAY / POSITION / BOUNCE; "
                             "регистр не важен)")
    parser.add_argument("--json", action="store_true", help="вывести summary как JSON")
    args = parser.parse_args(argv)

    data = _load(args)
    outcomes = data.get("outcomes", []) or []
    forecasts = filter_by_analysis_type(data.get("forecasts", []) or [],
                                        args.analysis_type)

    summary = realized_r_summary(forecasts, outcomes)
    by_type = summarize_by_analysis_type(forecasts, outcomes)
    if args.json:
        # Глобальный summary остаётся на верхнем уровне: старые потребители
        # JSON (parsed["computed"]) продолжают работать без изменений.
        payload = {**summary,
                   "analysis_type_filter": (args.analysis_type or None),
                   "by_analysis_type": by_type}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(summary, by_type, args.analysis_type))
    return 0


if __name__ == "__main__":
    sys.exit(main())
