"""Stage 10: offline per-forecast realized R calculator (Option B).

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

Запуск:

    .venv/bin/python -m tools.realized_r --input tests/fixtures/forecast_metrics_sample.json
    .venv/bin/python -m tools.realized_r --database-url "$DATABASE_URL"
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from typing import Any

import tools.forecast_metrics as fm

ENTER = "ENTER"

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
# Pure core
# ---------------------------------------------------------------------------

def _levels(forecast: dict[str, Any]) -> tuple[float | None, float | None]:
    tps = forecast.get("take_profit_levels") or []
    tp1 = float(tps[0]) if len(tps) > 0 and tps[0] else None
    tp2 = float(tps[1]) if len(tps) > 1 and tps[1] else None
    return tp1, tp2


def _ref(forecast: dict[str, Any], outcome: dict[str, Any] | None) -> float | None:
    """Якорь-«вход»: reference_price из outcome, иначе та же дефиниция, что в
    analyzer/outcomes.measure_outcome (executable_price → signal_close)."""
    if outcome is not None and outcome.get("reference_price") is not None:
        return float(outcome["reference_price"])
    ref = (forecast.get("executable_price_at_decision")
           or forecast.get("signal_close_price"))
    return float(ref) if ref else None


def _mtm(return_72h: Any, ref: float, risk: float) -> float | None:
    """Mark-to-market R из sign-adjusted %-возврата за 72h."""
    if return_72h is None:
        return None
    return (float(return_72h) / 100.0) * ref / risk


def classify(forecast: dict[str, Any],
             outcome: dict[str, Any] | None) -> tuple[float | None, str]:
    """Вернуть (realized_r | None, kind). kind описывает выбранную ветку либо
    причину недоступности (для summary/CLI-разбивки)."""
    if forecast.get("analysis_status") != ENTER:
        return None, "none_not_enter"          # WAIT / NO_TRADE / blocked
    direction = forecast.get("candidate_direction")
    if direction not in ("long", "short"):
        return None, "none_no_direction"
    if outcome is None:
        return None, "none_no_outcome"
    if not outcome.get("resolved"):
        return None, "none_unresolved"          # censored — окно не закрыто

    ref = _ref(forecast, outcome)
    if ref is None:
        return None, "none_missing_reference"
    stop = forecast.get("stop_loss")
    if stop is None:
        return None, "none_missing_stop"
    risk = abs(ref - float(stop))
    if risk == 0:
        return None, "none_zero_risk"

    sign = 1.0 if direction == "long" else -1.0
    tp1, tp2 = _levels(forecast)
    tp1_hit = bool(outcome.get("tp1_hit"))
    tp2_hit = bool(outcome.get("tp2_hit"))
    stop_hit = bool(outcome.get("stop_hit"))
    r72 = outcome.get("return_72h")

    if stop_hit and not tp1_hit:
        return -1.0, "loss"
    if tp2_hit and tp2 is not None:
        return sign * (tp2 - ref) / risk, "tp2_win"
    if tp1_hit and stop_hit:
        return 0.0, "breakeven"
    if tp1_hit and not tp2_hit and not stop_hit:
        if tp2 is None and tp1 is not None:
            return sign * (tp1 - ref) / risk, "tp1_single_win"
        mtm = _mtm(r72, ref, risk)
        return (mtm, "mark_to_market") if mtm is not None else (None, "none_no_return72h")
    # Ни TP, ни стоп (или tp2_hit без сохранённого tp2) — resolved -> MTM.
    mtm = _mtm(r72, ref, risk)
    return (mtm, "mark_to_market") if mtm is not None else (None, "none_no_return72h")


def realized_r(forecast: dict[str, Any],
               outcome: dict[str, Any] | None) -> float | None:
    """Честный per-forecast R или None (см. модульный docstring)."""
    return classify(forecast, outcome)[0]


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
# Report / CLI
# ---------------------------------------------------------------------------

def format_report(summary: dict[str, Any]) -> str:
    lines = ["=== per-forecast realized R (Stage 10, offline read-only) ==="]
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
    lines.append("")
    lines.append("[LIMITATIONS]")
    lines.append(f"  - {MID_CASE_NOTE}")
    lines.append(f"  - {REF_NOTE}")
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
    parser.add_argument("--json", action="store_true", help="вывести summary как JSON")
    args = parser.parse_args(argv)

    data = _load(args)
    summary = realized_r_summary(data.get("forecasts", []) or [],
                                 data.get("outcomes", []) or [])
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
