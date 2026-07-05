"""Stage 15A Step 2: read-only offline lifecycle report.

Прогоняет локальный JSON-экспорт forecasts через чистое ядро
analyzer.setup_lifecycle.classify_transition и агрегирует переходы сетапов
(NEW_SETUP / CONTINUATION / UPGRADED / DOWNGRADED / INVALIDATED / EXPIRED /
RESOLVED) плюс объясняющие флаги — чтобы offline видеть, как менялись решения
движка во времени.

Полностью offline, read-only:
  * forecasts — из локального JSON-файла (--input): голый список либо
    {"forecasts": [...]};
  * НЕ ходит в сеть, НЕ подключается к БД/Binance, НЕ пишет в БД, НЕ мигрирует,
    НЕ бэкфиллит, НЕ трогает Railway/env, DRY_RUN, scheduler, pipeline,
    signal_engine, risk, AI, Telegram или decision-path;
  * НЕ импортируется runtime-кодом; единственная бизнес-зависимость —
    analyzer.setup_lifecycle (чистое ядро).

RESOLVED требует исхода предыдущего ENTER; этот CLI читает только forecasts
(без forecast_outcomes), поэтому RESOLVED здесь не возникает и остаётся 0 —
честно, а не выдумывается.

Запуск:

    python -m tools.setup_lifecycle_report --input forecasts.json
    python -m tools.setup_lifecycle_report --input forecasts.json --json --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from analyzer.setup_lifecycle import (
    STATUSES, Thresholds, classify_transition,
)


# ---------------------------------------------------------------------------
# Загрузка (read-only, только локальный файл)
# ---------------------------------------------------------------------------

class InputError(Exception):
    """Понятная ошибка входа (нет файла / битый JSON / не тот формат)."""


def load_forecasts(path: str) -> list[dict[str, Any]]:
    """Прочитать forecasts из локального JSON: голый список или
    {"forecasts": [...]}. Бросает InputError с внятным текстом."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise InputError(f"файл не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"некорректный JSON в {path}: {exc}") from exc

    if isinstance(raw, dict):
        rows = raw.get("forecasts", [])
    elif isinstance(raw, list):
        rows = raw
    else:
        raise InputError(
            "ожидался JSON-список или объект с ключом 'forecasts', "
            f"получено: {type(raw).__name__}")
    if not isinstance(rows, list):
        raise InputError("'forecasts' должен быть списком объектов")
    return rows


# ---------------------------------------------------------------------------
# Детерминированная сортировка + построение переходов (чистые функции)
# ---------------------------------------------------------------------------

def _sort_key(f: dict[str, Any]) -> tuple:
    """Порядок: symbol, analysis_type, strategy_version, context_version,
    время (created_at/signal_candle_close_time), id. None-поля не роняют
    сравнение (приводятся к сортируемым дефолтам)."""
    return (
        str(f.get("symbol") or ""),
        str(f.get("analysis_type") or ""),
        str(f.get("strategy_version") or ""),
        str(f.get("context_version") or ""),
        _anchor_epoch(f),
        (f.get("id") is None, f.get("id") if f.get("id") is not None else 0),
    )


def sort_forecasts(forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Отсортировать копию списка детерминированно (вход не мутируется)."""
    return sorted(forecasts, key=_sort_key)


def build_transitions(forecasts: list[dict[str, Any]],
                      thresholds: Thresholds) -> list[dict[str, Any]]:
    """Классифицировать каждый ряд относительно предыдущего сопоставимого в его
    потоке symbol+analysis_type. Возвращает записи переходов (чистые dict'ы)."""
    ordered = sort_forecasts(forecasts)
    last_in_stream: dict[tuple[Any, Any], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for fc in ordered:
        stream = (fc.get("symbol"), fc.get("analysis_type"))
        previous = last_in_stream.get(stream)
        res = classify_transition(previous, fc, thresholds=thresholds)
        records.append({
            "previous_id": res.previous_id,
            "current_id": fc.get("id"),
            "status": res.status,
            "reasons": list(res.reasons),
            "analysis_status": fc.get("analysis_status"),
            "final_bias": fc.get("final_bias"),
            "raw_confidence": fc.get("raw_confidence"),
            "symbol": fc.get("symbol"),
            "analysis_type": fc.get("analysis_type"),
            "_anchor": _anchor_epoch(fc),
        })
        last_in_stream[stream] = fc
    return records


# ---------------------------------------------------------------------------
# Агрегация
# ---------------------------------------------------------------------------

_RECENT_FIELDS = ("previous_id", "current_id", "status", "reasons",
                  "analysis_status", "final_bias", "raw_confidence")


def summarize(forecasts: list[dict[str, Any]], thresholds: Thresholds,
              limit: int = 20) -> dict[str, Any]:
    """Собрать offline-сводку жизненного цикла из forecasts."""
    records = build_transitions(forecasts, thresholds)

    by_status = {s: 0 for s in STATUSES}
    reason_counts: dict[str, int] = {}
    by_stream: dict[str, dict[str, Any]] = {}

    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        for reason in r["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        key = f"{r['symbol']}|{r['analysis_type']}"
        stream = by_stream.setdefault(key, {"total": 0, "by_status":
                                            {s: 0 for s in STATUSES}})
        stream["total"] += 1
        stream["by_status"][r["status"]] += 1

    top_reasons = dict(sorted(reason_counts.items(),
                              key=lambda kv: (-kv[1], kv[0])))

    # Недавние переходы — по времени убыванием, затем по current_id (стабильно).
    recent_sorted = sorted(
        records,
        key=lambda r: (r["_anchor"],
                       r["current_id"] if r["current_id"] is not None else 0),
        reverse=True,
    )
    recent = [{k: r[k] for k in _RECENT_FIELDS}
              for r in recent_sorted[:max(limit, 0)]]

    return {
        "total_forecasts": len(forecasts),
        "total_transitions": len(records),
        "by_status": by_status,
        "top_reasons": top_reasons,
        "by_stream": {k: by_stream[k] for k in sorted(by_stream)},
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def _fmt_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "    (нет данных)"
    return "\n".join(f"    {k}: {v}" for k, v in counts.items())


def format_report(s: dict[str, Any]) -> str:
    lines = [
        "=== Setup Lifecycle Report (Stage 15A, offline read-only) ===",
        f"Всего forecasts:      {s['total_forecasts']}",
        f"Всего переходов:      {s['total_transitions']}",
        "",
        "По статусу жизненного цикла:",
        _fmt_counts(s["by_status"]),
        "",
        "Топ причин изменений:",
        _fmt_counts(s["top_reasons"]),
        "",
        "По потокам (symbol|analysis_type):",
    ]
    if s["by_stream"]:
        for key, stream in s["by_stream"].items():
            active = {k: v for k, v in stream["by_status"].items() if v}
            inner = ", ".join(f"{k}={v}" for k, v in active.items()) or "—"
            lines.append(f"    {key}: total={stream['total']} ({inner})")
    else:
        lines.append("    (нет данных)")

    lines.append("")
    lines.append(f"Недавние переходы (до {len(s['recent'])}):")
    if s["recent"]:
        for r in s["recent"]:
            reasons = ", ".join(r["reasons"]) or "—"
            conf = r["raw_confidence"]
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "н/д"
            lines.append(
                f"    #{r['previous_id']}→#{r['current_id']} {r['status']} "
                f"[{r['analysis_status']}/{r['final_bias']} conf={conf_s}] "
                f"reasons: {reasons}")
    else:
        lines.append("    (нет данных)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Время (локальный парсер — без импорта приватных помощников ядра)
# ---------------------------------------------------------------------------

def _anchor_epoch(f: dict[str, Any]) -> float:
    """Секунды-эпоха якоря ряда для сортировки; отсутствующее время -> -inf
    (недатированные ряды идут первыми, детерминированно)."""
    for key in ("signal_candle_close_time", "decision_time", "created_at"):
        dt = _to_datetime(f.get(key))
        if dt is not None:
            return dt.timestamp()
    return float("-inf")


def _to_datetime(value: Any) -> datetime | None:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _thresholds(args: argparse.Namespace) -> Thresholds:
    defaults = Thresholds()
    max_gap = (args.max_gap_hours * 3600 if args.max_gap_hours is not None
               else defaults.max_gap_seconds)
    return Thresholds(
        confidence_delta=(args.confidence_delta if args.confidence_delta is not None
                          else defaults.confidence_delta),
        score_delta=(args.score_delta if args.score_delta is not None
                     else defaults.score_delta),
        max_gap_seconds=max_gap,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline setup lifecycle report над локальным JSON "
                    "forecasts (Stage 15A).")
    parser.add_argument("--input", required=True,
                        help="локальный JSON: список forecasts или {'forecasts': [...]}")
    parser.add_argument("--limit", type=int, default=20,
                        help="сколько недавних переходов показать (по умолчанию 20)")
    parser.add_argument("--json", action="store_true",
                        help="вывести сводку как JSON")
    parser.add_argument("--max-gap-hours", type=float, default=None,
                        help="окно EXPIRED в часах (Thresholds.max_gap_seconds)")
    parser.add_argument("--confidence-delta", type=float, default=None,
                        help="порог дельты confidence (Thresholds)")
    parser.add_argument("--score-delta", type=float, default=None,
                        help="порог дельты score (Thresholds)")
    args = parser.parse_args(argv)

    try:
        forecasts = load_forecasts(args.input)
    except InputError as exc:
        print(f"Ошибка входа: {exc}", file=sys.stderr)
        return 2

    summary = summarize(forecasts, _thresholds(args), limit=args.limit)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
