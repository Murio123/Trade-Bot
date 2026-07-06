"""Stage 16B: read-only offline trade pattern report (CLI over the core).

Тонкая CLI-обёртка над чистым ядром ``analyzer.trade_patterns``. Читает уже
персистируемые forecast/outcome-строки из локального JSON/JSONL файла и строит
факторный отчёт: какие факторы «токсичны» для плохих сделок, какие связаны с
пропущенными сетапами, и контраст с хорошими/избегнутыми исходами — чтобы
человек учитывал их в РУЧНОМ трейдинге. Manual analytics only.

Полностью offline, read-only:
  * строки — ТОЛЬКО из локального файла (--input): JSON-массив объектов либо
    JSONL (по объекту на строку);
  * НЕ ходит в сеть/БД/Binance, НЕ подключается к базе, НЕ пишет и НЕ мигрирует,
    НЕ бэкфиллит, НЕ трогает Railway/env, DRY_RUN, scheduler, signal_engine,
    risk, AI, Telegram или decision-path;
  * НЕ содержит exchange/API/order-кода;
  * НЕ импортируется runtime-кодом; единственная бизнес-зависимость —
    analyzer.trade_patterns (чистое ядро, здесь НЕ меняется).

Запуск:

    python -m tools.trade_patterns_report --input rows.json
    python -m tools.trade_patterns_report --input rows.jsonl --json
    python -m tools.trade_patterns_report --input rows.json --min-count 5 --limit 30
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from analyzer.trade_patterns import mine_trade_patterns


class InputError(Exception):
    """Понятная ошибка входа (нет файла / битый JSON / не тот формат)."""


# ---------------------------------------------------------------------------
# Загрузка (read-only, только локальный файл): JSON-массив или JSONL
# ---------------------------------------------------------------------------

def load_rows(path: str) -> list[dict[str, Any]]:
    """Прочитать строки из локального JSON-массива или JSONL.

    Пустые строки в JSONL пропускаются. Битый JSON -> InputError с внятным
    текстом (CLI переводит это в ненулевой код возврата).
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError as exc:
        raise InputError(f"файл не найден: {path}") from exc

    stripped = text.strip()
    if not stripped:
        return []

    # 1) Попытка разобрать как единый JSON (массив или одиночный объект).
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    else:
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        raise InputError(
            "ожидался JSON-массив объектов или JSONL, получено: "
            f"{type(parsed).__name__}")

    # 2) Иначе — JSONL: по объекту на непустую строку.
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise InputError(
                f"некорректный JSON в {path}, строка {lineno}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# Форматирование текстового отчёта
# ---------------------------------------------------------------------------

def _bad_sort_key(p: dict[str, Any]) -> tuple:
    """Сильнее токсичность раньше: выше bad_lift, больше count_bad; None-lift
    тонет; стабильный tiebreak по ключу."""
    lift = p["bad_lift"] if p["bad_lift"] is not None else float("-inf")
    return (-lift, -p["count_bad"], p["key"])


def _missed_sort_key(p: dict[str, Any]) -> tuple:
    lift = p["missed_lift"] if p["missed_lift"] is not None else float("-inf")
    return (-lift, -p["count_missed"], p["key"])


def _fmt_lift(value: float | None) -> str:
    return f"{value:.2f}x" if isinstance(value, (int, float)) else "н/д"


def _fmt_factor_line(p: dict[str, Any], *, kind: str) -> str:
    """Одна строка фактора. kind: 'bad' -> показывает bad_rate/bad_lift;
    'missed' -> missed_rate/missed_lift."""
    if kind == "bad":
        rate, lift = p["bad_rate"], p["bad_lift"]
    else:
        rate, lift = p["missed_rate"], p["missed_lift"]
    low = "  [low_sample]" if p["low_sample"] else ""
    return (
        f"    {p['dimension']}={p['value']}  "
        f"total={p['total']} "
        f"(bad={p['count_bad']} good={p['count_good']} "
        f"missed={p['count_missed']} avoided={p['count_avoided']}) "
        f"rate={rate:.2f} lift={_fmt_lift(lift)}{low}")


def format_report(result: dict[str, Any], limit: int) -> str:
    s = result["summary"]
    patterns = result["patterns"]

    lines = [
        "=== Trade Pattern Report (Stage 16B, offline read-only) ===",
        "",
        "Сводка:",
        f"    строк (rows):            {s['rows']}",
        f"    с исходом (classified):  {s['classified_rows']}",
        f"    bad_trade:               {s['bad_trade_count']}",
        f"    good_trade:              {s['good_trade_count']}",
        f"    missed_opportunity:      {s['missed_opportunity_count']}",
        f"    correct_skip (avoided):  {s['correct_skip_count']}",
        f"    insufficient_data:       {s['insufficient_data_count']}",
        "",
    ]

    n = max(limit, 0)
    bad = sorted((p for p in patterns if p["count_bad"] > 0),
                 key=_bad_sort_key)[:n]
    lines.append(f"Топ факторов плохих сделок (до {len(bad)}):")
    if bad:
        lines += [_fmt_factor_line(p, kind="bad") for p in bad]
    else:
        lines.append("    (нет данных)")
    lines.append("")

    missed = sorted((p for p in patterns if p["count_missed"] > 0),
                    key=_missed_sort_key)[:n]
    lines.append(f"Топ факторов пропущенных возможностей (до {len(missed)}):")
    if missed:
        lines += [_fmt_factor_line(p, kind="missed") for p in missed]
    else:
        lines.append("    (нет данных)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline trade pattern report над локальным JSON/JSONL "
                    "forecast/outcome-строк (Stage 16B).")
    parser.add_argument("--input", required=True,
                        help="локальный файл: JSON-массив объектов или JSONL")
    parser.add_argument("--min-count", type=int, default=3,
                        help="порог выборки: total < min_count -> low_sample")
    parser.add_argument("--limit", type=int, default=20,
                        help="сколько факторов показать в каждой секции")
    parser.add_argument("--json", action="store_true",
                        help="вывести точный результат ядра как JSON")
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.input)
    except InputError as exc:
        print(f"Ошибка входа: {exc}", file=sys.stderr)
        return 2

    result = mine_trade_patterns(rows, min_count=args.min_count)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result, args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
