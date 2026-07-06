"""Stage 17B: read-only offline similar-setups report (CLI over the core).

Тонкая CLI-обёртка над чистым ядром ``analyzer.similar_setups``. По «текущему»
сетапу (target) находит похожие прошлые прогнозы бота из локального JSON/JSONL и
показывает, чем они закончились: средний/медианный realized_r, лучший/худший,
разбивку win/loss, пропущенные/избегнутые и примеры. Manual analytics only.

Полностью offline, read-only:
  * история — ТОЛЬКО из локального файла (--input): JSON-массив объектов либо
    JSONL; target — из отдельного файла (--target) либо строка истории по
    (--target-id);
  * НЕ ходит в сеть/БД/Binance, НЕ подключается к базе, НЕ пишет и НЕ мигрирует,
    НЕ бэкфиллит, НЕ трогает Railway/env, DRY_RUN, scheduler, signal_engine,
    risk, AI, Telegram или decision-path;
  * НЕ содержит exchange/API/order-кода;
  * НЕ импортируется runtime-кодом; единственная бизнес-зависимость —
    analyzer.similar_setups (чистое ядро, здесь НЕ меняется).

Запуск:

    python -m tools.similar_setups_report --input history.json --target current.json
    python -m tools.similar_setups_report --input history.jsonl --target-id 42 --json
    python -m tools.similar_setups_report --input history.json --target current.json \
        --limit 20 --min-score 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from analyzer.similar_setups import find_similar_setups


class InputError(Exception):
    """Понятная ошибка входа (нет файла / битый JSON / target не найден)."""


# ---------------------------------------------------------------------------
# Загрузка (read-only, только локальные файлы): JSON-массив или JSONL
# ---------------------------------------------------------------------------

def load_rows(path: str) -> list[dict[str, Any]]:
    """Прочитать строки из локального JSON-массива или JSONL.

    Пустые строки в JSONL пропускаются. Битый JSON -> InputError.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError as exc:
        raise InputError(f"файл не найден: {path}") from exc

    stripped = text.strip()
    if not stripped:
        return []

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


def load_target(path: str) -> dict[str, Any]:
    """Прочитать target-сетап из отдельного файла: одиночный объект или первый
    объект массива/JSONL. Пустой/битый вход -> InputError."""
    rows = load_rows(path)
    if not rows:
        raise InputError(f"в файле target нет ни одного объекта: {path}")
    return rows[0]


def resolve_target(rows: list[dict[str, Any]], target_id: str) -> dict[str, Any]:
    """Найти target-строку в истории по id (сравнение как строк). Не найдено ->
    InputError."""
    for row in rows:
        if str(row.get("id")) == str(target_id):
            return row
    raise InputError(f"строка с id={target_id} не найдена в --input")


# ---------------------------------------------------------------------------
# Форматирование текстового отчёта
# ---------------------------------------------------------------------------

def _fmt_num(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "н/д"


def _fmt_factors(items: list[str]) -> str:
    return ", ".join(items) if items else "—"


def format_report(result: dict[str, Any]) -> str:
    t = result["target"]
    s = result["summary"]
    matches = result["matches"]

    lines = [
        "=== Similar Setups Report (Stage 17B, offline read-only) ===",
        "",
        "Target:",
        f"    id={t.get('id')} {t.get('symbol')} {t.get('analysis_type')} "
        f"{t.get('direction')}",
        f"    setup_type={t.get('setup_type')} regime={t.get('market_regime')} "
        f"vol={t.get('volatility_regime')} conf={t.get('confidence_bucket')} "
        f"lifecycle={t.get('setup_lifecycle_status')}",
        "",
        "Сводка по похожим:",
        f"    кандидатов:          {s['candidates']}",
        f"    похожих (matches):   {s['matches']}",
        f"    с realized_r:        {s['with_realized_r']}",
        f"    wins / losses:       {s['wins']} / {s['losses']}",
        f"    missed / avoided:    {s['missed']} / {s['avoided']}",
        f"    unresolved:          {s['unresolved']}",
        f"    win_rate:            {_fmt_num(s['win_rate'])}",
        f"    avg realized_r:      {_fmt_num(s['avg_realized_r'])}",
        f"    median realized_r:   {_fmt_num(s['median_realized_r'])}",
        f"    best / worst R:      {_fmt_num(s['best_realized_r'])} / "
        f"{_fmt_num(s['worst_realized_r'])}",
        "",
        f"Похожие сетапы (до {len(matches)}):",
    ]
    if matches:
        for m in matches:
            lines.append(
                f"    #{m.get('id')} {m.get('timestamp') or ''} "
                f"{m.get('symbol')} {m.get('analysis_type')} "
                f"{m.get('direction')} sim={m['similarity_score']:.3f} "
                f"[{m.get('outcome_label')}] R={_fmt_num(m.get('realized_r'))}")
            lines.append(
                f"        shared: {_fmt_factors(m.get('shared_factors') or [])}"
                f"  |  diff: {_fmt_factors(m.get('different_factors') or [])}")
    else:
        lines.append("    (нет данных)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline similar-setups report над локальным JSON/JSONL "
                    "истории forecast/outcome-строк (Stage 17B).")
    parser.add_argument("--input", required=True,
                        help="локальный файл истории: JSON-массив или JSONL")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target",
                       help="отдельный файл с target-сетапом (один объект)")
    group.add_argument("--target-id",
                       help="id строки истории, взять её как target")
    parser.add_argument("--limit", type=int, default=10,
                        help="сколько похожих примеров показать (по умолчанию 10)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="порог сходства 0..1 (по умолчанию 0.0)")
    parser.add_argument("--json", action="store_true",
                        help="вывести точный результат ядра как JSON")
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.input)
        if args.target is not None:
            target = load_target(args.target)
        else:
            target = resolve_target(rows, args.target_id)
    except InputError as exc:
        print(f"Ошибка входа: {exc}", file=sys.stderr)
        return 2

    result = find_similar_setups(target, rows, limit=args.limit,
                                 min_score=args.min_score)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
