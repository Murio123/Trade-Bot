"""Stage 13B Step 2: read-only offline counterfactual report.

Прогоняет non-ENTER прогнозы (WAIT / NO_TRADE / blocked / journal-only) через
чистое ядро analyzer.counterfactual и агрегирует гипотетические исходы: где бот
избежал убытка (avoided_loss), а где пропустил движение (missed_opportunity), с
разбивкой по gate/типу/режиму — чтобы видеть, какие гейты слишком строгие.

Полностью offline, read-only:
  * forecasts — из локального JSON/JSONL файла (--input);
  * klines — из локального JSON/CSV файла (--klines);
  * НЕ ходит в сеть, НЕ подключается к Binance, НЕ пишет в БД, НЕ мигрирует,
    НЕ бэкфиллит, НЕ трогает Railway/env, DRY_RUN, scheduler, Telegram или
    decision-path; НЕ импортируется runtime-кодом.

ENTER-строки не оцениваются как гипотетические (у них есть фактический исход) —
они помечаются skipped_enter. Пропуски (нет направления/уровней/окна) честно
попадают в no_direction / no_levels / unresolved, а не в win/loss.

Запуск:

    python -m tools.counterfactual_report --input forecasts.jsonl --klines klines.csv
    python -m tools.counterfactual_report --input forecasts.json --klines klines.json --json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from analyzer.counterfactual import (
    EVALUATED, NO_DIRECTION, NO_LEVELS, UNRESOLVED, UNSUPPORTED,
    counterfactual_outcome,
)

ENTER = "ENTER"
SKIPPED_ENTER = "skipped_enter"
AVOIDED_LOSS = "avoided_loss"
MISSED_OPPORTUNITY = "missed_opportunity"
NONE = "none"

# Взаимоисключающие категории строки в отчёте.
CATEGORIES = (
    SKIPPED_ENTER, AVOIDED_LOSS, MISSED_OPPORTUNITY, NONE,
    NO_DIRECTION, NO_LEVELS, UNRESOLVED, UNSUPPORTED,
)

_BREAKDOWN_KEYS = ("analysis_type", "candidate_direction", "blocked_gate",
                   "market_regime", "volatility_regime")


# ---------------------------------------------------------------------------
# Нормализация строки forecast (без мутации ядра)
# ---------------------------------------------------------------------------

def _normalize_forecast(f: dict[str, Any]) -> dict[str, Any]:
    """Привести входную строку к тому, что ждёт analyzer.counterfactual:
    reference price -> executable_price_at_decision (если не задан), распарсить
    take_profit_levels. Возвращает НОВЫЙ dict — вход не мутируется."""
    out = dict(f)
    if out.get("executable_price_at_decision") is None:
        for k in ("reference_price", "entry_price", "signal_close_price"):
            if out.get(k) is not None:
                out["executable_price_at_decision"] = out[k]
                break
    tps = out.get("take_profit_levels")
    if isinstance(tps, str):
        try:
            out["take_profit_levels"] = json.loads(tps)
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------------------
# Категоризация + агрегация (чистые функции)
# ---------------------------------------------------------------------------

def categorize(forecast: dict[str, Any], df: pd.DataFrame,
               now: datetime) -> str:
    """Одна взаимоисключающая категория для строки прогноза."""
    if forecast.get("analysis_status") == ENTER:
        return SKIPPED_ENTER
    res = counterfactual_outcome(_normalize_forecast(forecast), df, now)
    if res["status"] == EVALUATED:
        return res["classification_hint"]  # avoided_loss / missed_opportunity / none
    return res["status"]                   # no_direction / no_levels / unresolved / unsupported


def _breakdown(rows: list[tuple[dict[str, Any], str]],
               key: str) -> dict[str, dict[str, int]]:
    """group_value -> {category: count}. None-группа -> 'unknown'."""
    out: dict[str, dict[str, int]] = {}
    for forecast, cat in rows:
        g = forecast.get(key)
        g = str(g) if g is not None else "unknown"
        out.setdefault(g, {})
        out[g][cat] = out[g].get(cat, 0) + 1
    return {g: dict(sorted(c.items())) for g, c in sorted(out.items())}


def summarize(forecasts: list[dict[str, Any]], df: pd.DataFrame,
              now: datetime) -> dict[str, Any]:
    rows = [(f, categorize(f, df, now)) for f in forecasts]
    counts = {c: 0 for c in CATEGORIES}
    for _, cat in rows:
        counts[cat] = counts.get(cat, 0) + 1

    evaluated = counts[AVOIDED_LOSS] + counts[MISSED_OPPORTUNITY] + counts[NONE]
    return {
        "total_rows": len(forecasts),
        "evaluated": evaluated,
        "skipped_enter": counts[SKIPPED_ENTER],
        "avoided_loss": counts[AVOIDED_LOSS],
        "missed_opportunity": counts[MISSED_OPPORTUNITY],
        "none": counts[NONE],
        "no_direction": counts[NO_DIRECTION],
        "no_levels": counts[NO_LEVELS],
        "unresolved": counts[UNRESOLVED],
        "unsupported": counts[UNSUPPORTED],
        "by_analysis_type": _breakdown(rows, "analysis_type"),
        "by_candidate_direction": _breakdown(rows, "candidate_direction"),
        "by_blocked_gate": _breakdown(rows, "blocked_gate"),
        "by_market_regime": _breakdown(rows, "market_regime"),
        "by_volatility_regime": _breakdown(rows, "volatility_regime"),
    }


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def _fmt_breakdown(bd: dict[str, dict[str, int]]) -> str:
    if not bd:
        return "    (нет данных)"
    lines = []
    for group, cats in bd.items():
        inner = ", ".join(f"{k}={v}" for k, v in cats.items())
        lines.append(f"    {group}: {inner}")
    return "\n".join(lines)


def format_report(s: dict[str, Any]) -> str:
    lines = [
        "=== Counterfactual Report (Stage 13B, offline read-only) ===",
        f"Всего строк:        {s['total_rows']}",
        f"Оценено (non-ENTER): {s['evaluated']}",
        f"  avoided_loss:       {s['avoided_loss']}",
        f"  missed_opportunity: {s['missed_opportunity']}",
        f"  none (ни то ни то): {s['none']}",
        f"ENTER пропущено:     {s['skipped_enter']}",
        f"no_direction:        {s['no_direction']}",
        f"no_levels:           {s['no_levels']}",
        f"unresolved:          {s['unresolved']}",
        f"unsupported:         {s['unsupported']}",
        "",
        "По analysis_type:",
        _fmt_breakdown(s["by_analysis_type"]),
        "По candidate_direction:",
        _fmt_breakdown(s["by_candidate_direction"]),
        "По blocked_gate:",
        _fmt_breakdown(s["by_blocked_gate"]),
        "По market_regime:",
        _fmt_breakdown(s["by_market_regime"]),
        "По volatility_regime:",
        _fmt_breakdown(s["by_volatility_regime"]),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Загрузчики (read-only, только локальные файлы)
# ---------------------------------------------------------------------------

def load_forecasts(path: str) -> list[dict[str, Any]]:
    """JSON ({forecasts:[...]} или голый список) либо JSONL (по объекту в строке)."""
    text = _read_text(path)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            raw = json.loads(text)
            if isinstance(raw, dict):
                return list(raw.get("forecasts", []))
            if isinstance(raw, list):
                return list(raw)
        except json.JSONDecodeError:
            pass  # возможно JSONL со строкой-объектом в начале
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_klines(path: str) -> pd.DataFrame:
    """Локальный JSON или CSV -> DataFrame[open_time, open, high, low, close,
    close_time]. close_time достраивается из шага open_time, если не задан."""
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raw = json.loads(_read_text(path))
        rows = raw.get("klines", raw) if isinstance(raw, dict) else raw
    return _build_klines_df(rows)


def _build_klines_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for r in rows:
        ot = _to_ts(r.get("open_time", r.get("timestamp")))
        rec = {
            "open_time": ot,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }
        if r.get("close_time") is not None:
            rec["close_time"] = _to_ts(r["close_time"])
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low",
                                     "close", "close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    if "close_time" not in df.columns or df["close_time"].isna().any():
        step = _infer_step(df["open_time"])
        df["close_time"] = df["open_time"] + step
    return df


def _infer_step(open_times: "pd.Series") -> timedelta:
    diffs = open_times.sort_values().diff().dropna()
    positive = diffs[diffs > timedelta(0)]
    if len(positive):
        return positive.min().to_pytimedelta()
    return timedelta(hours=1)


def _to_ts(value: Any) -> pd.Timestamp:
    """ISO-строка или epoch (сек/мс) -> tz-aware pd.Timestamp (UTC)."""
    if isinstance(value, (int, float)) or (
            isinstance(value, str) and value.isdigit()):
        num = int(value)
        unit = "ms" if num > 1_000_000_000_000 else "s"
        return pd.to_datetime(num, unit=unit, utc=True)
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _resolve_now(df: pd.DataFrame, override: str | None) -> datetime:
    if override:
        return _to_ts(override).to_pydatetime()
    if not df.empty:
        return df["close_time"].max().to_pydatetime()
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline counterfactual report для non-ENTER прогнозов (Stage 13B).")
    parser.add_argument("--input", required=True,
                        help="локальный JSON/JSONL файл с forecast-строками")
    parser.add_argument("--klines", required=True,
                        help="локальный JSON/CSV файл со свечами (open/high/low/close)")
    parser.add_argument("--now", help="ISO-время cutoff (по умолчанию — последняя свеча)")
    parser.add_argument("--json", action="store_true", help="вывести сводку как JSON")
    args = parser.parse_args(argv)

    forecasts = load_forecasts(args.input)
    df = load_klines(args.klines)
    now = _resolve_now(df, args.now)
    summary = summarize(forecasts, df, now)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
