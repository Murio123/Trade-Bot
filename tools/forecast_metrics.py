"""Stage 6: offline forecast outcome analytics (Option B).

Read-only measurement / analytics / reporting. НЕ меняет стратегию, сигналы,
scoring, thresholds или production-поведение; НЕ пишет в БД; НЕ импортируется
runtime-кодом. Считает метрики качества прогнозов/сигналов из УЖЕ
персистируемых данных (forecasts + forecast_outcomes + trades_journal).

Источник данных (read-only):
  * JSON-экспорт (--input FILE): полностью offline, детерминированно, для CI;
  * опционально живая БД (--database-url URL или config.DATABASE_URL): раннер
    открывает СВОЙ коннект и выполняет ТОЛЬКО SELECT. В database.py ничего не
    добавляется, ни одной пишущей операции не выполняется.

Ядро метрик — чистые функции над list[dict]; их можно тестировать без сети и
без БД. Всё, что пока нельзя посчитать честно (market/volatility regime,
calibrated_confidence, TP3, честный per-forecast R, реальный execution PnL),
явно перечислено в секции MISSING_DATA — не выдумывается.

Запуск:

    .venv/bin/python -m tools.forecast_metrics --input export.json
    .venv/bin/python -m tools.forecast_metrics --database-url "$DATABASE_URL"
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

# Модельная round-trip стоимость (из config, но без побочных импортов runtime).
TAKER_FEE_PCT = 0.05
SLIPPAGE_PCT = 0.03

ENTER, WAIT, NO_TRADE = "ENTER", "WAIT", "NO_TRADE"

CONFIDENCE_EDGES = (0.0, 0.3, 0.5, 0.7, 1.01)
SCORE_EDGES = (0.0, 5.0, 7.0, 9.0, float("inf"))
FRESHNESS_EDGES = (0.0, 300.0, 900.0, 3600.0, float("inf"))
LATENCY_EDGES = (0.0, 5.0, 30.0, 120.0, float("inf"))


# ---------------------------------------------------------------------------
# Чистые числовые помощники
# ---------------------------------------------------------------------------

def _mean(xs: Iterable[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def _median(xs: Iterable[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _pct(part: int, whole: int) -> float | None:
    return round(part / whole * 100, 2) if whole else None


def _safe_div(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _bucket_label(value: float | None, edges: tuple[float, ...]) -> str:
    if value is None:
        return "unknown"
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            hi = edges[i + 1]
            hi_s = "inf" if hi == float("inf") else f"{hi:g}"
            return f"[{edges[i]:g},{hi_s})"
    return "unknown"


# ---------------------------------------------------------------------------
# Джойн forecast + outcome
# ---------------------------------------------------------------------------

def join_records(forecasts: list[dict[str, Any]],
                 outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Слить каждый forecast с его outcome-строкой (или None) по forecast_id."""
    by_id = {o.get("forecast_id"): o for o in outcomes}
    joined = []
    for f in forecasts:
        joined.append({"forecast": f, "outcome": by_id.get(f.get("id"))})
    return joined


def _measured(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Прогнозы с направлением и наличием outcome-строки."""
    return [j for j in joined
            if j["outcome"] is not None
            and j["forecast"].get("candidate_direction") in ("long", "short")]


def _resolved(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["outcome"].get("resolved")]


def _favorable(outcome: dict[str, Any]) -> bool:
    """Исход в пользу прогноза: TP2, либо TP1 без выбитого стопа."""
    return bool(outcome.get("tp2_hit")
                or (outcome.get("tp1_hit") and not outcome.get("stop_hit")))


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def core_metrics(forecasts: list[dict[str, Any]],
                 outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    joined = join_records(forecasts, outcomes)
    total = len(forecasts)
    status = {ENTER: 0, WAIT: 0, NO_TRADE: 0}
    for f in forecasts:
        st = f.get("analysis_status")
        if st in status:
            status[st] += 1

    measured = _measured(joined)
    resolved = _resolved(measured)
    unresolved = [r for r in measured if not r["outcome"].get("resolved")]

    tp1 = sum(1 for r in resolved if r["outcome"].get("tp1_hit"))
    tp2 = sum(1 for r in resolved if r["outcome"].get("tp2_hit"))
    stop = sum(1 for r in resolved if r["outcome"].get("stop_hit"))

    # Direction accuracy: знак реализованного (sign-adjusted) return_72h.
    dir_rows = [r for r in measured if r["outcome"].get("return_72h") is not None]
    dir_correct = sum(1 for r in dir_rows if r["outcome"]["return_72h"] > 0)

    enter_resolved = [r for r in resolved
                      if r["forecast"].get("analysis_status") == ENTER]
    enter_favorable = sum(1 for r in enter_resolved if _favorable(r["outcome"]))

    mfe = [r["outcome"].get("mfe_points") for r in measured]
    mae = [r["outcome"].get("mae_points") for r in measured]
    avg_mfe, avg_mae = _mean(mfe), _mean(mae)

    reached = {}
    for col in ("reached_500", "reached_1500", "reached_3000"):
        hits = sum(1 for r in measured if r["outcome"].get(col))
        reached[col + "_rate"] = _pct(hits, len(measured))

    return {
        "total_forecasts": total,
        "counts": status,
        "coverage_enter_pct": _pct(status[ENTER], total),
        "measured": len(measured),
        "resolved": len(resolved),
        "unresolved_censored": len(unresolved),
        "unmeasured": len(joined) - len(measured),
        "tp1_hit_rate_pct": _pct(tp1, len(resolved)),
        "tp2_hit_rate_pct": _pct(tp2, len(resolved)),
        "stop_hit_rate_pct": _pct(stop, len(resolved)),
        "direction_accuracy_pct": _pct(dir_correct, len(dir_rows)),
        "direction_accuracy_n": len(dir_rows),
        "enter_precision_pct": _pct(enter_favorable, len(enter_resolved)),
        "enter_precision_n": len(enter_resolved),
        "avg_mfe_points": avg_mfe,
        "median_mfe_points": _median(mfe),
        "avg_mae_points": avg_mae,
        "median_mae_points": _median(mae),
        "mfe_mae_ratio": _safe_div(avg_mfe or 0.0, avg_mae or 0.0),
        **reached,
    }


# ---------------------------------------------------------------------------
# Trade journal metrics (чистый R — только по закрытым сделкам journal)
# ---------------------------------------------------------------------------

def journal_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [t for t in trades
              if t.get("outcome") in ("win", "loss", "breakeven")
              and t.get("pnl_r") is not None]
    total = len(closed)
    wins = sum(1 for t in closed if t["outcome"] == "win")
    losses = sum(1 for t in closed if t["outcome"] == "loss")
    be = sum(1 for t in closed if t["outcome"] == "breakeven")
    rs = [float(t["pnl_r"]) for t in closed]

    gross_win = sum(r for r in rs if r > 0)
    gross_loss = sum(-r for r in rs if r < 0)

    # Equity/drawdown строятся ТОЛЬКО из непересекающихся закрытых сделок
    # journal (упорядочены по closed_at), а НЕ из перекрывающихся per-candle
    # forecast-горизонтов — иначе double-counting и lookahead.
    ordered = sorted(closed, key=lambda t: t.get("closed_at") or "")
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for t in ordered:
        equity += float(t["pnl_r"])
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return {
        "total_trades": total,
        "wins": wins, "losses": losses, "breakeven": be,
        "winrate_pct": _pct(wins, total),
        "avg_r": _mean(rs),
        "median_r": _median(rs),
        "expectancy_r": _mean(rs),
        "profit_factor": _safe_div(gross_win, gross_loss),
        "max_drawdown_r": round(max_dd, 4),
        "final_equity_r": round(equity, 4),
        "peak_equity_r": round(peak, 4),
    }


# ---------------------------------------------------------------------------
# Bucket metrics
# ---------------------------------------------------------------------------

def _bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Компактная сводка по одному бакету (forecast+outcome строки)."""
    measured = [r for r in rows if r["outcome"] is not None
                and r["forecast"].get("candidate_direction") in ("long", "short")]
    resolved = _resolved(measured)
    fav = sum(1 for r in resolved if _favorable(r["outcome"]))
    dir_rows = [r for r in measured if r["outcome"].get("return_72h") is not None]
    dir_ok = sum(1 for r in dir_rows if r["outcome"]["return_72h"] > 0)
    return {
        "n": len(rows),
        "measured": len(measured),
        "resolved": len(resolved),
        "win_rate_pct": _pct(fav, len(resolved)),
        "direction_acc_pct": _pct(dir_ok, len(dir_rows)),
        "avg_mfe": _mean([r["outcome"].get("mfe_points") for r in measured]),
        "avg_mae": _mean([r["outcome"].get("mae_points") for r in measured]),
    }


def bucket_by(joined: list[dict[str, Any]],
              key_fn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for j in joined:
        groups.setdefault(key_fn(j["forecast"]), []).append(j)
    return {k: _bucket_summary(v) for k, v in sorted(groups.items())}


def _weekday_hour(f: dict[str, Any]) -> str:
    dt = _parse_dt(f.get("decision_time"))
    if dt is None:
        return "unknown"
    return f"{dt.strftime('%a')}-{dt.hour:02d}h"


def bucket_metrics(forecasts: list[dict[str, Any]],
                   outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    joined = join_records(forecasts, outcomes)
    blocked = [j for j in joined
               if j["forecast"].get("analysis_status") == NO_TRADE
               and j["forecast"].get("blocked_gate")]
    return {
        "by_analysis_type": bucket_by(joined, lambda f: f.get("analysis_type") or "unknown"),
        "by_timeframe": bucket_by(joined, lambda f: f.get("timeframe") or "unknown"),
        "by_direction": bucket_by(joined, lambda f: f.get("candidate_direction") or "none"),
        "by_analysis_status": bucket_by(joined, lambda f: f.get("analysis_status") or "unknown"),
        "by_confidence_bucket": bucket_by(
            joined, lambda f: _bucket_label(f.get("raw_confidence"), CONFIDENCE_EDGES)),
        "by_score_bucket": bucket_by(
            joined, lambda f: _bucket_label(_top_score(f), SCORE_EDGES)),
        "by_freshness_bucket": bucket_by(
            joined, lambda f: _bucket_label(f.get("data_freshness_seconds"), FRESHNESS_EDGES)),
        "by_weekday_hour": bucket_by(joined, _weekday_hour),
        "by_blocked_gate": {
            k: v["n"] for k, v in
            bucket_by(blocked, lambda f: f.get("blocked_gate") or "unknown").items()},
    }


def _top_score(f: dict[str, Any]) -> float | None:
    ls, ss = f.get("long_score"), f.get("short_score")
    vals = [v for v in (ls, ss) if v is not None]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_metrics(forecasts: list[dict[str, Any]],
                        outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    joined = join_records(forecasts, outcomes)
    resolved = [j for j in _resolved(_measured(joined))
                if j["forecast"].get("raw_confidence") is not None]

    buckets: dict[str, dict[str, Any]] = {}
    brier_terms: list[float] = []
    conf_sum = out_sum = 0.0
    for r in resolved:
        conf = float(r["forecast"]["raw_confidence"])
        outcome = 1.0 if _favorable(r["outcome"]) else 0.0
        brier_terms.append((conf - outcome) ** 2)
        conf_sum += conf
        out_sum += outcome
        label = _bucket_label(conf, CONFIDENCE_EDGES)
        b = buckets.setdefault(label, {"n": 0, "conf_sum": 0.0, "wins": 0})
        b["n"] += 1
        b["conf_sum"] += conf
        b["wins"] += int(outcome)

    table = {
        label: {
            "n": b["n"],
            "mean_confidence": round(b["conf_sum"] / b["n"], 4),
            "realized_win_rate_pct": _pct(b["wins"], b["n"]),
        }
        for label, b in sorted(buckets.items())
    }
    n = len(resolved)
    mean_conf = round(conf_sum / n, 4) if n else None
    realized = round(out_sum / n, 4) if n else None
    return {
        "n_resolved_with_confidence": n,
        "raw_brier_score": round(statistics.fmean(brier_terms), 4) if brier_terms else None,
        "mean_confidence": mean_conf,
        "realized_win_rate": realized,
        "overconfidence": (round(mean_conf - realized, 4)
                           if mean_conf is not None and realized is not None else None),
        "by_confidence_bucket": table,
        "note": ("raw_confidence — это base*modifiers, НЕ откалиброванная "
                 "вероятность; Brier здесь сырой, только как ориентир"),
    }


# ---------------------------------------------------------------------------
# Execution realism
# ---------------------------------------------------------------------------

def execution_metrics(forecasts: list[dict[str, Any]],
                      outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    joined = join_records(forecasts, outcomes)
    signed, abs_pct = [], []
    for f in forecasts:
        close = f.get("signal_close_price")
        execp = f.get("executable_price_at_decision")
        if close and execp:
            diff = float(execp) - float(close)
            signed.append(diff)
            abs_pct.append(abs(diff) / float(close) * 100)

    fresh = bucket_by(joined, lambda f: _bucket_label(
        f.get("data_freshness_seconds"), FRESHNESS_EDGES))
    latency = bucket_by(joined, lambda f: _bucket_label(
        f.get("decision_latency_seconds"), LATENCY_EDGES))

    net = [o.get("net_after_costs") for o in outcomes
           if o.get("net_after_costs") is not None]

    return {
        "close_vs_executable": {
            "n": len(signed),
            "avg_signed_points": _mean(signed),
            "avg_abs_slippage_pct": _mean(abs_pct),
            "median_abs_slippage_pct": _median(abs_pct),
        },
        "modeled_round_trip_cost_pct": round(2 * TAKER_FEE_PCT + SLIPPAGE_PCT, 4),
        "stale_freshness_impact": {k: {"n": v["n"], "win_rate_pct": v["win_rate_pct"]}
                                   for k, v in fresh.items()},
        "decision_latency_impact": {k: {"n": v["n"], "win_rate_pct": v["win_rate_pct"]}
                                    for k, v in latency.items()},
        "net_after_costs": {
            "n": len(net),
            "avg_pct": _mean(net),
            "median_pct": _median(net),
        },
    }


# ---------------------------------------------------------------------------
# Missing / unavailable data (честная граница анализа)
# ---------------------------------------------------------------------------

MISSING_DATA: tuple[tuple[str, str], ...] = (
    ("market_regime bucket", "market_regime не персистится в forecasts — нужна schema-extension (Stage 7/C)"),
    ("volatility_regime bucket", "режим волатильности не сохраняется — только expected_move_atr"),
    ("calibrated_confidence metrics", "forecasts.calibrated_confidence всегда NULL (будущая калибровка)"),
    ("TP3 metrics", "третий тейк нигде не хранится — только tp1/tp2"),
    ("honest per-forecast R", "forecast_outcomes хранит % за горизонт, не clean R; R доступен только в trades_journal"),
    ("full live-execution PnL", "реальные ордера/исполнение не подключены; PnL модельный (net_after_costs)"),
)


# ---------------------------------------------------------------------------
# Сборка и отчёт
# ---------------------------------------------------------------------------

def compute_all(data: dict[str, Any]) -> dict[str, Any]:
    """Единая точка: все метрики из одного экспорта (чистая функция)."""
    forecasts = data.get("forecasts", []) or []
    outcomes = data.get("outcomes", []) or []
    trades = data.get("trades", []) or []
    return {
        "core": core_metrics(forecasts, outcomes),
        "journal": journal_metrics(trades),
        "buckets": bucket_metrics(forecasts, outcomes),
        "calibration": calibration_metrics(forecasts, outcomes),
        "execution": execution_metrics(forecasts, outcomes),
        "missing_data": [{"metric": m, "reason": r} for m, r in MISSING_DATA],
    }


def _fmt_kv(d: dict[str, Any], indent: int = 2) -> list[str]:
    pad = " " * indent
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out += _fmt_kv(v, indent + 2)
        else:
            out.append(f"{pad}{k}: {v}")
    return out


def format_report(metrics: dict[str, Any]) -> str:
    lines = ["=== forecast outcome analytics (Stage 6, read-only) ==="]
    lines.append("[CORE]")
    lines += _fmt_kv(metrics["core"])
    lines.append("[TRADE JOURNAL] (clean R, только закрытые journal-сделки)")
    lines += _fmt_kv(metrics["journal"])
    lines.append("[BUCKETS]")
    lines += _fmt_kv(metrics["buckets"])
    lines.append("[CALIBRATION]")
    lines += _fmt_kv(metrics["calibration"])
    lines.append("[EXECUTION REALISM]")
    lines += _fmt_kv(metrics["execution"])
    lines.append("[MISSING / UNAVAILABLE — не считается честно]")
    for m in metrics["missing_data"]:
        lines.append(f"  - {m['metric']}: {m['reason']}")
    lines.append("")
    lines.append("NB: прошлые результаты не гарантируют будущие; учитывайте размер "
                 "выборки, censored-долю и модельные (не биржевые) издержки.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Загрузчики (read-only)
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict[str, Any]:
    """Прочитать JSON-экспорт {forecasts, outcomes, trades}."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in ("forecasts", "outcomes", "trades"):
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
    "SELECT id, analysis_type, timeframe, analysis_status, candidate_direction, "
    "blocked_gate, long_score, short_score, raw_confidence, expected_move_points, "
    "expected_move_percent, expected_move_atr, signal_close_price, "
    "executable_price_at_decision, stop_loss, take_profit_levels, tp2_source, "
    "risk_reward, data_freshness_seconds, decision_latency_seconds, decision_time "
    "FROM forecasts WHERE symbol = $1"
)
_SQL_OUTCOMES = (
    "SELECT o.forecast_id, o.mfe_points, o.mae_points, o.reached_500, o.reached_1500, "
    "o.reached_3000, o.tp1_hit, o.tp2_hit, o.stop_hit, o.return_1h, o.return_4h, "
    "o.return_12h, o.return_24h, o.return_72h, o.net_after_costs, o.resolved "
    "FROM forecast_outcomes o JOIN forecasts f ON f.id = o.forecast_id "
    "WHERE f.symbol = $1"
)
_SQL_TRADES = (
    "SELECT outcome, pnl_r, closed_at, analysis_type, timeframe, direction "
    "FROM trades_journal WHERE (symbol = $1 OR symbol IS NULL) AND outcome IS NOT NULL"
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


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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
        description="Offline read-only forecast outcome analytics (Stage 6).")
    parser.add_argument("--input", help="JSON export {forecasts, outcomes, trades}")
    parser.add_argument("--database-url", help="Postgres DSN (только SELECT)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--json", action="store_true", help="вывести метрики как JSON")
    args = parser.parse_args(argv)

    data = _load(args)
    metrics = compute_all(data)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
