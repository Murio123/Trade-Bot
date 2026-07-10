"""Stage B3a: offline observation report for the BOUNCE forecast stream.

Read-only measurement over data the forecast ledger ALREADY persists. This
module never writes, never touches the exchange, never imports ``database`` or
any runtime decision module, and is never imported by runtime code. It answers
the questions the bounce profile was created to answer:

  * are bounce runs happening at all, and how do they classify
    (ENTER / WAIT / NO_TRADE)?
  * when they are blocked, which gate blocked them?
  * does a bounce LONG ever survive the argmax that runs BEFORE the
    ``require_exhaustion`` HTF policy (see the limitations below)?
  * for the ENTER rows, what R was actually realised?

Nothing here re-derives a trading decision, a score, or the R formula. R is
READ from ``forecast_outcomes.realized_r`` where it was persisted by
``analyzer.outcomes``; where it was not, the row is reported as unknown rather
than recomputed from a second copy of the formula.

Data sources (read-only, mirroring the other tools):

    .venv/bin/python -m tools.bounce_observation_report --input export.json
    .venv/bin/python -m tools.bounce_observation_report --database-url "$DATABASE_URL"
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from typing import Any

BOUNCE = "BOUNCE"
ENTER = "ENTER"

# realized_r availability for an ENTER row, as mutually exclusive kinds.
R_COMPUTED = "computed"
R_NO_OUTCOME = "no_outcome"
R_NOT_PERSISTED = "not_persisted"

UNKNOWN = "unknown"

# Signed buckets for (long_score - short_score). A positive delta means the
# LONG side won the argmax; the sign is what decides whether a counter-trend
# bounce direction ever reaches the HTF policy at all.
_DELTA_BUCKETS = (
    "delta <= -3",
    "-3 < delta <= -1",
    "-1 < delta < 0",
    "delta == 0 (tie)",
    "0 < delta < 1",
    "1 <= delta < 3",
    "delta >= 3",
)

LIMITATIONS = (
    "Blocked rows usually carry NO stop_loss / take_profit_levels: the cascade "
    "returns before position sizing. Their hypothetical outcome is unmeasurable "
    "and they are counted under no_levels — never as a win or a loss.",

    "The argmax (long_score vs short_score) runs BEFORE the require_exhaustion "
    "HTF policy. The policy only ever sees the direction that already won, so a "
    "bounce LONG in a bearish regime reaches it only when long_score > short_score.",

    "Consequence: the BOUNCE stream can look like a 1H trend-follower. When "
    "short_score wins first in a bearish regime the direction is with-trend, "
    "require_exhaustion passes it through untouched, and no exhaustion check "
    "ever runs. Read by_candidate_direction together with the argmax census.",

    "The backtest cannot corroborate these rows: its per-bar context carries no "
    "funding history and no reversal candle-confirmation keys, so the exhaustion "
    "predicates fail closed there and a bounce backtest reports zero entries.",

    "Offline analytics only. Nothing here feeds scoring, gating, risk, "
    "confidence or Telegram.",
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def is_bounce(forecast: dict[str, Any]) -> bool:
    return forecast.get("analysis_type") == BOUNCE


def _tp1(forecast: dict[str, Any]) -> Any:
    tps = forecast.get("take_profit_levels") or []
    return tps[0] if len(tps) > 0 else None


def has_levels(forecast: dict[str, Any]) -> bool:
    """Enough levels to have measured a hypothetical outcome: stop + TP1."""
    return forecast.get("stop_loss") is not None and _tp1(forecast) is not None


def _number(value: Any) -> float | None:
    """A real number, or None. bool is not a score."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def delta_bucket(delta: float) -> str:
    if delta <= -3:
        return _DELTA_BUCKETS[0]
    if delta <= -1:
        return _DELTA_BUCKETS[1]
    if delta < 0:
        return _DELTA_BUCKETS[2]
    if delta == 0:
        return _DELTA_BUCKETS[3]
    if delta < 1:
        return _DELTA_BUCKETS[4]
    if delta < 3:
        return _DELTA_BUCKETS[5]
    return _DELTA_BUCKETS[6]


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _argmax_census(bounce: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of (long_score - short_score) over the bounce rows.

    Rows whose scores were not persisted (pre-scoring blocks such as
    ``stale_data``) are counted under missing_scores, not bucketed into a
    fabricated 0.
    """
    buckets: Counter[str] = Counter()
    deltas: list[float] = []
    missing = long_wins = short_wins = ties = 0

    for f in bounce:
        long_score = _number(f.get("long_score"))
        short_score = _number(f.get("short_score"))
        if long_score is None or short_score is None:
            missing += 1
            continue
        delta = long_score - short_score
        deltas.append(delta)
        buckets[delta_bucket(delta)] += 1
        if delta > 0:
            long_wins += 1
        elif delta < 0:
            short_wins += 1
        else:
            ties += 1

    return {
        "counted": len(deltas),
        "missing_scores": missing,
        "long_wins_argmax": long_wins,
        "short_wins_argmax": short_wins,
        "ties": ties,
        "average_delta": round(statistics.fmean(deltas), 4) if deltas else None,
        "median_delta": round(statistics.median(deltas), 4) if deltas else None,
        "min_delta": round(min(deltas), 4) if deltas else None,
        "max_delta": round(max(deltas), 4) if deltas else None,
        "buckets": {b: buckets[b] for b in _DELTA_BUCKETS if buckets[b]},
    }


def _realized_r(bounce: list[dict[str, Any]],
                outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Read persisted realized_r for the ENTER rows. Never recompute it."""
    by_id = {o.get("forecast_id"): o for o in outcomes}
    kinds: Counter[str] = Counter()
    values: list[float] = []

    enter = [f for f in bounce if f.get("analysis_status") == ENTER]
    for f in enter:
        outcome = by_id.get(f.get("id"))
        if outcome is None:
            kinds[R_NO_OUTCOME] += 1
            continue
        value = _number(outcome.get("realized_r"))
        if value is None:
            kinds[R_NOT_PERSISTED] += 1
            continue
        kinds[R_COMPUTED] += 1
        values.append(value)

    return {
        "enter_rows": len(enter),
        "kind_counts": dict(sorted(kinds.items())),
        "computed": len(values),
        "unknown": len(enter) - len(values),
        "average_realized_r": round(statistics.fmean(values), 4) if values else None,
        "median_realized_r": round(statistics.median(values), 4) if values else None,
        "min_realized_r": round(min(values), 4) if values else None,
        "max_realized_r": round(max(values), 4) if values else None,
        "sum_realized_r": round(sum(values), 4) if values else None,
    }


def summarize(forecasts: list[dict[str, Any]],
              outcomes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Bounce-only summary over already-persisted fields. Pure function."""
    outcomes = outcomes or []
    bounce = [f for f in forecasts if is_bounce(f)]

    blocked = [f for f in bounce if f.get("blocked_gate")]
    without_levels = [f for f in bounce if not has_levels(f)]

    return {
        "total_forecasts": len(forecasts),
        "bounce_forecasts": len(bounce),
        "by_analysis_status": _counts(f.get("analysis_status") or UNKNOWN
                                      for f in bounce),
        "by_blocked_gate": _counts(f["blocked_gate"] for f in blocked),
        "by_candidate_direction": _counts(f.get("candidate_direction") or "none"
                                          for f in bounce),
        "argmax_census": _argmax_census(bounce),
        "realized_r": _realized_r(bounce, outcomes),
        "no_levels": len(without_levels),
        "no_levels_blocked": sum(1 for f in without_levels if f.get("blocked_gate")),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _block(title: str, mapping: dict[str, Any], indent: str = "  ") -> list[str]:
    lines = [f"{indent}{title}:"]
    child = indent + "  "
    if not mapping:
        lines.append(f"{child}(none)")
        return lines
    lines.extend(f"{child}{k}: {v}" for k, v in mapping.items())
    return lines


def format_report(summary: dict[str, Any]) -> str:
    census = summary["argmax_census"]
    r = summary["realized_r"]

    lines = ["=== BOUNCE observation report (Stage B3a, offline read-only) ==="]
    lines.append(f"  total_forecasts: {summary['total_forecasts']}")
    lines.append(f"  bounce_forecasts: {summary['bounce_forecasts']}")
    lines.append(f"  no_levels: {summary['no_levels']} "
                 f"(blocked: {summary['no_levels_blocked']}) — unmeasurable, "
                 f"NOT counted as win/loss")
    lines.append("")

    lines += _block("by_analysis_status", summary["by_analysis_status"])
    lines += _block("by_blocked_gate", summary["by_blocked_gate"])
    lines += _block("by_candidate_direction", summary["by_candidate_direction"])
    lines.append("")

    lines.append("  argmax census (long_score - short_score):")
    for key in ("counted", "missing_scores", "long_wins_argmax",
                "short_wins_argmax", "ties", "average_delta", "median_delta",
                "min_delta", "max_delta"):
        lines.append(f"    {key}: {census[key]}")
    lines += _block("buckets", census["buckets"], indent="    ")
    lines.append("")

    lines.append("  realized R (ENTER rows, read from forecast_outcomes):")
    for key in ("enter_rows", "computed", "unknown", "average_realized_r",
                "median_realized_r", "min_realized_r", "max_realized_r",
                "sum_realized_r"):
        lines.append(f"    {key}: {r[key]}")
    lines += _block("kind_counts", r["kind_counts"], indent="    ")
    lines.append("")

    lines.append("[LIMITATIONS]")
    lines.extend(f"  - {note}" for note in LIMITATIONS)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loaders (read-only)
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict[str, Any]:
    """Read a JSON export {forecasts, outcomes}. Offline, deterministic."""
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


# SELECT-only. forecast_outcomes.realized_r is selected here because the shared
# export in tools/forecast_metrics does not carry it.
_SQL_FORECASTS = (
    "SELECT id, analysis_type, analysis_status, blocked_gate, "
    "candidate_direction, long_score, short_score, stop_loss, "
    "take_profit_levels "
    "FROM forecasts WHERE symbol = $1 AND analysis_type = 'BOUNCE'"
)
_SQL_OUTCOMES = (
    "SELECT o.forecast_id, o.realized_r, o.resolved "
    "FROM forecast_outcomes o JOIN forecasts f ON f.id = o.forecast_id "
    "WHERE f.symbol = $1 AND f.analysis_type = 'BOUNCE'"
)


def load_db(dsn: str, symbol: str) -> dict[str, Any]:
    """Read from Postgres over an OWN connection, SELECT only.

    Adds nothing to database.py and performs no write of any kind.
    """
    import asyncio

    import asyncpg  # local import: the JSON path needs no driver

    async def _run() -> dict[str, Any]:
        conn = await asyncpg.connect(dsn)
        try:
            forecasts = await conn.fetch(_SQL_FORECASTS, symbol)
            outcomes = await conn.fetch(_SQL_OUTCOMES, symbol)
        finally:
            await conn.close()
        return {"forecasts": [dict(r) for r in forecasts],
                "outcomes": [dict(r) for r in outcomes]}

    data = asyncio.run(_run())
    for f in data["forecasts"]:
        tps = f.get("take_profit_levels")
        if isinstance(tps, str):
            try:
                f["take_profit_levels"] = json.loads(tps)
            except (TypeError, ValueError):
                pass
    return data


def _load(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        return load_json(args.input)
    if args.database_url:
        return load_db(args.database_url, args.symbol)
    return {"forecasts": [], "outcomes": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline read-only observation report for the BOUNCE stream")
    parser.add_argument("--input", help="JSON export {forecasts, outcomes}")
    parser.add_argument("--database-url", help="Postgres DSN (SELECT only)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as JSON")
    args = parser.parse_args(argv)

    data = _load(args)
    summary = summarize(data["forecasts"], data["outcomes"])
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
