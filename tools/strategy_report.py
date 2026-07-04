"""Stage 7: offline strategy evaluation report (Option B).

Read-only interpretation / reporting layer ПОВЕРХ ``tools.forecast_metrics``.
Ничего не пересчитывает по-своему: берёт готовый результат
``forecast_metrics.compute_all`` и превращает его в человекочитаемый отчёт с
warnings / insights / recommendations_for_review.

Stage 7 НЕ меняет стратегию, сигналы, thresholds, scoring, risk, Telegram,
scheduler, БД или любое production-поведение. Модуль:
  * не пишет в БД (данные читает только через read-only загрузчики
    forecast_metrics, которые выполняют ТОЛЬКО SELECT);
  * не импортируется runtime-кодом;
  * не выполняет никаких auto-actions — все рекомендации помечены «for review»
    и являются кандидатами на РУЧНОЙ разбор, без изменения конфигурации.

Все warning-правила проходят через sample-size gates: при недостаточной выборке
правило молчит, а сильные выводы (ранжирование бакетов, суждения о качестве)
блокируются флагом low_sample_size. «Прибыльность» нигде не утверждается без
sample size, drawdown и costs.

Запуск:

    .venv/bin/python -m tools.strategy_report --input export.json
    .venv/bin/python -m tools.strategy_report --input export.json --format markdown
    .venv/bin/python -m tools.strategy_report --input export.json --format json
    .venv/bin/python -m tools.strategy_report --database-url "$DATABASE_URL"
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterator

import tools.forecast_metrics as fm

# ---------------------------------------------------------------------------
# Sample-size gates (минимальные выборки, ниже которых вывод не делается)
# ---------------------------------------------------------------------------

MIN_RESOLVED = 20          # резолвнутых прогнозов для hit-rate суждений
MIN_TRADES = 15            # закрытых journal-сделок для expectancy
MIN_DIRECTION_N = 20       # прогнозов с реализованным return для direction acc
MIN_MEASURED = 20          # measured-прогнозов для MFE/MAE
MIN_CONF_N = 20            # резолвнутых с confidence для калибровки
MIN_BUCKET_RESOLVED = 10   # резолвнутых внутри бакета для суждения по бакету
MIN_TOTAL = 20             # всего прогнозов для суждения о coverage

# ---------------------------------------------------------------------------
# Эвристические пороги — ФЛАГИ ДЛЯ РЕВЬЮ, не пороги стратегии
# ---------------------------------------------------------------------------

LOW_COVERAGE_PCT = 3.0
WEAK_DIRECTION_ACC_PCT = 50.0
LOW_TP1_HIT_PCT = 40.0
HIGH_STOP_HIT_PCT = 45.0
POOR_MFE_MAE_RATIO = 1.0
OVERCONFIDENCE_DELTA = 0.10
STALE_WIN_DROP_PCT = 20.0
BAD_BUCKET_WIN_PCT = 40.0

# Дименшены с dict-сводками {n, resolved, win_rate_pct, ...} для best/worst/bad.
_RANKED_DIMENSIONS = (
    "by_analysis_type",
    "by_timeframe",
    "by_direction",
    "by_confidence_bucket",
    "by_score_bucket",
)

DISCLAIMER = (
    "Historical results do not guarantee future performance. Interpret every "
    "number together with sample size, censored share, max drawdown and modeled "
    "(not exchange) costs. This report performs no automated action."
)


# ---------------------------------------------------------------------------
# Мелкие помощники
# ---------------------------------------------------------------------------

def _warn(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _flatten_buckets(buckets: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Пройтись по ранжируемым дименшенам, отдавая (dimension, key, summary)."""
    for dim in _RANKED_DIMENSIONS:
        table = buckets.get(dim) or {}
        if not isinstance(table, dict):
            continue
        for key, summary in table.items():
            if isinstance(summary, dict):
                yield dim, key, summary


# ---------------------------------------------------------------------------
# Правила: каждое возвращает warning-dict или None, все — под sample-gate
# ---------------------------------------------------------------------------

def _rule_low_sample_size(core: dict[str, Any], journal: dict[str, Any]) -> dict[str, str] | None:
    resolved = core.get("resolved") or 0
    trades = journal.get("total_trades") or 0
    if resolved < MIN_RESOLVED or trades < MIN_TRADES:
        return _warn(
            "low_sample_size",
            f"Sample too small for strong conclusions (resolved={resolved} "
            f"< {MIN_RESOLVED} or closed trades={trades} < {MIN_TRADES}); "
            f"strong quality/ranking statements are suppressed.")
    return None


def _rule_low_coverage(core: dict[str, Any]) -> dict[str, str] | None:
    total = core.get("total_forecasts") or 0
    cov = core.get("coverage_enter_pct")
    if total >= MIN_TOTAL and cov is not None and cov < LOW_COVERAGE_PCT:
        return _warn(
            "low_coverage",
            f"ENTER coverage is low ({cov}% over {total} forecasts, "
            f"threshold {LOW_COVERAGE_PCT}%).")
    return None


def _rule_weak_direction_accuracy(core: dict[str, Any]) -> dict[str, str] | None:
    n = core.get("direction_accuracy_n") or 0
    acc = core.get("direction_accuracy_pct")
    if n >= MIN_DIRECTION_N and acc is not None and acc < WEAK_DIRECTION_ACC_PCT:
        return _warn(
            "weak_direction_accuracy",
            f"Directional accuracy {acc}% over n={n} is below "
            f"{WEAK_DIRECTION_ACC_PCT}% (near/below coin-flip).")
    return None


def _rule_low_tp_hit_rate(core: dict[str, Any]) -> dict[str, str] | None:
    resolved = core.get("resolved") or 0
    tp1 = core.get("tp1_hit_rate_pct")
    if resolved >= MIN_RESOLVED and tp1 is not None and tp1 < LOW_TP1_HIT_PCT:
        return _warn(
            "low_tp_hit_rate",
            f"TP1 hit rate {tp1}% over {resolved} resolved is below "
            f"{LOW_TP1_HIT_PCT}%.")
    return None


def _rule_high_stop_hit_rate(core: dict[str, Any]) -> dict[str, str] | None:
    resolved = core.get("resolved") or 0
    stop = core.get("stop_hit_rate_pct")
    if resolved >= MIN_RESOLVED and stop is not None and stop >= HIGH_STOP_HIT_PCT:
        return _warn(
            "high_stop_hit_rate",
            f"Stop-hit rate {stop}% over {resolved} resolved is at/above "
            f"{HIGH_STOP_HIT_PCT}%.")
    return None


def _rule_negative_expectancy(journal: dict[str, Any]) -> dict[str, str] | None:
    trades = journal.get("total_trades") or 0
    exp = journal.get("expectancy_r")
    if trades >= MIN_TRADES and exp is not None and exp < 0:
        return _warn(
            "negative_expectancy",
            f"Expectancy {exp}R over {trades} closed trades is negative "
            f"(max drawdown {journal.get('max_drawdown_r')}R).")
    return None


def _rule_poor_mfe_mae_ratio(core: dict[str, Any]) -> dict[str, str] | None:
    measured = core.get("measured") or 0
    ratio = core.get("mfe_mae_ratio")
    if measured >= MIN_MEASURED and ratio is not None and ratio < POOR_MFE_MAE_RATIO:
        return _warn(
            "poor_mfe_mae_ratio",
            f"MFE/MAE ratio {ratio} over {measured} measured is below "
            f"{POOR_MFE_MAE_RATIO} (adverse excursion dominates).")
    return None


def _rule_overconfidence(calibration: dict[str, Any]) -> dict[str, str] | None:
    n = calibration.get("n_resolved_with_confidence") or 0
    oc = calibration.get("overconfidence")
    if n >= MIN_CONF_N and oc is not None and oc > OVERCONFIDENCE_DELTA:
        return _warn(
            "overconfidence",
            f"Mean raw confidence exceeds realized win rate by {oc} over n={n} "
            f"(threshold {OVERCONFIDENCE_DELTA}); confidence looks overstated.")
    return None


def _rule_stale_freshness_impact(execution: dict[str, Any]) -> dict[str, str] | None:
    impact = execution.get("stale_freshness_impact") or {}
    fresh = impact.get("[0,300)") or {}
    fresh_wr = fresh.get("win_rate_pct")
    if fresh_wr is None:
        return None
    for label, cell in impact.items():
        if label == "[0,300)":
            continue
        n = cell.get("n") or 0
        wr = cell.get("win_rate_pct")
        if n >= MIN_BUCKET_RESOLVED and wr is not None and fresh_wr - wr >= STALE_WIN_DROP_PCT:
            return _warn(
                "stale_freshness_impact",
                f"Win rate drops from {fresh_wr}% (fresh) to {wr}% on stale "
                f"freshness bucket {label} (n={n}).")
    return None


def _rule_bad_buckets(buckets: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for dim, key, summary in _flatten_buckets(buckets):
        resolved = summary.get("resolved") or 0
        wr = summary.get("win_rate_pct")
        if resolved >= MIN_BUCKET_RESOLVED and wr is not None and wr < BAD_BUCKET_WIN_PCT:
            out.append(_warn(
                "bad_bucket",
                f"Bucket {dim}={key} has win rate {wr}% over {resolved} "
                f"resolved (below {BAD_BUCKET_WIN_PCT}%)."))
    return out


# ---------------------------------------------------------------------------
# Ранжирование бакетов (только при достаточной выборке)
# ---------------------------------------------------------------------------

def _rank_buckets(buckets: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = []
    for dim, key, summary in _flatten_buckets(buckets):
        resolved = summary.get("resolved") or 0
        wr = summary.get("win_rate_pct")
        if resolved >= MIN_BUCKET_RESOLVED and wr is not None:
            ranked.append({
                "bucket": f"{dim}={key}",
                "resolved": resolved,
                "win_rate_pct": wr,
                "avg_mfe": summary.get("avg_mfe"),
                "avg_mae": summary.get("avg_mae"),
            })
    ranked.sort(key=lambda r: r["win_rate_pct"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Рекомендации «for review» (никаких auto-actions / императивов)
# ---------------------------------------------------------------------------

_REVIEW_HINTS: dict[str, str] = {
    "low_sample_size": "Keep collecting outcomes before drawing conclusions",
    "low_coverage": "Coverage of ENTER signals",
    "weak_direction_accuracy": "Directional edge of the signal",
    "low_tp_hit_rate": "TP1 reachability of current targets",
    "high_stop_hit_rate": "Stop placement / stop-hit frequency",
    "negative_expectancy": "Overall expectancy of closed trades",
    "poor_mfe_mae_ratio": "Excursion profile (MFE vs MAE)",
    "overconfidence": "Confidence calibration",
    "stale_freshness_impact": "Impact of stale market data on outcomes",
    "bad_bucket": "Underperforming segment",
}


def _recommendations(warnings: list[dict[str, str]]) -> list[str]:
    """По каждому уникальному warning-коду — нейтральная заметка для ручного
    разбора. Никаких disable/change/apply — только «for review»."""
    seen: set[str] = set()
    recs: list[str] = []
    for w in warnings:
        code = w["code"]
        if code in seen:
            continue
        seen.add(code)
        topic = _REVIEW_HINTS.get(code, code)
        recs.append(
            f"For review (no automated action taken): {topic}. See warning "
            f"'{code}'. Investigate manually with more data before any human "
            f"decision.")
    return recs


# ---------------------------------------------------------------------------
# evaluate: чистая функция над результатом compute_all
# ---------------------------------------------------------------------------

def evaluate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Построить read-only оценку стратегии из метрик forecast_metrics.

    Возвращает {sample, warnings, insights, recommendations_for_review,
    best_buckets, worst_buckets}. Ничего не меняет и не рекомендует к
    автоматическому применению.
    """
    core = metrics.get("core") or {}
    journal = metrics.get("journal") or {}
    buckets = metrics.get("buckets") or {}
    calibration = metrics.get("calibration") or {}
    execution = metrics.get("execution") or {}

    resolved = core.get("resolved") or 0
    strong_ok = resolved >= MIN_RESOLVED

    warnings: list[dict[str, str]] = []
    for rule in (
        lambda: _rule_low_sample_size(core, journal),
        lambda: _rule_low_coverage(core),
        lambda: _rule_weak_direction_accuracy(core),
        lambda: _rule_low_tp_hit_rate(core),
        lambda: _rule_high_stop_hit_rate(core),
        lambda: _rule_negative_expectancy(journal),
        lambda: _rule_poor_mfe_mae_ratio(core),
        lambda: _rule_overconfidence(calibration),
        lambda: _rule_stale_freshness_impact(execution),
    ):
        w = rule()
        if w is not None:
            warnings.append(w)
    warnings.extend(_rule_bad_buckets(buckets))

    # Best/worst бакеты — сильный вывод, только при достаточной общей выборке.
    ranked = _rank_buckets(buckets) if strong_ok else []
    best_buckets = ranked[:3]
    worst_buckets = list(reversed(ranked[-3:])) if ranked else []

    insights: list[dict[str, str]] = []
    if not strong_ok:
        insights.append(_warn(
            "insufficient_sample",
            "Not enough resolved outcomes to rank segments or judge quality; "
            "observations below are descriptive only."))
    else:
        exp = journal.get("expectancy_r")
        if exp is not None and exp > 0:
            insights.append(_warn(
                "positive_expectancy_observed",
                f"Historically observed positive expectancy {exp}R over "
                f"{journal.get('total_trades')} trades — NOT a profitability "
                f"claim; read with max drawdown {journal.get('max_drawdown_r')}R "
                f"and modeled costs before concluding."))

    return {
        "sample": {
            "total_forecasts": core.get("total_forecasts"),
            "measured": core.get("measured"),
            "resolved": resolved,
            "unresolved_censored": core.get("unresolved_censored"),
            "total_trades": journal.get("total_trades"),
            "strong_conclusions_allowed": strong_ok,
        },
        "warnings": warnings,
        "insights": insights,
        "best_buckets": best_buckets,
        "worst_buckets": worst_buckets,
        "recommendations_for_review": _recommendations(warnings),
    }


# ---------------------------------------------------------------------------
# Форматирование отчёта
# ---------------------------------------------------------------------------

def _fmt_pair(k: str, v: Any) -> str:
    return f"{k}: {v}"


def _section_lines(metrics: dict[str, Any], report: dict[str, Any]) -> list[str]:
    core = metrics.get("core") or {}
    journal = metrics.get("journal") or {}
    calib = metrics.get("calibration") or {}
    execu = metrics.get("execution") or {}
    sample = report["sample"]

    L: list[str] = []
    L.append("## Headline")
    L.append(_fmt_pair("strong_conclusions_allowed", sample["strong_conclusions_allowed"]))
    L.append(_fmt_pair("coverage_enter_pct", core.get("coverage_enter_pct")))
    L.append(_fmt_pair("expectancy_r", journal.get("expectancy_r")))
    L.append(_fmt_pair("profit_factor", journal.get("profit_factor")))
    L.append(_fmt_pair("max_drawdown_r", journal.get("max_drawdown_r")))

    L.append("")
    L.append("## Sample size / censored share")
    L.append(_fmt_pair("total_forecasts", sample["total_forecasts"]))
    L.append(_fmt_pair("measured", sample["measured"]))
    L.append(_fmt_pair("resolved", sample["resolved"]))
    L.append(_fmt_pair("unresolved_censored", sample["unresolved_censored"]))
    L.append(_fmt_pair("total_trades", sample["total_trades"]))

    L.append("")
    L.append("## Signal quality")
    L.append(_fmt_pair("direction_accuracy_pct", core.get("direction_accuracy_pct")))
    L.append(_fmt_pair("enter_precision_pct", core.get("enter_precision_pct")))
    L.append(_fmt_pair("tp1_hit_rate_pct", core.get("tp1_hit_rate_pct")))
    L.append(_fmt_pair("tp2_hit_rate_pct", core.get("tp2_hit_rate_pct")))
    L.append(_fmt_pair("stop_hit_rate_pct", core.get("stop_hit_rate_pct")))
    L.append(_fmt_pair("mfe_mae_ratio", core.get("mfe_mae_ratio")))

    L.append("")
    L.append("## Trade journal summary (clean R)")
    L.append(_fmt_pair("winrate_pct", journal.get("winrate_pct")))
    L.append(_fmt_pair("avg_r", journal.get("avg_r")))
    L.append(_fmt_pair("profit_factor", journal.get("profit_factor")))
    L.append(_fmt_pair("max_drawdown_r", journal.get("max_drawdown_r")))
    L.append(_fmt_pair("final_equity_r", journal.get("final_equity_r")))

    L.append("")
    L.append("## Best / worst buckets (only with sufficient sample)")
    if not sample["strong_conclusions_allowed"]:
        L.append("(suppressed: insufficient resolved sample)")
    elif not report["best_buckets"]:
        L.append("(no bucket reached the minimum resolved sample)")
    else:
        for b in report["best_buckets"]:
            L.append(f"BEST  {b['bucket']}: win_rate={b['win_rate_pct']}% "
                     f"(resolved={b['resolved']})")
        for b in report["worst_buckets"]:
            L.append(f"WORST {b['bucket']}: win_rate={b['win_rate_pct']}% "
                     f"(resolved={b['resolved']})")

    L.append("")
    L.append("## Calibration / overconfidence")
    L.append(_fmt_pair("mean_confidence", calib.get("mean_confidence")))
    L.append(_fmt_pair("realized_win_rate", calib.get("realized_win_rate")))
    L.append(_fmt_pair("overconfidence", calib.get("overconfidence")))
    L.append(_fmt_pair("n_resolved_with_confidence", calib.get("n_resolved_with_confidence")))

    L.append("")
    L.append("## Execution drag")
    cve = execu.get("close_vs_executable") or {}
    L.append(_fmt_pair("avg_abs_slippage_pct", cve.get("avg_abs_slippage_pct")))
    L.append(_fmt_pair("modeled_round_trip_cost_pct", execu.get("modeled_round_trip_cost_pct")))
    nac = execu.get("net_after_costs") or {}
    L.append(_fmt_pair("net_after_costs_avg_pct", nac.get("avg_pct")))

    L.append("")
    L.append("## Warnings")
    if not report["warnings"]:
        L.append("(none fired)")
    else:
        for w in report["warnings"]:
            L.append(f"[{w['code']}] {w['message']}")

    L.append("")
    L.append("## Insights")
    if not report["insights"]:
        L.append("(none)")
    else:
        for i in report["insights"]:
            L.append(f"[{i['code']}] {i['message']}")

    L.append("")
    L.append("## Recommendations for review")
    if not report["recommendations_for_review"]:
        L.append("(none)")
    else:
        for r in report["recommendations_for_review"]:
            L.append(f"- {r}")

    L.append("")
    L.append("## Data limitations / missing data")
    for m in metrics.get("missing_data") or []:
        L.append(f"- {m['metric']}: {m['reason']}")

    L.append("")
    L.append("## Disclaimer")
    L.append(DISCLAIMER)
    return L


def format_report(metrics: dict[str, Any], report: dict[str, Any],
                  fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps({"metrics": metrics, "evaluation": report},
                          ensure_ascii=False, indent=2, default=str)

    lines = _section_lines(metrics, report)
    if fmt == "markdown":
        header = "# Strategy Evaluation Report (Stage 7, read-only)\n"
        return header + "\n".join(lines)

    # text: те же секции, но заголовки без '#'
    text_lines = ["=== Strategy Evaluation Report (Stage 7, read-only) ==="]
    for ln in lines:
        text_lines.append(ln[3:] if ln.startswith("## ") else ln)
    return "\n".join(text_lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        return fm.load_json(args.input)
    if args.database_url:
        return fm.load_db(args.database_url, args.symbol)
    return {"forecasts": [], "outcomes": [], "trades": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline read-only strategy evaluation report (Stage 7).")
    parser.add_argument("--input", help="JSON export {forecasts, outcomes, trades}")
    parser.add_argument("--database-url", help="Postgres DSN (только SELECT)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--format", choices=("text", "markdown", "json"),
                        default="text")
    args = parser.parse_args(argv)

    data = _load(args)
    metrics = fm.compute_all(data)
    report = evaluate(metrics)
    print(format_report(metrics, report, args.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
