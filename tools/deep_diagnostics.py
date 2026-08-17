"""Stage C1.6: offline root-cause diagnostics над сетапами deep_backtest.

Диагноз, не тюнинг. C1.5 зафиксировал: все профили × пороги 5-10 дают
out-of-sample expectancy <= 0. Этот тул отвечает "ОТКУДА берётся минус":
раскладывает R по exit_reason / direction / regime / score-бакетам, меряет
монотонность score->R, для swing считает контрфакт таймаут-горизонта, для
bounce — подлинность контртренда, для intraday — вклад D13-рассинхрона.

Read-only measurement. НЕ торгует, НЕ отправляет ордера, НЕ ходит в сеть,
НЕ пишет в БД, НЕ меняет схему, scoring, thresholds, config или runtime.
НЕ импортируется runtime-кодом — только ручной CLI-запуск. Отчёты пишутся
в reports/c16/ (артефакты, не коммитятся). Никакого совета по порогу здесь
нет и быть не должно; disposition-ярлык — свидетельство, не торговое решение.

Вся R-семантика ПЕРЕИСПОЛЬЗУЕТСЯ из tools.deep_backtest (prepare, deep_walk,
resolve) — тул агрегирует те же числа, а не пересчитывает их по-своему.
Позиции (stop/targets) сетапы deep_walk не хранят, поэтому на время ЕДИНСТВЕННОГО
обхода resolve оборачивается рекордером (module-global вызов в deep_walk),
try/finally возвращает оригинал. deep_backtest.py не модифицируется.

Запуск:

    .venv/bin/python -m tools.deep_diagnostics \
      --dataset data/klines --exchange binance --symbol BTCUSDT \
      --profile intraday --bars 70080 --outdir reports/c16
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

import config
import tools.deep_backtest as deep_backtest
from tools.deep_backtest import DeepBacktestError, EXCHANGES, prepare
from signal_engine.profiles import PROFILES
from tools.swing_hypothesis_simulator import maybe_close_ms
from validation.funding import FundingSeries, load_funding_series
from validation.trade_costs import trade_costs, transaction_cost_r

STAGE = "C1.6"

# Бакеты счёта: нижняя граница включительно; сетапы ниже min(THRESHOLDS)=5
# отсечены гейтом below_min_threshold ещё в deep_walk.
SCORE_BUCKETS: list[tuple[int, int | None]] = [
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, None)]

# Пороги доказательности (описательные, не торговые).
MIN_GROUP_N = 50          # группа меньше — не улика, а шум
MIN_TOTAL_RESOLVED = 200  # всего resolved меньше — профилю нужен больший сэмпл
POSITIVE_EPS = 0.05       # mean_r >= этого = "в срезе есть неотрицательный сигнал"
FLAT_EPS = 0.02           # |шаг между бакетами| <= этого = "плоско"

# technical_dry_run_only НИКОГДА не назначается по данным — только вручную,
# отдельным решением (прогон plumbing без доверия сигналам и без капитала).
DISPOSITIONS = ("targeted_fix_candidate", "structural_negative",
                "needs_more_evidence", "technical_dry_run_only")

LIMITATIONS = [
    "Diagnosis only: no strategy, scoring, threshold or config change is "
    "proposed or implied; no threshold advice is emitted.",
    "Tables aggregate raw qualified setups (no cooldown filter), so counts "
    "differ from threshold_stats 'taken' trades in the C1.4 reports.",
    "R semantics are reused verbatim from tools.deep_backtest.resolve/deep_walk "
    "(cost-adjusted, conservative intrabar stop-first).",
    "Disposition labels are evidential summaries of these tables, not trading "
    "decisions; technical_dry_run_only is never auto-assigned.",
    "Single exchange / single symbol per run; unresolved setups are excluded "
    "from all R aggregates and reported as a count.",
]


# ---------------------------------------------------------------------------
# Чистые агрегаты
# ---------------------------------------------------------------------------

def resolved_only(setups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Только сетапы с исходом: unresolved не имеет R и не попадает в таблицы."""
    return [s for s in setups if s["outcome"] != "unresolved"]


def group_stats(setups: list[dict[str, Any]],
                key_fn: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, Any]]:
    """n / win_rate / mean_r / median_r / sum_r / timeout_share по группам."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in resolved_only(setups):
        groups.setdefault(key_fn(s), []).append(s)

    out: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        rows = groups[key]
        rs = [s["r"] for s in rows]
        out[key] = {
            "n": len(rows),
            "win_rate": round(sum(1 for s in rows if s["outcome"] == "win")
                              / len(rows), 4),
            "mean_r": round(sum(rs) / len(rs), 4),
            "median_r": round(statistics.median(rs), 4),
            "sum_r": round(sum(rs), 2),
            "timeout_share": round(sum(1 for s in rows
                                       if s["outcome"] == "timeout")
                                   / len(rows), 4),
        }
    return out


def score_bucket(score: float) -> str:
    for lo, hi in SCORE_BUCKETS:
        if hi is None:
            if score >= lo:
                return f"[{lo},inf)"
        elif lo <= score < hi:
            return f"[{lo},{hi})"
    return "below_min"  # недостижимо после гейта below_min_threshold


def trend_alignment(direction: str, regime: str) -> str:
    """Подлинность контртренда: с трендом, против тренда или режим range.

    Словарь режимов — как у detect_regime: trend_up / trend_down / range
    (у live-parity ещё high_volatility — не тренд, значит neutral).
    """
    if regime == "trend_up":
        return "with_trend" if direction == "long" else "counter_trend"
    if regime == "trend_down":
        return "with_trend" if direction == "short" else "counter_trend"
    return "neutral"


def score_monotonicity(bucket_table: dict[str, dict[str, Any]],
                       min_n: int = MIN_GROUP_N,
                       eps: float = FLAT_EPS) -> dict[str, Any]:
    """Растёт ли mean_r со счётом? improving/worsening/flat/mixed/insufficient.

    Учитываются только бакеты с n >= min_n (мелкие — шум). Тренд считается по
    шагам между СОСЕДНИМИ подходящими бакетами в порядке роста счёта.
    """
    ordered = [f"[{lo},{hi})" if hi is not None else f"[{lo},inf)"
               for lo, hi in SCORE_BUCKETS]
    eligible = [(b, bucket_table[b]["mean_r"]) for b in ordered
                if b in bucket_table and bucket_table[b]["n"] >= min_n]
    if len(eligible) < 2:
        return {"trend": "insufficient", "eligible_buckets": eligible,
                "min_n": min_n}

    # mean_r приходит округлённым до 4 знаков; разность двух таких чисел несёт
    # FP-шум (0.02 -> 0.020000000000000004), округление возвращает точность.
    diffs = [round(b2 - b1, 4) for (_, b1), (_, b2) in zip(eligible, eligible[1:])]
    if all(d > eps for d in diffs):
        trend = "improving"
    elif all(d < -eps for d in diffs):
        trend = "worsening"
    elif all(abs(d) <= eps for d in diffs):
        trend = "flat"
    else:
        trend = "mixed"
    return {
        "trend": trend,
        "eligible_buckets": eligible,
        "top_vs_bottom": round(eligible[-1][1] - eligible[0][1], 4),
        "min_n": min_n,
    }


def d13_summary(setups: list[dict[str, Any]]) -> dict[str, Any]:
    """Вклад D13: где regime_current расходится с live-parity тенью."""
    rows = resolved_only(setups)
    table = group_stats(rows, lambda s: "agree"
                        if s["regime_current"] == s["regime_live_parity"]
                        else "disagree")
    total_sum = round(sum(s["r"] for s in rows), 2)
    dis = table.get("disagree", {"n": 0, "sum_r": 0.0})
    agree = table.get("agree")
    return {
        "groups": table,
        "disagreement_share": round(dis["n"] / len(rows), 4) if rows else None,
        "total_sum_r": total_sum,
        "sum_r_from_disagreement": dis["sum_r"],
        "sum_r_share_from_disagreement": (
            round(dis["sum_r"] / total_sum, 4) if total_sum else None),
        "expectancy_excluding_disagreement": (
            agree["mean_r"] if agree else None),
        "note": "descriptive only; regime_live_parity is shadow-only (D13)",
    }


def classify_disposition(tables: dict[str, dict[str, dict[str, Any]]],
                         total_resolved: int) -> dict[str, Any]:
    """Ярлык по гетерогенности убытка. Свидетельство, не торговое решение.

    exit_reason в улики не входит: у win mean_r положителен по построению.
    technical_dry_run_only отсюда не возвращается никогда (ручной ярлык).
    """
    note = ("evidential label over these tables only; not a trading decision; "
            "no profile is promoted to a trading dry-run")
    if total_resolved < MIN_TOTAL_RESOLVED:
        return {"label": "needs_more_evidence",
                "reason": f"resolved setups {total_resolved} < "
                          f"{MIN_TOTAL_RESOLVED}", "note": note}

    evidence_dims = ("direction", "regime_current", "regime_agreement",
                     "score_bucket")
    eligible: list[tuple[str, str, int, float]] = []
    for dim in evidence_dims:
        for group, row in tables.get(dim, {}).items():
            if row["n"] >= MIN_GROUP_N:
                eligible.append((dim, group, row["n"], row["mean_r"]))

    if not eligible:
        return {"label": "needs_more_evidence",
                "reason": f"no group reaches n >= {MIN_GROUP_N}", "note": note}

    positive = [e for e in eligible if e[3] >= POSITIVE_EPS]
    if positive:
        best = max(positive, key=lambda e: e[3])
        return {"label": "targeted_fix_candidate",
                "reason": f"non-negative slice exists: {best[0]}={best[1]} "
                          f"(n={best[2]}, mean_r={best[3]:+.4f}); loss is "
                          f"heterogeneous", "note": note}
    if all(e[3] <= 0 for e in eligible):
        worst = min(eligible, key=lambda e: e[3])
        return {"label": "structural_negative",
                "reason": f"all {len(eligible)} eligible groups have "
                          f"mean_r <= 0 (worst {worst[0]}={worst[1]} "
                          f"mean_r={worst[3]:+.4f}); loss is uniform",
                "note": note}
    return {"label": "needs_more_evidence",
            "reason": "groups sit between 0 and the evidence threshold "
                      f"(+{POSITIVE_EPS}); neither uniform loss nor a clear "
                      "positive slice", "note": note}


# ---------------------------------------------------------------------------
# Обход с захватом позиций и контрфакт таймаут-горизонта
# ---------------------------------------------------------------------------

def walk_with_positions(frames: dict[str, Any], profile: dict[str, Any],
                        bars: int, *, funding: FundingSeries | Any
                        ) -> tuple[dict[str, Any],
                                            dict[int, dict[str, Any]]]:
    """Один deep_walk, попутно захватив pos (stop/targets) каждого сетапа.

    deep_walk вызывает resolve как module-global (deep_backtest.py:610), поэтому
    обёртка-рекордер видит каждый вызов. Оригинал возвращается в try/finally;
    сам deep_backtest.py не модифицируется.
    """
    original = deep_backtest.resolve
    positions: dict[int, dict[str, Any]] = {}

    def recording_resolve(df: pd.DataFrame, entry_idx: int, direction: str,
                          pos: dict[str, Any], hold_bars: int) -> dict[str, Any]:
        positions[entry_idx] = dict(pos)
        return original(df, entry_idx, direction, pos, hold_bars)

    deep_backtest.resolve = recording_resolve
    try:
        walk = deep_backtest.deep_walk(frames, profile, bars, funding=funding)
    finally:
        deep_backtest.resolve = original
    return walk, positions


def _cost_r(pos: dict[str, Any]) -> float:
    """Транзакционные издержки в R — канонический validation.trade_costs.

    P1: funding сюда НЕ входит, потому что здесь он был бы посчитан по
    контрфактическому горизонту; его начисляет _counterfactual_net_r, где
    известен новый exit_idx.
    """
    price = pos["entry_price"]
    risk_dist = abs(price - pos["stop_loss"])
    return transaction_cost_r(price, risk_dist)


def timeout_counterfactual(entry_df: pd.DataFrame,
                           setups: list[dict[str, Any]],
                           positions: dict[int, dict[str, Any]],
                           hold_bars: int,
                           multipliers: tuple[int, ...] = (2, 4),
                           *, funding: FundingSeries | Any) -> dict[str, Any]:
    """Что стало бы с timeout-сетапами при 2x/4x горизонте. Только измерение.

    Горизонт профиля НЕ меняется — мы пере-решаем те же входы тем же resolve
    на удлинённом hold_bars и считаем конверсии и дельту mean_r. Ставшие
    unresolved (история кончилась) из дельты исключаются и считаются отдельно.

    P1: удлинённый горизонт держит позицию через БОЛЬШЕ сеттлментов, поэтому
    funding пересчитывается на новом интервале. Оставить здесь funding
    базового горизонта означало бы бесплатное удержание — ровно тот перекос,
    который делает «подержать подольше» выгодным на бумаге.
    """
    entry_close_ms = maybe_close_ms(entry_df, funding)
    timeouts = [s for s in setups if s["outcome"] == "timeout"]
    result: dict[str, Any] = {
        "n_timeouts": len(timeouts),
        "baseline_hold_bars": hold_bars,
        "hit_tp1_share": (round(sum(1 for s in timeouts if s["hit_tp1"])
                                / len(timeouts), 4) if timeouts else None),
        "baseline_mean_r": (round(sum(s["r"] for s in timeouts)
                                  / len(timeouts), 4) if timeouts else None),
        "horizons": {},
        "note": "counterfactual measurement only; profile horizon unchanged",
    }
    for m in multipliers:
        extended = hold_bars * m
        conv = {"timeout_to_win": 0, "timeout_to_loss": 0,
                "timeout_to_breakeven": 0, "still_timeout": 0,
                "became_unresolved": 0}
        deltas: list[float] = []
        for s in timeouts:
            pos = positions[s["idx"]]
            new = deep_backtest.resolve(entry_df, s["idx"], s["direction"],
                                        pos, extended)
            if new["outcome"] == "unresolved":
                conv["became_unresolved"] += 1
                continue
            key = ("still_timeout" if new["outcome"] == "timeout"
                   else f"timeout_to_{new['outcome']}")
            conv[key] += 1
            if entry_close_ms is None:
                new_r = round(new["r"] - _cost_r(pos), 2)
            else:
                costs = trade_costs(
                    entry_price=pos["entry_price"],
                    risk_distance=abs(pos["entry_price"] - pos["stop_loss"]),
                    side=s["direction"],
                    entry_time=int(entry_close_ms[s["idx"]]),
                    exit_time=int(entry_close_ms[new["exit_idx"]]),
                    funding=funding, gross_r=new["r"])
                new_r = round(costs.net_r, 2)
            deltas.append(new_r - s["r"])
        result["horizons"][f"x{m}"] = {
            "hold_bars": extended,
            "conversions": conv,
            "delta_mean_r": (round(sum(deltas) / len(deltas), 4)
                             if deltas else None),
            "n_re_resolved": len(deltas),
        }
    return result


# ---------------------------------------------------------------------------
# Сборка отчёта
# ---------------------------------------------------------------------------

def build_tables(setups: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "exit_reason": group_stats(setups, lambda s: s["outcome"]),
        "direction": group_stats(setups, lambda s: s["direction"]),
        "regime_current": group_stats(setups, lambda s: s["regime_current"]),
        "regime_live_parity": group_stats(
            setups, lambda s: s["regime_live_parity"]),
        "regime_agreement": group_stats(
            setups, lambda s: "agree"
            if s["regime_current"] == s["regime_live_parity"] else "disagree"),
        "score_bucket": group_stats(setups, lambda s: score_bucket(s["score"])),
        "direction_x_regime": group_stats(
            setups, lambda s: f"{s['direction']}|{s['regime_current']}"),
        "trend_alignment": group_stats(
            setups, lambda s: trend_alignment(s["direction"],
                                              s["regime_current"])),
    }


def run_diagnostics(dataset: str, exchange: str, symbol: str,
                    profile_name: str, bars: int,
                    max_gap_ratio: float = 0.001,
                    allow_estimated_cvd: bool = False,
                    funding_dir: str = "data/funding") -> dict[str, Any]:
    frames, profile, _table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    # P1: real funding over the actual holding interval; missing data raises.
    funding = load_funding_series(funding_dir, exchange=exchange, symbol=symbol)
    walk, positions = walk_with_positions(frames, profile, bars,
                                          funding=funding)
    setups = walk["setups"]
    resolved = resolved_only(setups)

    tables = build_tables(setups)
    report: dict[str, Any] = {
        "stage": STAGE,
        "kind": "deep_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "exchange": exchange,
        "symbol": symbol,
        "bars_requested": bars,
        "cvd_method": cvd_method,
        "max_hold_bars": walk["max_hold_bars"],
        "counts": {"setups": len(setups), "resolved": len(resolved),
                   "unresolved": len(setups) - len(resolved)},
        "tables": tables,
        "score_monotonicity": score_monotonicity(tables["score_bucket"]),
        "d13": d13_summary(setups),
        "timeout_counterfactual": None,
        "disposition": classify_disposition(tables, len(resolved)),
        "limitations": list(LIMITATIONS),
    }
    if profile_name == "swing":
        entry_df = frames[profile["entry"]].df
        report["timeout_counterfactual"] = timeout_counterfactual(
            entry_df, setups, positions, walk["max_hold_bars"],
            funding=funding)
    report["funding_provenance"] = funding.provenance()
    report["funding"] = walk["funding"]
    return report


# ---------------------------------------------------------------------------
# Текстовый отчёт и CLI
# ---------------------------------------------------------------------------

_COLS = f"{'n':>6} {'win_rate':>8} {'mean_r':>8} {'median_r':>8} " \
        f"{'sum_r':>9} {'timeout':>8}"


def _table_lines(title: str, table: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"{title}:", f"  {'group':<18}{_COLS}"]
    for key, row in table.items():
        lines.append(f"  {key:<18}{row['n']:>6} {row['win_rate']:>8.4f} "
                     f"{row['mean_r']:>+8.4f} {row['median_r']:>+8.4f} "
                     f"{row['sum_r']:>+9.2f} {row['timeout_share']:>8.4f}")
    return lines


def format_report(report: dict[str, Any]) -> str:
    c = report["counts"]
    lines = [
        f"deep diagnostics ({report['stage']}) — {report['exchange']} "
        f"{report['symbol']} profile={report['profile']}",
        f"  generated_at   : {report['generated_at_utc']}",
        f"  bars_requested : {report['bars_requested']}",
        f"  cvd_method     : {report['cvd_method']}",
        f"  max_hold_bars  : {report['max_hold_bars']}",
        f"  setups         : {c['setups']} (resolved {c['resolved']}, "
        f"unresolved {c['unresolved']})",
        "",
    ]
    order = ["exit_reason", "direction", "regime_current", "regime_live_parity",
             "regime_agreement", "score_bucket", "direction_x_regime",
             "trend_alignment"]
    for name in order:
        lines.extend(_table_lines(name, report["tables"][name]))
        lines.append("")

    mono = report["score_monotonicity"]
    lines.append(f"score->R monotonicity: {mono['trend']} "
                 f"(eligible buckets n>={mono['min_n']}: "
                 f"{mono['eligible_buckets']}"
                 + (f", top_vs_bottom={mono['top_vs_bottom']:+.4f}"
                    if "top_vs_bottom" in mono else "") + ")")

    d13 = report["d13"]
    lines.append(f"D13: disagreement_share={d13['disagreement_share']} "
                 f"sum_r_from_disagreement={d13['sum_r_from_disagreement']} "
                 f"of total {d13['total_sum_r']} "
                 f"(expectancy excl. disagreement: "
                 f"{d13['expectancy_excluding_disagreement']})")

    cf = report["timeout_counterfactual"]
    if cf is not None:
        lines.append("")
        lines.append(f"timeout counterfactual (baseline hold="
                     f"{cf['baseline_hold_bars']}, n_timeouts="
                     f"{cf['n_timeouts']}, hit_tp1_share="
                     f"{cf['hit_tp1_share']}, baseline_mean_r="
                     f"{cf['baseline_mean_r']}):")
        for hz, row in cf["horizons"].items():
            lines.append(f"  {hz} (hold={row['hold_bars']}): "
                         f"{row['conversions']} delta_mean_r="
                         f"{row['delta_mean_r']} "
                         f"n_re_resolved={row['n_re_resolved']}")

    disp = report["disposition"]
    lines.extend(["",
                  f"disposition: {disp['label']}",
                  f"  reason: {disp['reason']}",
                  f"  note  : {disp['note']}",
                  "", "limitations:"])
    lines.extend(f"  - {item}" for item in report["limitations"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.deep_diagnostics",
        description="offline root-cause diagnostics over deep_backtest setups "
                    "(Stage C1.6). Reads cached files only; never touches the "
                    "network, the DB or the runtime.")
    parser.add_argument("--dataset", required=True,
                        help="directory written by tools.kline_cache")
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--bars", type=int, required=True,
                        help="entry-timeframe bars to walk")
    parser.add_argument("--max-gap-ratio", type=float, default=0.001)
    parser.add_argument("--allow-estimated-cvd", action="store_true")
    parser.add_argument("--outdir", default="reports/c16",
                        help="artifact directory for the JSON/text reports")
    parser.add_argument("--json", action="store_true",
                        help="print JSON to stdout instead of the text report")
    args = parser.parse_args(argv)

    if args.bars <= 0:
        parser.error("--bars must be positive")

    try:
        report = run_diagnostics(args.dataset, args.exchange, args.symbol,
                                 args.profile, args.bars, args.max_gap_ratio,
                                 args.allow_estimated_cvd)
    except DeepBacktestError as exc:
        print(f"deep_diagnostics: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir,
                        f"{args.profile}_{args.exchange}_diagnostics")
    with open(f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    text = format_report(report)
    with open(f"{stem}.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json
          else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
