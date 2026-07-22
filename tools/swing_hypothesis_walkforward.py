"""Stage C2.2c: walk-forward evaluation of the two frozen swing hypotheses.

Evaluates H1 (trend pullback continuation) and H2 (range mean reversion)
independently, using the real purge/embargo/sealed-holdout walk-forward
geometry from tools.deep_backtest (WFConfig/fold_windows/partition_setups,
unchanged), applied to the trade records tools.swing_hypothesis_simulator
already produces (also unchanged, reused, not duplicated).

НЕ торгует, НЕ отправляет ордера, НЕ ходит в сеть, НЕ пишет в БД, НЕ меняет
scoring/thresholds/config/runtime. НЕ импортируется runtime-кодом — только
ручной CLI-запуск. Отчёты пишутся в reports/c23/ (артефакты, не коммитятся).

Frozen: neither hypothesis's rules are touched here. This module only folds
and aggregates trade records tools.swing_hypothesis_simulator already
computed; it contains no entry/stop/target/eligibility logic of its own.

Sealed holdout: the underlying bar-by-bar walk (tools.swing_hypothesis_
simulator.walk_hypothesis) has no holdout concept and computes context over
the full requested `bars` window, same as tools.deep_backtest.walk_forward's
own convention. This module NEVER aggregates or reports any statistic for
idx >= holdout_lo — only the holdout's geometry (bar range, bar count) is
reported, mirroring tools.deep_backtest.walk_forward's own holdout section
exactly.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config
from tools.deep_backtest import (DeepBacktestError, EXCHANGES, WFConfig,
                                 fold_windows, max_hold_bars, partition_setups,
                                 prepare, walk_start)
from tools.regime_gate_confirmation import percentile_rank
from tools.swing_hypothesis_simulator import (current_swing_baseline,
                                              random_direction_baseline,
                                              regime_direction_baseline,
                                              summarize, walk_hypothesis)

STAGE = "C2.2c"

HOLDOUT_FRAC = 0.15
MIN_FOLDS = 3
N_RANDOM_DRAWS = 500
RANDOM_SEED_BASE = 42
MATCHED_COVERAGE_SEED_BASE = 1042

FOLD_POSITIVE_RATIO_MIN = 2 / 3
YEAR_POSITIVE_RATIO_MIN = 2 / 3   # applied as an integer comparison, not round()
MIN_YEAR_TRADES = 10
MIN_RESOLVED_TRADES = 50
SINGLE_FOLD_DOMINANCE_MAX = 0.5
MAX_TIMEOUT_SHARE = 0.9
MAX_UNRESOLVED_FRACTION = 0.05
MAX_COST_TO_GROSS_RATIO = 0.5
RANDOM_P95 = 0.95

LIMITATIONS = [
    "Confirmation only: this stage walk-forward-evaluates the two FROZEN "
    "C2.2a/C2.2b hypotheses; it does not redesign either spec or change "
    "runtime/config/score.",
    "Neither hypothesis's frozen constants were changed after seeing any "
    "result from this or any prior stage.",
    "Reuses tools.swing_hypothesis_simulator.walk_hypothesis/summarize/"
    "current_swing_baseline/regime_direction_baseline/random_direction_"
    "baseline unchanged; this module contains no simulation logic of its "
    "own, only fold geometry + aggregation.",
    "Reuses tools.deep_backtest.WFConfig/fold_windows/partition_setups "
    "unchanged for real purge/embargo/holdout geometry (same convention as "
    "C2.1's regime gate confirmation).",
    "Sealed holdout tail is never aggregated or scored — only its bar range "
    "and bar count are reported.",
    "H1 and H2 are evaluated and reported completely independently; no "
    "shared score, no blended statistic, no combined verdict.",
]


# ---------------------------------------------------------------------------
# Fold geometry (unchanged tools.deep_backtest machinery).
# ---------------------------------------------------------------------------

def build_geometry(profile: dict[str, Any], bars: int, n_entry_bars: int,
                   holdout_frac: float, min_folds: int):
    wf = WFConfig.for_profile(profile, walked_bars=bars,
                              holdout_frac=holdout_frac, min_folds=min_folds)
    span_lo = walk_start(n_entry_bars, bars)
    span_hi = n_entry_bars - 1
    folds = fold_windows(span_lo, span_hi, wf)
    return wf, folds, span_lo, span_hi


def holdout_section(span_hi: int, holdout_bars: int) -> dict[str, Any]:
    holdout_lo = span_hi - holdout_bars
    return {
        "idx_lo": holdout_lo, "idx_hi": span_hi, "n_bars": span_hi - holdout_lo,
        "sealed": True,
        "note": "geometry and bar count only; never aggregated or scored by C2.2c",
    }


# ---------------------------------------------------------------------------
# Year grouping (parses the ISO timestamp already on every trade record —
# no new field, no re-walk).
# ---------------------------------------------------------------------------

def _year_of(record: dict[str, Any]) -> int:
    return int(record["timestamp"][:4])


def year_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    taken = [r for r in records if r["eligible"] and r["outcome"] != "unresolved"]
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in taken:
        by_year[_year_of(r)].append(r)
    return {str(y): summarize(rows) for y, rows in sorted(by_year.items())}


def year_independence_check(yearly: dict[str, Any],
                            min_trades: int = MIN_YEAR_TRADES
                            ) -> tuple[bool, dict[str, Any]]:
    qualifying = [(y, s) for y, s in yearly.items()
                 if s["n"] >= min_trades and s["net_expectancy_r"] is not None]
    if len(qualifying) < 2:
        return False, {"reason": "fewer than 2 qualifying years",
                       "qualifying_years": len(qualifying)}
    positive = sum(1 for _, s in qualifying if s["net_expectancy_r"] > 0)
    # Integer comparison (positive*3 >= n*2), not round() — round(2*2/3)==1
    # would silently pass 1-of-2 years (the C2.1 bug this mirrors the fix for).
    ok = positive * 3 >= len(qualifying) * 2
    return ok, {"qualifying_years": len(qualifying), "positive_years": positive}


# ---------------------------------------------------------------------------
# Fold-level aggregation checks.
# ---------------------------------------------------------------------------

def fold_positive_ratio(per_fold: list[dict[str, Any]], min_trades: int = 1
                        ) -> tuple[float | None, int]:
    confident = [f for f in per_fold if f["n"] and f["n"] >= min_trades]
    if not confident:
        return None, 0
    positive = sum(1 for f in confident
                  if f["net_expectancy_r"] is not None and f["net_expectancy_r"] > 0)
    return positive / len(confident), len(confident)


def single_fold_dominance_ok(per_fold: list[dict[str, Any]],
                             max_fraction: float = SINGLE_FOLD_DOMINANCE_MAX
                             ) -> tuple[bool, dict[str, Any]]:
    sums = [f["sum_r"] for f in per_fold if f["sum_r"] is not None]
    total = sum(sums) if sums else 0.0
    if total <= 0 or not sums:
        return False, {"total_sum_r": total, "max_fold_sum_r": None,
                       "fraction": None}
    max_fold = max(sums)
    fraction = max_fold / total
    return fraction <= max_fraction, {"total_sum_r": round(total, 4),
                                      "max_fold_sum_r": round(max_fold, 4),
                                      "fraction": round(fraction, 4)}


# ---------------------------------------------------------------------------
# Random baselines: distributions (many seeded draws), not single points.
# ---------------------------------------------------------------------------

def random_direction_distribution(full_records: list[dict[str, Any]], entry_df,
                                  hold_bars: int, cost_pct: float,
                                  n_draws: int = N_RANDOM_DRAWS,
                                  seed_base: int = RANDOM_SEED_BASE
                                  ) -> list[float]:
    draws = []
    for i in range(n_draws):
        recs = random_direction_baseline(full_records, entry_df, hold_bars,
                                         cost_pct, seed=seed_base + i)
        s = summarize(recs)
        if s["net_expectancy_r"] is not None:
            draws.append(s["net_expectancy_r"])
    return draws


def matched_coverage_random_baseline(pool_records: list[dict[str, Any]],
                                     take_n: int,
                                     n_draws: int = N_RANDOM_DRAWS,
                                     seed_base: int = MATCHED_COVERAGE_SEED_BASE
                                     ) -> list[float]:
    """Randomly sample `take_n` trades from the broader regime-eligible pool
    (same stop/target construction, no extreme-timing filter) at H2's own
    taken-trade count — tests whether the extreme-timing condition beats
    plain random selection from the same eligible pool, not just random
    direction on its own picks (that's the B baseline)."""
    pool = [r for r in pool_records if r["eligible"] and r["outcome"] != "unresolved"]
    n_pool = len(pool)
    if n_pool == 0 or take_n <= 0 or take_n > n_pool:
        return []
    rng = np.random.RandomState(seed_base)
    draws = []
    for _ in range(n_draws):
        sample_idx = rng.choice(n_pool, size=take_n, replace=False)
        sample = [pool[j] for j in sample_idx]
        draws.append(float(np.mean([r["net_r"] for r in sample])))
    return draws


# ---------------------------------------------------------------------------
# H1 — zero/near-zero coverage classification.
# ---------------------------------------------------------------------------

def evaluate_h1(records: list[dict[str, Any]], folds, hold_bars: int
               ) -> dict[str, Any]:
    """Pooled figures are built from the UNION of fold validation windows
    only (never the full `records` list, which spans the sealed holdout and
    non-validation train-only regions too) — fold_windows() already
    subtracts holdout_bars from its region, so concatenating every fold's
    val partition can never include a holdout bar."""
    per_fold = []
    all_val: list[dict[str, Any]] = []
    for f in folds:
        _, val = partition_setups(records, f, hold_bars)
        s = summarize(val)
        per_fold.append({"fold": f.index, "val_lo": f.val_lo, "val_hi": f.val_hi,
                         "n_considered": len(val), **s})
        all_val.extend(val)

    summary = summarize(all_val)
    rc = summary["rejection_counts"]
    # Bars that reached the RR gate: passed regime/htf_bias/volatility/
    # timing/structure/reward-headroom, then were tested against RR>=1.5
    # (rr_below_floor), or passed RR and were tested against cost
    # (cost_r_exceeds_ceiling), or passed everything (taken).
    eligible_before_rr = (rc.get("rr_below_floor", 0)
                         + rc.get("cost_r_exceeds_ceiling", 0) + summary["n"])
    rr_gate_reached = (rc.get("rr_below_floor", 0) > 0
                      or rc.get("cost_r_exceeds_ceiling", 0) > 0)

    if summary["n"] > 0:
        # This stage's two-label contract presumes zero validation-region
        # coverage (per the C2.2c spec: "because H1 has zero pooled trades
        # ... classify as one of [these two]"). Nonzero coverage is outside
        # that contract — never observed on real data (0 taken pooled and
        # in every fold, confirmed in C2.2b and this stage's own real run)
        # — so fail loudly rather than silently mislabel a result the spec
        # never anticipated.
        raise ValueError(
            f"evaluate_h1's two-label classification only covers zero "
            f"validation-region coverage, but {summary['n']} trade(s) were "
            f"taken in validation folds; C2.2c's contract does not define "
            f"a verdict for nonzero H1 coverage")

    if rr_gate_reached:
        verdict = "REJECT_FROZEN_SPECIFICATION"
        reason = (f"{eligible_before_rr} bars in validation folds reached "
                 "the RR gate (passed regime/htf_bias/volatility/pullback-"
                 "timing/structure/reward-headroom checks) and 100% failed "
                 "the frozen RR>=1.5 floor or the cost ceiling; the "
                 "specification was genuinely exercised at its final gates "
                 "and never survives on this dataset, not merely absent "
                 "from it.")
    else:
        verdict = "INSUFFICIENT_EVIDENCE_DUE_TO_ZERO_COVERAGE"
        reason = ("No bar in any validation fold reached the RR gate at "
                 "all under the frozen specification; earlier gates "
                 "already eliminated every candidate, so the RR floor "
                 "itself was never tested.")

    return {
        "considered": summary["considered"],
        "eligible_before_rr_filter": eligible_before_rr,
        "rejection_counts": rc, "taken": summary["n"], "resolved": summary["n"],
        "pooled_summary": summary, "per_fold": per_fold,
        "verdict": verdict, "verdict_reason": reason,
    }


# ---------------------------------------------------------------------------
# H2 — full walk-forward + baseline comparison + stability checks.
# ---------------------------------------------------------------------------

def evaluate_h2(records: list[dict[str, Any]],
               regime_records: list[dict[str, Any]],
               baseline_setups: list[dict[str, Any]], entry_df, hold_bars: int,
               cost_pct: float, folds) -> dict[str, Any]:
    """Every pooled/baseline/distribution figure below is built from the
    UNION of fold validation windows only (never the full input lists,
    which span the sealed holdout and non-validation train-only regions
    too) — fold_windows() already subtracts holdout_bars from its region,
    so concatenating every fold's val partition can never include a
    holdout bar."""
    per_fold, baseline_per_fold, regime_per_fold = [], [], []
    all_val: list[dict[str, Any]] = []
    all_baseline_val: list[dict[str, Any]] = []
    all_regime_val: list[dict[str, Any]] = []
    for f in folds:
        _, val = partition_setups(records, f, hold_bars)
        s = summarize(val)
        per_fold.append({"fold": f.index, "train_lo": f.train_lo,
                         "train_hi": f.train_hi, "val_lo": f.val_lo,
                         "val_hi": f.val_hi, "purge_bars": f.purge_bars,
                         "embargo_bars": f.embargo_bars, **s})
        all_val.extend(val)

        _, bval = partition_setups(baseline_setups, f, hold_bars)
        b_net = (round(float(np.mean([r["r"] for r in bval])), 4)
                if bval else None)
        baseline_per_fold.append({"fold": f.index, "n": len(bval),
                                  "net_expectancy_r": b_net})
        all_baseline_val.extend(bval)

        _, rval = partition_setups(regime_records, f, hold_bars)
        regime_per_fold.append({"fold": f.index, **summarize(rval)})
        all_regime_val.extend(rval)

    pooled = summarize(all_val)
    regime_pooled = summarize(all_regime_val)

    yearly = year_stats(all_val)
    year_ok, year_detail = year_independence_check(yearly)
    fold_ratio, n_confident_folds = fold_positive_ratio(per_fold)
    dominance_ok, dominance_detail = single_fold_dominance_ok(per_fold)

    random_draws = random_direction_distribution(all_val, entry_df, hold_bars,
                                                 cost_pct)
    random_p95 = float(np.percentile(random_draws, 95)) if random_draws else None
    random_rank = (percentile_rank(pooled["net_expectancy_r"], random_draws)
                  if pooled["net_expectancy_r"] is not None else None)

    matched_draws = matched_coverage_random_baseline(all_regime_val, pooled["n"])
    matched_p95 = float(np.percentile(matched_draws, 95)) if matched_draws else None
    matched_rank = (percentile_rank(pooled["net_expectancy_r"], matched_draws)
                   if pooled["net_expectancy_r"] is not None else None)

    baseline_net = (round(float(np.mean([r["r"] for r in all_baseline_val])), 4)
                   if all_baseline_val else None)

    n_eligible_total = pooled["n"] + pooled["unresolved"]
    unresolved_fraction = (pooled["unresolved"] / n_eligible_total
                          if n_eligible_total else 0.0)

    checks = {
        "net_positive_after_costs": (pooled["net_expectancy_r"] is not None
                                     and pooled["net_expectancy_r"] > 0),
        "positive_in_2_of_3_folds_or_more": (fold_ratio is not None
                                             and fold_ratio >= FOLD_POSITIVE_RATIO_MIN),
        "not_dependent_on_one_calendar_year": year_ok,
        "sufficient_resolved_trades": pooled["n"] >= MIN_RESOLVED_TRADES,
        "beats_current_swing_baseline": (pooled["net_expectancy_r"] is not None
                                         and baseline_net is not None
                                         and pooled["net_expectancy_r"] > baseline_net),
        "beats_regime_direction_baseline": (pooled["net_expectancy_r"] is not None
                                            and regime_pooled["net_expectancy_r"] is not None
                                            and pooled["net_expectancy_r"] > regime_pooled["net_expectancy_r"]),
        "exceeds_random_direction_p95": (random_rank is not None
                                         and random_rank >= RANDOM_P95),
        "exceeds_matched_coverage_random_p95": (matched_rank is not None
                                                and matched_rank >= RANDOM_P95),
        "no_single_fold_dominance": dominance_ok,
        "timeout_share_not_severe": (pooled["timeout_share"] is not None
                                     and pooled["timeout_share"] <= MAX_TIMEOUT_SHARE),
        "unresolved_fraction_not_severe": unresolved_fraction <= MAX_UNRESOLVED_FRACTION,
        "cost_does_not_consume_most_of_gross_edge": (
            pooled["gross_expectancy_r"] is not None
            and pooled["gross_expectancy_r"] > 0
            and pooled["avg_cost_r"] is not None
            and pooled["avg_cost_r"] <= MAX_COST_TO_GROSS_RATIO * pooled["gross_expectancy_r"]),
    }
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    if all(checks.values()):
        verdict = "PROCEED_TO_REPORT_ONLY_FORWARD_TEST"
        reason = ("H2 passed every stability requirement: net-positive after "
                 "costs, positive in >=2/3 confident folds, not dependent on "
                 "a single calendar year, sufficient resolved trades, beats "
                 "the current swing baseline, beats the regime-direction "
                 "baseline, exceeds both random baselines at p95, no single "
                 "fold dominates total R, no severe timeout/unresolved "
                 "artifact, and cost does not consume most of the gross edge.")
    elif checks["net_positive_after_costs"] and passed >= total - 2:
        verdict = "NEEDS_MORE_EVIDENCE"
        reason = (f"H2 passed {passed}/{total} stability checks and remains "
                 "net-positive, but at least one required condition did not "
                 "clear its bar.")
    else:
        verdict = "REJECT_HYPOTHESIS"
        reason = (f"H2 passed only {passed}/{total} stability checks "
                 "(or turned net-negative); the extreme-timing effect does "
                 "not confirm as stable/robust enough on this evidence.")

    return {
        "pooled_summary": pooled, "per_fold": per_fold,
        "baselines": {
            "A_current_swing_baseline": {"pooled_net_expectancy_r": baseline_net,
                                         "per_fold": baseline_per_fold},
            "B_random_direction": {"n_draws": len(random_draws),
                                  "mean": (round(float(np.mean(random_draws)), 4)
                                          if random_draws else None),
                                  "std": (round(float(np.std(random_draws)), 4)
                                         if random_draws else None),
                                  "p95": (round(random_p95, 4)
                                         if random_p95 is not None else None),
                                  "actual_percentile_rank": random_rank},
            "C_regime_direction": {"pooled_summary": regime_pooled,
                                   "per_fold": regime_per_fold},
            "D_matched_coverage_random": {"n_draws": len(matched_draws),
                                         "mean": (round(float(np.mean(matched_draws)), 4)
                                                 if matched_draws else None),
                                         "std": (round(float(np.std(matched_draws)), 4)
                                                if matched_draws else None),
                                         "p95": (round(matched_p95, 4)
                                                if matched_p95 is not None else None),
                                         "actual_percentile_rank": matched_rank},
        },
        "year_stats": yearly, "year_independence": year_detail,
        "fold_positive_ratio": fold_ratio, "n_confident_folds": n_confident_folds,
        "single_fold_dominance": dominance_detail,
        "checks": checks, "verdict": verdict, "verdict_reason": reason,
    }


# ---------------------------------------------------------------------------
# Report assembly + CLI.
# ---------------------------------------------------------------------------

def run_walkforward(dataset: str, exchange: str, symbol: str,
                    profile_name: str, bars: int,
                    max_gap_ratio: float = 0.001,
                    allow_estimated_cvd: bool = False,
                    holdout_frac: float = HOLDOUT_FRAC,
                    min_folds: int = MIN_FOLDS) -> dict[str, Any]:
    frames, profile, _table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    entry_df = frames[profile["entry"]].df
    n_entry_bars = len(entry_df)
    hold_bars = max_hold_bars(profile)
    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100

    wf, folds, span_lo, span_hi = build_geometry(profile, bars, n_entry_bars,
                                                 holdout_frac, min_folds)
    holdout = holdout_section(span_hi, wf.holdout_bars)

    baseline_walk = current_swing_baseline(frames, profile, bars)
    baseline_setups = [s for s in baseline_walk["setups"]
                      if s["outcome"] != "unresolved"]

    h1_records = walk_hypothesis(frames, profile, bars, "H1", timing=True)
    h2_records = walk_hypothesis(frames, profile, bars, "H2", timing=True)
    h2_regime_records = regime_direction_baseline(frames, profile, bars, "H2")

    h1_report = evaluate_h1(h1_records, folds, hold_bars)
    h2_report = evaluate_h2(h2_records, h2_regime_records, baseline_setups,
                            entry_df, hold_bars, cost_pct, folds)

    return {
        "stage": STAGE, "kind": "swing_hypothesis_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name, "exchange": exchange, "symbol": symbol,
        "bars_requested": bars, "cvd_method": cvd_method,
        "wf_config": {"train_bars": wf.train_bars, "val_bars": wf.val_bars,
                     "step_bars": wf.step_bars, "purge_bars": wf.purge_bars,
                     "embargo_bars": wf.embargo_bars,
                     "holdout_bars": wf.holdout_bars, "min_folds": wf.min_folds,
                     "n_folds": len(folds)},
        "holdout": holdout,
        "H1": h1_report, "H2": h2_report,
        "limitations": list(LIMITATIONS),
    }


def format_h1_report(report: dict[str, Any]) -> str:
    h1 = report["H1"]
    lines = [
        f"H1 (trend pullback) walk-forward ({report['stage']}) — "
        f"{report['exchange']} {report['symbol']} profile={report['profile']}",
        f"  generated_at : {report['generated_at_utc']}",
        f"  wf_config    : {report['wf_config']}",
        f"  holdout      : {report['holdout']}",
        "",
        f"considered={h1['considered']} eligible_before_rr_filter="
        f"{h1['eligible_before_rr_filter']} taken={h1['taken']}",
        f"rejection_counts: {h1['rejection_counts']}",
        "",
        "per-fold coverage:",
    ]
    for f in h1["per_fold"]:
        lines.append(f"  fold {f['fold']}: considered={f['n_considered']} "
                     f"taken={f['n']}")
    lines.append("")
    lines.append(f"VERDICT: {h1['verdict']}")
    lines.append(f"  {h1['verdict_reason']}")
    return "\n".join(lines) + "\n"


def format_h2_report(report: dict[str, Any]) -> str:
    h2 = report["H2"]
    s = h2["pooled_summary"]
    lines = [
        f"H2 (range mean reversion) walk-forward ({report['stage']}) — "
        f"{report['exchange']} {report['symbol']} profile={report['profile']}",
        f"  generated_at : {report['generated_at_utc']}",
        f"  wf_config    : {report['wf_config']}",
        f"  holdout      : {report['holdout']}",
        "",
        f"pooled: n={s['n']} unresolved={s['unresolved']} gross={s['gross_expectancy_r']} "
        f"net={s['net_expectancy_r']} median={s['median_r']} win={s['win_rate']} "
        f"timeout={s['timeout_share']} sum_r={s['sum_r']} avg_cost_r={s['avg_cost_r']}",
        "",
        "per-fold:",
    ]
    for f in h2["per_fold"]:
        lines.append(f"  fold {f['fold']} val=[{f['val_lo']},{f['val_hi']}) "
                     f"purge={f['purge_bars']} embargo={f['embargo_bars']} "
                     f"n={f['n']} net={f['net_expectancy_r']} sum_r={f['sum_r']}")
    lines.append("")
    lines.append(f"fold_positive_ratio={h2['fold_positive_ratio']} "
                f"(n_confident_folds={h2['n_confident_folds']})")
    lines.append(f"year_independence: {h2['year_independence']}")
    lines.append(f"single_fold_dominance: {h2['single_fold_dominance']}")
    lines.append("")
    lines.append("baselines:")
    for name, b in h2["baselines"].items():
        lines.append(f"  {name}: {b}")
    lines.append("")
    lines.append("checks:")
    for k, v in h2["checks"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"VERDICT: {h2['verdict']}")
    lines.append(f"  {h2['verdict_reason']}")
    return "\n".join(lines) + "\n"


def format_decision_md(report: dict[str, Any]) -> str:
    h1, h2 = report["H1"], report["H2"]
    lines = [
        "# C2.2c — Swing Hypothesis Walk-Forward: DECISION", "",
        f"Profile: {report['profile']} · {report['exchange']} {report['symbol']} · "
        f"generated: {report['generated_at_utc']}", "",
        "Confirmation/evaluation result only — no strategy, scoring, threshold, "
        "or runtime change is proposed or implied.", "",
        f"## H1 (Trend Pullback Continuation): `{h1['verdict']}`", "",
        h1["verdict_reason"], "",
        f"## H2 (Range Mean Reversion): `{h2['verdict']}`", "",
        h2["verdict_reason"], "",
        "### H2 stability checks", "",
    ]
    for k, v in h2["checks"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Limitations", ""]
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C2.2c: walk-forward evaluation of frozen swing "
                    "hypotheses H1/H2 (offline, read-only)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bars", type=int, required=True)
    parser.add_argument("--max-gap-ratio", type=float, default=0.001)
    parser.add_argument("--allow-estimated-cvd", action="store_true")
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--min-folds", type=int, default=MIN_FOLDS)
    parser.add_argument("--outdir", default="reports/c23")
    args = parser.parse_args(argv)

    try:
        report = run_walkforward(args.dataset, args.exchange, args.symbol,
                                args.profile, args.bars, args.max_gap_ratio,
                                args.allow_estimated_cvd, args.holdout_frac,
                                args.min_folds)
    except DeepBacktestError as exc:
        import sys
        print(f"swing_hypothesis_walkforward: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    h1_only = {k: v for k, v in report.items() if k != "H2"}
    h2_only = {k: v for k, v in report.items() if k != "H1"}
    baseline_only = {"stage": report["stage"],
                     "generated_at_utc": report["generated_at_utc"],
                     "H2_baselines": report["H2"]["baselines"]}

    with open(os.path.join(args.outdir, "h1_walkforward.json"), "w") as f:
        json.dump(h1_only, f, indent=2, default=str)
    with open(os.path.join(args.outdir, "h1_walkforward.txt"), "w") as f:
        f.write(format_h1_report(report))
    with open(os.path.join(args.outdir, "h2_walkforward.json"), "w") as f:
        json.dump(h2_only, f, indent=2, default=str)
    with open(os.path.join(args.outdir, "h2_walkforward.txt"), "w") as f:
        f.write(format_h2_report(report))
    with open(os.path.join(args.outdir, "baseline_comparison.json"), "w") as f:
        json.dump(baseline_only, f, indent=2, default=str)
    with open(os.path.join(args.outdir, "DECISION.md"), "w") as f:
        f.write(format_decision_md(report))

    print(format_h1_report(report))
    print(format_h2_report(report))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
