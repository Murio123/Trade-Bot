"""Stage C2.1: Swing Regime Gate Confirmation.

Offline, read-only confirmation test of ONE frozen deterministic gate
discovered in C2a (reports/c20/archetype_discovery.md). This is NOT a new
strategy and NOT a score change: it tests whether excluding one discovered
market context (regime_current == "trend_up", optionally reinforced by
htf_bias == "bullish") improves the existing swing baseline's net expectancy.

НЕ торгует, НЕ отправляет ордера, НЕ ходит в сеть, НЕ пишет в БД, НЕ меняет
scoring/thresholds/config/runtime. НЕ импортируется runtime-кодом — только
ручной CLI-запуск. Отчёты пишутся в reports/c21/ (артефакты, не коммитятся).

Recorder reuse: this module does not modify tools/deep_backtest.py or
tools/deep_discovery.py. It calls tools.deep_discovery.record_walk (already
tested, recorder-only) to obtain regime_current/htf_bias per setup, and
tools.deep_backtest.WFConfig/fold_windows/partition_setups (already tested,
C1.3d) unchanged for fold geometry with purge/embargo/sealed holdout.

Frozen gate (fixed BEFORE this module's evaluation code was run against real
data — see reports/c20/archetype_discovery.md §2-3 for the cluster evidence
this translates):

  range-context cluster (C2a):  regime_current mostly "range"/"trend_down",
                                 htf_bias mostly "neutral"; net ~+0.10R.
  trend-up-context cluster:     regime_current 73.6% "trend_up", htf_bias
                                 73.3% "bullish" (near-perfectly correlated
                                 with regime_current in this dataset); net
                                 ~-0.18R, negative every calendar year.

  Variant B ("range_gate" — the frozen gate under test): REJECT (no trade)
  if regime_current == "trend_up" OR htf_bias == "bullish"; ALLOW otherwise.
  Variant C ("trend_up_exclusion_only" — simplest baseline): REJECT if
  regime_current == "trend_up"; ALLOW otherwise.
  Variant A ("baseline"): no gate, current swing setups unfiltered.

Both B and C use only regime_current/htf_bias — fields already computed live
today (tools/deep_backtest.get_htf_bias, regime_current from deep_walk) — no
new indicator, no numeric threshold invented, no CVD as a trigger. The rule
is frozen: it is not adjusted after seeing evaluation results (Phase 2/3
below only measure it, never retune it).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from tools.deep_backtest import (DeepBacktestError, EXCHANGES, WFConfig,
                                 fold_windows, partition_setups, prepare,
                                 walk_start)
from tools.deep_discovery import build_feature_rows, record_walk
from validation.funding import load_funding_series

STAGE = "C2.1"

GATE_VARIANTS = ("A_baseline", "B_range_gate", "C_trend_up_exclusion_only")

# Walk-forward geometry defaults for this confirmation test. holdout_frac > 0
# seals a real tail; this module never reads performance stats from it (see
# _holdout_section, mirrors tools.deep_backtest.walk_forward's own convention).
HOLDOUT_FRAC = 0.15
MIN_FOLDS = 3

N_RANDOM_DRAWS = 500
RANDOM_SEED = 42


def gate_allows(variant: str, regime_current: str | None,
                htf_bias: str | None) -> bool:
    """Pure, frozen routing rule. Only pre-entry fields; no look-ahead."""
    if variant == "A_baseline":
        return True
    if variant == "C_trend_up_exclusion_only":
        return regime_current != "trend_up"
    if variant == "B_range_gate":
        if regime_current == "trend_up":
            return False
        if htf_bias == "bullish":
            return False
        return True
    raise ValueError(f"unknown gate variant: {variant!r}")


# ---------------------------------------------------------------------------
# Dataset assembly (reuses deep_discovery.record_walk unchanged).
# ---------------------------------------------------------------------------

def build_rows(dataset: str, exchange: str, symbol: str, profile_name: str,
              bars: int, max_gap_ratio: float = 0.001,
              allow_estimated_cvd: bool = False,
              funding_dir: str = "data/funding"
              ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any],
                        int, str]:
    """Return (rows, walk, profile, n_entry_bars, cvd_method).

    rows: one dict per RESOLVED setup (build_feature_rows semantics — same
    convention as C1.8: unresolved setups excluded from all R aggregates).
    walk: the underlying deep_walk output (setups incl. unresolved, needed
    for fold-geometry bar counts).
    n_entry_bars: len(entry_df) — same quantity tools.deep_backtest.
    walk_forward uses for span_lo/span_hi, needed since walk_forward()'s own
    span computation is not exposed as a standalone helper.
    """
    frames, profile, _table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    # P1: real funding over the actual holding interval; missing data raises.
    funding = load_funding_series(funding_dir, exchange=exchange, symbol=symbol)
    walk, records = record_walk(frames, profile, bars, funding=funding)
    rows = build_feature_rows(walk, records)
    n_entry_bars = len(frames[profile["entry"]].df)
    return rows, walk, profile, n_entry_bars, cvd_method


# ---------------------------------------------------------------------------
# Full-sample statistics.
# ---------------------------------------------------------------------------

def _max_losing_streak(rows: list[dict[str, Any]]) -> int:
    streak = worst = 0
    for r in sorted(rows, key=lambda r: r["idx"]):
        if r["r"] < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def variant_stats(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    kept = [r for r in rows if gate_allows(variant, r.get("regime_current"),
                                           r.get("htf_bias"))]
    n_total = len(rows)
    n = len(kept)
    if n == 0:
        return {
            "n": 0, "coverage": 0.0, "rejection_rate": 1.0,
            "gross_expectancy_r": None, "net_expectancy_r": None,
            "median_r": None, "win_rate": None, "timeout_share": None,
            "sum_r": None, "avg_cost_r": None, "max_losing_streak": None,
        }
    net_rs = [r["r"] for r in kept]
    gross_rs = [r["gross_r"] for r in kept if r.get("gross_r") is not None]
    cost_rs = [r["cost_r"] for r in kept if r.get("cost_r") is not None]
    wins = sum(1 for r in kept if r["outcome"] == "win")
    timeouts = sum(1 for r in kept if r["outcome"] == "timeout")
    return {
        "n": n,
        "coverage": round(n / n_total, 4) if n_total else None,
        "rejection_rate": round(1 - n / n_total, 4) if n_total else None,
        "gross_expectancy_r": (round(float(np.mean(gross_rs)), 4)
                               if gross_rs else None),
        "net_expectancy_r": round(float(np.mean(net_rs)), 4),
        "median_r": round(float(np.median(net_rs)), 4),
        "win_rate": round(wins / n, 4),
        "timeout_share": round(timeouts / n, 4),
        "sum_r": round(float(np.sum(net_rs)), 4),
        "avg_cost_r": round(float(np.mean(cost_rs)), 4) if cost_rs else None,
        "max_losing_streak": _max_losing_streak(kept),
    }


def calendar_year_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r)
    out: dict[str, Any] = {}
    for variant in GATE_VARIANTS:
        out[variant] = {
            str(y): variant_stats(yr_rows, variant)
            for y, yr_rows in sorted(by_year.items())
        }
    return out


# ---------------------------------------------------------------------------
# Walk-forward evaluation (real C1.3d geometry: purge + embargo + holdout).
# ---------------------------------------------------------------------------

def run_walk_forward(rows: list[dict[str, Any]], walk: dict[str, Any],
                     profile: dict[str, Any], bars: int, n_entry_bars: int,
                     holdout_frac: float = HOLDOUT_FRAC,
                     min_folds: int = MIN_FOLDS) -> dict[str, Any]:
    """Fold rows (resolved setups only) using the SAME fold geometry as
    tools.deep_backtest.walk_forward — purge/embargo computed from the
    profile, sealed holdout carved out and never scored.
    """
    hold_bars = walk["max_hold_bars"]
    span_lo = walk_start(n_entry_bars, bars)
    span_hi = n_entry_bars - 1

    wf = WFConfig.for_profile(profile, walked_bars=bars,
                              holdout_frac=holdout_frac, min_folds=min_folds)
    folds = fold_windows(span_lo, span_hi, wf)

    holdout_lo = span_hi - wf.holdout_bars
    # Count from walk["setups"] (all setups, incl. unresolved), matching
    # tools.deep_backtest.walk_forward's own holdout setup count exactly —
    # not just resolved rows.
    holdout_setups = [s for s in walk["setups"]
                     if holdout_lo <= s["idx"] < span_hi]

    fold_reports = []
    per_variant_fold_net: dict[str, list[float | None]] = {
        v: [] for v in GATE_VARIANTS}
    for f in folds:
        _train, val = partition_setups(rows, f, hold_bars)
        fold_entry = {"fold": f.index, "val_lo": f.val_lo, "val_hi": f.val_hi,
                      "n_val_setups": len(val)}
        for variant in GATE_VARIANTS:
            stats = variant_stats(val, variant)
            fold_entry[variant] = stats
            per_variant_fold_net[variant].append(stats["net_expectancy_r"])
        fold_reports.append(fold_entry)

    fold_stability = {}
    for variant, nets in per_variant_fold_net.items():
        confident = [x for x in nets if x is not None]
        if not confident:
            fold_stability[variant] = {"positive_folds_ratio": None,
                                       "n_confident_folds": 0}
            continue
        positive = sum(1 for x in confident if x > 0)
        fold_stability[variant] = {
            "positive_folds_ratio": round(positive / len(confident), 4),
            "n_confident_folds": len(confident),
        }

    return {
        "config": {
            "train_bars": wf.train_bars, "val_bars": wf.val_bars,
            "step_bars": wf.step_bars, "purge_bars": wf.purge_bars,
            "embargo_bars": wf.embargo_bars, "holdout_bars": wf.holdout_bars,
            "min_folds": wf.min_folds,
        },
        "folds": fold_reports,
        "fold_stability": fold_stability,
        "holdout": {
            "idx_lo": holdout_lo, "idx_hi": span_hi,
            "n_setups": len(holdout_setups), "sealed": True,
            "note": ("geometry and setup count only; no performance stats "
                     "computed — sealed, not opened by C2.1"),
        },
    }


# ---------------------------------------------------------------------------
# Random-rejection baseline (matched coverage, many draws).
# ---------------------------------------------------------------------------

def random_rejection_baseline(rows: list[dict[str, Any]], rejection_rate: float,
                              n_draws: int = N_RANDOM_DRAWS,
                              seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Reject the same FRACTION of rows at random (uniform, not context-aware),
    n_draws times, and report the resulting net-expectancy distribution.
    """
    rng = np.random.RandomState(seed)
    n = len(rows)
    n_keep = max(0, round(n * (1 - rejection_rate)))
    net_rs = np.array([r["r"] for r in rows])
    draws = []
    for _ in range(n_draws):
        keep_idx = rng.choice(n, size=n_keep, replace=False)
        draws.append(float(np.mean(net_rs[keep_idx])) if n_keep else None)
    confident = [d for d in draws if d is not None]
    return {
        "rejection_rate": round(rejection_rate, 4),
        "n_keep": n_keep,
        "n_draws": n_draws,
        "random_mean_net_r": (round(float(np.mean(confident)), 4)
                              if confident else None),
        "random_std_net_r": (round(float(np.std(confident)), 4)
                             if confident else None),
        "draws": [round(d, 4) if d is not None else None for d in confident],
    }


def percentile_rank(value: float, draws: list[float]) -> float:
    if not draws:
        return float("nan")
    return round(sum(1 for d in draws if d <= value) / len(draws), 4)


# ---------------------------------------------------------------------------
# Robustness checks + decision.
# ---------------------------------------------------------------------------

def evaluate_robustness(full_sample: dict[str, Any], wf: dict[str, Any],
                        year_stats: dict[str, Any],
                        random_baselines: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    a_net = full_sample["A_baseline"]["net_expectancy_r"]
    for variant in ("B_range_gate", "C_trend_up_exclusion_only"):
        v = full_sample[variant]
        beats_baseline = (v["net_expectancy_r"] is not None and a_net is not None
                          and v["net_expectancy_r"] > a_net)
        net_positive = v["net_expectancy_r"] is not None and v["net_expectancy_r"] > 0
        fold_stab = wf["fold_stability"].get(variant, {})
        positive_folds_ok = (fold_stab.get("positive_folds_ratio") is not None
                             and fold_stab["positive_folds_ratio"] >= (2 / 3))
        years = year_stats.get(variant, {})
        year_nets = [y["net_expectancy_r"] for y in years.values()
                    if y["net_expectancy_r"] is not None and y["n"] >= 10]
        positive_years = sum(1 for x in year_nets if x > 0)
        # Integer form of "positive_years / len(year_nets) >= 2/3" avoids
        # round()/floor() silently lowering the bar for small year counts
        # (e.g. round(2 * 2/3) == 1 would wrongly pass 1-of-2 years).
        year_independent = (len(year_nets) >= 2
                            and positive_years * 3 >= len(year_nets) * 2)
        enough_trades = v["n"] is not None and v["n"] >= 50
        rb = random_baselines.get(variant, {})
        rank = percentile_rank(v["net_expectancy_r"], rb.get("draws", [])) \
            if v["net_expectancy_r"] is not None else float("nan")
        beats_random = rank >= 0.95 if rank == rank else False  # NaN-safe
        checks[variant] = {
            "beats_unfiltered_baseline_net_expectancy": beats_baseline,
            "net_positive_after_costs": net_positive,
            "positive_in_2_of_3_folds_or_more": positive_folds_ok,
            "not_dependent_on_one_calendar_year": year_independent,
            "retains_enough_trades": enough_trades,
            "beats_random_rejection_p95": beats_random,
            "random_rejection_percentile_rank": rank if rank == rank else None,
        }
    return checks


def decide(checks: dict[str, Any]) -> tuple[str, str]:
    b = checks.get("B_range_gate", {})
    all_pass = all([
        b.get("beats_unfiltered_baseline_net_expectancy"),
        b.get("net_positive_after_costs"),
        b.get("positive_in_2_of_3_folds_or_more"),
        b.get("not_dependent_on_one_calendar_year"),
        b.get("retains_enough_trades"),
        b.get("beats_random_rejection_p95"),
    ])
    if all_pass:
        return ("PROCEED_TO_REPORT_ONLY_FORWARD_TEST",
                "Frozen gate B (range_gate) beat the unfiltered baseline, "
                "stayed net-positive after costs, was positive in >=2/3 "
                "confident walk-forward folds, was not dependent on a "
                "single calendar year, retained enough trades, and beat "
                "random rejection at the matched coverage rate.")
    passed = sum(1 for k, v in b.items() if v is True)
    total = sum(1 for k in b if k != "random_rejection_percentile_rank")
    if b.get("net_positive_after_costs") and passed >= total - 2:
        return ("NEEDS_MORE_EVIDENCE",
                "Frozen gate B passed most but not all robustness checks "
                f"({passed}/{total}); net expectancy remains positive but at "
                "least one required condition (baseline beat, fold "
                "stability, year independence, sample size, or beating "
                "random rejection) did not clear its bar.")
    return ("REJECT_REGIME_GATE",
            "Frozen gate B failed multiple robustness checks or turned "
            "net-negative after costs; the discovered regime split does not "
            "confirm as a usable live gate on this evidence.")


# ---------------------------------------------------------------------------
# Report assembly + CLI.
# ---------------------------------------------------------------------------

LIMITATIONS = [
    "Confirmation only: this stage tests ONE frozen gate derived from C2a; "
    "it does not redesign the swing strategy or change runtime/config/score.",
    "Gate is frozen before evaluation (see module docstring) and was not "
    "adjusted after seeing results.",
    "R semantics reused verbatim from tools.deep_backtest.resolve/deep_walk "
    "(cost-adjusted, conservative intrabar stop-first).",
    "Walk-forward uses tools.deep_backtest.WFConfig/fold_windows/"
    "partition_setups unchanged (real purge/embargo geometry), not the "
    "simplified contiguous idx-block folds used for C1.8 attribution.",
    "Sealed holdout tail is carved out exactly as in "
    "tools.deep_backtest.walk_forward and never scored by this module.",
    "Single exchange / single symbol / single discovery-region run.",
]


def run_confirmation(dataset: str, exchange: str, symbol: str,
                     profile_name: str, bars: int,
                     max_gap_ratio: float = 0.001,
                     allow_estimated_cvd: bool = False,
                     holdout_frac: float = HOLDOUT_FRAC,
                     min_folds: int = MIN_FOLDS) -> dict[str, Any]:
    rows, walk, profile, n_entry_bars, cvd_method = build_rows(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)

    full_sample = {v: variant_stats(rows, v) for v in GATE_VARIANTS}
    year_stats = calendar_year_stats(rows)
    wf = run_walk_forward(rows, walk, profile, bars, n_entry_bars,
                          holdout_frac, min_folds)

    random_baselines = {}
    for variant in ("B_range_gate", "C_trend_up_exclusion_only"):
        rr = full_sample[variant]["rejection_rate"] or 0.0
        random_baselines[variant] = random_rejection_baseline(rows, rr)

    checks = evaluate_robustness(full_sample, wf, year_stats, random_baselines)
    decision, reason = decide(checks)

    resolved = len(rows)
    total_setups = len(walk["setups"])
    return {
        "stage": STAGE,
        "kind": "regime_gate_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name, "exchange": exchange, "symbol": symbol,
        "bars_requested": bars, "cvd_method": cvd_method,
        "counts": {"setups": total_setups, "resolved": resolved,
                  "unresolved": total_setups - resolved},
        "frozen_gate": {
            "A_baseline": "no gate; current swing setups unfiltered",
            "B_range_gate": ("REJECT if regime_current == 'trend_up' OR "
                            "htf_bias == 'bullish'; ALLOW otherwise"),
            "C_trend_up_exclusion_only": ("REJECT if regime_current == "
                                          "'trend_up'; ALLOW otherwise"),
        },
        "full_sample": full_sample,
        "calendar_year": year_stats,
        "walk_forward": wf,
        "random_rejection_baseline": random_baselines,
        "robustness_checks": checks,
        "decision": decision,
        "decision_reason": reason,
        "limitations": list(LIMITATIONS),
    }


def format_report(report: dict[str, Any]) -> str:
    c = report["counts"]
    lines = [
        f"regime gate confirmation ({report['stage']}) — "
        f"{report['exchange']} {report['symbol']} profile={report['profile']}",
        f"  generated_at : {report['generated_at_utc']}",
        f"  setups       : {c['setups']} (resolved {c['resolved']}, "
        f"unresolved {c['unresolved']})",
        "",
        "frozen gate:",
    ]
    for v, desc in report["frozen_gate"].items():
        lines.append(f"  {v}: {desc}")
    lines.append("")
    lines.append("full sample:")
    for v in GATE_VARIANTS:
        s = report["full_sample"][v]
        lines.append(
            f"  {v:28s} n={s['n']:>4} cov={s['coverage']} "
            f"gross={s['gross_expectancy_r']} net={s['net_expectancy_r']} "
            f"median={s['median_r']} win={s['win_rate']} "
            f"timeout={s['timeout_share']} sum_r={s['sum_r']} "
            f"cost_r={s['avg_cost_r']} max_losing_streak="
            f"{s['max_losing_streak']}")
    lines.append("")
    lines.append("walk-forward fold stability (share of confident folds "
                 "net-positive):")
    for v, st in report["walk_forward"]["fold_stability"].items():
        lines.append(f"  {v:28s} {st}")
    lines.append("")
    lines.append("random-rejection baseline (matched coverage, "
                 f"{N_RANDOM_DRAWS} draws):")
    for v, rb in report["random_rejection_baseline"].items():
        lines.append(f"  {v:28s} random_mean={rb['random_mean_net_r']} "
                     f"random_std={rb['random_std_net_r']} "
                     f"rejection_rate={rb['rejection_rate']}")
    lines.append("")
    lines.append("robustness checks:")
    for v, chk in report["robustness_checks"].items():
        lines.append(f"  {v}:")
        for k, val in chk.items():
            lines.append(f"    {k}: {val}")
    lines.append("")
    lines.append(f"DECISION: {report['decision']}")
    lines.append(f"  {report['decision_reason']}")
    lines.append("")
    lines.append("limitations:")
    for lim in report["limitations"]:
        lines.append(f"  - {lim}")
    return "\n".join(lines) + "\n"


def format_decision_md(report: dict[str, Any]) -> str:
    lines = [
        "# C2.1 — Swing Regime Gate Confirmation: DECISION",
        "",
        f"Profile: {report['profile']} · {report['exchange']} "
        f"{report['symbol']} · resolved setups: {report['counts']['resolved']}"
        f" · generated: {report['generated_at_utc']}",
        "",
        "This is a confirmation test result, not a strategy or runtime "
        "change. No score, threshold, config, or runtime behavior is "
        "modified by this report.",
        "",
        f"## Verdict: `{report['decision']}`",
        "",
        report["decision_reason"],
        "",
        "## Frozen gate under test",
        "",
    ]
    for v, desc in report["frozen_gate"].items():
        lines.append(f"- **{v}**: {desc}")
    lines += ["", "## Full-sample results", "",
             "| variant | n | coverage | gross R | net R | win rate | "
             "max losing streak |",
             "|---|---|---|---|---|---|---|"]
    for v in GATE_VARIANTS:
        s = report["full_sample"][v]
        lines.append(f"| {v} | {s['n']} | {s['coverage']} | "
                     f"{s['gross_expectancy_r']} | {s['net_expectancy_r']} | "
                     f"{s['win_rate']} | {s['max_losing_streak']} |")
    lines += ["", "## Robustness checks (B_range_gate)", ""]
    for k, val in report["robustness_checks"].get("B_range_gate", {}).items():
        lines.append(f"- {k}: {val}")
    lines += ["", "## Limitations", ""]
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C2.1: swing regime gate confirmation (offline, "
                    "read-only)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bars", type=int, required=True)
    parser.add_argument("--max-gap-ratio", type=float, default=0.001)
    parser.add_argument("--allow-estimated-cvd", action="store_true")
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--min-folds", type=int, default=MIN_FOLDS)
    parser.add_argument("--outdir", default="reports/c21")
    parser.add_argument("--json", action="store_true",
                       help="also print the JSON report to stdout")
    args = parser.parse_args(argv)

    try:
        report = run_confirmation(
            args.dataset, args.exchange, args.symbol, args.profile,
            args.bars, args.max_gap_ratio, args.allow_estimated_cvd,
            args.holdout_frac, args.min_folds)
    except DeepBacktestError as exc:
        print(f"regime_gate_confirmation: {exc}", file=__import__("sys").stderr)
        return 2

    import os
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "regime_gate_confirmation.json"),
              "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(args.outdir, "regime_gate_confirmation.txt"),
              "w") as f:
        f.write(format_report(report))
    with open(os.path.join(args.outdir, "DECISION.md"), "w") as f:
        f.write(format_decision_md(report))

    print(format_report(report))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
