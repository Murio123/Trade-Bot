"""C4.3d: does Ridge out-RANK a fair, time-varying rolling baseline?

Measurement only. Nothing here retrains Ridge, touches features, labels, WF
geometry or the frozen C4.1 criteria, reads the sealed holdout, installs
LightGBM, or begins C4.4. It re-runs the existing C4.3 walk-forward
unchanged and reads two things off it: the published Ridge metrics (which
must reproduce) and the new point-in-time baseline comparison.

Why this stage exists. C4.3c rejected Ridge as a point predictor: it lost on
MAE to both rolling-mean-60 and the plain train mean. Its one surviving
positive signal was a stable Spearman of ~0.31-0.40 — but that was compared
against baselines that are CONSTANT within a fold, and a constant cannot rank
anything, so any non-degenerate model wins by construction. C4.3c said so
itself and noted the honest comparison was not in the saved artifacts. This
stage supplies it.

A positive verdict would establish ranking utility ONLY. It would not
reinstate Ridge as a precise volatility predictor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

from tools.deep_backtest import DeepBacktestError
from tools.ridge_volatility_run import (RANKING_DECISION_RULE, MIN_YEAR_ROWS,
                                        decide_ranking_verdict)
from tools.ridge_volatility_run import run as ridge_run

STAGE = "C4.3d"
PUBLISHED_C43 = "reports/c43/ridge_evaluation.json"

# Ridge's closed-form solve carries no randomness, and C4.3's own
# `reproducible` check already asserts bit-equality between two independent
# trainings (atol=0). The same standard applies here; the tolerance only
# absorbs JSON float round-tripping.
REPRODUCTION_TOLERANCE = 1e-12

# Metrics that must come back unchanged. Per-fold keys are checked for every
# fold, pooled keys once.
REPRODUCED_FOLD_KEYS = ("mae_model", "mae_persistence", "mae_rolling_mean_60",
                        "mae_train_mean", "rmse_model", "spearman",
                        "chosen_alpha", "n_train", "n_val", "confident")
REPRODUCED_POOLED_KEYS = ("n", "mae_model", "mae_persistence",
                          "mae_rolling_mean_60", "mae_train_mean",
                          "rmse_model", "spearman")


class ReproductionMismatch(Exception):
    """Raised when a previously published Ridge number moved."""


_MISSING = object()


def _close(a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return False  # a required metric that vanished is drift, not a match
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0,
                            abs_tol=REPRODUCTION_TOLERANCE)
    return a == b


def _show(value: Any) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def compare_to_published(fresh: dict[str, Any], published: dict[str, Any]
                         ) -> list[str]:
    """Every C4.3 Ridge number that moved, as human-readable diffs."""
    diffs: list[str] = []

    old_folds = {f["fold"]: f for f in published.get("folds", [])}
    new_folds = {f["fold"]: f for f in fresh.get("folds", [])}
    if set(old_folds) != set(new_folds):
        diffs.append(f"fold set changed: {sorted(old_folds)} -> "
                     f"{sorted(new_folds)}")
    for fold in sorted(set(old_folds) & set(new_folds)):
        for key in REPRODUCED_FOLD_KEYS:
            old = old_folds[fold].get(key, _MISSING)
            new = new_folds[fold].get(key, _MISSING)
            if not _close(old, new):
                diffs.append(f"fold {fold}.{key}: {_show(old)} -> {_show(new)}")

    old_pooled = published.get("pooled") or {}
    new_pooled = fresh.get("pooled") or {}
    for key in REPRODUCED_POOLED_KEYS:
        old = old_pooled.get(key, _MISSING)
        new = new_pooled.get(key, _MISSING)
        if not _close(old, new):
            diffs.append(f"pooled.{key}: {_show(old)} -> {_show(new)}")

    if published.get("verdict") != fresh.get("verdict"):
        diffs.append(f"verdict: {published.get('verdict')!r} -> "
                     f"{fresh.get('verdict')!r}")
    old_checks, new_checks = published.get("checks") or {}, fresh.get("checks") or {}
    for key in sorted(set(old_checks) | set(new_checks)):
        if old_checks.get(key) != new_checks.get(key):
            diffs.append(f"checks.{key}: {old_checks.get(key)!r} -> "
                         f"{new_checks.get(key)!r}")
    return diffs


def _int_pair(value: Any) -> tuple[int, int] | None:
    """(lo, hi) if `value` is exactly two integer-like bounds, else None.

    Deliberately total: any shape this does not understand yields None so the
    caller fails the holdout gate instead of raising.
    """
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple)):
        return None
    if len(value) != 2:
        return None
    try:
        lo, hi = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return lo, hi


def extract_ranking_inputs(fresh: dict[str, Any]) -> dict[str, Any]:
    """Pull the C4.3d comparison out of a completed run."""
    fold_adv: list[float | None] = []
    for f in fresh.get("folds", []):
        if not f.get("confident"):
            continue
        fold_adv.append((f.get("c43d") or {}).get("spearman_advantage"))

    c43d = (fresh.get("pooled") or {}).get("c43d") or {}
    pooled = c43d.get("pooled") or {}
    year_slices = fresh.get("pooled", {}).get("year_slices") or {}

    year_adv: list[float | None] = []
    for year, stats in (c43d.get("year_slices") or {}).items():
        if (year_slices.get(year) or {}).get("n", 0) < MIN_YEAR_ROWS:
            continue
        year_adv.append(stats.get("spearman_advantage"))

    # Real check, not a formality: count validation rows any fold would have
    # scored at or beyond the sealed holdout's first index. Fold geometry
    # should make this structurally impossible, so a non-zero here means the
    # geometry itself broke and the verdict must be withheld.
    holdout_lo = (fresh.get("holdout") or {}).get("idx_lo")
    holdout_rows = 0
    if holdout_lo is None:
        holdout_rows = -1  # cannot prove exclusion -> fails the gate
    else:
        for f in fresh.get("folds", []):
            # Anything that is not a clean pair of integers means exclusion
            # cannot be proven, which must fail the gate rather than raise —
            # a crash here would skip the check entirely instead of failing
            # it, which is the opposite of fail-closed.
            bounds = _int_pair(f.get("val_idx_range"))
            if bounds is None:
                holdout_rows = -1
                break
            val_lo, val_hi = bounds
            if val_hi > holdout_lo:
                holdout_rows += int(val_hi - max(val_lo, holdout_lo))

    return {
        "fold_advantages": fold_adv,
        "pooled_advantage": pooled.get("spearman_advantage"),
        "year_advantages": year_adv,
        "holdout_rows_used": holdout_rows,
        "pooled_detail": pooled,
        "year_detail": c43d.get("year_slices") or {},
    }


def run(dataset: str, exchange: str, symbol: str, bars: int,
        published_path: str = PUBLISHED_C43, outdir: str = "reports/c43d",
        c43_outdir: str | None = None) -> dict[str, Any]:
    if not os.path.exists(published_path):
        raise DeepBacktestError(
            f"{published_path} not found — C4.3d compares against the "
            "published C4.3 artifacts and cannot run without them")
    with open(published_path) as fh:
        published = json.load(fh)

    # Re-run C4.3 into a scratch dir so the published artifacts are never
    # overwritten before the reproduction guard has had its say.
    scratch = c43_outdir or os.path.join(outdir, "_c43_rerun")
    fresh = ridge_run(dataset, exchange, symbol, bars, outdir=scratch,
                      registry_dir=os.path.join(scratch, "model_registry"))

    per_row = fresh.pop("c43d_per_row", {})
    diffs = compare_to_published(fresh, published)
    reproduced = not diffs

    inputs = extract_ranking_inputs(fresh)
    verdict, checks = decide_ranking_verdict(
        fold_advantages=inputs["fold_advantages"],
        pooled_advantage=inputs["pooled_advantage"],
        year_advantages=inputs["year_advantages"],
        holdout_rows_used=inputs["holdout_rows_used"],
        reproduced=reproduced)

    result = {
        "stage": STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange, "symbol": symbol, "bars": bars,
        "code_commit": fresh.get("code_commit"),
        "published_c43": published_path,
        "reproduction": {
            "reproduced": reproduced,
            "tolerance": REPRODUCTION_TOLERANCE,
            "diffs": diffs,
        },
        "wf_config": fresh.get("wf_config"),
        "holdout": fresh.get("holdout"),
        "folds": [{"fold": f["fold"], "confident": f["confident"],
                   "c43d": f.get("c43d")} for f in fresh.get("folds", [])],
        "pooled": inputs["pooled_detail"],
        "year_slices": inputs["year_detail"],
        "ranking_inputs": {k: inputs[k] for k in
                           ("fold_advantages", "pooled_advantage",
                            "year_advantages", "holdout_rows_used")},
        "checks": checks,
        "decision_rule": RANKING_DECISION_RULE,
        "verdict": verdict,
    }

    os.makedirs(outdir, exist_ok=True)
    # Per-row dump so significance testing can run without rebuilding the
    # dataset. Confident-fold validation rows only — the same population the
    # pooled numbers are computed from.
    with open(os.path.join(outdir, "per_row.json"), "w") as fh:
        json.dump(per_row, fh, indent=2, default=str)
    with open(os.path.join(outdir, "ridge_vs_rolling_series.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    with open(os.path.join(outdir, "ridge_vs_rolling_series.txt"), "w") as fh:
        fh.write(format_report(result))
    with open(os.path.join(outdir, "DECISION.md"), "w") as fh:
        fh.write(format_decision_md(result))
    return result


def format_report(result: dict[str, Any]) -> str:
    rep = result["reproduction"]
    lines = [
        f"Ridge vs time-varying rolling baseline ({result['stage']}) — "
        f"{result['exchange']} {result['symbol']}",
        f"  generated_at : {result['generated_at_utc']}",
        f"  commit       : {result['code_commit']}",
        f"  wf_config    : {result['wf_config']}",
        f"  holdout      : {result['holdout']}",
        "",
        f"reproduction of published C4.3: {'OK' if rep['reproduced'] else 'FAILED'}"
        f" (tolerance {rep['tolerance']})",
    ]
    for d in rep["diffs"]:
        lines.append(f"  DIFF {d}")
    lines.append("")
    lines.append("per-fold (Spearman: ridge vs rolling-series):")
    for f in result["folds"]:
        c = f.get("c43d") or {}
        lines.append(
            f"  fold {f['fold']}: confident={f['confident']} n={c.get('n')} "
            f"dropped={c.get('n_dropped_no_history')} "
            f"ridge={c.get('spearman_model')} "
            f"series={c.get('spearman_rolling_series')} "
            f"adv={c.get('spearman_advantage')} | "
            f"MAE ridge={c.get('mae_model')} series={c.get('mae_rolling_series')}")
    lines.append("")
    lines.append(f"pooled: {result['pooled']}")
    lines.append("")
    lines.append("per-year:")
    for year, stats in sorted(result["year_slices"].items()):
        lines.append(f"  {year}: {stats}")
    lines.append("")
    lines.append(f"ranking_inputs: {result['ranking_inputs']}")
    lines.append(f"checks: {result['checks']}")
    lines.append("")
    lines.append(f"VERDICT: {result['verdict']}")
    lines.append(f"decision_rule: {result['decision_rule']}")
    return "\n".join(lines) + "\n"


def format_decision_md(result: dict[str, Any]) -> str:
    rep = result["reproduction"]
    pooled = result["pooled"] or {}
    lines = [
        "# C4.3d — Ridge Ranking vs a Time-Varying Baseline: DECISION",
        "",
        f"{result['exchange']} {result['symbol']} · generated: "
        f"{result['generated_at_utc']} · commit {result['code_commit']}",
        "",
        "Measurement only. Ridge was not retrained differently, no feature, "
        "label, walk-forward geometry or frozen criterion changed, the sealed "
        "holdout was not read, LightGBM was not installed, and C4.4 does not "
        "begin here.",
        "",
        f"## Verdict: `{result['verdict']}`",
        "",
        "## Why this comparison was needed",
        "",
        "C4.3c rejected Ridge as a point predictor — it lost on MAE to both "
        "rolling-mean-60 and the plain train mean. Its one surviving positive "
        "signal was a stable Spearman of roughly 0.31-0.40. But the baselines "
        "it was measured against are CONSTANT within a fold, and a constant "
        "has no rank-order power by construction, so any non-degenerate model "
        "wins that comparison automatically. C4.3c said as much and recorded "
        "that the honest comparison was absent from the artifacts.",
        "",
        "The baseline used here is a trailing mean an observer could actually "
        "have tracked bar to bar: at query bar t it averages the last 60 "
        "labels with index <= t-12, because the label of bar u is computed "
        "from bars u+1..u+12 and is not knowable before u+12.",
        "",
        "## Reproduction guard",
        "",
        f"- reproduced: **{rep['reproduced']}** (tolerance {rep['tolerance']})",
    ]
    if rep["diffs"]:
        lines.append("- differences found:")
        for d in rep["diffs"]:
            lines.append(f"  - {d}")
        lines.append("")
        lines.append("The published C4.3 numbers did not reproduce, so the new "
                     "comparison is NOT interpreted.")
    else:
        lines.append("- every published C4.3 per-fold and pooled Ridge metric "
                     "came back unchanged, so the new comparison is "
                     "interpretable.")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- pooled n = {pooled.get('n')} "
                 f"(rows dropped for no observable history: "
                 f"{pooled.get('n_dropped_no_history')})")
    lines.append(f"- pooled Spearman: ridge={pooled.get('spearman_model')} "
                 f"rolling-series={pooled.get('spearman_rolling_series')} "
                 f"advantage={pooled.get('spearman_advantage')}")
    lines.append(f"- pooled MAE: ridge={pooled.get('mae_model')} "
                 f"rolling-series={pooled.get('mae_rolling_series')}")
    lines.append(f"- pooled RMSE: ridge={pooled.get('rmse_model')} "
                 f"rolling-series={pooled.get('rmse_rolling_series')}")
    lines.append("")
    lines.append("### Per fold")
    lines.append("")
    lines.append("| fold | confident | n | ridge Spearman | series Spearman | advantage |")
    lines.append("|---|---|---|---|---|---|")
    for f in result["folds"]:
        c = f.get("c43d") or {}
        lines.append(f"| {f['fold']} | {f['confident']} | {c.get('n')} | "
                     f"{c.get('spearman_model')} | "
                     f"{c.get('spearman_rolling_series')} | "
                     f"{c.get('spearman_advantage')} |")
    lines.append("")
    lines.append("### Per year")
    lines.append("")
    lines.append("| year | n | ridge Spearman | series Spearman | advantage |")
    lines.append("|---|---|---|---|---|")
    for year, s in sorted(result["year_slices"].items()):
        lines.append(f"| {year} | {s.get('n')} | {s.get('spearman_model')} | "
                     f"{s.get('spearman_rolling_series')} | "
                     f"{s.get('spearman_advantage')} |")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for name, value in result["checks"].items():
        lines.append(f"- {name}: {value}")
    lines.append("")
    lines.append(f"Decision rule: {result['decision_rule']}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- A CONFIRMED verdict would establish ranking utility only. "
                 "Ridge remains rejected as a point predictor by C4.3c; "
                 "nothing here reinstates it as a precise volatility "
                 "forecast.")
    lines.append(f"- **The trailing baseline is ANTI-correlated with this "
                 f"label** (pooled Spearman "
                 f"{pooled.get('spearman_rolling_series')}). The label is a "
                 f"range normalised by current ATR, and its own "
                 f"autocorrelation flips sign at the 12-bar observability "
                 f"lag, so a rising trailing ratio precedes a falling one. "
                 f"A signed advantage therefore flatters the model: an "
                 f"observer who simply inverted the baseline would score "
                 f"|rho|. Against that inverted baseline Ridge's pooled "
                 f"advantage is only "
                 f"{pooled.get('spearman_advantage_vs_abs')}, not "
                 f"{pooled.get('spearman_advantage')}. The frozen rule uses "
                 f"the signed number and was not rewritten after the fact, "
                 f"but the verdict must not be read without this.")
    lines.append("- Rows where the trailing baseline has no observable history "
                 "yet are dropped from BOTH sides, so the two are always "
                 "scored on identical rows.")
    lines.append("- The sealed holdout is excluded twice over: fold geometry "
                 "never places a validation row inside it, and the baseline's "
                 "source pool is hard-capped below its first index.")
    lines.append("- This stage does not start C4.4 and implies no deployment.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C4.3d: ridge ranking vs a time-varying rolling baseline")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bars", type=int, default=8000)
    parser.add_argument("--published", default=PUBLISHED_C43)
    parser.add_argument("--outdir", default="reports/c43d")
    args = parser.parse_args(argv)

    try:
        result = run(args.dataset, args.exchange, args.symbol, args.bars,
                     args.published, args.outdir)
    except DeepBacktestError as exc:
        import sys
        print(f"ridge_ranking_check: {exc}", file=sys.stderr)
        return 2

    print(format_report(result))
    print(f"VERDICT: {result['verdict']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
