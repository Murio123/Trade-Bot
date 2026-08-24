"""S3 runner: declare the trials, then measure one family at a time.

Offline. Reads S1's panel, S2's write-once universe snapshots and S2's frozen
labels; writes one registry file and one report artifact. Ranks nothing in
production, trains nothing, and combines no features.

Order matters and is enforced by the code rather than by intention: **every
trial is declared before a single outcome is read.** `_declare_first()` runs to
completion before any label file is opened.

    .venv/bin/python -m tools.s3_evaluate --outdir data/spot \
        --reportdir reports/s3
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from spot.evaluate import (MIN_UNIVERSE, DateResult, aggregate, block_stability,
                           classify)
from spot.evaluate import evaluate_date as eval_one_date
from spot.features import FEATURES
from spot.labels import RESOLVED
from spot.trials import BY_FEATURE, SCOPE, counts, declare_all
from spot.universe import btc_regime, load_snapshot, snapshot_path, visible
from tools.spot_dataset import load_panel
from tools.spot_symbols import DEFAULT_OUTDIR

LABELS_REL = os.path.join("labels", "clean_2x_180d.jsonl")
UNIVERSE_REL = "universe"
REGISTRY_REL = os.path.join("trials", "spot_trials.jsonl")
BTC = "BTCUSDT"


def _declare_first(outdir: str) -> dict[str, Any]:
    """Register the six trials. Nothing about any outcome is read here."""
    from validation.trial_registry import TrialRegistry
    path = os.path.join(outdir, REGISTRY_REL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    registry = TrialRegistry(path)
    ids = declare_all(registry)
    return {"registry": path, "trial_ids": ids,
            "family": SCOPE.as_dict(), "counts": counts(registry)}


def load_labels(outdir: str) -> dict[str, dict[str, dict[str, Any]]]:
    """S2 outcomes, keyed by date then symbol. Resolved events only."""
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    with open(os.path.join(outdir, LABELS_REL), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["outcome"] in RESOLVED:
                by_date.setdefault(row["asof"], {})[row["symbol"]] = row
    return by_date


def evaluation_dates(outdir: str, by_date: dict[str, Any]) -> list[str]:
    """Monthly dates whose eligible universe reached the frozen minimum."""
    out = []
    universe_dir = os.path.join(outdir, UNIVERSE_REL)
    for name in sorted(os.listdir(universe_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(universe_dir, name), encoding="utf-8") as fh:
            snap = json.load(fh)
        if len(snap["eligible"]) >= MIN_UNIVERSE and snap["asof"] in by_date:
            out.append(snap["asof"])
    return out


def run(outdir: str) -> dict[str, Any]:
    governance = _declare_first(outdir)          # before any outcome is read

    panel = {s.symbol: s.df for s in load_panel(outdir)}
    btc = panel[BTC]
    by_date = load_labels(outdir)
    dates = evaluation_dates(outdir, by_date)

    universe_dir = os.path.join(outdir, UNIVERSE_REL)
    snapshots = {d: json.load(open(os.path.join(universe_dir, f"{d}.json"),
                                   encoding="utf-8")) for d in dates}

    # Feature values are computed once per (date, symbol) and reused by every
    # rule's evaluation, so no rule can perturb another's inputs.
    values: dict[str, dict[str, dict[str, float]]] = {f.fid: {} for f in FEATURES}
    regimes: dict[str, str] = {}
    for d in dates:
        snap = snapshots[d]
        asof_ms = snap["asof_ms"]
        regimes[d] = btc_regime(btc, asof_ms)
        btc_seen = visible(btc, asof_ms)
        for symbol in snap["eligible"]:
            seen = visible(panel[symbol], asof_ms)
            for feature in FEATURES:
                v = feature.value(seen, btc_seen)
                if v is not None:
                    values[feature.fid].setdefault(d, {})[symbol] = v

    families: dict[str, Any] = {}
    for feature in FEATURES:
        rng = np.random.default_rng(20260824)
        results: list[DateResult] = []
        for d in dates:
            snap = snapshots[d]
            results.append(eval_one_date(
                asof=d, regime=regimes[d],
                values=values[feature.fid].get(d, {}),
                labels=by_date[d], descending=feature.descending,
                n_eligible=len(snap["eligible"]), rng=rng))
        agg = aggregate(results)
        by_regime = {r: aggregate([x for x in results if x.regime == r])
                     for r in sorted({x.regime for x in results})}
        stability = block_stability(results)
        verdict, checks = classify(agg, stability, by_regime)
        families[feature.fid] = {
            "id": feature.fid, "name": feature.name, "family": feature.family,
            "definition": feature.definition,
            "trial_id": BY_FEATURE[feature.fid].trial_id,
            "overall": agg, "by_regime": by_regime, "stability": stability,
            "classification": verdict, "checks": checks,
            "per_date": [vars(r) for r in results],
        }

    return {
        "stage": "S3",
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "governance": governance,
        "dates": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "min_universe": MIN_UNIVERSE,
        "baselines": baselines(panel, snapshots, by_date, dates, regimes),
        "families": families,
    }


def baselines(panel: dict[str, pd.DataFrame], snapshots: dict[str, Any],
              by_date: dict[str, Any], dates: list[str],
              regimes: dict[str, str]) -> dict[str, Any]:
    """B0-B3. B4/B5 are ranking rules and live with the families."""
    btc = panel[BTC]
    uni_rate, uni_ret, uni_ex, uni_mae, btc_rows = [], [], [], [], []
    for d in dates:
        labels = by_date[d]
        eligible = [s for s in snapshots[d]["eligible"] if s in labels]
        outcomes = [labels[s]["clean_2x"] for s in eligible
                    if labels[s]["clean_2x"] is not None]
        if outcomes:
            uni_rate.append(float(np.mean(outcomes)))
        # A symbol delisted on the decision bar itself resolves as a failure
        # with no window, so its vector fields are None. It counts in the rate
        # and cannot count in a median.
        ret = [labels[s]["terminal_return"] for s in eligible
               if labels[s]["terminal_return"] is not None]
        if ret:
            uni_ret.append(float(np.median(ret)))
        ex = [labels[s]["excess_vs_btc"] for s in eligible
              if labels[s]["excess_vs_btc"] is not None]
        if ex:
            uni_ex.append(float(np.median(ex)))
        mae = [labels[s]["mae"] for s in eligible
               if labels[s]["mae"] is not None]
        if mae:
            uni_mae.append(float(np.median(mae)))
        if BTC in labels:
            btc_rows.append(labels[BTC])

    def med(rows, field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return float(np.median(vals)) if vals else None

    return {
        "B0_random": "reported per family as random_mean / random_p95",
        "B1_btc_hold": {
            "dates": len(btc_rows),
            "clean_2x_rate": (float(np.mean([r["clean_2x"] for r in btc_rows]))
                              if btc_rows else None),
            "median_return": med(btc_rows, "terminal_return"),
            "median_mae": med(btc_rows, "mae"),
        },
        "B2_equal_weight_universe": {
            "clean_2x_rate": float(np.mean(uni_rate)) if uni_rate else None,
            "median_return": float(np.mean(uni_ret)) if uni_ret else None,
            "median_excess_vs_btc": float(np.mean(uni_ex)) if uni_ex else None,
            "median_mae": float(np.mean(uni_mae)) if uni_mae else None,
        },
        "B3_cap_weighted": "UNAVAILABLE - no trustworthy point-in-time market "
                           "cap (S0 §6); today's cap applied historically is a "
                           "leak, so it is not approximated",
    }


def format_report(s: dict[str, Any]) -> str:
    b = s["baselines"]
    lines = [
        "S3 — independent families vs strong baselines",
        f"  dates {s['dates']} ({s['first_date']} .. {s['last_date']}), "
        f"min universe {s['min_universe']}",
        f"  spot family n_trials: "
        f"{s['governance']['counts']['conservative']}",
        "",
        "  BASELINES",
        f"    B1 BTC hold      : clean2x "
        f"{b['B1_btc_hold']['clean_2x_rate']}  median ret "
        f"{_r(b['B1_btc_hold']['median_return'])}  MAE "
        f"{_r(b['B1_btc_hold']['median_mae'])}",
        f"    B2 equal-weight  : clean2x "
        f"{_r(b['B2_equal_weight_universe']['clean_2x_rate'])}  median ret "
        f"{_r(b['B2_equal_weight_universe']['median_return'])}  vsBTC "
        f"{_r(b['B2_equal_weight_universe']['median_excess_vs_btc'])}  MAE "
        f"{_r(b['B2_equal_weight_universe']['median_mae'])}",
        "    B3 cap-weighted  : UNAVAILABLE",
        "",
        "  RANKING RULES (top quintile)",
        f"    {'id':4} {'rate':>7} {'univ':>7} {'lift':>6} {'vsBTC':>8} "
        f"{'MAE':>7} {'rho':>7} {'diff 90% CI':>22}  class",
    ]
    for fid, f in s["families"].items():
        a = f["overall"]
        d = a["rate_difference"]
        ci = f"[{_r(d['low'])}, {_r(d['high'])}]"
        lines.append(
            f"    {fid:4} {_r(a['selection_rate']):>7} "
            f"{_r(a['universe_rate']):>7} {_r(a['lift']):>6} "
            f"{_r(a['selection_median_excess']):>8} "
            f"{_r(a['selection_median_mae']):>7} "
            f"{_r(a['spearman']['mean']):>7} {ci:>22}  {f['classification']}")
    lines.append("")
    for fid, f in s["families"].items():
        reg = {r: _r(v.get("lift")) for r, v in f["by_regime"].items()}
        lines.append(f"    {fid} lift by regime {reg}  "
                     f"blocks>1: {f['stability']['blocks_with_lift_above_1']}"
                     f"/{f['stability']['blocks']}")
    return "\n".join(lines)


def _r(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--reportdir", default="reports/s3")
    args = parser.parse_args(argv)

    summary = run(args.outdir)
    os.makedirs(args.reportdir, exist_ok=True)
    with open(os.path.join(args.reportdir, "s3_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
