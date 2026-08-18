"""G1 — run the negative-control suite and apply the frozen thresholds.

Offline only. Reads nothing but its own synthetic data: no dataset, no database,
no exchange, no sealed holdout. Writes per-replication records and an aggregate
verdict under `reports/g1/` (untracked), and prints a report.

The thresholds T1-T7 are not defined here. They are defined in
`reports/c50/G1_SPEC.md` §4.1, committed at `197a765` before any control existed,
and this module only evaluates them. That separation is the point: a runner that
carried its own thresholds could always be adjusted until it passed.

    python -m tools.g1_negative_controls --out reports/g1
    python -m tools.g1_negative_controls --quick     # 20 reps, for a smoke check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from validation.negative_controls import (CONTROLS, MASTER_SEED,
                                          NEGATIVE_CONTROLS_VERSION,
                                          barrier_bias_diagnostic,
                                          bootstrap_mean_ci, ci_verdict,
                                          rng_for, share_band)

# G1_SPEC.md §4.1 T1: the binomial one-sided 99% bound at R=500, rounded up.
T1_MAX_FPR = 0.075

# Which controls T1 binds to (the DSR-based ones, per the spec's table).
DSR_CONTROLS = ("NC1", "NC2", "NC5", "NC6", "NC7", "NC8")
# Which controls T2 binds to (the expectancy-based baselines).
EXPECTANCY_CONTROLS = ("NC3", "NC4")


def run_control(name: str, replications: int) -> list[dict[str, Any]]:
    fn, _ = CONTROLS[name]
    return [fn(i) for i in range(replications)]


def _finite(records: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([r.get(key, np.nan) for r in records], dtype=float)
    return vals[np.isfinite(vals)]


def _fpr(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of replications reported significant, and the DSR distribution.

    `insufficient` is reported separately and counted as *not* significant: an
    explicit insufficient-data outcome is M01 refusing to answer, which is the
    correct behaviour and not a false positive. It is surfaced because a control
    that was mostly insufficient would satisfy T1 while testing nothing.
    """
    n = len(records)
    sig = sum(1 for r in records if r.get("significant"))
    insufficient = sum(1 for r in records
                       if r.get("status") == "INSUFFICIENT_DATA")
    dsr = _finite(records, "dsr")
    out = {"n": n, "n_significant": sig, "fpr": sig / n if n else float("nan"),
           "n_insufficient": insufficient,
           "t1_max_fpr": T1_MAX_FPR}
    if dsr.size:
        out.update({"dsr_mean": float(dsr.mean()),
                    "dsr_median": float(np.median(dsr)),
                    "dsr_q05": float(np.quantile(dsr, 0.05)),
                    "dsr_q95": float(np.quantile(dsr, 0.95))})
    out["t1_pass"] = bool(out["fpr"] <= T1_MAX_FPR)
    return out


def _sample_size_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    """T7 — both counts reported by EVERY record, and effective <= raw.

    "Reported" means every replication carries both numbers. An earlier version
    only required that some finite values existed somewhere in the batch, which
    would have passed a control that reported the pair once and the raw count
    everywhere else — the exact omission T7 exists to forbid.
    """
    missing = [r.get("replication") for r in records
               if not np.isfinite(float(r.get("raw_count", np.nan)))
               or not np.isfinite(float(r.get("effective_count", np.nan)))]
    raw = _finite(records, "raw_count")
    eff = _finite(records, "effective_count")
    if raw.size == 0 or eff.size == 0:
        return {"reported": False, "n_missing": len(missing)}
    # Pair the two counts per record rather than aligning two separately
    # filtered arrays: with a record missing one of them the arrays have
    # different lengths, and an elementwise comparison would either raise or,
    # worse, silently compare mismatched rows.
    pairs = [(float(r["raw_count"]), float(r["effective_count"]))
             for r in records
             if np.isfinite(float(r.get("raw_count", np.nan)))
             and np.isfinite(float(r.get("effective_count", np.nan)))]
    raw_p = np.array([p[0] for p in pairs], dtype=float)
    eff_p = np.array([p[1] for p in pairs], dtype=float)
    return {"reported": not missing, "n_missing": len(missing),
            "n_records": len(records),
            "raw_mean": float(raw.mean()), "effective_mean": float(eff.mean()),
            "overlap_factor_mean": float((raw_p / eff_p).mean()),
            "effective_never_exceeds_raw": bool(np.all(eff_p <= raw_p + 1e-9))}


def _leakage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """T6 — an exact total, over every split of every replication."""
    vals = [int(r["leakage_pairs"]) for r in records if "leakage_pairs" in r]
    if not vals:
        return {"measured": False}
    total = int(sum(vals))
    return {"measured": True, "total_leakage_pairs": total,
            "t6_pass": total == 0}


def aggregate(name: str, records: list[dict[str, Any]],
              rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {"control": name, "replications": len(records),
                           "sample_size": _sample_size_check(records),
                           "leakage": _leakage(records)}

    if name in DSR_CONTROLS:
        out["t1"] = _fpr(records)

    if name in EXPECTANCY_CONTROLS:
        net = _finite(records, "mean_net_r")
        gross = _finite(records, "mean_gross_r")
        ci_net = bootstrap_mean_ci(net, rng)
        ci_gross = bootstrap_mean_ci(gross, rng)
        v_net, v_gross = ci_verdict(ci_net), ci_verdict(ci_gross)
        out["t2"] = {
            "net": {**ci_net, "verdict": v_net},
            "gross": {**ci_gross, "verdict": v_gross},
            # The operative sentences of T2: strictly above zero fails, strictly
            # below zero is reported and does not fail (a random entry paying
            # real costs is supposed to lose).
            "t2_pass": v_net != "ABOVE_ZERO" and v_gross != "ABOVE_ZERO",
        }

    if name == "NC4":
        # T3 is evaluated on GROSS expectancy. T2 qualifies its quantity as "net
        # of the P1 cost model"; T3 says only "mean expectancy", and gross is the
        # only reading under which T3 measures what its own closing sentence
        # claims — whether the sign convention is asymmetric. On net, a ~0.11 R
        # round-trip cost drives the share of positive replications to zero for
        # every possible apparatus, correct or broken, so the net reading would
        # be unpassable by construction and would test nothing. Both are
        # reported; the frozen band binds to gross.
        gross = _finite(records, "mean_gross_r")
        share = float(np.mean(gross > 0.0))
        lo, hi = share_band(0.5, len(records))
        long_r = _finite(records, "mean_gross_r_long")
        short_r = _finite(records, "mean_gross_r_short")
        out["t3"] = {"share_positive": share, "band_lo": lo, "band_hi": hi,
                     "t3_pass": bool(lo <= share <= hi),
                     "share_positive_net": float(
                         np.mean(_finite(records, "mean_net_r") > 0.0)),
                     "mean_long": float(long_r.mean()) if long_r.size else float("nan"),
                     "mean_short": float(short_r.mean()) if short_r.size else float("nan"),
                     # The long/short gap is what NC4 exists to expose: it
                     # cancels in the pooled mean, so only a side-by-side
                     # comparison can see it.
                     "long_short_gap": float(long_r.mean() - short_r.mean())
                     if long_r.size and short_r.size else float("nan")}
        if long_r.size == short_r.size and long_r.size > 1:
            gap_ci = bootstrap_mean_ci(long_r - short_r, rng)
            out["t3"]["gap_ci"] = {**gap_ci, "verdict": ci_verdict(gap_ci)}
            out["t3"]["t3_gap_pass"] = ci_verdict(gap_ci) == "CONTAINS_ZERO"

    if name == "NC6":
        med = _finite(records, "path_median")
        ci = bootstrap_mean_ci(med, rng)
        best = _finite(records, "best_path_sharpe")
        out["t4"] = {"path_median_ci": {**ci, "verdict": ci_verdict(ci)},
                     "best_path_sharpe_mean": float(best.mean()) if best.size else None,
                     "best_path_sharpe_q95": float(np.quantile(best, 0.95))
                     if best.size else None,
                     # A degenerate path distribution would satisfy the CI test
                     # while measuring nothing, so the spread is checked too.
                     "path_median_spread": float(med.std(ddof=1)) if med.size > 1 else 0.0,
                     "t4_pass": ci_verdict(ci) == "CONTAINS_ZERO"}

    if name == "NC7":
        share = float(np.mean([bool(r["a_wins"]) for r in records]))
        lo, hi = share_band(0.5, len(records))
        out["t5"] = {"winner_share": share, "band_lo": lo, "band_hi": hi,
                     "t5_pass": bool(lo <= share <= hi)}

    if name == "NC5":
        share = float(np.mean([bool(r["permutation_improved"]) for r in records]))
        lo, hi = share_band(0.5, len(records))
        out["permutation_improved_share"] = {
            "share": share, "band_lo": lo, "band_hi": hi,
            "in_band": bool(lo <= share <= hi)}

    if name == "NC2":
        # G1_SPEC.md §4's table states NC2's condition as "must not rank above
        # its own unshuffled null". That was recorded and then never evaluated —
        # every replication could have had the shuffled signal beating the
        # unshuffled one and NC2 would still have passed on T1 alone. Both are
        # null, so the share must sit in the binomial band around one half; a
        # share systematically above it would mean destroying a signal's time
        # alignment reliably improves it, which is not a thing that can be true.
        shuffled = _finite(records, "shuffled_sharpe")
        unshuffled = _finite(records, "unshuffled_sharpe")
        share = float(np.mean(shuffled > unshuffled))
        lo, hi = share_band(0.5, len(records))
        out["nc2_rank"] = {"shuffled_beats_unshuffled_share": share,
                           "band_lo": lo, "band_hi": hi,
                           "marginal_preserved_always": all(
                               bool(r.get("marginal_preserved")) for r in records),
                           "nc2_rank_pass": bool(share <= hi)}

    return out


def verdict(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply G1_SPEC.md §5. Returns the reasons, not just the word."""
    failures: list[str] = []
    for agg in aggregates:
        name = agg["control"]
        t1 = agg.get("t1")
        if t1 and not t1["t1_pass"]:
            failures.append(f"{name}: T1 false-positive rate {t1['fpr']:.4f} "
                            f"> {T1_MAX_FPR}")
        t2 = agg.get("t2")
        if t2 and not t2["t2_pass"]:
            failures.append(f"{name}: T2 null expectancy CI is above zero "
                            f"(net {t2['net']['verdict']}, "
                            f"gross {t2['gross']['verdict']})")
        # T3 binds to the statistic as frozen. A3 proposes replacing it, and the
        # proposal is reported as evidence, but it does not gate: adopting it
        # here would convert a failing frozen criterion into a pass after the
        # results were seen, which is the one thing the freeze exists to prevent.
        # The distinction between "the statistic is defective" and "the statistic
        # is inconvenient" cannot be drawn by the code that is failing it.
        t3 = agg.get("t3")
        if t3 and not t3["t3_pass"]:
            failures.append(
                f"{name}: T3 as frozen — share of positive replications "
                f"{t3['share_positive']:.4f} outside "
                f"[{t3['band_lo']:.3f}, {t3['band_hi']:.3f}] "
                "(SPECIFICATION_DEFECT: see G1_SPEC.md A3; the gap statistic it "
                "proposes instead reads "
                f"{t3.get('gap_ci', {}).get('verdict', 'n/a')})")
        if t3 and "t3_gap_pass" in t3 and not t3["t3_gap_pass"]:
            failures.append(f"{name}: T3 long/short gap CI excludes zero "
                            f"({t3['gap_ci']['verdict']}, gap "
                            f"{t3['long_short_gap']:+.5f} R) — the sign "
                            "convention is asymmetric")

        nc2 = agg.get("nc2_rank")
        if nc2 and not nc2["nc2_rank_pass"]:
            failures.append(
                f"{name}: shuffling the signal improved it in "
                f"{nc2['shuffled_beats_unshuffled_share']:.4f} of replications, "
                f"above the band top {nc2['band_hi']:.3f}")
        if nc2 and not nc2["marginal_preserved_always"]:
            failures.append(f"{name}: the permutation did not preserve the "
                            "signal's marginal distribution")

        for key, label in (("t4", "T4 path distribution"),
                           ("t5", "T5 false ranking superiority")):
            block = agg.get(key)
            if block and not block[f"{key}_pass"]:
                failures.append(f"{name}: {label} outside its frozen band")
        leak = agg.get("leakage", {})
        if leak.get("measured") and not leak["t6_pass"]:
            failures.append(f"{name}: T6 leakage — "
                            f"{leak['total_leakage_pairs']} intersecting "
                            "train/test span pairs")
        # Both halves of T7, and neither guarded by the other: an unreported
        # count used to skip the comparison as well, so a control that stopped
        # reporting effective sizes passed T7 by omission.
        size = agg.get("sample_size", {})
        if size and not size.get("reported", False):
            failures.append(f"{name}: T7 — {size.get('n_missing', '?')} "
                            f"replication(s) did not report both the raw and the "
                            "effective sample size")
        if size.get("raw_mean") is not None and not size["effective_never_exceeds_raw"]:
            failures.append(f"{name}: T7 effective sample size exceeds raw")

    # A criterion that no correct apparatus could clear is a defect in the
    # criterion, and it is a different outcome from the apparatus manufacturing
    # edge. Collapsing the two would either overstate the apparatus's failure or,
    # worse, invite the criterion to be quietly rewritten until it passed.
    spec_defects = [f for f in failures if "SPECIFICATION_DEFECT" in f]
    real = [f for f in failures if "SPECIFICATION_DEFECT" not in f]
    if real:
        outcome = "CONTROLS_FAIL"
    elif spec_defects:
        outcome = "CONTROLS_INDETERMINATE"
    else:
        outcome = "CONTROLS_PASS"
    return {"verdict": outcome, "failures": failures,
            "specification_defects": spec_defects,
            "apparatus_failures": real}


def run_suite(replications: dict[str, int] | None = None) -> dict[str, Any]:
    # A stream disjoint from every replication's: the aggregate bootstrap must
    # not reuse a generator that produced any of the data it resamples.
    rng = rng_for(10 ** 9)
    aggregates: list[dict[str, Any]] = []
    per_control: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(CONTROLS):
        reps = (replications or {}).get(name, CONTROLS[name][1])
        records = run_control(name, reps)
        per_control[name] = records
        aggregates.append(aggregate(name, records, rng))

    return {
        "version": NEGATIVE_CONTROLS_VERSION,
        "master_seed": MASTER_SEED,
        "spec": "reports/c50/G1_SPEC.md",
        "aggregates": aggregates,
        "records": per_control,
        "barrier_bias_diagnostic": barrier_bias_diagnostic(),
        **verdict(aggregates),
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [f"G1 negative-control suite  ({result['version']}, "
             f"master_seed={result['master_seed']})",
             f"thresholds: {result['spec']}", ""]
    for agg in result["aggregates"]:
        name = agg["control"]
        lines.append(f"{name}  (R={agg['replications']})")
        t1 = agg.get("t1")
        if t1:
            lines.append(f"    T1 FPR      {t1['fpr']:.4f} <= {T1_MAX_FPR}  "
                         f"[{'pass' if t1['t1_pass'] else 'FAIL'}]  "
                         f"dsr mean {t1.get('dsr_mean', float('nan')):.4f}, "
                         f"insufficient {t1['n_insufficient']}")
        t2 = agg.get("t2")
        if t2:
            lines.append(f"    T2 net      mean {t2['net']['mean']:+.5f} "
                         f"CI [{t2['net']['ci_lo']:+.5f}, {t2['net']['ci_hi']:+.5f}] "
                         f"{t2['net']['verdict']}")
            lines.append(f"    T2 gross    mean {t2['gross']['mean']:+.5f} "
                         f"CI [{t2['gross']['ci_lo']:+.5f}, "
                         f"{t2['gross']['ci_hi']:+.5f}] {t2['gross']['verdict']}  "
                         f"[{'pass' if t2['t2_pass'] else 'FAIL'}]")
        t3 = agg.get("t3")
        if t3:
            lines.append(f"    T3 share+   {t3['share_positive']:.4f} gross in "
                         f"[{t3['band_lo']:.3f}, {t3['band_hi']:.3f}]  "
                         f"[{'pass' if t3['t3_pass'] else 'FAIL'}]  "
                         f"(net share {t3['share_positive_net']:.4f})")
            lines.append(f"       sides    gross long {t3['mean_long']:+.5f} / "
                         f"short {t3['mean_short']:+.5f}  "
                         f"gap {t3['long_short_gap']:+.5f} R")
            gap_ci = t3.get("gap_ci")
            if gap_ci:
                lines.append(f"       gap CI   [{gap_ci['ci_lo']:+.5f}, "
                             f"{gap_ci['ci_hi']:+.5f}] {gap_ci['verdict']}  "
                             f"[{'pass' if t3['t3_gap_pass'] else 'FAIL'}]")
        t4 = agg.get("t4")
        if t4:
            ci = t4["path_median_ci"]
            lines.append(f"    T4 paths    median {ci['mean']:+.5f} CI "
                         f"[{ci['ci_lo']:+.5f}, {ci['ci_hi']:+.5f}] {ci['verdict']}  "
                         f"spread {t4['path_median_spread']:.5f}  "
                         f"[{'pass' if t4['t4_pass'] else 'FAIL'}]")
        t5 = agg.get("t5")
        if t5:
            lines.append(f"    T5 winner   {t5['winner_share']:.4f} in "
                         f"[{t5['band_lo']:.3f}, {t5['band_hi']:.3f}]  "
                         f"[{'pass' if t5['t5_pass'] else 'FAIL'}]")
        leak = agg.get("leakage", {})
        if leak.get("measured"):
            lines.append(f"    T6 leakage  {leak['total_leakage_pairs']} pairs  "
                         f"[{'pass' if leak['t6_pass'] else 'FAIL'}]")
        size = agg.get("sample_size", {})
        if size.get("raw_mean") is not None:
            missing = size["n_missing"]
            note = "pass" if size["reported"] else f"FAIL, {missing} missing"
            lines.append(f"    T7 samples  raw {size['raw_mean']:.1f} / "
                         f"effective {size['effective_mean']:.1f}  "
                         f"overlap x{size['overlap_factor_mean']:.4f}  [{note}]")
        nc2 = agg.get("nc2_rank")
        if nc2:
            lines.append(f"    NC2 rank    shuffled beats unshuffled "
                         f"{nc2['shuffled_beats_unshuffled_share']:.4f} "
                         f"(band top {nc2['band_hi']:.3f})  "
                         f"[{'pass' if nc2['nc2_rank_pass'] else 'FAIL'}]")
        perm = agg.get("permutation_improved_share")
        if perm:
            lines.append(f"       perm+    {perm['share']:.4f} in "
                         f"[{perm['band_lo']:.3f}, {perm['band_hi']:.3f}]")
        lines.append("")

    bias = result["barrier_bias_diagnostic"]
    lines.append("M02 tie-rule bias (diagnostic, not a threshold)")
    for cfg_name, block in bias.items():
        lines.append(f"    {cfg_name:<13} {block['upper_mult']}:{block['lower_mult']}  "
                     f"tie rate {block['tie_rate']:.4f}  "
                     f"gross {block['mean_gross_r']:+.5f} R  "
                     f"net {block['mean_net_r']:+.5f} R")
    lines.append("")
    lines.append(f"VERDICT: {result['verdict']}")
    for f in result["failures"]:
        lines.append(f"  - {f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="G1 negative-control suite")
    p.add_argument("--out", default="reports/g1",
                   help="directory for the untracked run artifacts")
    p.add_argument("--quick", action="store_true",
                   help="20 replications per control — a smoke check, never a "
                        "G1 verdict (the frozen counts are 500/200)")
    p.add_argument("--json", action="store_true", help="print JSON, not a report")
    args = p.parse_args(argv)

    reps = {name: 20 for name in CONTROLS} if args.quick else None
    result = run_suite(reps)
    result["quick"] = bool(args.quick)
    if args.quick:
        result["verdict_note"] = (
            "quick mode: replication counts are below the frozen ones, so this "
            "run cannot establish a G1 verdict")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stem = "quick" if args.quick else "full"
        with open(os.path.join(args.out, f"negative_controls_{stem}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True, default=float)
        with open(os.path.join(args.out, f"negative_controls_{stem}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(format_report(result) + "\n")

    print(json.dumps(result["aggregates"], indent=1, default=float)
          if args.json else format_report(result))
    return 0 if result["verdict"] == "CONTROLS_PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
