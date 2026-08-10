"""C4.3e: is Ridge's ranking advantage distinguishable from luck?

C4.3d returned RIDGE_RANKING_VALUE_CONFIRMED, but the honest size of the
effect is small. The trailing baseline is ANTI-correlated with this label, so
the headline signed advantage (+0.48) flatters the model: an observer who
simply inverted the baseline would score |rho|. Against that inverted
baseline Ridge's edge is about +0.13 Spearman. This stage asks whether +0.13
survives a null.

WHY A PLAIN TEST WOULD LIE. The rows are 12-bar overlapping windows: the
label's own autocorrelation is ~+0.93 at lag 1. Treating 4062 rows as 4062
independent observations would inflate significance enormously — the
effective sample is closer to 4062/12 ≈ 340, and less than that once regime
persistence is counted. Everything here therefore uses MOVING BLOCK
resampling with block lengths at or above the 12-bar horizon, and reports
the result at several block lengths rather than picking one.

Measurement only. Reads the per-row dump produced by C4.3d; retrains
nothing, rebuilds no dataset, touches no sealed holdout, changes no frozen
criterion.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from tools.forecast_platform.evaluation_engine import spearman_corr

STAGE = "C4.3e"
DEFAULT_INPUT = "reports/c43d/per_row.json"

# --- frozen before the test was run ----------------------------------------
BLOCK_LENGTHS = (12, 24, 60, 120)   # >= the 12-bar label horizon
N_RESAMPLES = 5000
SEED = 42
CI_ALPHA = 0.05                     # two-sided 95%

SIGNIFICANCE_RULE = (
    "ADVANTAGE_SIGNIFICANT only if, at EVERY tested block length "
    f"{BLOCK_LENGTHS}, the lower bound of the {int((1-CI_ALPHA)*100)}% moving-"
    "block bootstrap CI for (|rho_ridge| - |rho_baseline|) is strictly above "
    "zero, AND the block permutation test rejects at p < 0.05 at every block "
    "length. Otherwise ADVANTAGE_NOT_SIGNIFICANT. Block lengths are all >= "
    "the 12-bar label horizon because the rows are overlapping windows; the "
    "statistic is computed on |rho| so that the anti-correlated baseline is "
    "credited with the skill an observer could get by inverting it. Fixed "
    "here BEFORE the test was run."
)


def advantage(y: np.ndarray, ridge: np.ndarray, base: np.ndarray) -> float | None:
    """|rho(y, ridge)| - |rho(y, baseline)|.

    Absolute values on purpose: the baseline is anti-correlated, so a signed
    comparison would hand Ridge credit for skill the baseline already has in
    inverted form.
    """
    a, b = spearman_corr(y, ridge), spearman_corr(y, base)
    if a is None or b is None:
        return None
    return abs(a) - abs(b)


def _block_starts(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    return rng.integers(0, n, size=n_blocks)


def _circular_take(arr: np.ndarray, starts: np.ndarray, block: int, n: int
                   ) -> np.ndarray:
    offsets = np.arange(block)
    idx = (starts[:, None] + offsets[None, :]).ravel() % n
    return arr[idx[:n]]


def block_bootstrap_ci(y: np.ndarray, ridge: np.ndarray, base: np.ndarray,
                       block: int, n_resamples: int = N_RESAMPLES,
                       seed: int = SEED) -> dict[str, Any]:
    """Moving-block bootstrap CI for the advantage.

    Blocks are drawn circularly and the SAME row indices are used for all
    three series, so the pairing between a label and both predictions is
    never broken.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    offsets = np.arange(block)
    stats: list[float] = []
    for _ in range(n_resamples):
        starts = _block_starts(rng, n, block)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n] % n
        stat = advantage(y[idx], ridge[idx], base[idx])
        if stat is not None:
            stats.append(stat)
    arr = np.array(stats)
    lo, hi = np.quantile(arr, [CI_ALPHA / 2, 1 - CI_ALPHA / 2])
    return {"block": block, "n_resamples": len(arr),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "mean": float(arr.mean()), "frac_le_zero": float((arr <= 0).mean())}


def block_permutation_p(y: np.ndarray, ridge: np.ndarray, base: np.ndarray,
                        block: int, n_resamples: int = N_RESAMPLES,
                        seed: int = SEED) -> dict[str, Any]:
    """Null: Ridge carries no ranking information beyond the baseline.

    Ridge's predictions are circularly block-shifted against the labels,
    which destroys their alignment while preserving their autocorrelation
    structure. The baseline stays aligned, so the null keeps whatever skill
    the baseline genuinely has.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(y)
    observed = advantage(y, ridge, base)
    count = 0
    total = 0
    for _ in range(n_resamples):
        shift = int(rng.integers(block, n - block)) if n > 2 * block else 1
        stat = advantage(y, np.roll(ridge, shift), base)
        if stat is None:
            continue
        total += 1
        if stat >= observed:
            count += 1
    # +1/+1 keeps p strictly positive (Davison & Hinkley convention).
    p = (count + 1) / (total + 1)
    return {"block": block, "n_resamples": total, "p_value": float(p),
            "observed": observed}


def effective_sample_size(y: np.ndarray, horizon: int = 12) -> dict[str, Any]:
    """Crude but honest: overlapping windows of `horizon` bars mean roughly
    n/horizon independent observations. Reported so the raw row count is
    never mistaken for the real sample size."""
    lag1 = spearman_corr(y[1:], y[:-1])
    return {"n_rows": int(len(y)), "naive_independent": int(len(y) / horizon),
            "label_autocorr_lag1": lag1}


def decide_significance(bootstraps: list[dict[str, Any]],
                        permutations: list[dict[str, Any]]) -> tuple[str, dict]:
    """Pure application of SIGNIFICANCE_RULE."""
    ci_ok = all(b["ci_lo"] > 0 for b in bootstraps) and bool(bootstraps)
    perm_ok = all(p["p_value"] < 0.05 for p in permutations) and bool(permutations)
    checks = {
        "ci_lower_positive_at_every_block": ci_ok,
        "permutation_rejects_at_every_block": perm_ok,
        "worst_ci_lo": min((b["ci_lo"] for b in bootstraps), default=None),
        "worst_p_value": max((p["p_value"] for p in permutations), default=None),
    }
    verdict = ("ADVANTAGE_SIGNIFICANT" if ci_ok and perm_ok
               else "ADVANTAGE_NOT_SIGNIFICANT")
    return verdict, checks


def required_n_for_power(observed_adv: float, eff_n: int, n_rows: int,
                         target_se_ratio: float = 2.0) -> dict[str, Any]:
    """How many MATURED forward forecasts a live ledger would need.

    Uses the standard SE of a Spearman coefficient, 1/sqrt(m-1), on EFFECTIVE
    observations, and asks for the advantage to sit `target_se_ratio` SEs
    above zero. Forward forecasts at a 48h horizon on 4h bars overlap the
    same way, so the row requirement is scaled back up by the same factor.
    """
    if not observed_adv or observed_adv <= 0:
        return {"applicable": False}
    m_eff = (target_se_ratio / observed_adv) ** 2 + 1
    overlap = max(n_rows / eff_n, 1.0) if eff_n else 1.0
    return {"applicable": True, "target_se_ratio": target_se_ratio,
            "effective_obs_needed": int(np.ceil(m_eff)),
            "matured_forecasts_needed": int(np.ceil(m_eff * overlap))}


def run(input_path: str = DEFAULT_INPUT, outdir: str = "reports/c43e"
        ) -> dict[str, Any]:
    with open(input_path) as fh:
        rows = json.load(fh)
    if not rows:
        raise SystemExit(f"{input_path} is empty — run C4.3d first")

    y = np.asarray(rows["label"], dtype=float)
    ridge = np.asarray(rows["ridge_pred"], dtype=float)
    base = np.asarray(rows["rolling_series_pred"], dtype=float)
    ok = ~np.isnan(base)
    y, ridge, base = y[ok], ridge[ok], base[ok]

    observed_signed = None
    a, b = spearman_corr(y, ridge), spearman_corr(y, base)
    if a is not None and b is not None:
        observed_signed = a - b
    observed = advantage(y, ridge, base)

    bootstraps = [block_bootstrap_ci(y, ridge, base, bl) for bl in BLOCK_LENGTHS]
    permutations = [block_permutation_p(y, ridge, base, bl) for bl in BLOCK_LENGTHS]
    verdict, checks = decide_significance(bootstraps, permutations)
    eff = effective_sample_size(y)
    power = required_n_for_power(observed or 0.0, eff["naive_independent"],
                                 eff["n_rows"])

    result = {
        "stage": STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_path,
        "n_rows_used": int(len(y)),
        "spearman_ridge": a, "spearman_baseline": b,
        "advantage_signed": observed_signed,
        "advantage_vs_abs": observed,
        "effective_sample": eff,
        "bootstrap": bootstraps,
        "permutation": permutations,
        "checks": checks,
        "forward_sample_requirement": power,
        "decision_rule": SIGNIFICANCE_RULE,
        "verdict": verdict,
    }

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "ranking_significance.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    with open(os.path.join(outdir, "ranking_significance.txt"), "w") as fh:
        fh.write(format_report(result))
    return result


def format_report(r: dict[str, Any]) -> str:
    eff = r["effective_sample"]
    lines = [
        f"Ridge ranking significance ({r['stage']})",
        f"  generated_at : {r['generated_at_utc']}",
        f"  rows used    : {r['n_rows_used']}",
        f"  effective    : ~{eff['naive_independent']} independent "
        f"(label autocorr lag1 = {eff['label_autocorr_lag1']})",
        "",
        f"  spearman ridge    : {r['spearman_ridge']}",
        f"  spearman baseline : {r['spearman_baseline']}",
        f"  advantage signed  : {r['advantage_signed']}",
        f"  advantage vs |rho|: {r['advantage_vs_abs']}   <- the honest one",
        "",
        "moving-block bootstrap (95% CI for advantage vs |rho|):",
    ]
    for b in r["bootstrap"]:
        lines.append(f"  block {b['block']:>4}: [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] "
                     f"mean {b['mean']:+.4f}  P(adv<=0)={b['frac_le_zero']:.4f}")
    lines.append("")
    lines.append("block permutation (null: ridge adds no ranking information):")
    for p in r["permutation"]:
        lines.append(f"  block {p['block']:>4}: p = {p['p_value']:.5f}  "
                     f"(n={p['n_resamples']})")
    lines.append("")
    lines.append(f"checks: {r['checks']}")
    lines.append(f"forward sample requirement: {r['forward_sample_requirement']}")
    lines.append("")
    lines.append(f"VERDICT: {r['verdict']}")
    lines.append(f"decision_rule: {r['decision_rule']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C4.3e: ranking-advantage significance")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--outdir", default="reports/c43e")
    args = p.parse_args(argv)
    result = run(args.input, args.outdir)
    print(format_report(result))
    print(f"VERDICT: {result['verdict']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
