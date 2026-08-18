"""G1.1 — materialize the reconstructed trial ledger and report its counts.

Offline. Reads no market data, opens no sealed holdout, evaluates no strategy.
It writes the registry file and prints the confirmed / uncertain / conservative
decomposition for each family, which is the only form in which a trial count
may be published (TRIAL_REGISTRY_SPEC.md §8.3).

    python -m tools.g1_1_trial_ledger --outdir reports/g1_1

Re-running is safe and produces the same file: declarations are idempotent, so
a second run adds nothing and changes no count.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from validation.trial_history import ALL_TRIALS, populate, summary
from validation.trial_registry import FamilyScope, TrialRegistry

DEFAULT_OUTDIR = "reports/g1_1"
REGISTRY_NAME = "trial_registry.jsonl"

# The families a count would actually be quoted for. `A_PRIME_SCOPE` is the one
# Phase A' would use: swing trade expectancy on BTCUSDT.
A_PRIME_SCOPE = FamilyScope(research_objective="trade_expectancy",
                            profile="swing", symbol="BTCUSDT",
                            target_family="trade_r")

SCOPES: dict[str, FamilyScope] = {
    "a_prime_swing_trade_r": A_PRIME_SCOPE,
    "trade_expectancy_all_profiles": FamilyScope(
        research_objective="trade_expectancy", symbol="BTCUSDT",
        target_family="trade_r"),
    "volatility_forecast": FamilyScope(research_objective="volatility_forecast",
                                       symbol="BTCUSDT",
                                       target_family="volatility_ratio"),
    "everything": FamilyScope(),
}


def build(outdir: str) -> dict[str, Any]:
    path = os.path.join(outdir, REGISTRY_NAME)
    registry = TrialRegistry(path)
    populate(registry)
    counts = {name: registry.n_trials(scope).as_dict()
              for name, scope in SCOPES.items()}
    return {
        "stage": "G1.1",
        "registry": registry.snapshot(),
        "history": summary(),
        "counts": counts,
    }


def format_report(payload: dict[str, Any]) -> str:
    lines = ["G1.1 — trial registry", ""]
    snap = payload["registry"]
    lines.append(f"registry      : {snap['path']}")
    lines.append(f"sha256        : {snap['content_sha256']}")
    lines.append(f"declarations  : {snap['n_declarations']}")
    hist = payload["history"]
    lines.append(f"history       : {hist['history_version']}  "
                 f"records={hist['n_records']}")
    lines.append("")
    lines.append("Trial counts by family "
                 "(confirmed / uncertain / CONSERVATIVE = n_trials)")
    for name, count in payload["counts"].items():
        lines.append(
            f"  {name:34s} {count['confirmed']:5d} / "
            f"{count['uncertain']:5d} / {count['conservative']:5d}"
            f"   [{count['n_records']} records]")
    lines.append("")
    lines.append("By stage (records / confirmed / conservative)")
    for stage, bucket in hist["by_stage"].items():
        lines.append(f"  {stage:10s} {bucket['records']:4d} "
                     f"{bucket['confirmed']:6d} {bucket['conservative']:6d}")
    lines.append("")
    lines.append("A conservative count is the one DSR uses. The confirmed "
                 "count is reported for")
    lines.append("transparency and may not be used to deflate a published "
                 "result (§6.4).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    payload = build(args.outdir)
    report = format_report(payload)

    with open(os.path.join(args.outdir, "trial_counts.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.outdir, "trial_counts.txt"), "w") as fh:
        fh.write(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
