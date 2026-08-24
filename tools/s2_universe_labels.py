"""S2 runner: build point-in-time universe snapshots, then label them.

Offline. Reads the S1 panel, writes universe snapshots and one labels file.
Evaluates no strategy, ranks nothing, and computes no feature.

    .venv/bin/python -m tools.s2_universe_labels --outdir data/spot \
        --reportdir reports/s2
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from spot.labels import CONFIG, Label, bar_index_asof, label_event, summarize
from spot.universe import (FIRST_DECISION_DATE, btc_regime, build_snapshot,
                           month_starts, write_snapshot)
from tools.spot_dataset import load_panel
from tools.spot_symbols import DEFAULT_OUTDIR, load_table

UNIVERSE_SUBDIR = "universe"
LABELS_SUBDIR = "labels"
LABELS_NAME = "clean_2x_180d.jsonl"
BTC = "BTCUSDT"


def load_inputs(outdir: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    spine = {r["symbol"]: r for r in load_table(outdir)["symbols"]}
    panel: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for series in load_panel(outdir):
        panel[series.symbol] = series.df
        row = spine.get(series.symbol, {})
        meta[series.symbol] = {
            "base_asset": row.get("base_asset") or series.symbol[:-4],
            "trading_now": bool(series.manifest.get("trading_now")),
        }
    return panel, meta


def run(outdir: str, *, limit_dates: int = 0) -> dict[str, Any]:
    panel, meta = load_inputs(outdir)
    btc = panel[BTC]
    last_ms = max(int(df["close_time"].iloc[-1]) for df in panel.values())
    dates = month_starts(FIRST_DECISION_DATE, last_ms)
    if limit_dates:
        dates = dates[:limit_dates]

    universe_dir = os.path.join(outdir, UNIVERSE_SUBDIR)
    labels: list[Label] = []
    per_date: list[dict[str, Any]] = []

    for asof in dates:
        snapshot = build_snapshot(panel, meta, asof)
        write_snapshot(snapshot, universe_dir)
        asof_ms = snapshot["asof_ms"]
        regime = btc_regime(btc, asof_ms)
        date_labels: list[Label] = []
        for symbol in snapshot["eligible"]:
            df = panel[symbol]
            idx = bar_index_asof(df, asof_ms)
            if idx is None:
                continue
            lab = label_event(df, idx, symbol=symbol,
                              asof=snapshot["asof"], asof_ms=asof_ms,
                              trading_now=meta[symbol]["trading_now"], btc=btc)
            date_labels.append(lab)
        labels.extend(date_labels)
        resolved = [l for l in date_labels if l.resolved]
        hits = sum(1 for l in resolved if l.clean_2x == 1)
        per_date.append({
            "asof": snapshot["asof"],
            "regime": regime,
            "eligible": len(snapshot["eligible"]),
            "resolved": len(resolved),
            "hits": hits,
            "base_rate": round(hits / len(resolved), 4) if resolved else None,
        })

    labels_dir = os.path.join(outdir, LABELS_SUBDIR)
    os.makedirs(labels_dir, exist_ok=True)
    with open(os.path.join(labels_dir, LABELS_NAME), "w", encoding="utf-8") as fh:
        for lab in labels:
            fh.write(json.dumps(lab.as_dict(), ensure_ascii=False) + "\n")

    regime_by_date = {d["asof"]: d["regime"] for d in per_date}
    by_regime: dict[str, dict[str, Any]] = {}
    for regime in sorted({d["regime"] for d in per_date}):
        subset = [l for l in labels if regime_by_date.get(l.asof) == regime]
        by_regime[regime] = summarize(subset)

    return {
        "stage": "S2",
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "barrier_config": {
            "upper_mult": CONFIG.upper_mult, "lower_mult": CONFIG.lower_mult,
            "vertical_bars": CONFIG.vertical_bars,
            "content_sha256": CONFIG.content_sha256(),
        },
        "decision_dates": len(dates),
        "first_date": dates[0].date().isoformat() if dates else None,
        "last_date": dates[-1].date().isoformat() if dates else None,
        # The honest N: 180-day windows that do not overlap each other.
        "independent_windows": round(len(dates) / 6.0, 1),
        "overall": summarize(labels),
        "by_regime": by_regime,
        "per_date": per_date,
    }


def format_report(s: dict[str, Any]) -> str:
    o = s["overall"]
    lines = [
        "S2 — universe + CLEAN_2X(180d) labels",
        f"  decision dates   : {s['decision_dates']} "
        f"({s['first_date']} .. {s['last_date']})",
        f"  independent windows: {s['independent_windows']}",
        f"  events           : {o['events']}  resolved {o['resolved']}",
        f"  outcomes         : {o['outcomes']}",
        f"  BASE RATE        : {o['base_rate']}  ({o['hits']} hits)",
        f"  ties             : {o['ties']}",
        f"  MAE   p10/med/p90: {o['mae']['p10']} / {o['mae']['median']} / "
        f"{o['mae']['p90']}",
        f"  MFE   med/p90    : {o['mfe']['median']} / {o['mfe']['p90']}",
        f"  term  med/p90    : {o['terminal_return']['median']} / "
        f"{o['terminal_return']['p90']}",
        f"  vs BTC med       : {o['excess_vs_btc']['median']}",
        f"  days to 2x (med) : {o['days_to_2x']['median']}",
        f"  MAE of hits (med): {o['hit_mae']['median']}",
        "",
        "  by regime:",
    ]
    for regime, r in s["by_regime"].items():
        lines.append(f"    {regime:8} events {r['events']:6}  "
                     f"resolved {r['resolved']:6}  base rate {r['base_rate']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--reportdir", default="reports/s2")
    parser.add_argument("--limit-dates", type=int, default=0)
    args = parser.parse_args(argv)

    summary = run(args.outdir, limit_dates=args.limit_dates)
    os.makedirs(args.reportdir, exist_ok=True)
    with open(os.path.join(args.reportdir, "s2_labels.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
