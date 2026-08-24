"""S1: the survivorship-free symbol spine.

Read-only, offline. Writes one file, `data/spot/symbols.json`, and nothing
else. No filtering, no eligibility, no judgement about what is investable —
that is S2's job and doing it here would bake a universe decision into the
data layer where it could never be revisited.

What this module is actually for. `exchangeInfo` is the present tense: it
lists what trades today. Building a historical universe from it is the single
most common way a crypto backtest lies to itself — every coin that died is
missing, so the past looks populated entirely by survivors. The public
data-dump index still lists pairs that were fully delisted years ago, so the
union of the two is the closest thing available to "every USDT pair that ever
existed".

Each symbol is recorded with *why* it is here:

    in_dump_index      — the venue published historical data for it
    in_exchange_info   — the venue still knows about it today
    status             — the venue's own word: TRADING, BREAK, ... or GONE
                         when the symbol survives only in the dump index
    trading_now        — derived: status == TRADING

A pair that stops trading is **not** removed from `exchangeInfo`. It stays,
with `status = "BREAK"` — BCCUSDT, delisted in 2018, is still listed today.
Deriving "delisted" from absence therefore finds almost nothing, and a first
run of this module reported exactly one dead pair out of 734. The real answer
is `status != "TRADING"`, and it is 249 — **a third of every USDT pair that
ever existed**. That number is the size of the survivorship problem this stage
exists to control.

`GONE` and the dead/alive split are derived, not sourced: the venue publishes
no delisting list. The derivation is stated in the artifact so a reader can
disagree with it without re-deriving it.

    .venv/bin/python -m tools.spot_symbols --outdir data/spot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from tools.spot_client import SpotClient, usdt_pairs

DEFAULT_OUTDIR = "data/spot"
SYMBOLS_NAME = "symbols.json"
STATUS_GONE = "GONE"
STATUS_TRADING = "TRADING"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_table(dump_symbols: list[str],
                exchange_info: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the two views into one table, keeping the provenance of each."""
    info_by_symbol = {row["symbol"]: row for row in exchange_info}
    dump_usdt = set(usdt_pairs(dump_symbols))
    info_usdt = {s for s, row in info_by_symbol.items()
                 if row.get("quoteAsset") == "USDT"}

    records: list[dict[str, Any]] = []
    for symbol in sorted(dump_usdt | info_usdt):
        row = info_by_symbol.get(symbol)
        records.append({
            "symbol": symbol,
            "base_asset": (row or {}).get("baseAsset"),
            "quote_asset": "USDT",
            "in_dump_index": symbol in dump_usdt,
            "in_exchange_info": row is not None,
            # The venue's own word where it has one. A symbol that survives
            # only in the dump index gets GONE, which is ours.
            "status": (row.get("status") if row else STATUS_GONE),
            "trading_now": bool(row) and row.get("status") == STATUS_TRADING,
        })

    dead = [r for r in records if not r["trading_now"]]
    trading = [r for r in records if r["trading_now"]]
    return {
        "stage": "S1",
        "fetched_at": _utcnow_iso(),
        "quote_asset": "USDT",
        "sources": {
            "exchange_info": {
                "endpoint": "/api/v3/exchangeInfo",
                "symbols_total": len(info_by_symbol),
                "usdt_pairs": len(info_usdt),
            },
            "dump_index": {
                "endpoint": "data.binance.vision spot/monthly/klines index",
                "symbols_total": len(dump_symbols),
                "usdt_pairs": len(dump_usdt),
            },
        },
        "counts": {
            "usdt_pairs_ever": len(records),
            "trading_now": len(trading),
            "not_trading": len(dead),
            "in_dump_only": sum(1 for r in records if r["in_dump_index"]
                                and not r["in_exchange_info"]),
            "in_exchange_info_only": sum(1 for r in records
                                         if r["in_exchange_info"]
                                         and not r["in_dump_index"]),
        },
        "derivation": (
            "trading_now is derived as status == 'TRADING'. A pair that stops "
            "trading keeps its exchangeInfo entry with status 'BREAK' rather "
            "than disappearing, so absence from exchangeInfo (status 'GONE') "
            "identifies almost none of the dead pairs and must not be used as "
            "the delisting test. The venue publishes no delisting list; this "
            "is an inference from two sources, not a fact read from one."
        ),
        "symbols": records,
    }


async def collect(base: str | None = None) -> dict[str, Any]:
    client = SpotClient() if base is None else SpotClient(base=base)
    try:
        dumps, info = await asyncio.gather(client.dump_symbols(),
                                           client.exchange_info())
    finally:
        await client.close()
    return build_table(dumps, info)


def write_table(table: dict[str, Any], outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, SYMBOLS_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def load_table(outdir: str = DEFAULT_OUTDIR) -> dict[str, Any]:
    with open(os.path.join(outdir, SYMBOLS_NAME), encoding="utf-8") as fh:
        return json.load(fh)


def format_report(table: dict[str, Any], path: str) -> str:
    c = table["counts"]
    return "\n".join([
        "S1 — spot symbol spine (USDT)",
        f"  USDT pairs ever : {c['usdt_pairs_ever']}",
        f"  trading now     : {c['trading_now']}",
        f"  NOT trading     : {c['not_trading']}  "
        f"<- the survivorship-free part",
        f"  dump index only : {c['in_dump_only']}",
        f"  exchangeInfo only: {c['in_exchange_info_only']}  "
        f"(too new to have dumps)",
        f"  written         : {path}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    table = asyncio.run(collect())
    path = write_table(table, args.outdir)
    print(json.dumps(table["counts"], indent=2) if args.json
          else format_report(table, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
