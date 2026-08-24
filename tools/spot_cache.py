"""S1: daily spot OHLCV for every USDT pair that ever existed.

Read-only research ingestion. It does NOT trade, does NOT write to the
database, does NOT touch scheduler, pipeline, signal_engine or futures logic,
and is NOT imported by runtime code.

Conventions are inherited from `tools/kline_cache.py` rather than reinvented:
the venue is pinned, pagination has no silent cap, a sidecar manifest records
provenance, and gaps are reported honestly instead of being smoothed away. Two
things are new, and both exist because this is a panel rather than one series:

  * **Dead pairs are ingested with the same priority as live ones.** The symbol
    list comes from `tools/spot_symbols.py`, where a third of the names no
    longer trade. Fetching only what trades today is precisely the survivorship
    bias the spot track is built to avoid, so a run that quietly skips them is
    a failed run, not a partial one — `_ingestion.json` records the dead/alive
    split so the omission cannot hide.
  * **Every dataset is content-hashed.** With 700+ files, "the manifest
    describes the file" cannot rest on the file having been written a moment
    ago. `tools/spot_dataset.py` re-checks the hash before it will load a
    symbol.

Columns. `taker_buy_base`, `taker_buy_quote` and Binance's `ignore` field are
dropped deliberately: CVD is a microstructure concept with no role in a
180-day question, and keeping them would double the panel's size on disk for
data nothing downstream reads. `quote_volume` is kept because dollar volume is
the liquidity screen S2 depends on.

    .venv/bin/python -m tools.spot_cache --outdir data/spot
    .venv/bin/python -m tools.spot_cache --outdir data/spot --symbols BTCUSDT
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from tools.kline_cache import gap_report
from tools.spot_client import SpotClient, SpotClientError
from tools.spot_symbols import DEFAULT_OUTDIR, load_table

KLINES_SUBDIR = "klines"
INGESTION_NAME = "_ingestion.json"
TIMEFRAME = "1d"

# What is kept from Binance's 12-column kline row, and in this order.
COLUMNS = ("open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_volume", "trades")
NUMERIC = ("open", "high", "low", "close", "volume", "quote_volume")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iso(ms: Any) -> str | None:
    if ms is None or (isinstance(ms, float) and pd.isna(ms)):
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


DAY_MS = 86_400_000


def to_frame(rows: list[list[Any]]) -> tuple[pd.DataFrame, int, int]:
    """Binance rows -> a sorted, deduplicated, self-consistent frame.

    Returns (df, duplicates_removed, malformed_removed).

    A daily bar must close exactly one day minus a millisecond after it opens.
    KLAYUSDT's final bar does not: it carries a `close_time` **earlier than its
    own `open_time`** and earlier than the previous bar's close. `open_time`
    stays monotonic throughout, so a frame like that passes an ordering check
    on open_time and then breaks any scan keyed on close_time — M02 refused it,
    which is what M02's timestamp guard is for.

    Malformed bars are dropped rather than repaired: a bar whose own timestamps
    contradict each other does not have a knowable close, and inventing one
    would be guessing at data.
    """
    if not rows:
        return pd.DataFrame(columns=list(COLUMNS)), 0, 0
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "ignore"])[list(COLUMNS)]
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("int64")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = (df.drop_duplicates(subset="open_time", keep="last")
            .sort_values("open_time")
            .reset_index(drop=True))
    dupes = before - len(df)

    consistent = df["close_time"] == df["open_time"] + DAY_MS - 1
    malformed = int((~consistent).sum())
    if malformed:
        df = df[consistent].reset_index(drop=True)
    return df, dupes, malformed


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_stem(symbol: str) -> str:
    return f"binance_spot_{symbol}_{TIMEFRAME}"


def build_manifest(df: pd.DataFrame, *, symbol: str, spine: dict[str, Any],
                   duplicates_removed: int, malformed_removed: int,
                   data_file: str, content_sha256: str) -> dict[str, Any]:
    first = int(df["open_time"].iloc[0]) if len(df) else None
    last = int(df["open_time"].iloc[-1]) if len(df) else None
    return {
        "stage": "S1",
        "exchange": "binance",
        "market": "spot",
        "endpoint": "/api/v3/klines",
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "fetched_at": _utcnow_iso(),
        "rows": int(len(df)),
        "first_open_time": first,
        "first_open_time_iso": _iso(first),
        "last_open_time": last,
        "last_open_time_iso": _iso(last),
        "columns": list(COLUMNS),
        "dropped_columns": ["taker_buy_base", "taker_buy_quote", "ignore"],
        "duplicate_count_removed": int(duplicates_removed),
        "malformed_bars_removed": int(malformed_removed),
        # Carried from the symbol spine so a reader of one file can see
        # whether they are holding a survivor or a casualty.
        "venue_status": spine.get("status"),
        "trading_now": spine.get("trading_now"),
        "data_file": os.path.basename(data_file),
        "content_sha256": content_sha256,
        **gap_report(df, TIMEFRAME),
    }


def write_symbol(df: pd.DataFrame, outdir: str, *, symbol: str,
                 spine: dict[str, Any], duplicates_removed: int,
                 malformed_removed: int = 0) -> dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    stem = dataset_stem(symbol)
    data_path = os.path.join(outdir, f"{stem}.csv")
    df.to_csv(data_path, index=False)
    manifest = build_manifest(df, symbol=symbol, spine=spine,
                              duplicates_removed=duplicates_removed,
                              malformed_removed=malformed_removed,
                              data_file=data_path,
                              content_sha256=sha256_file(data_path))
    with open(os.path.join(outdir, f"{stem}.manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return manifest


def already_cached(outdir: str, symbol: str) -> bool:
    stem = dataset_stem(symbol)
    return (os.path.exists(os.path.join(outdir, f"{stem}.csv"))
            and os.path.exists(os.path.join(outdir, f"{stem}.manifest.json")))


async def fetch_symbol(client: SpotClient, symbol: str, spine: dict[str, Any],
                       outdir: str) -> dict[str, Any]:
    rows = await client.full_history(symbol, TIMEFRAME)
    df, dupes, malformed = to_frame(rows)
    if df.empty:
        # Recorded, not raised: a symbol the venue lists but never published a
        # daily bar for is a fact about the venue, and a run that died on it
        # would lose the other 733.
        return {"symbol": symbol, "rows": 0, "empty": True}
    return write_symbol(df, outdir, symbol=symbol, spine=spine,
                        duplicates_removed=dupes,
                        malformed_removed=malformed)


async def ingest(symbols: Iterable[dict[str, Any]], outdir: str, *,
                 concurrency: int = 6, refresh: bool = False,
                 progress_every: int = 25) -> dict[str, Any]:
    klines_dir = os.path.join(outdir, KLINES_SUBDIR)
    os.makedirs(klines_dir, exist_ok=True)
    todo = list(symbols)
    client = SpotClient()
    sem = asyncio.Semaphore(concurrency)
    done: list[dict[str, Any]] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    empty: list[str] = []

    async def one(spine: dict[str, Any]) -> None:
        symbol = spine["symbol"]
        if not refresh and already_cached(klines_dir, symbol):
            skipped.append(symbol)
            return
        async with sem:
            try:
                res = await fetch_symbol(client, symbol, spine, klines_dir)
            except (SpotClientError, OSError, ValueError) as exc:
                failed.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
                return
        if res.get("empty"):
            empty.append(symbol)
        else:
            done.append(res)
        n = len(done) + len(failed) + len(empty)
        if progress_every and n % progress_every == 0:
            print(f"  ... {n}/{len(todo) - len(skipped)} fetched "
                  f"({len(failed)} failed)", flush=True)

    try:
        await asyncio.gather(*(one(s) for s in todo))
    finally:
        await client.close()

    alive = [m for m in done if m.get("trading_now")]
    dead = [m for m in done if not m.get("trading_now")]
    summary = {
        "stage": "S1",
        "finished_at": _utcnow_iso(),
        "timeframe": TIMEFRAME,
        "requested": len(todo),
        "written": len(done),
        "skipped_already_cached": len(skipped),
        "empty_no_bars": len(empty),
        "failed": len(failed),
        "written_trading_now": len(alive),
        "written_not_trading": len(dead),
        "total_rows": int(sum(m["rows"] for m in done)),
        "symbols_with_gaps": sum(1 for m in done if m.get("has_gaps")),
        "malformed_bars_removed": int(sum(m.get("malformed_bars_removed", 0)
                                          for m in done)),
        "empty_symbols": sorted(empty),
        "failures": failed,
    }
    with open(os.path.join(klines_dir, INGESTION_NAME), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return summary


def format_report(s: dict[str, Any]) -> str:
    return "\n".join([
        f"S1 — spot daily klines ({s['timeframe']})",
        f"  requested        : {s['requested']}",
        f"  written          : {s['written']}  "
        f"({s['written_trading_now']} trading, "
        f"{s['written_not_trading']} dead)",
        f"  skipped (cached) : {s['skipped_already_cached']}",
        f"  empty / failed   : {s['empty_no_bars']} / {s['failed']}",
        f"  rows             : {s['total_rows']:,}",
        f"  with gaps        : {s['symbols_with_gaps']}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--symbols", nargs="*",
                        help="restrict to these symbols (default: the spine)")
    parser.add_argument("--limit", type=int, default=0,
                        help="ingest at most N symbols (smoke runs)")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch symbols that are already cached")
    args = parser.parse_args(argv)

    spine = load_table(args.outdir)["symbols"]
    if args.symbols:
        wanted = set(args.symbols)
        spine = [s for s in spine if s["symbol"] in wanted]
        missing = wanted - {s["symbol"] for s in spine}
        if missing:
            raise SystemExit(f"not in the symbol spine: {sorted(missing)}")
    if args.limit:
        spine = spine[:args.limit]

    summary = asyncio.run(ingest(spine, args.outdir,
                                 concurrency=args.concurrency,
                                 refresh=args.refresh))
    print(format_report(summary))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
