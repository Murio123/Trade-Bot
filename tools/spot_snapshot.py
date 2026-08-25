"""S4A: build the current-market snapshot the Telegram screener reads.

Offline builder. It is the only piece of S4A that reaches the network or the
73 MB research panel, and it deliberately sits on the research side of the
boundary: `spot_market/` reads the file this writes and never computes
anything itself, so a Telegram handler can never accidentally start a
multi-year job.

    .venv/bin/python -m tools.spot_snapshot --source live
    .venv/bin/python -m tools.spot_snapshot --source panel

Two sources, one schema:

  * **live** — `/api/v3/exchangeInfo` for what actually trades right now, plus
    the last ~400 daily bars per candidate. Needs no local panel, which is
    what makes the screener deployable somewhere the panel does not exist.
  * **panel** — the verified S1 panel on disk. No network. Used by tests and
    by anyone who wants to rebuild a snapshot from data they already trust.

**Current eligibility is not the S2 rule and must not be confused with it.**
S2 asks "could a person have bought this on 2019-04-01", answered from bars
visible at that instant. This asks "can a person buy this today", and the
difference is the whole S2 E8 defect: checking that a symbol's last 30
*observed* bars are consecutive is satisfied forever by a coin that died in
2022. So every recency test here is anchored to **now**, not to the end of the
symbol's own history, and the venue's live status is required on top of it.

What this file computes are facts. Three of them share their definitions with
S3 features (B5 relative strength, F3 relative participation, F4 drawdown) and
that reuse is intentional: the numbers shown to a user should be the same
numbers the research measured. It is not a rehabilitation of those features.
S3's verdict stands — none of them ranked `CLEAN_2X(180d)` better than the
universe — and nothing here combines two of them or attaches a weight to one.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np
import pandas as pd

from spot.features import (abnormal_participation, distance_above_ma,
                           drawdown_from_high, relative_strength_90)
from spot.universe import (LIQUIDITY_WINDOW, MIN_BARS, MIN_LISTING_AGE_DAYS,
                           MIN_MEDIAN_DOLLAR_VOLUME, PEGGED_ASSETS,
                           WRAPPED_ASSETS, btc_regime, is_leveraged_token,
                           visible)
from tools.spot_cache import to_frame
from tools.spot_client import SpotClient, SpotClientError
from tools.spot_symbols import DEFAULT_OUTDIR, STATUS_TRADING

SCHEMA = "spot.snapshot/1"
BTC_SYMBOL = "BTCUSDT"
DAY_MS = 86_400_000
HOUR_MS = 3_600_000

DEFAULT_SNAPSHOT_DIR = os.path.join(DEFAULT_OUTDIR, "snapshot")
SNAPSHOT_NAME = "current.json"
LOCK_NAME = ".refresh.lock"
TMP_PREFIX = ".current.json."

# Exit codes. The scheduled refresher reads these, so they are part of the
# contract between the CLI and its caller rather than incidental.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_LOCKED = 2

# --- frozen current-eligibility thresholds ---------------------------------
# The first four are S2's, unchanged, because "enough history and enough
# liquidity to be worth looking at" does not become a different question in
# the present tense. The last one is new and is the one that replaces E8.
MAX_LAST_BAR_AGE_MS = 48 * HOUR_MS
ACTIVITY_WINDOW_DAYS = 30

# How many daily bars live mode pulls. 400 covers the longest window used
# here (200-day mean) with room for gaps; asking for less would silently drop
# symbols that have the history but not the bars in hand.
LIVE_BARS = 400

RET_30 = 31   # close[-1] / close[-31] - 1
RET_90 = 91
VOL_WINDOW = 30
TRADING_DAYS_PER_YEAR = 365  # crypto trades every day; no 252 here

# The numeric fields every coin carries, and the ones percentiles are taken
# over. Named once so the snapshot cannot gain a field the reader rejects.
FIELDS = ("price", "ret_30d", "ret_90d", "rel_btc_90d", "drawdown_180d",
          "median_quote_volume_30d", "realized_vol_30d", "relative_volume")


class SnapshotBuildError(Exception):
    """The snapshot cannot be built from what is available."""


class SnapshotLocked(Exception):
    """Another build already holds the lock for this snapshot directory."""


@contextlib.contextmanager
def build_lock(outdir: str) -> Iterator[None]:
    """Refuse to build while another build is running. Advisory, per directory.

    `flock` rather than a pid file, for one reason: the kernel releases it when
    the holding process dies, however it dies. A pid file survives a SIGKILL
    and then blocks every later run until somebody notices and deletes it,
    which is a worse failure than the one it prevents.

    This guards *every* invocation path, not just the scheduled one — the
    operator running the CLI by hand while the two-hourly job is mid-build is
    the overlap most likely to actually happen.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, LOCK_NAME)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - not a deployment target
            yield
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SnapshotLocked(
                f"another build holds {path}; skipping this run") from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {_utcnow_iso()}\n".encode())
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _utcnow_iso() -> str:
    return _iso(_utcnow_ms())


def _utcnow_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# --- current eligibility ----------------------------------------------------

def screen_now(symbol: str, df: pd.DataFrame, now_ms: int, *,
               base_asset: str, trading_now: bool,
               known_assets: frozenset[str]) -> str:
    """Why this symbol may or may not appear in today's scanner.

    Returns "eligible" or a reason token. Order matters only for reporting:
    the cheapest and most categorical tests run first so the counts read
    sensibly.
    """
    if not trading_now:
        return "not_trading"
    if base_asset in PEGGED_ASSETS:
        return "pegged"
    if base_asset in WRAPPED_ASSETS:
        return "wrapped"
    if is_leveraged_token(base_asset, known_assets):
        return "leveraged"

    seen = visible(df, now_ms)
    if len(seen) < MIN_BARS:
        return "too_few_bars"

    age_days = (now_ms - int(seen["open_time"].iloc[0])) // DAY_MS
    if age_days < MIN_LISTING_AGE_DAYS:
        return "too_young"

    tail = seen.tail(LIQUIDITY_WINDOW)
    if float(tail["quote_volume"].median()) < MIN_MEDIAN_DOLLAR_VOLUME:
        return "illiquid"

    # The E8 correction, and the order of the two checks IS the correction.
    #
    # `stale` is anchored to `now_ms`: a symbol whose last bar closed weeks ago
    # fails it, no matter how tidy that symbol's own history is. That is the
    # test S2 was missing, and it is the one a dead coin cannot pass.
    #
    # `inactive` then looks for holes, and it is allowed to count backwards
    # from the symbol's last bar because staleness has already established
    # that the last bar is recent. On its own this check is exactly the defect
    # — a coin delisted in 2022 has thirty perfectly consecutive final bars
    # forever — which is why it never runs on its own.
    last_close = int(seen["close_time"].iloc[-1])
    if now_ms - last_close > MAX_LAST_BAR_AGE_MS:
        return "stale"

    tail_days = seen.tail(ACTIVITY_WINDOW_DAYS)
    span_days = (int(tail_days["open_time"].iloc[-1])
                 - int(tail_days["open_time"].iloc[0])) // DAY_MS
    if (len(tail_days) < ACTIVITY_WINDOW_DAYS
            or span_days != ACTIVITY_WINDOW_DAYS - 1):
        return "inactive"

    return "eligible"


# --- facts ------------------------------------------------------------------

def _ret(closes: np.ndarray, lookback: int) -> float | None:
    if len(closes) < lookback:
        return None
    past = float(closes[-lookback])
    return float(closes[-1]) / past - 1.0 if past > 0 else None


def realized_vol(df: pd.DataFrame, window: int = VOL_WINDOW) -> float | None:
    """Annualised standard deviation of daily log returns.

    Reported because "this coin moves twice as much as that one" is a fact a
    person needs before sizing anything, and it is one of the few numbers here
    that is genuinely stable out of sample.
    """
    c = df["close"].to_numpy(dtype=float)
    if len(c) < window + 1:
        return None
    tail = c[-(window + 1):]
    if np.any(tail <= 0):
        return None
    rets = np.diff(np.log(tail))
    sd = float(np.std(rets, ddof=1))
    return sd * math.sqrt(TRADING_DAYS_PER_YEAR) if math.isfinite(sd) else None


def coin_facts(symbol: str, df: pd.DataFrame, btc: pd.DataFrame, now_ms: int,
               *, base_asset: str) -> dict[str, Any] | None:
    """Every displayed number for one asset, or None if one cannot be had.

    All-or-nothing on purpose: a card with a blank where the drawdown belongs
    invites the reader to fill the blank in themselves.
    """
    seen = visible(df, now_ms)
    btc_seen = visible(btc, now_ms)
    closes = seen["close"].to_numpy(dtype=float)

    values: dict[str, float | None] = {
        "price": float(closes[-1]) if len(closes) else None,
        "ret_30d": _ret(closes, RET_30),
        "ret_90d": _ret(closes, RET_90),
        "rel_btc_90d": relative_strength_90(seen, btc_seen)
        if len(seen) >= RET_90 else None,
        "drawdown_180d": drawdown_from_high(seen, btc_seen)
        if len(seen) >= 180 else None,
        "median_quote_volume_30d": float(
            seen.tail(LIQUIDITY_WINDOW)["quote_volume"].median()),
        "realized_vol_30d": realized_vol(seen),
        "relative_volume": abnormal_participation(seen, btc_seen)
        if len(seen) >= 180 else None,
    }
    if any(v is None or not math.isfinite(float(v))
           for v in values.values()):
        return None

    return {
        "symbol": symbol,
        "base_asset": base_asset,
        "bars": int(len(seen)),
        "listing_age_days": int((now_ms - int(seen["open_time"].iloc[0]))
                                // DAY_MS),
        "last_close_ms": int(seen["close_time"].iloc[-1]),
        **{k: float(v) for k, v in values.items()},  # type: ignore[arg-type]
    }


def btc_context(btc: pd.DataFrame, now_ms: int) -> dict[str, Any]:
    seen = visible(btc, now_ms)
    closes = seen["close"].to_numpy(dtype=float)
    if len(closes) < 200:
        raise SnapshotBuildError(
            f"{BTC_SYMBOL}: {len(closes)} visible bars, need at least 200 for "
            f"the benchmark context")
    for name, value in (("ret_30d", _ret(closes, RET_30)),
                        ("ret_90d", _ret(closes, RET_90)),
                        ("distance_above_ma200", distance_above_ma(seen, seen)),
                        ("realized_vol_30d", realized_vol(seen))):
        if value is None or not math.isfinite(value):
            raise SnapshotBuildError(f"{BTC_SYMBOL}: {name} is unavailable")
    return {
        "symbol": BTC_SYMBOL,
        "price": float(closes[-1]),
        "ret_30d": float(_ret(closes, RET_30)),      # type: ignore[arg-type]
        "ret_90d": float(_ret(closes, RET_90)),      # type: ignore[arg-type]
        "distance_above_ma200": float(distance_above_ma(seen, seen)),
        "realized_vol_30d": float(realized_vol(seen)),  # type: ignore[arg-type]
        "regime": btc_regime(btc, now_ms),
        "last_close_ms": int(seen["close_time"].iloc[-1]),
    }


def percentiles(coins: list[dict[str, Any]]) -> None:
    """Attach each coin's place in the current universe, field by field.

    Mid-rank: the share strictly below, plus half the ties. It is symmetric,
    it does not depend on sort order, and with a single coin it answers 50
    rather than dividing by zero.

    These are per-field positions and they stay per-field. Averaging them
    across fields would be a composite score by another name, which is the one
    thing S3's result forbids.
    """
    for field in FIELDS:
        vals = sorted(c[field] for c in coins)
        n = len(vals)
        for c in coins:
            v = c[field]
            below = sum(1 for u in vals if u < v)
            equal = sum(1 for u in vals if u == v)
            c.setdefault("percentiles", {})[field] = int(
                round(100.0 * (below + 0.5 * equal) / n))


# --- assembling -------------------------------------------------------------

def build_snapshot(frames: dict[str, pd.DataFrame], meta: dict[str, dict],
                   now_ms: int, *, source: str) -> dict[str, Any]:
    """Screen, measure and package. Pure: no I/O, no network, no clock."""
    if BTC_SYMBOL not in frames:
        raise SnapshotBuildError(
            f"{BTC_SYMBOL} is missing; it is the benchmark and every "
            f"BTC-relative number depends on it")
    btc = frames[BTC_SYMBOL]

    known_assets = frozenset(m.get("base_asset") or "" for m in meta.values())
    reasons: dict[str, int] = {}
    coins: list[dict[str, Any]] = []

    for symbol in sorted(frames):
        m = meta.get(symbol, {})
        base = m.get("base_asset") or symbol.removesuffix("USDT")
        reason = screen_now(symbol, frames[symbol], now_ms,
                            base_asset=base,
                            trading_now=bool(m.get("trading_now")),
                            known_assets=known_assets)
        if reason != "eligible":
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        facts = coin_facts(symbol, frames[symbol], btc, now_ms,
                           base_asset=base)
        if facts is None:
            reasons["incomplete_facts"] = reasons.get("incomplete_facts", 0) + 1
            continue
        coins.append(facts)

    if not coins:
        raise SnapshotBuildError(
            f"no symbol passed current eligibility; exclusions: {reasons}")

    percentiles(coins)
    btc_block = btc_context(btc, now_ms)

    # The oldest bar behind anything shown, not the newest. The reader's
    # freshness promise has to hold for every coin on the screen, so the
    # weakest link is the number that gets published.
    data_asof_ms = min([c["last_close_ms"] for c in coins]
                       + [btc_block["last_close_ms"]])

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "stage": "S4A",
        "venue": "binance_spot",
        "source": source,
        "generated_at_ms": now_ms,
        "generated_at": _iso(now_ms),
        "data_asof_ms": data_asof_ms,
        "data_asof": _iso(data_asof_ms),
        "thresholds": {
            "min_bars": MIN_BARS,
            "min_listing_age_days": MIN_LISTING_AGE_DAYS,
            "liquidity_window": LIQUIDITY_WINDOW,
            "min_median_dollar_volume": MIN_MEDIAN_DOLLAR_VOLUME,
            "max_last_bar_age_hours": MAX_LAST_BAR_AGE_MS // HOUR_MS,
            "activity_window_days": ACTIVITY_WINDOW_DAYS,
        },
        "counts": {
            "screened": len(frames),
            "eligible": len(coins),
            "excluded": dict(sorted(reasons.items())),
        },
        "btc": btc_block,
        "coins": sorted(coins, key=lambda c: c["symbol"]),
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _sweep_stale_temps(outdir: str) -> None:
    """Remove temp files left by a build that was killed mid-write.

    A process that takes a SIGKILL between opening its temp file and renaming
    it cannot clean up after itself, so somebody has to. They are never read
    and never harmful — but a directory that accumulates them is a directory
    nobody can look at and tell whether something is wrong.
    """
    try:
        names = os.listdir(outdir)
    except OSError:
        return
    for name in names:
        if name.startswith(TMP_PREFIX):
            try:
                os.unlink(os.path.join(outdir, name))
            except OSError:
                pass


def write_snapshot(payload: dict[str, Any],
                   outdir: str = DEFAULT_SNAPSHOT_DIR) -> str:
    """Replace the current snapshot atomically and durably.

    Unlike the S2 universe snapshots this one is *meant* to be overwritten —
    it describes now, and now moves. Three things make the overwrite safe:

      * **A temp file and a rename.** A Telegram handler may be reading at the
        moment of the replace; `os.replace` on the same filesystem is atomic,
        so it sees the whole old file or the whole new one. Writing in place
        would give it a half.
      * **fsync, on the file and then on the directory.** Without the first,
        a host that dies after the rename can come back with the directory
        entry pointing at unflushed content — an atomic rename to nothing. The
        second is what makes the rename itself durable; on most filesystems a
        rename is not persisted just because the process returned from it.
      * **A unique temp name.** Two writers, however unlikely given the lock,
        must not share a scratch file.
    """
    os.makedirs(outdir, exist_ok=True)
    _sweep_stale_temps(outdir)
    path = os.path.join(outdir, SNAPSHOT_NAME)
    fd, tmp = tempfile.mkstemp(prefix=TMP_PREFIX, dir=outdir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Includes KeyboardInterrupt and SystemExit: an interrupted publish
        # must not leave its scratch file behind either.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(outdir)
    return path


def _fsync_dir(outdir: str) -> None:
    """Persist the rename itself. Best-effort: not every filesystem allows it."""
    try:
        dfd = os.open(outdir, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


# --- sources ----------------------------------------------------------------

def _candidates(exchange_info: list[dict[str, Any]]) -> dict[str, dict]:
    """USDT pairs the venue reports as TRADING right now, with their assets."""
    out: dict[str, dict] = {}
    for row in exchange_info:
        if row.get("quoteAsset") != "USDT":
            continue
        out[row["symbol"]] = {
            "base_asset": row.get("baseAsset"),
            "trading_now": row.get("status") == STATUS_TRADING,
        }
    return out


async def fetch_live(*, concurrency: int = 8, bars: int = LIVE_BARS,
                     limit_symbols: int = 0
                     ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Current bars straight from the venue. No local panel required."""
    client = SpotClient()
    try:
        info = await client.exchange_info()
        meta = _candidates(info)
        known = frozenset(m["base_asset"] or "" for m in meta.values())
        wanted = [s for s, m in meta.items()
                  if m["trading_now"]
                  and (m["base_asset"] or "") not in PEGGED_ASSETS
                  and (m["base_asset"] or "") not in WRAPPED_ASSETS
                  and not is_leveraged_token(m["base_asset"] or "", known)]
        if BTC_SYMBOL not in wanted and BTC_SYMBOL in meta:
            wanted.append(BTC_SYMBOL)
        wanted.sort()
        if limit_symbols:
            wanted = sorted(set(wanted[:limit_symbols]) | {BTC_SYMBOL})

        sem = asyncio.Semaphore(concurrency)
        frames: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        async def one(symbol: str) -> None:
            async with sem:
                try:
                    rows = await client.klines(symbol, "1d", limit=bars)
                except SpotClientError:
                    failed.append(symbol)
                    return
            df, _dupes, _bad = to_frame(rows)
            if len(df):
                frames[symbol] = df

        await asyncio.gather(*(one(s) for s in wanted))
    finally:
        await client.close()

    if BTC_SYMBOL not in frames:
        raise SnapshotBuildError(
            f"{BTC_SYMBOL} did not return bars; refusing to build a snapshot "
            f"without the benchmark (failed: {len(failed)} symbols)")
    return frames, meta


def load_panel_frames(outdir: str = DEFAULT_OUTDIR
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """The verified S1 panel, read through its own fail-closed loader."""
    from tools.spot_dataset import load_panel

    with open(os.path.join(outdir, "symbols.json"), encoding="utf-8") as fh:
        table = json.load(fh)
    meta = {r["symbol"]: {"base_asset": r.get("base_asset"),
                          "trading_now": bool(r.get("trading_now"))}
            for r in table["symbols"]}
    frames = {s.symbol: s.df for s in load_panel(outdir)}
    return frames, meta


def format_report(payload: dict[str, Any], *, elapsed_s: float | None = None
                  ) -> str:
    counts = payload["counts"]
    top = sorted(counts["excluded"].items(), key=lambda kv: -kv[1])[:6]
    lines = [
        f"S4A — current spot snapshot ({payload['source']})",
        f"  generated_at : {payload['generated_at']}",
        f"  data_asof    : {payload['data_asof']}",
        f"  screened     : {counts['screened']}",
        f"  eligible     : {counts['eligible']}",
        f"  btc regime   : {payload['btc']['regime']}",
        "  excluded     : " + ", ".join(f"{k}={v}" for k, v in top),
    ]
    if elapsed_s is not None:
        lines.append(f"  duration     : {elapsed_s:.1f}s")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch or load, then screen and measure. No lock, no write."""
    if args.source == "live":
        frames, meta = asyncio.run(fetch_live(
            concurrency=args.concurrency, limit_symbols=args.limit_symbols))
    else:
        frames, meta = load_panel_frames(args.outdir)
    return build_snapshot(frames, meta, _utcnow_ms(), source=args.source)


def main(argv: list[str] | None = None) -> int:
    """The one canonical entry point. The scheduler calls exactly this.

    Order matters and is the whole of the failure contract: the lock is taken
    first, the snapshot is built and validated entirely in memory, and only a
    complete payload ever reaches the filesystem — through a temp file and an
    atomic rename. There is no point in this function at which the previous
    snapshot has been removed and the new one does not yet exist, so a build
    that dies anywhere leaves the last good snapshot exactly where it was.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("live", "panel"), default="live")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="where the S1 panel lives (panel source)")
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit-symbols", type=int, default=0,
                        help="live smoke runs: fetch at most N symbols")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and report, write nothing")
    args = parser.parse_args(argv)

    started = _utcnow_ms()
    print(f"spot-snapshot: refresh started (source={args.source}, "
          f"dest={args.snapshot_dir})", flush=True)
    try:
        with build_lock(args.snapshot_dir):
            payload = build(args)
            if not args.dry_run:
                path = write_snapshot(payload, args.snapshot_dir)
                print(f"  written      : {path}")
    except SnapshotLocked as exc:
        print(f"spot-snapshot: SKIPPED — {exc}; previous snapshot preserved",
              file=sys.stderr)
        return EXIT_LOCKED
    except Exception as exc:  # noqa: BLE001
        # Broad on purpose. This is an operational entry point, and every way
        # a build can fail — the venue, the panel loader's own fail-closed
        # checks, a malformed row, an OOM in pandas — has to end the same way:
        # exit 1, one legible line, previous snapshot untouched. A traceback
        # escaping here would still preserve the file (the write is the last
        # statement inside the lock and never runs on this path), but the
        # caller would have to parse a stack trace to learn that.
        print(f"spot-snapshot: FAILED — {type(exc).__name__}: {exc}; "
              f"previous snapshot preserved", file=sys.stderr)
        return EXIT_FAILED

    elapsed = (_utcnow_ms() - started) / 1000.0
    print(format_report(payload, elapsed_s=elapsed))
    print(f"spot-snapshot: refresh finished in {elapsed:.1f}s "
          f"(eligible={payload['counts']['eligible']}, "
          f"screened={payload['counts']['screened']}, "
          f"data_asof={payload['data_asof']})", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
