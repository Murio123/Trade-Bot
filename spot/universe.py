"""S2: the point-in-time investable universe.

Read-only. Builds one eligibility snapshot per decision date from bars that
closed strictly before that date, and nothing else. Criteria are frozen in
`reports/s2/S2_SPEC.md` §2 and must not be edited here — an amendment goes in
the spec first.

The whole module exists to answer one question honestly: *which symbols could a
person actually have bought on this date, knowing only what was knowable then.*
Every rule below is therefore evaluated on a trailing window, and the exclusion
lists describe what an asset **is** rather than how it turned out.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

DAY_MS = 86_400_000

# --- frozen thresholds (S2_SPEC §2) ---------------------------------------
MIN_LISTING_AGE_DAYS = 180
MIN_BARS = 200
LIQUIDITY_WINDOW = 30
MIN_MEDIAN_DOLLAR_VOLUME = 5_000_000.0
ACTIVITY_WINDOW_DAYS = 30
UNIVERSE_CAP = 250

FIRST_DECISION_DATE = "2018-01-01"
HORIZON_DAYS = 180

# Pegged assets: fiat- or commodity-pegged by design. Membership is a property
# of what the asset IS, decided before any outcome is known — UST is on this
# list for every date, including the ones where it moved violently, because it
# was designed as a peg on those dates too.
PEGGED_ASSETS = frozenset({
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "UST", "USTC", "FDUSD",
    "SUSD", "FRAX", "USDE", "USDS", "USDSB", "USDSOLD", "USD1", "RLUSD",
    "BFUSD", "XUSD", "EUR", "EURI", "AEUR", "GBP", "PAXG",
})

# Wrapped / staked representations of an asset already in the universe.
# Keeping both would count one bet twice.
WRAPPED_ASSETS = frozenset({"WBTC", "WBETH", "BETH", "WETH", "STETH", "BTCB",
                            "CBETH", "RETH", "SOLVBTC"})

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def is_leveraged_token(base: str, known_assets: frozenset[str]) -> bool:
    """A leveraged token is `<asset><UP|DOWN|BULL|BEAR>`, or bare BULL/BEAR.

    The suffix alone is not enough, and this is not a hypothetical: **JUP**
    (Jupiter) and **SYRUP** (Maple) both end in "UP" and are ordinary tokens.
    A naive suffix rule silently deletes them from every universe. So the
    prefix must itself be an asset the venue lists — ADAUP strips to ADA, JUP
    strips to "J", which is nothing.
    """
    for suffix in LEVERAGED_SUFFIXES:
        if not base.endswith(suffix):
            continue
        stem = base[: -len(suffix)]
        if stem == "" or stem in known_assets:
            return True
    return False


@dataclass(frozen=True)
class Screen:
    """One symbol's measured values at a decision date, and its verdict."""
    symbol: str
    eligible: bool
    reason: str
    bars: int
    listing_age_days: int
    median_dollar_volume: float
    last_close: float
    trading_now: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "eligible": self.eligible,
            "reason": self.reason, "bars": self.bars,
            "listing_age_days": self.listing_age_days,
            "median_dollar_volume": round(self.median_dollar_volume, 2),
            "last_close": self.last_close, "trading_now": self.trading_now,
        }


def month_starts(first: str, last_ms: int, horizon_days: int = HORIZON_DAYS
                 ) -> list[datetime]:
    """First-of-month decision dates with a full horizon inside the panel."""
    start = datetime.fromisoformat(first).replace(tzinfo=timezone.utc)
    end = (datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)
           - timedelta(days=horizon_days))
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def visible(df: pd.DataFrame, asof_ms: int) -> pd.DataFrame:
    """Bars that had closed before the decision instant. The whole PIT rule."""
    return df[df["close_time"] < asof_ms]


def screen_symbol(symbol: str, df: pd.DataFrame, asof_ms: int, *,
                  base_asset: str, trading_now: bool,
                  known_assets: frozenset[str]) -> Screen:
    """Apply E1-E8 to one symbol. E9 (the size cap) is cross-sectional."""
    def verdict(reason: str, seen: pd.DataFrame | None = None) -> Screen:
        n = 0 if seen is None else len(seen)
        age = 0
        vol = 0.0
        close = float("nan")
        if seen is not None and n:
            age = int((asof_ms - int(seen["open_time"].iloc[0])) // DAY_MS)
            tail = seen.tail(LIQUIDITY_WINDOW)
            vol = float(tail["quote_volume"].median())
            close = float(seen["close"].iloc[-1])
        return Screen(symbol=symbol, eligible=(reason == "eligible"),
                      reason=reason, bars=n, listing_age_days=age,
                      median_dollar_volume=vol, last_close=close,
                      trading_now=trading_now)

    if base_asset in PEGGED_ASSETS:
        return verdict("pegged")
    if base_asset in WRAPPED_ASSETS:
        return verdict("wrapped")
    if is_leveraged_token(base_asset, known_assets):
        return verdict("leveraged")

    seen = visible(df, asof_ms)
    if len(seen) < MIN_BARS:
        return verdict("too_few_bars", seen)

    age_days = (asof_ms - int(seen["open_time"].iloc[0])) // DAY_MS
    if age_days < MIN_LISTING_AGE_DAYS:
        return verdict("too_young", seen)

    tail = seen.tail(LIQUIDITY_WINDOW)
    if float(tail["quote_volume"].median()) < MIN_MEDIAN_DOLLAR_VOLUME:
        return verdict("illiquid", seen)

    # E8: a bar on each of the last 30 calendar days *before the decision
    # date*. Two conditions, and the second one is the one that matters.
    #
    # The first version checked only that the last 30 observed bars were
    # consecutive, which a symbol that stopped trading in 2022 satisfies
    # forever: its final 30 bars stay consecutive no matter how much later the
    # decision date is. An audit found 938 such stale rows still passing
    # eligibility, 927 of which then resolved as delisted failures — a coin
    # nobody could have bought, counted as a loss.
    #
    # This is the opposite error from survivorship bias and it is just as
    # wrong: a symbol is eligible at `t` only if it was still trading at `t`.
    # A symbol that trades at `t` and dies at `t + 90` remains eligible here,
    # and its death remains a failure — that part is unchanged.
    last_close = int(seen["close_time"].iloc[-1])
    if asof_ms - last_close > DAY_MS:
        return verdict("stale", seen)

    span_days = (int(tail["open_time"].iloc[-1])
                 - int(tail["open_time"].iloc[0])) // DAY_MS
    if len(tail) < ACTIVITY_WINDOW_DAYS or span_days != ACTIVITY_WINDOW_DAYS - 1:
        return verdict("inactive", seen)

    return verdict("eligible", seen)


def build_snapshot(panel: dict[str, pd.DataFrame], meta: dict[str, dict],
                   asof: datetime, *, cap: int = UNIVERSE_CAP) -> dict[str, Any]:
    """One decision date's eligibility, with every screen value recorded."""
    asof_ms = int(asof.timestamp() * 1000)
    known_assets = frozenset(m["base_asset"] for m in meta.values()
                             if m.get("base_asset"))

    screens = [screen_symbol(sym, df, asof_ms,
                             base_asset=meta[sym]["base_asset"],
                             trading_now=meta[sym]["trading_now"],
                             known_assets=known_assets)
               for sym, df in panel.items()]

    passed = sorted((s for s in screens if s.eligible),
                    key=lambda s: (-s.median_dollar_volume, s.symbol))
    capped = passed[:cap]
    capped_out = {s.symbol for s in passed[cap:]}

    rows = []
    for s in screens:
        d = s.as_dict()
        if s.symbol in capped_out:
            d["eligible"] = False
            d["reason"] = "below_size_cap"
        rows.append(d)
    rows.sort(key=lambda r: r["symbol"])

    eligible = [s.symbol for s in capped]
    payload = {
        "stage": "S2",
        "asof": asof.date().isoformat(),
        "asof_ms": asof_ms,
        "thresholds": {
            "min_listing_age_days": MIN_LISTING_AGE_DAYS,
            "min_bars": MIN_BARS,
            "liquidity_window": LIQUIDITY_WINDOW,
            "min_median_dollar_volume": MIN_MEDIAN_DOLLAR_VOLUME,
            "activity_window_days": ACTIVITY_WINDOW_DAYS,
            "cap": cap,
        },
        "counts": {
            "screened": len(screens),
            "eligible": len(eligible),
            "eligible_delisted_later": sum(1 for s in capped
                                           if not s.trading_now),
        },
        "eligible": eligible,
        "screens": rows,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "content_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def snapshot_path(outdir: str, asof: datetime) -> str:
    return os.path.join(outdir, f"{asof.date().isoformat()}.json")


def write_snapshot(payload: dict[str, Any], outdir: str) -> tuple[str, bool]:
    """Write once. An existing date is verified, never overwritten.

    Returns (path, written). A snapshot that reproduces byte-identically is a
    no-op; one that does not raises, because a universe that changes under a
    rerun cannot be the universe anything was measured against.
    """
    os.makedirs(outdir, exist_ok=True)
    asof = datetime.fromisoformat(payload["asof"]).replace(tzinfo=timezone.utc)
    path = snapshot_path(outdir, asof)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("content_sha256") != payload["content_sha256"]:
            raise ValueError(
                f"snapshot {path} already exists with different content "
                f"({str(existing.get('content_sha256'))[:12]} vs "
                f"{payload['content_sha256'][:12]}); snapshots are write-once")
        return path, False
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path, True


def load_snapshot(outdir: str, asof: datetime) -> dict[str, Any]:
    with open(snapshot_path(outdir, asof), encoding="utf-8") as fh:
        return json.load(fh)


# --- BTC regime (S2_SPEC §6) ----------------------------------------------

REGIME_SMA = 200
REGIME_SLOPE_LOOKBACK = 20


def btc_regime(btc: pd.DataFrame, asof_ms: int) -> str:
    """bull / bear / range from BTC's own trend, knowable at `asof_ms`."""
    seen = visible(btc, asof_ms)
    if len(seen) < REGIME_SMA + REGIME_SLOPE_LOOKBACK:
        return "unknown"
    sma = seen["close"].rolling(REGIME_SMA).mean()
    close = float(seen["close"].iloc[-1])
    now = float(sma.iloc[-1])
    then = float(sma.iloc[-1 - REGIME_SLOPE_LOOKBACK])
    if close > now and now > then:
        return "bull"
    if close < now and now < then:
        return "bear"
    return "range"
