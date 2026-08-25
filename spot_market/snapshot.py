"""S4A: the fail-closed reader for the current-market snapshot.

One job: hand back facts that are actually current, or refuse. The refusal
matters more than the facts. A screener that quietly shows a three-week-old
price is worse than one that is unavailable, because the user cannot tell the
difference and the numbers look exactly as authoritative either way.

Two independent clocks are therefore checked, and both must pass:

    data_asof_ms     the oldest last-closed daily bar behind any coin shown
    generated_at_ms  when the snapshot file itself was built

The first catches a build that ran against a stale panel. The second catches a
snapshot that was fresh when it was written and has since been left to rot,
which is the failure mode of any cached artifact whose producer stops running.
Neither implies the other, so neither is enough on its own.

No network, no database, no research imports. Reading is all this module does.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA = "spot.snapshot/1"
DEFAULT_SNAPSHOT_PATH = os.path.join("data", "spot", "snapshot", "current.json")

# BTC is the benchmark and the market's context, and it is named here once so
# that no other module needs a hard-coded symbol. It is NOT a default subject:
# every function below takes the symbol it operates on explicitly.
BTC_SYMBOL = "BTCUSDT"

# --- frozen freshness thresholds -------------------------------------------
# A daily bar closes at 00:00 UTC. 48 hours lets a snapshot built early in the
# day still be served late the next day, and refuses anything that has missed
# a whole session. 24 hours on the file itself is the weaker of the two and
# binds first in practice, which is the intended order.
MAX_BAR_AGE_HOURS = 48
MAX_SNAPSHOT_AGE_HOURS = 24

HOUR_MS = 3_600_000

# Every numeric field a coin must carry. A snapshot missing one of these is
# rejected whole rather than rendered with a blank where a number belongs.
COIN_FIELDS = (
    "price", "ret_30d", "ret_90d", "rel_btc_90d", "drawdown_180d",
    "median_quote_volume_30d", "realized_vol_30d", "relative_volume",
)


class SnapshotUnavailable(Exception):
    """The current market cannot be described honestly right now.

    `reason` is a short machine-readable token so the caller can distinguish
    "never built" from "stale" without parsing prose.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class CoinFacts:
    """One currently tradable asset, as measured, with nothing derived from it.

    `percentiles` place each field inside the current eligible universe. They
    are context, not a score: they are never summed, averaged or weighted, and
    there is deliberately no field here that combines two others.
    """
    symbol: str
    base_asset: str
    price: float
    ret_30d: float
    ret_90d: float
    rel_btc_90d: float
    drawdown_180d: float
    median_quote_volume_30d: float
    realized_vol_30d: float
    relative_volume: float
    bars: int
    listing_age_days: int
    last_close_ms: int
    percentiles: dict[str, int]

    def field(self, name: str) -> float:
        if name not in COIN_FIELDS:
            raise KeyError(name)
        return float(getattr(self, name))

    def percentile(self, name: str) -> int | None:
        v = self.percentiles.get(name)
        return None if v is None else int(v)

    @property
    def excess_over_btc_90d(self) -> float:
        """The log excess expressed as a plain fraction, for display only."""
        return math.expm1(self.rel_btc_90d)


@dataclass(frozen=True)
class BtcContext:
    """The benchmark's own state. Context for altcoin reading, not a signal."""
    symbol: str
    price: float
    ret_30d: float
    ret_90d: float
    distance_above_ma200: float
    realized_vol_30d: float
    regime: str
    last_close_ms: int


@dataclass(frozen=True)
class MarketSnapshot:
    """A whole verified snapshot. Immutable, and already known to be fresh."""
    generated_at_ms: int
    data_asof_ms: int
    source: str
    venue: str
    coins: tuple[CoinFacts, ...]
    btc: BtcContext
    thresholds: dict[str, Any]
    counts: dict[str, Any]

    @property
    def universe_size(self) -> int:
        return len(self.coins)

    def by_symbol(self, symbol: str) -> CoinFacts | None:
        """Explicit lookup. There is no 'current' or 'default' symbol here."""
        for c in self.coins:
            if c.symbol == symbol:
                return c
        return None

    def symbols(self) -> list[str]:
        """Alphabetical — a stable order that asserts nothing about merit."""
        return sorted(c.symbol for c in self.coins)

    def age_hours(self, now_ms: int | None = None) -> float:
        now = _now_ms() if now_ms is None else now_ms
        return (now - self.data_asof_ms) / HOUR_MS


def _now_ms() -> int:
    return int(time.time() * 1000)


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _require(payload: dict[str, Any], key: str, path: str) -> Any:
    if key not in payload:
        raise SnapshotUnavailable("malformed",
                                  f"{path}: missing key {key!r}")
    return payload[key]


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotUnavailable("malformed", f"{where} is not a number")
    v = float(value)
    if not math.isfinite(v):
        raise SnapshotUnavailable("malformed", f"{where} is not finite")
    return v


def _coin(raw: dict[str, Any], path: str) -> CoinFacts:
    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise SnapshotUnavailable("malformed", f"{path}: a coin has no symbol")
    values = {name: _number(_require(raw, name, path), f"{symbol}.{name}")
              for name in COIN_FIELDS}
    pct_raw = raw.get("percentiles")
    if not isinstance(pct_raw, dict):
        raise SnapshotUnavailable("malformed",
                                  f"{symbol}: percentiles are missing")
    percentiles = {}
    for name in COIN_FIELDS:
        if name not in pct_raw:
            continue
        p = _number(pct_raw[name], f"{symbol}.percentiles.{name}")
        percentiles[name] = int(round(p))
    return CoinFacts(
        symbol=symbol,
        base_asset=str(raw.get("base_asset") or symbol.removesuffix("USDT")),
        bars=int(_number(_require(raw, "bars", path), f"{symbol}.bars")),
        listing_age_days=int(_number(_require(raw, "listing_age_days", path),
                                     f"{symbol}.listing_age_days")),
        last_close_ms=int(_number(_require(raw, "last_close_ms", path),
                                  f"{symbol}.last_close_ms")),
        percentiles=percentiles,
        **values)


def _btc(raw: Any, path: str) -> BtcContext:
    if not isinstance(raw, dict):
        raise SnapshotUnavailable("malformed", f"{path}: no btc block")
    fields = ("price", "ret_30d", "ret_90d", "distance_above_ma200",
              "realized_vol_30d")
    values = {n: _number(_require(raw, n, path), f"btc.{n}") for n in fields}
    regime = raw.get("regime")
    if regime not in ("bull", "bear", "range", "unknown"):
        raise SnapshotUnavailable("malformed", f"btc.regime is {regime!r}")
    return BtcContext(
        symbol=str(raw.get("symbol") or BTC_SYMBOL),
        regime=regime,
        last_close_ms=int(_number(_require(raw, "last_close_ms", path),
                                  "btc.last_close_ms")),
        **values)


def parse_snapshot(payload: dict[str, Any], *, path: str = "<memory>",
                   now_ms: int | None = None) -> MarketSnapshot:
    """Validate a snapshot payload and its freshness, or raise.

    Split out from `load_snapshot` so the freshness rule can be tested without
    a file, and so the rule lives in exactly one place for both callers.
    """
    now = _now_ms() if now_ms is None else int(now_ms)

    if payload.get("schema") != SCHEMA:
        raise SnapshotUnavailable(
            "schema", f"{path}: schema {payload.get('schema')!r} is not "
                      f"{SCHEMA!r}")

    generated_at_ms = int(_number(_require(payload, "generated_at_ms", path),
                                  "generated_at_ms"))
    data_asof_ms = int(_number(_require(payload, "data_asof_ms", path),
                               "data_asof_ms"))

    raw_coins = _require(payload, "coins", path)
    if not isinstance(raw_coins, list) or not raw_coins:
        raise SnapshotUnavailable("empty",
                                  f"{path}: the snapshot lists no coins")
    coins = tuple(_coin(c, path) for c in raw_coins)
    if len({c.symbol for c in coins}) != len(coins):
        raise SnapshotUnavailable("malformed", f"{path}: duplicate symbols")

    btc = _btc(payload.get("btc"), path)

    # Freshness last, so a malformed file is reported as malformed rather than
    # as stale — the two need different fixes.
    snapshot_age_h = (now - generated_at_ms) / HOUR_MS
    if snapshot_age_h > MAX_SNAPSHOT_AGE_HOURS:
        raise SnapshotUnavailable(
            "stale_snapshot",
            f"built {snapshot_age_h:.1f}h ago, limit {MAX_SNAPSHOT_AGE_HOURS}h")
    data_age_h = (now - data_asof_ms) / HOUR_MS
    if data_age_h > MAX_BAR_AGE_HOURS:
        raise SnapshotUnavailable(
            "stale_data",
            f"last closed bar is {data_age_h:.1f}h old, limit "
            f"{MAX_BAR_AGE_HOURS}h")
    # A snapshot from the future is a clock fault, not freshness. Refuse it:
    # the alternative is serving data whose timestamps cannot be trusted.
    if snapshot_age_h < -1.0 or data_age_h < -1.0:
        raise SnapshotUnavailable(
            "clock", f"{path}: timestamps are ahead of now "
                     f"(snapshot {snapshot_age_h:.1f}h, data {data_age_h:.1f}h)")

    return MarketSnapshot(
        generated_at_ms=generated_at_ms,
        data_asof_ms=data_asof_ms,
        source=str(payload.get("source") or "unknown"),
        venue=str(payload.get("venue") or "unknown"),
        coins=coins,
        btc=btc,
        thresholds=dict(payload.get("thresholds") or {}),
        counts=dict(payload.get("counts") or {}),
    )


def load_snapshot(path: str = DEFAULT_SNAPSHOT_PATH, *,
                  now_ms: int | None = None) -> MarketSnapshot:
    """Read and verify the snapshot at `path`, or raise `SnapshotUnavailable`."""
    if not os.path.exists(path):
        raise SnapshotUnavailable("missing", f"no snapshot at {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SnapshotUnavailable(
            "unreadable", f"{path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnapshotUnavailable("malformed", f"{path}: not an object")
    return parse_snapshot(payload, path=path, now_ms=now_ms)
