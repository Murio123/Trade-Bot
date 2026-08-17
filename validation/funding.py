"""P1: realized funding over an actual holding interval.

A perpetual position pays (or receives) funding at every settlement it is held
through. Every cost path in this project charged fees and slippage and charged
nothing for funding, so every net number it produced was optimistic. This
module is the one place that answers "what funding did this trade actually
pay", and `validation.trade_costs` is the one place that composes it with fees.

Four decisions, each of which could have been made wrongly:

**Funding is signed, not a cost.** A long pays when the rate is positive and is
*credited* when it is negative. 15% of BTCUSDT settlements in the measured
history are negative, so clamping funding to a non-negative cost would be a
falsification, not a conservatism. `realized_funding_pct` returns a signed
percent: positive means the position paid.

**The boundary convention is half-open: `(entry, exit]`.** A settlement exactly
at the entry stamp is NOT charged — the position was not held when that
snapshot was taken. A settlement exactly at the exit stamp IS charged — it was.
The convention has to be stated because both halves are defensible and picking
silently would make two implementations disagree by one whole settlement.

**Actual settlement rates, never an extrapolation.** `latest_rate x holding
time` is forbidden by P1 and would be wrong for the same reason the sign
matters: the rate changes, and it changes sign, inside a single swing hold.

**Missing data fails closed.** If the interval reaches outside the observed
window, or a gap in the observed series overlaps it, the answer is
`FundingDataUnavailable` — not zero. Zero is returned only when the series
proves that no settlement was crossed.

One documented approximation: funding is charged on the notional at ENTRY
price. Binance's historical `fundingRate` records carry `markPrice` only for
part of the history (63.8% on the measured BTCUSDT set), so per-settlement
mark-to-market notional is not available for the whole window. The residual
error is a few percent of a term worth ~0.06pp per 48h trade — below 0.005pp —
and it keeps the funding term normalised exactly like the fee term, which is
also charged on entry notional.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from validation.bar_grid import GAP_TOLERANCE, gap_intervals, modal_interval_ms

FUNDING_ACCOUNTING_VERSION = "p1_funding_v1"

# GAP_TOLERANCE is re-exported from validation.bar_grid, where the same rule
# serves M02's kline grid. It is not redefined here: two copies of a tolerance
# is how two modules come to disagree about what a gap is.

LONG = "long"
SHORT = "short"


class FundingError(Exception):
    """Malformed input — a programming error, not a data-coverage limit."""


class FundingDataUnavailable(FundingError):
    """The interval cannot be answered from the observed series.

    Deliberately a separate type: callers are expected to propagate this as
    "unavailable", never to swallow it into a zero.
    """


def _sign(side: Any) -> float:
    """+1 for a long, -1 for a short. Accepts the project's two conventions."""
    if isinstance(side, str):
        s = side.strip().lower()
        if s == LONG:
            return 1.0
        if s == SHORT:
            return -1.0
        raise FundingError(f"side must be 'long' or 'short', got {side!r}")
    if isinstance(side, bool):
        raise FundingError("side must not be a bool")
    if isinstance(side, (int, float, np.integer, np.floating)):
        if side == 1:
            return 1.0
        if side == -1:
            return -1.0
    raise FundingError(f"side must be 'long'/'short' or +1/-1, got {side!r}")


def to_ms(value: Any) -> int:
    """Timestamps arrive as epoch ms, datetimes and pandas stamps alike.

    Naive stamps are read as UTC, matching every dataset in the project.
    """
    if isinstance(value, bool):
        raise FundingError("timestamp must not be a bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise FundingError(f"timestamp is not finite: {value!r}")
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    if pd.isna(ts):
        raise FundingError(f"timestamp is not parseable: {value!r}")
    return int(ts.value // 1_000_000)


@dataclass(frozen=True)
class FundingSeries:
    """Observed funding settlements, sorted, with an explicit coverage window.

    `rates` are fractions (0.0001 == 0.01%), the units Binance returns.
    Construct through `from_records` / `load_funding_series` so the sorting,
    de-duplication and gap detection are never skipped.
    """
    times_ms: np.ndarray
    rates: np.ndarray
    symbol: str
    exchange: str
    modal_interval_ms: int
    gaps: tuple[tuple[int, int], ...] = ()
    source_sha256: str | None = None
    version: str = FUNDING_ACCOUNTING_VERSION

    # -- construction ------------------------------------------------------

    @classmethod
    def from_records(cls, times: Sequence[Any], rates: Sequence[float], *,
                     symbol: str, exchange: str,
                     source_sha256: str | None = None) -> "FundingSeries":
        if len(times) != len(rates):
            raise FundingError(
                f"times and rates differ in length: {len(times)} vs {len(rates)}")
        if len(times) == 0:
            raise FundingError("a funding series cannot be empty")

        t = np.array([to_ms(v) for v in times], dtype=np.int64)
        r = np.asarray(rates, dtype=float)
        if not np.isfinite(r).all():
            raise FundingError("funding rates must all be finite")

        order = np.argsort(t, kind="stable")
        t, r = t[order], r[order]
        keep = np.ones(t.size, dtype=bool)
        keep[1:] = t[1:] != t[:-1]
        t, r = t[keep], r[keep]

        modal = modal_interval_ms(t)
        return cls(times_ms=t, rates=r, symbol=symbol, exchange=exchange,
                   modal_interval_ms=modal, gaps=gap_intervals(t, modal),
                   source_sha256=source_sha256)

    # -- coverage ----------------------------------------------------------

    @property
    def first_ms(self) -> int:
        return int(self.times_ms[0])

    @property
    def last_ms(self) -> int:
        return int(self.times_ms[-1])

    def __len__(self) -> int:
        return int(self.times_ms.size)

    def covers(self, entry_ms: int, exit_ms: int) -> bool:
        """Can every settlement inside `(entry, exit]` be named?

        Requires the interval to sit inside the observed window and to miss
        every known gap. `entry >= first_ms` is deliberately strict: before the
        first observed settlement the series says nothing, and regular spacing
        is an assumption this module refuses to make.
        """
        if exit_ms < entry_ms:
            raise FundingError(
                f"exit {exit_ms} precedes entry {entry_ms}")
        if entry_ms < self.first_ms or exit_ms > self.last_ms:
            return False
        # A gap is the OPEN interval between two observed stamps: the missing
        # settlements lie strictly inside it. Comparing non-strictly would
        # refuse a trade that merely starts or ends on the gap's edge, where
        # nothing is actually unknown.
        return not any(g_lo < exit_ms and g_hi > entry_ms
                       for g_lo, g_hi in self.gaps)

    # -- accounting --------------------------------------------------------

    def settlement_rates(self, entry_ms: int, exit_ms: int) -> np.ndarray:
        """Rates settled in `(entry, exit]`, in chronological order.

        Raises `FundingDataUnavailable` rather than returning a short answer:
        an incomplete settlement list is indistinguishable from a cheap trade.
        """
        entry_ms, exit_ms = to_ms(entry_ms), to_ms(exit_ms)
        if not self.covers(entry_ms, exit_ms):
            raise FundingDataUnavailable(
                f"funding series {self.exchange}:{self.symbol} does not cover "
                f"({entry_ms}, {exit_ms}]; observed "
                f"[{self.first_ms}, {self.last_ms}], {len(self.gaps)} gap(s)")
        lo = int(np.searchsorted(self.times_ms, entry_ms, side="right"))
        hi = int(np.searchsorted(self.times_ms, exit_ms, side="right"))
        return self.rates[lo:hi]

    def realized_funding_pct(self, entry_ms: Any, exit_ms: Any,
                             side: Any) -> float:
        """Funding paid over `(entry, exit]`, in percent of entry notional.

        Positive = the position paid. Negative = the position was credited.
        """
        rates = self.settlement_rates(to_ms(entry_ms), to_ms(exit_ms))
        if rates.size == 0:
            return 0.0
        return float(_sign(side) * rates.sum() * 100.0)

    def settlement_count(self, entry_ms: Any, exit_ms: Any) -> int:
        return int(self.settlement_rates(to_ms(entry_ms), to_ms(exit_ms)).size)

    def provenance(self) -> dict[str, Any]:
        """What a report has to print so the number can be reproduced."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "settlements": len(self),
            "first_funding_time": self.first_ms,
            "last_funding_time": self.last_ms,
            "modal_interval_ms": self.modal_interval_ms,
            "gap_count": len(self.gaps),
            "source_sha256": self.source_sha256,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def dataset_stem(exchange: str, symbol: str) -> str:
    """Single source of truth for the cache filename.

    Defined here rather than in tools.funding_cache because production code
    imports this module, and production must never import the offline tools
    layer (the invariant tests/test_realized_r.py guards). The writer imports
    the name from here; the dependency points tools -> validation, never back.
    """
    return f"{exchange}_{symbol}_funding"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_funding_series(outdir: str = "data/funding", *,
                        exchange: str = "binance", symbol: str = "BTCUSDT",
                        verify_sha256: bool = True) -> FundingSeries:
    """Load the cache written by `tools.funding_cache`.

    The manifest's `data_sha256` is verified by default: a research number
    whose input silently changed between runs is not reproducible, which is the
    same defect C4.4 fixed for model artifacts.
    """
    stem = dataset_stem(exchange, symbol)
    data_path = os.path.join(outdir, f"{stem}.csv")
    manifest_path = os.path.join(outdir, f"{stem}.manifest.json")
    if not os.path.exists(data_path):
        raise FundingDataUnavailable(
            f"no funding dataset at {data_path}; build it with "
            f"`python -m tools.funding_cache --exchange {exchange} "
            f"--symbol {symbol} --start <date>`")

    sha = _sha256_file(data_path)
    if verify_sha256:
        if not os.path.exists(manifest_path):
            raise FundingDataUnavailable(
                f"funding dataset at {data_path} has no manifest; provenance "
                f"cannot be established")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        pinned = manifest.get("data_sha256")
        if pinned and pinned != sha:
            raise FundingDataUnavailable(
                f"funding dataset sha mismatch: manifest pins {pinned[:16]}, "
                f"file is {sha[:16]}")

    df = pd.read_csv(data_path)
    for col in ("funding_time", "funding_rate"):
        if col not in df.columns:
            raise FundingError(f"funding dataset is missing column {col!r}")
    return FundingSeries.from_records(
        df["funding_time"].tolist(), df["funding_rate"].tolist(),
        symbol=symbol, exchange=exchange, source_sha256=sha)


def series_from_binance_records(records: Sequence[dict[str, Any]], *,
                                symbol: str = "BTCUSDT") -> FundingSeries:
    """Build a series from raw `/fapi/v1/fundingRate` payloads.

    Used by paths that hold a live client instead of the offline cache — the
    accounting stays identical, only the transport differs.
    """
    if not records:
        raise FundingDataUnavailable("no funding records returned")
    return FundingSeries.from_records(
        [r["fundingTime"] for r in records],
        [float(r["fundingRate"]) for r in records],
        symbol=symbol, exchange="binance")
