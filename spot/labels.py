"""S2: the `CLEAN_2X(180d)` label and its record vector.

Read-only. The label itself is computed by M02 (`labeling/triple_barrier.py`),
**unmodified** — this module configures it and interprets what it returns.

Why M02 rather than a fresh scan. The barrier geometry has one property that
decides whether the answer means anything (which barrier was touched *first*),
one convention that quietly biases the result if it is wrong (what to do when a
single bar's range spans both barriers), and one refusal that protects it (an
event whose window crosses a hole in the bar grid has no answer). All three
were built and audited under G1. Writing a second scan here would mean
re-deriving them and getting the tie rule wrong in a new way.

The configuration, frozen in `reports/s2/S2_SPEC.md` §4:

    upper_mult = 1.0, lower_mult = 0.4, vertical_bars = 180, sigma = 1.0

M02 scales barriers by volatility (`entry * (1 ± mult * sigma)`), so a constant
sigma of 1.0 turns them into fixed ratios: +100% and −40%. No change to M02 was
needed, exactly as S0 §6 predicted.

**The one interpretation that is ours, not M02's: death.** M02 reports
TRUNCATED when the horizon runs past the end of the data, and cannot know
whether that is a coin that was delisted or a panel that has not reached the
future yet. Those are opposite things — one is a resolved failure, the other is
missing data — and collapsing them either drops the deaths (survivorship bias
again) or invents outcomes for coins that are still trading. §4.1 of the spec
splits them and this module implements that split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from labeling.triple_barrier import (BarrierConfig, BarrierDataError,
                                     BarrierError, apply_barriers)

DAY_MS = 86_400_000

# Frozen (S2_SPEC §4). sigma is constant, so these are literal price ratios.
UPPER_MULT = 1.0      # +100%
LOWER_MULT = 0.4      # -40%
HORIZON_BARS = 180

CONFIG = BarrierConfig(upper_mult=UPPER_MULT, lower_mult=LOWER_MULT,
                       vertical_bars=HORIZON_BARS)

# Outcomes this module reports. UPPER/LOWER/TIME come from M02; DEAD and
# CENSORED are the two halves of M02's TRUNCATED; GAP is its refusal.
HIT = "hit_2x"
STOPPED = "stopped_out"
TIMED_OUT = "timed_out"
DEAD = "delisted"
CENSORED = "censored"
GAP = "gap_unresolved"

RESOLVED = (HIT, STOPPED, TIMED_OUT, DEAD)


@dataclass(frozen=True)
class Label:
    """One (symbol, decision date) outcome, with the record vector."""
    symbol: str
    asof: str
    outcome: str
    clean_2x: int | None
    entry_price: float
    mfe: float | None
    mae: float | None
    days_to_2x: int | None
    terminal_return: float | None
    excess_vs_btc: float | None
    window_bars: int
    tie: bool
    trading_now: bool

    @property
    def resolved(self) -> bool:
        return self.outcome in RESOLVED

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bar_index_asof(df: pd.DataFrame, asof_ms: int) -> int | None:
    """Index of the last bar that closed before the decision instant."""
    closed = df["close_time"].to_numpy()
    pos = int(np.searchsorted(closed, asof_ms, side="left")) - 1
    return pos if pos >= 0 else None


def _btc_return(btc: pd.DataFrame, start_ms: int, end_ms: int) -> float | None:
    """BTC's return over the identical calendar window.

    Matched on timestamps rather than on bar counts: the comparison must be to
    the same days, and a coin's window can be shorter than 180 bars when it
    dies inside the horizon.
    """
    i = bar_index_asof(btc, start_ms)
    if i is None:
        return None
    closed = btc["close_time"].to_numpy()
    j = int(np.searchsorted(closed, end_ms, side="right")) - 1
    if j <= i:
        return None
    p0 = float(btc["close"].iloc[i])
    p1 = float(btc["close"].iloc[j])
    return p1 / p0 - 1.0 if p0 > 0 else None


def label_event(df: pd.DataFrame, idx: int, *, symbol: str, asof: str,
                asof_ms: int, trading_now: bool,
                btc: pd.DataFrame | None = None) -> Label:
    """Resolve one decision date for one symbol."""
    n = len(df)
    entry = float(df["close"].iloc[idx])
    sigma = np.ones(n, dtype=float)

    def empty(outcome: str) -> Label:
        return Label(symbol=symbol, asof=asof, outcome=outcome, clean_2x=None,
                     entry_price=entry, mfe=None, mae=None, days_to_2x=None,
                     terminal_return=None, excess_vs_btc=None, window_bars=0,
                     tie=False, trading_now=trading_now)

    if idx >= n - 1:
        return empty(CENSORED if trading_now else DEAD)

    try:
        events = apply_barriers(df, [idx], sigma, CONFIG)
    except BarrierDataError as exc:
        if "spans a gap" in str(exc):
            return empty(GAP)
        raise
    except BarrierError:
        raise
    ev = events[0]

    lo = idx + 1
    hi = min(idx + HORIZON_BARS, n - 1)
    window = df.iloc[lo:hi + 1]
    mfe = float(window["high"].max()) / entry - 1.0
    mae = float(window["low"].min()) / entry - 1.0
    terminal = float(window["close"].iloc[-1]) / entry - 1.0

    if ev.barrier == "UPPER":
        outcome, clean = HIT, 1
    elif ev.barrier == "LOWER":
        outcome, clean = STOPPED, 0
    elif ev.barrier == "TIME":
        outcome, clean = TIMED_OUT, 0
    else:
        # TRUNCATED: the path ran out before the horizon did. A coin that is
        # still trading means the panel has not reached that future yet
        # (censored, excluded); a coin that is not means it died inside the
        # window, which is a resolved failure and must not be dropped.
        outcome, clean = (CENSORED, None) if trading_now else (DEAD, 0)

    # Bar distance, not a timestamp subtraction. The decision instant sits one
    # millisecond after bar `idx` closes, so differencing close times lands a
    # day short; and M02 has already refused any event whose window crosses a
    # hole in the grid, so one bar is exactly one day here.
    days = int(ev.end_idx - idx) if outcome == HIT else None

    excess = None
    if btc is not None:
        end_ms = int(window["close_time"].iloc[-1])
        btc_ret = _btc_return(btc, asof_ms, end_ms)
        if btc_ret is not None and terminal > -1.0 and btc_ret > -1.0:
            excess = math.log1p(terminal) - math.log1p(btc_ret)

    return Label(symbol=symbol, asof=asof, outcome=outcome, clean_2x=clean,
                 entry_price=entry, mfe=mfe, mae=mae, days_to_2x=days,
                 terminal_return=terminal, excess_vs_btc=excess,
                 window_bars=int(len(window)), tie=bool(ev.tie),
                 trading_now=trading_now)


def summarize(labels: list[Label]) -> dict[str, Any]:
    """Base rate and the record-vector distribution, resolved events only."""
    resolved = [l for l in labels if l.resolved]
    hits = [l for l in resolved if l.clean_2x == 1]
    counts: dict[str, int] = {}
    for l in labels:
        counts[l.outcome] = counts.get(l.outcome, 0) + 1

    def pct(values: list[float], q: float) -> float | None:
        return round(float(np.percentile(values, q)), 4) if values else None

    maes = [l.mae for l in resolved if l.mae is not None]
    mfes = [l.mfe for l in resolved if l.mfe is not None]
    terms = [l.terminal_return for l in resolved if l.terminal_return is not None]
    excess = [l.excess_vs_btc for l in resolved if l.excess_vs_btc is not None]
    ttt = [l.days_to_2x for l in hits if l.days_to_2x is not None]
    hit_maes = [l.mae for l in hits if l.mae is not None]

    return {
        "events": len(labels),
        "resolved": len(resolved),
        "outcomes": dict(sorted(counts.items())),
        "base_rate": (round(len(hits) / len(resolved), 4) if resolved else None),
        "hits": len(hits),
        "ties": sum(1 for l in labels if l.tie),
        "mae": {"p10": pct(maes, 10), "median": pct(maes, 50),
                "p90": pct(maes, 90)},
        "mfe": {"median": pct(mfes, 50), "p90": pct(mfes, 90)},
        "terminal_return": {"median": pct(terms, 50), "p90": pct(terms, 90)},
        "excess_vs_btc": {"median": pct(excess, 50), "p90": pct(excess, 90)},
        "days_to_2x": {"median": pct([float(d) for d in ttt], 50)},
        "hit_mae": {"median": pct(hit_maes, 50), "p10": pct(hit_maes, 10)},
    }
