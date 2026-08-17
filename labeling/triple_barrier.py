"""M02 — what actually happened, not what the range was 12 bars later.

The project's existing label is fixed-horizon: it asks what the price range was
`h` bars after the decision, unconditionally. Nobody trades that. A real
position has a stop and a target, and which one it reaches first decides the
outcome; a trade that ran to +2R and came back to zero is a win, and the
fixed-horizon label calls it flat. Every downstream module inherits that defect,
which is why this primitive sits at the root of the G1 dependency chain.

Three barriers per event: an upper and a lower price barrier scaled by the
volatility observed **at the decision bar**, and a vertical barrier `v` bars
out. The first one touched resolves the label.

The conventions below are frozen in `reports/c50/G1_SPEC.md` §2 and each one is
a choice that could have been made the other way:

**The decision bar is never scanned.** An event at bar `i` scans bars
`i+1 .. i+v` inclusive. Bar `i`'s own high and low are already known when its
barriers are set from its own close, so admitting them would be look-ahead of
the trivial kind — and it would resolve a large fraction of events instantly.

**The vertical bar is inside the scan.** A touch on bar `i+v` is a touch, not a
timeout. `TIME` means the closed range `i+1 .. i+v` contained no touch.

**Touches are non-strict.** `high >= upper` and `low <= lower`. A price that
reaches the barrier exactly has reached it.

**A same-bar tie resolves to the position's STOP.** When one bar's range spans
both barriers, OHLC does not say which came first, so the adverse one is
assumed: LOWER for a long, UPPER for a short. A documented pessimistic bias, not
a hidden one.

This was originally specified as "resolve to LOWER", unconditionally, and that
was wrong. LOWER is the stop only for a long; for a short it is the *target*, so
the rule that was described as conservative was handing every short a free win.
G1's NC4 control caught it — random-direction entries came out at -0.22 R long
against -0.01 R short, an asymmetry that cancels in the pooled mean and would
have been invisible in any aggregate. Recorded here rather than quietly fixed,
and amended in `reports/c50/G1_SPEC.md` §2 with the same account.

For direction-neutral labeling there is no stop, so the tie resolves to LOWER as
a frozen deterministic convention. "Pessimistic" has no meaning without a side;
what matters there is only that the choice is fixed and stated.

**A truncated horizon is not a timeout.** An event whose vertical barrier falls
past the end of the frame, with no touch before it, resolves `TRUNCATED` with a
null label. Calling it `TIME` would assert that a full horizon was observed and
found empty, which is exactly the claim the data cannot support.

**A gap inside the horizon fails closed.** If the bar grid is missing bars
inside an event's scan window, the first touch may have happened in the missing
bars. `BarrierDataError`, not a best guess. The continuity check needs a
timestamp column; a caller with a synthetic frame must pass `time_col=None`
explicitly, so that skipping the check is always a named decision at the call
site (the same discipline as P1's `FUNDING_NOT_MODELLED`).

Fills are assumed at the barrier price itself. Execution cost is the P1 cost
model's business (`validation.trade_costs`), not the label's; mixing them would
put slippage inside the label geometry where no test could see it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from validation.bar_grid import gap_intervals, modal_interval_ms

TRIPLE_BARRIER_VERSION = "m02_triple_barrier_v1"

LONG = 1
SHORT = -1
NEUTRAL = 0

# Barrier identity, independent of which side the position took.
UPPER = "UPPER"
LOWER = "LOWER"
TIME = "TIME"
TRUNCATED = "TRUNCATED"

# Outcome relative to the position's side.
TP = "TP"
SL = "SL"


class BarrierError(Exception):
    """Malformed input — a programming error."""


class BarrierDataError(BarrierError):
    """The frame cannot answer the question for this event.

    A separate type because it is a data-coverage limit, not a bug: callers
    report the event as unavailable and never as a resolved label.
    """


@dataclass(frozen=True)
class BarrierConfig:
    """The two multipliers and the horizon, hashed so a run can be pinned.

    These multipliers are the most obvious overfitting surface in the whole
    module — two free parameters that could be tuned to any desired outcome.
    ARCHITECTURE.md M02 requires them frozen and recorded in the trial ledger
    before any run; `content_sha256` is what makes "frozen" checkable rather
    than asserted. G1 does not choose them: it only proves the geometry is
    correct for whatever they are.
    """
    upper_mult: float
    lower_mult: float
    vertical_bars: int

    def __post_init__(self) -> None:
        if not (self.upper_mult > 0.0 and self.lower_mult > 0.0):
            raise BarrierError("barrier multipliers must be positive")
        if self.vertical_bars < 1:
            raise BarrierError(
                "vertical_bars must be at least 1; a zero horizon would scan no "
                "bars at all and resolve every event as TIME")

    def content_sha256(self) -> str:
        payload = json.dumps({"version": TRIPLE_BARRIER_VERSION,
                              "upper_mult": self.upper_mult,
                              "lower_mult": self.lower_mult,
                              "vertical_bars": self.vertical_bars},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class BarrierEvent:
    """One resolved event, with every input kept for inspection."""
    start_idx: int
    end_idx: int
    start_time: int | None
    end_time: int | None
    side: int
    entry_price: float
    upper: float
    lower: float
    sigma: float
    barrier: str          # UPPER / LOWER / TIME / TRUNCATED
    outcome: str          # TP / SL / TIME / TRUNCATED
    label: int | None     # +1 / -1 / 0, None when TRUNCATED
    exit_price: float | None
    ret: float | None     # signed return in the side's direction, fraction
    r_multiple: float | None
    timeout: bool
    resolved: bool
    # True when one bar's range spanned both barriers and the tie rule decided
    # the outcome. Recorded per event because the rule is a deliberate
    # pessimistic bias, and a bias whose frequency is unmeasured is
    # indistinguishable from a bug. G1 measures it (reports/c50/G1_RESULT.md).
    tie: bool = False

    @property
    def span(self) -> tuple[int, int]:
        """The closed bar interval over which the label was determined."""
        return (self.start_idx, self.end_idx)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_events(events: Iterable[Any]) -> list[tuple[int, int]]:
    """Accept bare indices or `(idx, side)` pairs.

    A bare index means direction-neutral: the question is which barrier the
    price reached first, with no position implied.
    """
    out: list[tuple[int, int]] = []
    for e in events:
        if isinstance(e, (tuple, list)):
            if len(e) != 2:
                raise BarrierError(f"event {e!r} must be (idx, side)")
            idx, side = int(e[0]), int(e[1])
        elif isinstance(e, dict):
            idx, side = int(e["idx"]), int(e.get("side", NEUTRAL))
        else:
            idx, side = int(e), NEUTRAL
        if side not in (LONG, SHORT, NEUTRAL):
            raise BarrierError(
                f"side must be +1 (long), -1 (short) or 0 (neutral), got {side}")
        out.append((idx, side))
    return out


def _time_column(ohlcv: pd.DataFrame, time_col: str | None) -> np.ndarray | None:
    if time_col is None:
        return None
    if time_col not in ohlcv.columns:
        raise BarrierDataError(
            f"time_col {time_col!r} is not in the frame; pass time_col=None to "
            "skip the bar-continuity check, but only for a frame whose "
            "timestamps are synthetic — on real data skipping it means a gap "
            "can hide the first touch")
    return ohlcv[time_col].to_numpy(dtype=np.int64)


def apply_barriers(ohlcv: pd.DataFrame, events: Iterable[Any],
                   sigma: Sequence[float] | np.ndarray,
                   config: BarrierConfig, *,
                   time_col: str | None = "close_time",
                   price_col: str = "close") -> list[BarrierEvent]:
    """Resolve every event against its three barriers.

    `sigma` is the volatility estimate at each bar, as a fraction of price
    (0.015 = 1.5%), aligned to `ohlcv` rows. Barriers scale with it rather than
    with a fixed percentage, so the same configuration means the same thing in a
    quiet and a violent regime — ARCHITECTURE.md M02.

    Deterministic: the same frame and events give byte-identical output.
    """
    for col in ("high", "low", price_col):
        if col not in ohlcv.columns:
            raise BarrierError(f"frame is missing required column {col!r}")

    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    price = ohlcv[price_col].to_numpy(dtype=float)
    sig = np.asarray(sigma, dtype=float).ravel()
    n = int(len(ohlcv))

    if sig.size != n:
        raise BarrierError(
            f"sigma has {sig.size} values for {n} bars; it must be aligned to "
            "the frame, because a shifted sigma silently sets every barrier "
            "from the wrong bar's volatility")

    times = _time_column(ohlcv, time_col)
    gaps: tuple[tuple[int, int], ...] = ()
    if times is not None:
        if times.size != n:
            raise BarrierError("time column length does not match the frame")
        if np.any(np.diff(times) <= 0):
            raise BarrierDataError(
                "bar timestamps are not strictly increasing; the frame is "
                "unsorted or duplicated and no scan over it is meaningful")
        gaps = gap_intervals(times, modal_interval_ms(times))

    resolved: list[BarrierEvent] = []
    for idx, side in _as_events(events):
        if not (0 <= idx < n):
            raise BarrierError(f"event index {idx} is outside the frame (n={n})")
        s = float(sig[idx])
        if not np.isfinite(s) or s <= 0.0:
            raise BarrierDataError(
                f"sigma at bar {idx} is {s!r}; barriers scaled by a zero or "
                "undefined volatility would collapse onto the entry price")

        entry = float(price[idx])
        upper = entry * (1.0 + config.upper_mult * s)
        lower = entry * (1.0 - config.lower_mult * s)

        scan_lo = idx + 1
        scan_hi = min(idx + config.vertical_bars, n - 1)   # inclusive
        truncated_horizon = idx + config.vertical_bars > n - 1

        if times is not None and scan_hi >= scan_lo:
            # A gap is an open interval between two observed stamps. It matters
            # only if it lies strictly inside the window actually scanned.
            lo_ms, hi_ms = int(times[idx]), int(times[scan_hi])
            if any(g_lo < hi_ms and g_hi > lo_ms for g_lo, g_hi in gaps):
                raise BarrierDataError(
                    f"event at bar {idx} spans a gap in the bar grid; the first "
                    "touch may have happened in the missing bars, so this event "
                    "has no answer rather than a guessed one")

        hit_idx: int | None = None
        hit_barrier = TIME
        hit_tie = False
        for j in range(scan_lo, scan_hi + 1):
            up = high[j] >= upper
            dn = low[j] <= lower
            if up and dn:
                # Both barriers inside one bar's range. OHLC cannot order them,
                # so the adverse one is assumed: the stop for this side, LOWER
                # for a neutral event by frozen convention. Resolving to LOWER
                # regardless of side would be pessimistic for longs and
                # generous to shorts (see the module docstring).
                hit_idx, hit_tie = j, True
                hit_barrier = UPPER if side == SHORT else LOWER
                break
            if up:
                hit_idx, hit_barrier = j, UPPER
                break
            if dn:
                hit_idx, hit_barrier = j, LOWER
                break

        if hit_idx is None and truncated_horizon:
            end_idx = max(scan_hi, idx + 1) if n > idx + 1 else idx + 1
            end_idx = min(end_idx, n - 1)
            if end_idx <= idx:
                raise BarrierDataError(
                    f"event at bar {idx} is the last bar of the frame; there is "
                    "no forward bar to scan and therefore no label")
            resolved.append(BarrierEvent(
                start_idx=idx, end_idx=end_idx,
                start_time=None if times is None else int(times[idx]),
                end_time=None if times is None else int(times[end_idx]),
                side=side, entry_price=entry, upper=upper, lower=lower, sigma=s,
                barrier=TRUNCATED, outcome=TRUNCATED, label=None,
                exit_price=None, ret=None, r_multiple=None,
                timeout=False, resolved=False))
            continue

        if hit_idx is None:
            end_idx = scan_hi
            exit_price = float(price[end_idx])
        else:
            end_idx = hit_idx
            exit_price = upper if hit_barrier == UPPER else lower

        if end_idx <= idx:
            raise BarrierError(
                f"event at bar {idx} produced a zero-length span; this is a bug "
                "in the scan, not a fast fill")

        # Signed return in the side's own direction. Neutral events are measured
        # as if long, so `ret`'s sign matches `label`'s and the two never
        # disagree about what happened.
        direction = 1.0 if side in (LONG, NEUTRAL) else -1.0
        ret = direction * (exit_price - entry) / entry

        # The stop is whichever barrier is adverse for this side.
        stop_frac = (config.lower_mult if side in (LONG, NEUTRAL)
                     else config.upper_mult) * s
        r_multiple = ret / stop_frac

        if hit_barrier == TIME:
            label = 0
            outcome = TIME
        elif side == NEUTRAL:
            label = 1 if hit_barrier == UPPER else -1
            outcome = TP if hit_barrier == UPPER else SL
        else:
            favourable = UPPER if side == LONG else LOWER
            label = 1 if hit_barrier == favourable else -1
            outcome = TP if hit_barrier == favourable else SL

        resolved.append(BarrierEvent(
            start_idx=idx, end_idx=end_idx,
            start_time=None if times is None else int(times[idx]),
            end_time=None if times is None else int(times[end_idx]),
            side=side, entry_price=entry, upper=upper, lower=lower, sigma=s,
            barrier=hit_barrier, outcome=outcome, label=label,
            exit_price=exit_price, ret=ret, r_multiple=r_multiple,
            timeout=hit_barrier == TIME, resolved=True, tie=hit_tie))

    return resolved


def label_spans(events: Sequence[BarrierEvent],
                *, resolved_only: bool = True) -> np.ndarray:
    """The `(start_idx, end_idx)` closed intervals, as an `(n, 2)` int array.

    This is the only thing M03 and M04 need from M02, and the reason both take
    spans rather than events: leakage protection is a property of intervals, not
    of labels.

    Unresolved (`TRUNCATED`) events are dropped by default. Keeping them would
    give the weighting and the splitting a span for an observation that has no
    label, inflating the effective sample size with rows that cannot train
    anything.
    """
    kept = [e for e in events if e.resolved or not resolved_only]
    if not kept:
        return np.empty((0, 2), dtype=np.int64)
    return np.array([[e.start_idx, e.end_idx] for e in kept], dtype=np.int64)


def knowable_at(events: Sequence[BarrierEvent], t: int) -> list[BarrierEvent]:
    """Events whose label is known by bar `t` — `end_idx <= t`.

    ARCHITECTURE.md §5.1: the label of bar `u` is knowable only at
    `u + span(u)`. Any point-in-time computation filters through this; doing it
    inline at each call site is how one of them eventually forgets.
    """
    return [e for e in events if e.resolved and e.end_idx <= t]


def outcome_counts(events: Sequence[BarrierEvent]) -> dict[str, int]:
    """TP / SL / TIME / TRUNCATED tally, plus the tie count.

    `TIE` is reported alongside rather than folded into `SL`, because it answers
    a different question: how often the label was decided by the convention
    instead of by the data.
    """
    counts = {TP: 0, SL: 0, TIME: 0, TRUNCATED: 0, "TIE": 0}
    for e in events:
        counts[e.outcome] = counts.get(e.outcome, 0) + 1
        if e.tie:
            counts["TIE"] += 1
    return counts
