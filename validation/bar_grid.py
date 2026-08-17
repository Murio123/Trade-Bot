"""Where the bars actually are, and where they are missing.

Two questions that both P1's funding series and M02's barrier scan have to
answer identically: what is the real spacing of this timestamp series, and
between which pairs of stamps is something known to be absent. P1 answered them
privately inside `validation/funding.py`; M02 needs the same answers about the
kline grid. Two copies of "detect the modal interval" would drift, so the
implementation lives here and both callers delegate.

Nothing here assumes a specific interval. Binance moved some symbols from 8h to
4h funding, and the project's kline profiles run at 15m and 4h; a hardcoded
step would mis-detect every gap on whichever series it was not written for.
"""
from __future__ import annotations

import numpy as np

# Spacing above modal x tolerance counts as a gap. Not 1.0: exchange timestamps
# carry millisecond jitter (…400002 for a nominal …400000), which is not a
# missing observation.
GAP_TOLERANCE = 1.5


def modal_interval_ms(times: np.ndarray) -> int:
    """The spacing this series actually has, rounded to the minute.

    The mode, not the mean or the median: a series with one large gap has a
    mean spacing that matches no real pair of stamps, and using it would hide
    the gap it was inflated by.

    Returns 0 for a series too short to have a spacing, which callers must read
    as "unknown" rather than as "zero".
    """
    if times.size < 2:
        return 0
    deltas = np.diff(times)
    if deltas.size == 0:
        return 0
    minutes = np.rint(deltas / 60_000).astype(np.int64)
    values, counts = np.unique(minutes, return_counts=True)
    return int(values[int(np.argmax(counts))]) * 60_000


def gap_intervals(times: np.ndarray, modal_ms: int,
                  tolerance: float = GAP_TOLERANCE
                  ) -> tuple[tuple[int, int], ...]:
    """Open intervals in which observations are known to be missing.

    Each returned pair is `(last_seen, next_seen)` — an OPEN interval. The
    missing observations lie strictly inside it, so an interval that merely
    touches a gap's edge is not affected by that gap, and callers must compare
    strictly.
    """
    if modal_ms <= 0 or times.size < 2:
        return ()
    deltas = np.diff(times)
    idx = np.nonzero(deltas > modal_ms * tolerance)[0]
    return tuple((int(times[i]), int(times[i + 1])) for i in idx)
