"""C4.4: the shipped ranker — an inverted trailing mean of past labels.

Why inverted. The label is a forward range normalised by the ATR at the
prediction bar, and its own autocorrelation flips sign at the 12-bar
horizon: +0.93 at lag 1, but -0.13 at lag 12 and negative beyond. After a
burst of range, ATR catches up and the ratio falls back. So a HIGH trailing
ratio precedes a LOW one, and the trailing mean must be negated before it
can be read as an expectation.

C4.3e measured this baseline at |rho| = 0.176 against Ridge's 0.307 — 57% of
the model's ranking skill, for none of its machinery. It needs no features,
no model artifact and no retraining: only labels of bars already closed.
"""
from __future__ import annotations

import numpy as np

# Frozen alongside the C4.1 volatility spec — the horizon the label is built
# on, and the trailing window C4.3 evaluated.
HORIZON_BARS = 12
TRAILING_WINDOW = 60
RANKER_VERSION = "c44_inverted_trailing_v1"


class NotEnoughHistory(Exception):
    """Raised rather than returning a number nobody could have computed."""


def trailing_mean(label_idx, labels, at_idx: int, *,
                  window: int = TRAILING_WINDOW,
                  horizon_bars: int = HORIZON_BARS) -> float:
    """Mean of the last `window` labels observable at bar `at_idx`.

    The label of bar u spans bars u+1..u+horizon_bars, so it is knowable only
    at u+horizon_bars. Nothing later than at_idx - horizon_bars may enter —
    the same rule the C4.3d baseline is built on, restated here because this
    one runs on live data where a mistake is not caught by a walk-forward.
    """
    idx = np.asarray(label_idx, dtype=np.int64)
    lab = np.asarray(labels, dtype=float)
    if idx.shape != lab.shape:
        raise ValueError("label_idx and labels must be the same length")
    if idx.size and np.any(np.diff(idx) <= 0):
        raise ValueError("label_idx must be strictly increasing and unique")

    end = int(np.searchsorted(idx, at_idx - horizon_bars, side="right"))
    start = max(end - window, 0)
    if end - start <= 0:
        raise NotEnoughHistory(
            f"no label observable at bar {at_idx} (need at least one bar "
            f"with idx <= {at_idx - horizon_bars})")
    return float(lab[start:end].mean())


def rank_score(label_idx, labels, at_idx: int, **kw) -> float:
    """The ranker's output, in "higher means more expected volatility" units.

    This is the negated trailing mean, so it is NOT a predicted ratio and
    must never be shown as one. Only its RANK against the frozen training
    distribution is meaningful, which is why callers go through
    percentile.py rather than reading this number.
    """
    return -trailing_mean(label_idx, labels, at_idx, **kw)
