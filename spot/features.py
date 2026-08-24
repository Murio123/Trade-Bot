"""S3: the six ranking rules, frozen in `reports/s3/S3_SPEC.md` §4-§5.

Read-only, offline. Each rule is a pure function of **visible bars** — those
that closed strictly before the decision instant — and of nothing else. No rule
may read another rule's output, and there is deliberately no function here that
combines any of them: S3 measures families one at a time, and the absence of a
combiner is the mechanism, not an oversight.

Two properties are load-bearing and are asserted by tests rather than trusted:

  * **A feature value cannot depend on the future.** Every definition indexes
    backwards from the last visible bar, so appending later bars to a frame
    must not change any value already computed.
  * **A ranking is deterministic.** Ties break on symbol name, so the same
    inputs always produce the same order — otherwise a rate measured over a
    top quintile would wobble for reasons that have nothing to do with the
    feature.

Index convention: `close[-1]` is the last visible bar, `close[-31]` is thirty
bars before it. Windows are stated in bars, and daily bars are days here
because the panel is daily and gap-crossing events were already refused by M02
at label time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# --- frozen windows (S3_SPEC §4-§5) ---------------------------------------
MOM_SKIP = 31            # B4: skip the most recent month
MOM_FORMATION = 181      # B4: five months of formation before the skip
RS_WINDOW = 91           # B5: one quarter
RS_PERSIST_WINDOW = 120  # F1: how far back persistence is measured
RS_PERSIST_MEAN = 60     # F1: the ratio's own trailing mean
MA_WINDOW = 200          # F2
VOL_RECENT = 30          # F3
VOL_BASE = 180           # F3
DRAWDOWN_WINDOW = 180    # F4


@dataclass(frozen=True)
class Feature:
    """One frozen ranking rule."""
    fid: str
    name: str
    family: str
    definition: str
    min_bars: int
    fn: Callable[[pd.DataFrame, pd.DataFrame], float | None]
    # Every rule here ranks descending; kept explicit so a future rule that
    # does not cannot be added silently.
    descending: bool = True

    def value(self, df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
        if len(df) < self.min_bars:
            return None
        v = self.fn(df, btc)
        if v is None:
            return None
        return v if math.isfinite(v) else None


def _closes(df: pd.DataFrame) -> np.ndarray:
    return df["close"].to_numpy(dtype=float)


def momentum_6_1(df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
    """B4 — five months of formation, the most recent month skipped."""
    c = _closes(df)
    past, recent = c[-MOM_FORMATION], c[-MOM_SKIP]
    return recent / past - 1.0 if past > 0 else None


def relative_strength_90(df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
    """B5 — the asset's 90-day log return minus BTC's over the same bars.

    BTC is aligned on `close_time`, not on position: the two frames can have
    different lengths, and comparing bar 91 of one with bar 91 of the other
    would silently compare different calendar windows.
    """
    c = _closes(df)
    b = _btc_window(btc, df, RS_WINDOW)
    if b is None:
        return None
    if c[-RS_WINDOW] <= 0 or c[-1] <= 0:
        return None
    return math.log(c[-1] / c[-RS_WINDOW]) - math.log(b[-1] / b[0])


def _btc_window(btc: pd.DataFrame, df: pd.DataFrame,
                window: int) -> np.ndarray | None:
    """BTC closes over the same calendar span as `df`'s trailing `window`."""
    start_ms = int(df["close_time"].iloc[-window])
    end_ms = int(df["close_time"].iloc[-1])
    sub = btc[(btc["close_time"] >= start_ms) & (btc["close_time"] <= end_ms)]
    if len(sub) < 2:
        return None
    return sub["close"].to_numpy(dtype=float)


def rs_persistence(df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
    """F1 — how often the coin/BTC ratio held above its own trailing mean.

    A level says the asset outran BTC once; this says it kept doing so. The
    two are different claims and are therefore different trials.
    """
    need = RS_PERSIST_WINDOW + RS_PERSIST_MEAN
    start_ms = int(df["close_time"].iloc[-need])
    end_ms = int(df["close_time"].iloc[-1])
    sub = btc[(btc["close_time"] >= start_ms) & (btc["close_time"] <= end_ms)]
    merged = df[["close_time", "close"]].merge(
        sub[["close_time", "close"]], on="close_time", suffixes=("", "_btc"))
    if len(merged) < need:
        return None
    ratio = (merged["close"] / merged["close_btc"]).to_numpy(dtype=float)
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0):
        return None
    trailing = pd.Series(ratio).rolling(RS_PERSIST_MEAN).mean().to_numpy()
    tail_ratio = ratio[-RS_PERSIST_WINDOW:]
    tail_mean = trailing[-RS_PERSIST_WINDOW:]
    ok = np.isfinite(tail_mean)
    if ok.sum() < RS_PERSIST_WINDOW:
        return None
    return float(np.mean(tail_ratio[ok] > tail_mean[ok]))


def distance_above_ma(df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
    """F2 — where price stands now against a 200-day average."""
    c = _closes(df)
    ma = float(np.mean(c[-MA_WINDOW:]))
    return c[-1] / ma - 1.0 if ma > 0 else None


def abnormal_participation(df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
    """F3 — recent dollar volume against the asset's own longer history.

    A ratio of an asset to itself, so a permanently large coin scores zero
    rather than scoring high — which is the whole point of normalising within
    the asset (S3_SPEC §5).
    """
    q = df["quote_volume"].to_numpy(dtype=float)
    recent = float(np.median(q[-VOL_RECENT:]))
    base = float(np.median(q[-VOL_BASE:]))
    if not (recent > 0 and base > 0):
        return None
    return math.log(recent / base)


def drawdown_from_high(df: pd.DataFrame, btc: pd.DataFrame) -> float | None:
    """F4 — distance below the 180-day high; 0 means at the high."""
    c = _closes(df)
    peak = float(np.max(c[-DRAWDOWN_WINDOW:]))
    return c[-1] / peak - 1.0 if peak > 0 else None


FEATURES: tuple[Feature, ...] = (
    Feature("B4", "momentum_6_1", "baseline_momentum",
            "close[-31] / close[-181] - 1", MOM_FORMATION, momentum_6_1),
    Feature("B5", "relative_strength_90", "baseline_relative_strength",
            "log(close[-1]/close[-91]) - log(btc[-1]/btc[-91])",
            RS_WINDOW, relative_strength_90),
    Feature("F1", "rs_persistence", "relative_strength",
            "share of last 120 days with close/btc above its 60d mean",
            RS_PERSIST_WINDOW + RS_PERSIST_MEAN, rs_persistence),
    Feature("F2", "distance_above_ma", "momentum_trend_quality",
            "close[-1] / mean(close[-200:]) - 1", MA_WINDOW, distance_above_ma),
    Feature("F3", "abnormal_participation", "volume_participation",
            "log(median(quote_volume[-30:]) / median(quote_volume[-180:]))",
            VOL_BASE, abnormal_participation),
    Feature("F4", "drawdown_from_high", "volatility_drawdown_state",
            "close[-1] / max(close[-180:]) - 1", DRAWDOWN_WINDOW,
            drawdown_from_high),
)

BY_ID = {f.fid: f for f in FEATURES}


def rank_symbols(values: dict[str, float], descending: bool = True
                 ) -> list[str]:
    """Deterministic order. Ties break on symbol name, never on input order."""
    return [s for s, _ in sorted(values.items(),
                                 key=lambda kv: (-kv[1] if descending
                                                 else kv[1], kv[0]))]


def top_quantile_k(n: int, fraction: float = 0.20) -> int:
    """`K = ceil(fraction * N)` — the frozen top-quintile size (§3)."""
    return max(1, math.ceil(fraction * n))
