"""C4.2 §2: Feature Store — central, point-in-time ML feature registry.

Offline-only. Never imported by runtime/signal_engine/execution. Computes
every feature exactly once per bar via `build_bar_context`, then exposes
each feature to the Dataset Builder through a small, declarative
`FeatureSpec` — no feature's underlying computation is duplicated anywhere
in this module; every raw quantity (ATR, HTF bias, equilibrium, regime,
CVD, zone levels) is imported unchanged from analyzer/*, signal_engine/*,
pipeline.py, and tools/deep_backtest.py, following the exact same as-of
slicing pattern already audited in tools/swing_hypothesis_simulator.py's
`_bar_context`.

`FEATURE_REGISTRY` currently holds the 20 features frozen for the first
(volatility) model in reports/c41/volatility_model_frozen_spec.md §3.
Future models (direction/magnitude/regime) register additional
`FeatureSpec` entries in this same registry rather than building a parallel
one — this is the "no duplicated feature logic" requirement from C4.2's
brief.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from analyzer.cvd import compute_cvd_from_klines
from analyzer.equilibrium import compute_equilibrium
from analyzer.indicators import compute_indicators
from analyzer.volatility import analyze_volatility
from pipeline import _build_htf_zones
from signal_engine.htf_filter import get_htf_bias
from tools.deep_backtest import (HTF_WINDOW, ZONE_MIN_BARS, ZONE_WINDOW,
                                 _IndicatorCache, _asof_end, regime_current)
from tools.deep_discovery import _ema_slope, _nearest_level_distance
from tools.kline_dataset import ENTRY_WARMUP, HTF_WARMUP

FEATURE_VERSION = "c41_v1"  # bump whenever FEATURE_REGISTRY's contents change

# Prior-window lookbacks used by the OHLCV-only persistence features. All are
# comfortably smaller than ENTRY_WARMUP (300), so — given the dataset
# builder always starts at walk_start() >= ENTRY_WARMUP — these bounds are
# never the binding constraint in practice; they exist for correctness at
# the dataset's very first eligible bar and for any future caller that
# doesn't go through walk_start().
_PREV_RANGE_SHORT = 12
_PREV_RANGE_LONG = 24
_ABS_RETURN_MEAN_WINDOW = 12
_ROLLING_STD_WINDOW = 20


@dataclass(frozen=True)
class BarContext:
    """Every point-in-time quantity computed once per bar (C4.1 §3).

    `score` is intentionally `None` here — see FEATURE_REGISTRY's `score`
    entry docstring for why it is a documented placeholder, not a bug.
    """
    open_time: Any
    price: float
    atr: float
    atr_pct: float | None
    volatility_atr_percentile: float | None
    realized_range_prev_12: float | None
    realized_range_prev_24: float | None
    abs_return_1: float | None
    abs_return_prev_12_mean: float | None
    rolling_std_return_20: float | None
    ema_slope: float | None
    htf_bias: str
    regime_current: str
    eq_pos: float | None
    cvd_last_delta: float
    bbw: float | None
    bb_squeeze: bool
    distance_to_level_atr: float | None
    weekday: int
    month: int
    distance_to_weekend: int
    score: None = None


def _prev_window_ratio(entry_df: pd.DataFrame, i: int, window: int,
                       atr: float) -> float | None:
    """Same high-low realized-range construction as the C4.1 §2 label, but
    over the `window` bars strictly BEFORE `i` — the autoregressive
    "yesterday's version of the label" persistence feature."""
    lo = i - window
    if lo < 0:
        return None
    sl = entry_df.iloc[lo:i]
    return float((sl["high"].max() - sl["low"].min()) / atr) if atr else None


def build_bar_context(entry_df: pd.DataFrame, i: int, frames: dict[str, Any],
                      profile: dict[str, Any], close_ms: dict[str, np.ndarray],
                      cache: _IndicatorCache
                      ) -> tuple[BarContext | None, str | None]:
    """Build one bar's full feature context, or (None, reason) if it must be
    excluded (C4.1 §1/§3 missing-value policy: excluded, never imputed, for
    every feature marked "row excluded" in the frozen spec).

    Mirrors tools.swing_hypothesis_simulator._bar_context's as-of pattern
    exactly, extended with the additional C4.1-frozen features. Reuses the
    SAME pure functions that module and tools.deep_backtest.deep_walk already
    call — no new indicator/zone logic is introduced here.
    """
    entry_tf = profile["entry"]
    htf = profile["htf"]
    zone_tfs = profile["zone_tfs"]
    open_time = entry_df["open_time"].iloc[i]

    entry_sub = entry_df.iloc[max(0, i - (ENTRY_WARMUP - 1)):i + 1]
    ind = compute_indicators(entry_sub)
    atr = ind.get("atr")
    if not atr:
        return None, "no_atr"

    t_ms = int(close_ms[entry_tf][i])
    price = float(entry_df["close"].iloc[i])

    htf_end = _asof_end(close_ms[htf], t_ms)
    htf_slice = frames[htf].df.iloc[max(0, htf_end - HTF_WINDOW):htf_end]
    if len(htf_slice) < HTF_WARMUP:
        return None, "htf_insufficient_history"
    ind_htf = cache.get(htf, htf_end, HTF_WINDOW,
                        lambda: compute_indicators(htf_slice))
    htf_bias = get_htf_bias(ind_htf)
    eq = compute_equilibrium(htf_slice)
    ema_slope = _ema_slope(htf_slice, price)
    vol = analyze_volatility(entry_sub, tf_per_day=24.0)
    regime_cur = regime_current(profile, ind_htf)
    cvd = compute_cvd_from_klines(entry_sub)

    zdfs: dict[str, Any] = {}
    zinds: dict[str, Any] = {}
    for tf in zone_tfs:
        end = _asof_end(close_ms[tf], t_ms)
        s = frames[tf].df.iloc[max(0, end - ZONE_WINDOW):end]
        if len(s) >= ZONE_MIN_BARS:
            zdfs[tf] = s
            zinds[tf] = cache.get(tf, end, ZONE_WINDOW,
                                  lambda s=s: compute_indicators(s))
    if not zdfs:
        return None, "no_zone_frames"
    zones = _build_htf_zones(zdfs, zinds, list(zdfs.keys()), price, entry_sub)
    distance_to_level_atr = _nearest_level_distance(zones["levels"], price, atr)

    range_12 = _prev_window_ratio(entry_df, i, _PREV_RANGE_SHORT, atr)
    range_24 = _prev_window_ratio(entry_df, i, _PREV_RANGE_LONG, atr)

    abs_return_1 = None
    if i >= 1:
        prev_close = float(entry_df["close"].iloc[i - 1])
        if prev_close:
            abs_return_1 = abs(price - prev_close) / prev_close

    abs_return_prev_12_mean = None
    if i >= _ABS_RETURN_MEAN_WINDOW:
        closes = entry_df["close"].iloc[i - _ABS_RETURN_MEAN_WINDOW:i + 1]
        rets = closes.pct_change().abs().iloc[1:]
        if len(rets):
            abs_return_prev_12_mean = float(rets.mean())

    rolling_std_return_20 = None
    if i >= _ROLLING_STD_WINDOW:
        closes = entry_df["close"].iloc[i - _ROLLING_STD_WINDOW:i + 1]
        rets = closes.pct_change().iloc[1:]
        if len(rets) >= 2:
            rolling_std_return_20 = float(rets.std())

    weekday = int(pd.Timestamp(open_time).weekday())
    month = int(pd.Timestamp(open_time).month)

    return BarContext(
        open_time=open_time, price=price, atr=float(atr),
        atr_pct=vol.get("atr_pct"),
        volatility_atr_percentile=vol.get("atr_percentile"),
        realized_range_prev_12=range_12, realized_range_prev_24=range_24,
        abs_return_1=abs_return_1,
        abs_return_prev_12_mean=abs_return_prev_12_mean,
        rolling_std_return_20=rolling_std_return_20,
        ema_slope=ema_slope, htf_bias=htf_bias, regime_current=regime_cur,
        eq_pos=eq.get("pos"), cvd_last_delta=float(cvd.get("last_delta", 0.0)),
        bbw=ind.get("bbw"), bb_squeeze=bool(ind.get("bb_squeeze", False)),
        distance_to_level_atr=distance_to_level_atr,
        weekday=weekday, month=month,
        distance_to_weekend=min(weekday, 6 - weekday),
    ), None


@dataclass(frozen=True)
class FeatureSpec:
    """One registry entry (C4.2 §2): metadata + a pure extractor over
    BarContext. The extractor never recomputes anything — it only reads a
    field already populated by `build_bar_context`."""
    name: str
    dtype: str
    timeframe: str
    source: str
    version: str
    live_availability: str
    extractor: Callable[[BarContext], Any]


def _feature(name: str, dtype: str, timeframe: str, source: str,
            live_availability: str, attr: str) -> FeatureSpec:
    return FeatureSpec(name=name, dtype=dtype, timeframe=timeframe,
                      source=source, version=FEATURE_VERSION,
                      live_availability=live_availability,
                      extractor=lambda ctx, a=attr: getattr(ctx, a))


# Frozen 20-feature set (reports/c41/volatility_model_frozen_spec.md §3).
# `score` is registered but excluded from every model's actual input vector
# by both ridge_input_names()/lightgbm_input_names() below — see its own
# docstring for why its extractor is a documented placeholder.
FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "atr": _feature("atr", "float", "entry", "analyzer/indicators.py::atr",
                    "bar t close", "atr"),
    "atr_pct": _feature("atr_pct", "float", "entry",
                        "analyzer/volatility.py::analyze_volatility",
                        "bar t close", "atr_pct"),
    "volatility_atr_percentile": _feature(
        "volatility_atr_percentile", "float", "entry",
        "analyzer/volatility.py::analyze_volatility", "bar t close",
        "volatility_atr_percentile"),
    "realized_range_prev_12": _feature(
        "realized_range_prev_12", "float", "entry", "raw OHLCV, [t-12, t)",
        "bar t close", "realized_range_prev_12"),
    "realized_range_prev_24": _feature(
        "realized_range_prev_24", "float", "entry", "raw OHLCV, [t-24, t)",
        "bar t close", "realized_range_prev_24"),
    "abs_return_1": _feature("abs_return_1", "float", "entry", "raw OHLCV",
                             "bar t close", "abs_return_1"),
    "abs_return_prev_12_mean": _feature(
        "abs_return_prev_12_mean", "float", "entry", "raw OHLCV, [t-12, t]",
        "bar t close", "abs_return_prev_12_mean"),
    "rolling_std_return_20": _feature(
        "rolling_std_return_20", "float", "entry", "raw OHLCV, [t-20, t]",
        "bar t close", "rolling_std_return_20"),
    "ema_slope": _feature("ema_slope", "float", "htf",
                          "tools/deep_discovery.py::_ema_slope",
                          "latest htf bar closed as-of t", "ema_slope"),
    "htf_bias": _feature("htf_bias", "categorical", "htf",
                         "signal_engine/htf_filter.py::get_htf_bias",
                         "latest htf bar closed as-of t", "htf_bias"),
    "regime_current": _feature(
        "regime_current", "categorical", "1d",
        "tools/deep_backtest.py::regime_current", "latest 1d bar closed as-of t",
        "regime_current"),
    "eq_pos": _feature("eq_pos", "float", "htf",
                       "analyzer/equilibrium.py::compute_equilibrium",
                       "latest htf bar closed as-of t", "eq_pos"),
    "cvd_last_delta": _feature("cvd_last_delta", "float", "entry",
                               "analyzer/cvd.py::compute_cvd_from_klines",
                               "bar t close", "cvd_last_delta"),
    "bbw": _feature("bbw", "float", "entry",
                    "analyzer/indicators.py::compute_indicators", "bar t close",
                    "bbw"),
    "bb_squeeze": _feature("bb_squeeze", "bool", "entry",
                           "analyzer/indicators.py::compute_indicators",
                           "bar t close", "bb_squeeze"),
    "distance_to_level_atr": _feature(
        "distance_to_level_atr", "float", "entry/htf",
        "pipeline.py::_build_htf_zones + "
        "tools/deep_discovery.py::_nearest_level_distance",
        "latest zone-tf bars closed as-of t", "distance_to_level_atr"),
    "weekday": _feature("weekday", "int", "-", "calendar of bar t",
                        "bar t close", "weekday"),
    "month": _feature("month", "int", "-", "calendar of bar t", "bar t close",
                      "month"),
    "distance_to_weekend": _feature("distance_to_weekend", "int", "-",
                                    "calendar of bar t", "bar t close",
                                    "distance_to_weekend"),
    "score": FeatureSpec(
        name="score", dtype="float", timeframe="entry+zone_tfs",
        source="signal_engine/confluence.py::calculate_confluence_score "
              "(NOT wired — see docstring)",
        version=FEATURE_VERSION, live_availability="bar t close",
        extractor=lambda ctx: ctx.score),
}

# `score` requires the full liquidity/reversal/divergence pipeline
# (tools.deep_backtest.deep_walk's flat dict) to reproduce the real runtime
# confluence score; wiring that whole pipeline solely to populate a column
# excluded from both models' actual inputs (C4.1 §3: "benchmark only") is
# out of scope for C4.2's "implement only what the first model needs"
# instruction. `score` is registered (so the schema/cap accounting in C4.1
# is satisfied) but always extracts `None` until a later stage wires it —
# documented in reports/c42/implementation_summary.md, not a silent gap.
BENCHMARK_ONLY_FEATURES = frozenset({"score"})


def model_input_names() -> list[str]:
    """The 19 real model inputs — everything in the registry except the
    documented benchmark-only column (C4.1 §3)."""
    return [name for name in FEATURE_REGISTRY if name not in BENCHMARK_ONLY_FEATURES]


def extract_row(ctx: BarContext, names: list[str] | None = None
                ) -> dict[str, Any]:
    """Extract a plain dict of feature values from a BarContext for the
    given feature names (defaults to every registered feature, including
    the benchmark-only column, so dataset rows keep it for correlation
    analysis while training code uses model_input_names() to select only
    real inputs)."""
    names = names if names is not None else list(FEATURE_REGISTRY)
    return {name: FEATURE_REGISTRY[name].extractor(ctx) for name in names}


def feature_schema() -> list[dict[str, Any]]:
    """The full feature_schema.json contents (C4.1 §12), one row per
    registered feature."""
    return [
        {"name": spec.name, "dtype": spec.dtype, "timeframe": spec.timeframe,
         "source": spec.source, "version": spec.version,
         "live_availability": spec.live_availability,
         "benchmark_only": spec.name in BENCHMARK_ONLY_FEATURES}
        for spec in FEATURE_REGISTRY.values()
    ]
