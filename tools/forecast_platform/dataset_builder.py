"""C4.2 §1: Dataset Builder — generic point-in-time dataset construction.

Offline-only. Reuses tools.deep_backtest.prepare/WFConfig/fold_windows/
partition_setups/_close_ms/_IndicatorCache/walk_start/provenance unchanged
(deep_backtest.py itself is never modified — see the module docstring
convention already established by tools/swing_hypothesis_simulator.py).

Generic across future models: `build_dataset` takes a pluggable `label_fn`
(any (entry_df, i, ctx, horizon_bars) -> float | None) and a `feature_names`
list, so a future direction/magnitude/regime model reuses this same builder
with its own label function and its own feature subset from
tools.forecast_platform.feature_store — no new dataset-construction code is
needed for those models.

The concrete label used by the first (volatility) model is
`volatility_range_label`, implementing reports/c41/
volatility_model_frozen_spec.md §2 exactly (frozen: horizon=12 bars, clip
[0,20], log(+1e-6)).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from tools.deep_backtest import (DeepBacktestError, Fold, WFConfig,
                                 _IndicatorCache, _close_ms, _iso,
                                 fold_windows, partition_setups, prepare,
                                 provenance, walk_start)
from tools.forecast_platform.feature_store import (FEATURE_VERSION,
                                                    BarContext, extract_row,
                                                    build_bar_context)

DATASET_BUILDER_VERSION = "c42_v1"

# --- C4.1 §2 frozen volatility label constants (not tunable, not re-derived) ---
VOLATILITY_HORIZON_BARS = 12
VOLATILITY_LABEL_VERSION = "c41_volatility_range_v1"
VOLATILITY_CLIP_LO = 0.0
VOLATILITY_CLIP_HI = 20.0
VOLATILITY_LOG_EPS = 1e-6

Row = dict[str, Any]
LabelFn = Callable[[pd.DataFrame, int, BarContext, int], Optional[float]]


def volatility_range_label(entry_df: pd.DataFrame, i: int, ctx: BarContext,
                           horizon_bars: int = VOLATILITY_HORIZON_BARS
                           ) -> float | None:
    """Frozen primary target (reports/c41/volatility_model_frozen_spec.md §2):

        raw_ratio = (max(high[t+1..t+H]) - min(low[t+1..t+H])) / atr[t]
        label     = log(clip(raw_ratio, 0, 20) + 1e-6)

    Returns None (row excluded) if atr[t] <= 0 or the forward window doesn't
    fully exist (final-incomplete-horizon policy, C4.1 §1) — never
    truncated, never imputed.
    """
    if not ctx.atr or ctx.atr <= 0:
        return None
    lo_idx, hi_idx = i + 1, i + horizon_bars
    if hi_idx >= len(entry_df):
        return None  # incomplete horizon at the tail — excluded, not truncated
    window = entry_df.iloc[lo_idx:hi_idx + 1]
    raw_ratio = (float(window["high"].max()) - float(window["low"].min())) / ctx.atr
    clipped = min(max(raw_ratio, VOLATILITY_CLIP_LO), VOLATILITY_CLIP_HI)
    return math.log(clipped + VOLATILITY_LOG_EPS)


@dataclass(frozen=True)
class DatasetBuildResult:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    span_lo: int
    span_hi: int


def build_dataset(dataset: str, exchange: str, symbol: str, profile_name: str,
                  bars: int, *, label_fn: LabelFn = volatility_range_label,
                  label_version: str = VOLATILITY_LABEL_VERSION,
                  horizon_bars: int = VOLATILITY_HORIZON_BARS,
                  feature_names: list[str] | None = None,
                  max_gap_ratio: float = 0.001,
                  allow_estimated_cvd: bool = False) -> DatasetBuildResult:
    """Build a bar-level, point-in-time dataset (C4.1 §4: bar-level rows, no
    subsampling, deterministic ordering by construction since the walk is
    already index-ordered).

    Row exclusion follows C4.1 §1/§2/§3 exactly: a bar is skipped (not
    imputed) if feature context is unavailable (no_atr/htf_insufficient_
    history/no_zone_frames — same fail-closed gates as tools.deep_backtest.
    deep_walk) or if the label's forward window doesn't fully exist. No
    other reason excludes a row.
    """
    frames, profile, _table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    entry_df = frames[profile["entry"]].df
    n = len(entry_df)
    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    cache = _IndicatorCache()
    span_lo = walk_start(n, bars)
    span_hi = n - 1  # matches tools.deep_backtest's own walked-region convention

    rows: list[Row] = []
    skip_counts: dict[str, int] = {}
    for i in range(span_lo, n - horizon_bars):
        ctx, reason = build_bar_context(entry_df, i, frames, profile, close_ms,
                                        cache)
        if ctx is None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue
        label = label_fn(entry_df, i, ctx, horizon_bars)
        if label is None:
            skip_counts["label_unavailable"] = skip_counts.get(
                "label_unavailable", 0) + 1
            continue
        row = extract_row(ctx, feature_names)
        row["idx"] = i
        row["prediction_timestamp_utc"] = _iso(ctx.open_time)
        row["label"] = label
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

    manifest = {
        "dataset_builder_version": DATASET_BUILDER_VERSION,
        "feature_version": FEATURE_VERSION,
        "label_version": label_version,
        "horizon_bars": horizon_bars,
        "profile": profile_name,
        "exchange": exchange,
        "symbol": symbol,
        "bars_requested": bars,
        "cvd_method": cvd_method,
        "span_lo": span_lo,
        "span_hi": span_hi,
        "row_count": len(frame),
        "skip_counts": skip_counts,
        "source_manifest_refs": provenance(frames),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["dataset_version"] = _dataset_version_hash(manifest)
    return DatasetBuildResult(frame=frame, manifest=manifest, span_lo=span_lo,
                              span_hi=span_hi)


def _dataset_version_hash(manifest: dict[str, Any]) -> str:
    from tools.forecast_platform.contracts import spec_hash
    fingerprint = {k: v for k, v in manifest.items()
                  if k not in ("generated_at_utc",)}
    return spec_hash(fingerprint)


# ---------------------------------------------------------------------------
# Fold assignment: reuses tools.deep_backtest.WFConfig/fold_windows/
# partition_setups UNCHANGED — bar-level rows are fold-compatible with this
# machinery because it only requires each row to carry an "idx" key
# (reports/c41/volatility_model_frozen_spec.md §5).
# ---------------------------------------------------------------------------

MIN_VAL_ROWS_PER_FOLD = 300  # C4.1 §5, a safety floor, not expected to bind


def build_wf_config(profile: dict[str, Any], bars: int, *,
                    horizon_bars: int = VOLATILITY_HORIZON_BARS,
                    embargo_bars: int = 2, holdout_frac: float = 0.15,
                    min_folds: int = 3) -> WFConfig:
    """C4.1 §5 frozen geometry: purge = horizon EXACTLY (the frozen spec is
    explicit that this is the minimum required, "no extra margin" — a
    specification choice, not just a safety floor), embargo = 2 (smallest
    existing convention), holdout_frac = 0.15, min_folds = 3.

    WFConfig.for_profile's own `purge = max(purge_bars, max_hold_bars(
    profile))` floor exists to stop a PROFILE's own trade horizon leaking
    into validation — irrelevant here, since `profile` is only reused for
    its entry/htf/zone_tfs timeframe structure, not because this model
    cares about that profile's trades. Sizing (train/val/step bars) is
    still derived the normal way via for_profile; purge_bars is then
    replaced with the frozen horizon so it is never silently widened by an
    unrelated profile's own max_hold_bars (a strictly MORE conservative
    substitution would be safe too, but would silently contradict the
    frozen "exactly 12, no extra margin" spec choice)."""
    wf = WFConfig.for_profile(profile, walked_bars=bars,
                              embargo_bars=embargo_bars,
                              holdout_frac=holdout_frac, min_folds=min_folds)
    return dataclasses.replace(wf, purge_bars=horizon_bars)


def build_folds(span_lo: int, span_hi: int, wf: WFConfig) -> list[Fold]:
    return fold_windows(span_lo, span_hi, wf)


def partition_rows(rows: list[Row], fold: Fold, horizon_bars: int
                   ) -> tuple[list[Row], list[Row]]:
    """Train/validation split for one fold, purge-aware (C4.1 §4/§5): reuses
    tools.deep_backtest.partition_setups unchanged, passing `horizon_bars`
    as the "hold_bars" purge window (every row's label already needs
    exactly `horizon_bars` forward bars, so this is the correct purge
    horizon for THIS dataset, not the strategy's max_hold_bars)."""
    return partition_setups(rows, fold, horizon_bars)


def is_confident_fold(val_rows: list[Row],
                      min_rows: int = MIN_VAL_ROWS_PER_FOLD) -> bool:
    return len(val_rows) >= min_rows
