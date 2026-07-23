"""C4.2: tests for tools/forecast_platform/feature_store.py.

Reuses the swing synthetic-dataset generator already established in
test_deep_discovery.py (4h/1d/12h/6h, real random-walk OHLCV with
taker_buy_base) rather than building a parallel fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_deep_discovery import SWING_BARS, _build_swing_dataset
from tools import kline_cache
from tools.deep_backtest import _IndicatorCache, _close_ms, prepare
from tools.forecast_platform.feature_store import (BENCHMARK_ONLY_FEATURES,
                                                    FEATURE_REGISTRY,
                                                    _prev_window_ratio,
                                                    build_bar_context,
                                                    extract_row,
                                                    feature_schema,
                                                    model_input_names)


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


@pytest.fixture(scope="module")
def swing_inputs(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c42fs") / "binance"
    _build_swing_dataset(outdir)
    frames, profile, _table, _cvd = prepare(
        str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS, 0.001, False)
    return frames, profile


def test_registry_has_exactly_twenty_features_per_c41_spec():
    assert len(FEATURE_REGISTRY) == 20


def test_score_is_the_only_benchmark_only_feature():
    assert BENCHMARK_ONLY_FEATURES == frozenset({"score"})


def test_model_input_names_excludes_benchmark_only_and_has_nineteen_entries():
    names = model_input_names()
    assert len(names) == 19
    assert "score" not in names


def test_feature_schema_matches_registry_and_flags_benchmark_only():
    schema = feature_schema()
    assert len(schema) == 20
    by_name = {row["name"]: row for row in schema}
    assert by_name["score"]["benchmark_only"] is True
    assert by_name["atr"]["benchmark_only"] is False


def test_build_bar_context_returns_populated_context(swing_inputs):
    frames, profile = swing_inputs
    entry_df = frames[profile["entry"]].df
    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    cache = _IndicatorCache()
    # bar late enough in the walked region to clear every warmup gate.
    i = len(entry_df) - 20
    ctx, reason = build_bar_context(entry_df, i, frames, profile, close_ms,
                                    cache)
    assert reason is None
    assert ctx is not None
    assert ctx.atr > 0
    assert ctx.htf_bias in ("bullish", "bearish", "neutral")
    assert ctx.regime_current in ("trend_up", "trend_down", "range",
                                  "high_volatility")
    assert isinstance(ctx.bb_squeeze, bool)
    assert 0 <= ctx.weekday <= 6
    assert 1 <= ctx.month <= 12
    assert ctx.score is None  # documented placeholder, never wired in C4.2


def test_build_bar_context_excludes_row_before_entry_warmup(swing_inputs):
    frames, profile = swing_inputs
    entry_df = frames[profile["entry"]].df
    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    cache = _IndicatorCache()
    ctx, reason = build_bar_context(entry_df, 5, frames, profile, close_ms,
                                    cache)
    # Too early for a full 300-bar entry warmup slice to exist meaningfully —
    # either no_atr or htf_insufficient_history, never a populated context.
    assert ctx is None
    assert reason in ("no_atr", "htf_insufficient_history", "no_zone_frames")


def test_prev_window_ratio_never_reads_the_current_bar():
    """No-look-ahead check mirroring tools.swing_hypothesis_simulator's own
    _rolling_extremes test: _prev_window_ratio only reads bars strictly
    BEFORE i, so mutating bar i itself must never change the result (unlike
    a full build_bar_context() comparison, which would also legitimately
    change because atr[i] itself depends on bar i's own true range —
    that's correct ATR semantics, not a look-ahead bug, so it is tested
    here directly against the pure helper instead)."""
    df = pd.DataFrame({"high": [100.0] * 30, "low": [90.0] * 30})
    i = 20
    before = _prev_window_ratio(df, i, window=12, atr=2.0)

    mutated = df.copy()
    mutated.loc[i, ["high", "low"]] = [1e9, -1e9]
    after = _prev_window_ratio(mutated, i, window=12, atr=2.0)

    assert before == after == (100.0 - 90.0) / 2.0


def test_extract_row_pulls_exactly_the_requested_features(swing_inputs):
    frames, profile = swing_inputs
    entry_df = frames[profile["entry"]].df
    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    cache = _IndicatorCache()
    i = len(entry_df) - 20
    ctx, _ = build_bar_context(entry_df, i, frames, profile, close_ms, cache)
    row = extract_row(ctx, ["atr", "htf_bias"])
    assert set(row) == {"atr", "htf_bias"}
    assert row["atr"] == ctx.atr


def test_extract_row_defaults_to_every_registered_feature(swing_inputs):
    frames, profile = swing_inputs
    entry_df = frames[profile["entry"]].df
    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    cache = _IndicatorCache()
    i = len(entry_df) - 20
    ctx, _ = build_bar_context(entry_df, i, frames, profile, close_ms, cache)
    row = extract_row(ctx)
    assert set(row) == set(FEATURE_REGISTRY)
    assert row["score"] is None
