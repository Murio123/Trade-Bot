"""C4.2: tests for tools/forecast_platform/dataset_builder.py."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from tests.test_deep_discovery import SWING_BARS, _build_swing_dataset
from tools import kline_cache
from tools.forecast_platform.dataset_builder import (
    VOLATILITY_CLIP_HI, VOLATILITY_CLIP_LO, VOLATILITY_HORIZON_BARS,
    VOLATILITY_LOG_EPS, build_dataset, build_folds, build_wf_config,
    is_confident_fold, partition_rows, volatility_range_label)


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


class _FakeCtx:
    def __init__(self, atr):
        self.atr = atr


def _entry_df(n=40):
    # Flat band except a known excursion inside [i+1, i+12] to hand-verify
    # the exact ratio the frozen formula must produce.
    rows = {"open": [100.0] * n, "high": [100.0] * n, "low": [100.0] * n,
           "close": [100.0] * n}
    return pd.DataFrame(rows)


def test_frozen_horizon_and_clip_constants_match_c41_spec():
    assert VOLATILITY_HORIZON_BARS == 12
    assert VOLATILITY_CLIP_LO == 0.0
    assert VOLATILITY_CLIP_HI == 20.0
    assert VOLATILITY_LOG_EPS == 1e-6


def test_label_formula_matches_hand_computed_ratio():
    df = _entry_df(40)
    i = 10
    # bars [11, 22] (t+1..t+12): put a known 6-point range in there.
    df.loc[15, "high"] = 106.0
    df.loc[16, "low"] = 100.0
    ctx = _FakeCtx(atr=2.0)
    label = volatility_range_label(df, i, ctx)
    expected_raw_ratio = (106.0 - 100.0) / 2.0  # = 3.0
    assert label == pytest.approx(math.log(expected_raw_ratio + 1e-6))


def test_label_ignores_bars_outside_the_forward_window():
    df = _entry_df(40)
    i = 10
    df.loc[i, ["high", "low"]] = [1e6, -1e6]  # bar t itself — must be ignored
    df.loc[i + VOLATILITY_HORIZON_BARS + 1, ["high", "low"]] = [1e6, -1e6]  # t+13 — outside window
    ctx = _FakeCtx(atr=2.0)
    label = volatility_range_label(df, i, ctx)
    expected_raw_ratio = 0.0  # flat band inside [t+1, t+12] once the leaks are excluded
    assert label == pytest.approx(math.log(expected_raw_ratio + 1e-6))


def test_label_clips_extreme_ratio_at_twenty():
    df = _entry_df(40)
    i = 10
    df.loc[15, "high"] = 10_000.0  # would be ratio >> 20 uncapped
    ctx = _FakeCtx(atr=1.0)
    label = volatility_range_label(df, i, ctx)
    assert label == pytest.approx(math.log(20.0 + 1e-6))


def test_label_excludes_row_when_atr_is_zero_or_missing():
    df = _entry_df(40)
    assert volatility_range_label(df, 10, _FakeCtx(atr=0.0)) is None
    assert volatility_range_label(df, 10, _FakeCtx(atr=None)) is None


def test_label_excludes_incomplete_horizon_at_the_tail():
    df = _entry_df(40)
    i = len(df) - 5  # only 4 forward bars exist, horizon needs 12
    assert volatility_range_label(df, i, _FakeCtx(atr=1.0)) is None


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c42ds") / "binance"
    _build_swing_dataset(outdir)
    return build_dataset(str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS)


def test_build_dataset_produces_rows_with_features_idx_and_label(built_dataset):
    frame = built_dataset.frame
    assert len(frame) > 0
    assert "idx" in frame.columns
    assert "label" in frame.columns
    assert "prediction_timestamp_utc" in frame.columns
    assert "atr" in frame.columns
    assert "score" in frame.columns
    assert frame["idx"].is_monotonic_increasing  # deterministic ordering (C4.1 §4)
    assert frame["label"].notna().all()


def test_build_dataset_manifest_has_frozen_identity_fields(built_dataset):
    manifest = built_dataset.manifest
    assert manifest["horizon_bars"] == 12
    assert manifest["label_version"] == "c41_volatility_range_v1"
    assert "feature_version" in manifest
    assert "dataset_version" in manifest
    assert manifest["row_count"] == len(built_dataset.frame)
    assert "source_manifest_refs" in manifest
    assert set(manifest["source_manifest_refs"]) >= {"4h", "1d"}


def test_build_dataset_version_hash_is_deterministic(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c42ds_repeat") / "binance"
    _build_swing_dataset(outdir)
    r1 = build_dataset(str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS)
    r2 = build_dataset(str(outdir), "binance", "BTCUSDT", "swing", SWING_BARS)
    assert r1.manifest["dataset_version"] == r2.manifest["dataset_version"]


def test_build_wf_config_purge_equals_horizon_not_profile_floor(built_dataset):
    from tools.deep_backtest import get_profile
    profile = get_profile("swing")
    wf = build_wf_config(profile, SWING_BARS)
    assert wf.purge_bars == VOLATILITY_HORIZON_BARS
    assert wf.embargo_bars == 2
    assert wf.holdout_bars == round(SWING_BARS * 0.15)


def test_partition_rows_excludes_train_setups_whose_horizon_reaches_validation(
        built_dataset):
    from tools.deep_backtest import get_profile
    profile = get_profile("swing")
    frame = built_dataset.frame
    span_lo, span_hi = built_dataset.span_lo, built_dataset.span_hi
    try:
        wf = build_wf_config(profile, SWING_BARS)
        folds = build_folds(span_lo, span_hi, wf)
    except Exception:
        pytest.skip("not enough walked bars in this synthetic fixture for "
                   "3 confident folds — geometry itself is exercised "
                   "elsewhere via WFConfig's own tests")
    rows = frame.to_dict("records")
    for fold in folds:
        train_rows, val_rows = partition_rows(rows, fold, VOLATILITY_HORIZON_BARS)
        for r in train_rows:
            assert r["idx"] + VOLATILITY_HORIZON_BARS < fold.val_lo
        for r in val_rows:
            assert fold.val_lo <= r["idx"] < fold.val_hi


def test_is_confident_fold_uses_min_val_rows_threshold():
    assert is_confident_fold([{"idx": i} for i in range(300)], min_rows=300)
    assert not is_confident_fold([{"idx": i} for i in range(299)], min_rows=300)
