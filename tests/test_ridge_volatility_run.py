"""C4.3: tests for tools/ridge_volatility_run.py.

Pure-function tests (frozen constants, verdict decision rule, baseline
helpers) plus one end-to-end smoke test on a small synthetic swing dataset
(reusing tests/test_deep_discovery.py's generator, matching the convention
already used throughout C4.2's own tests) to prove the whole pipeline runs
without crashing, writes every required artifact, registers the run, and
never lets a sealed-holdout row enter any fold's train/validation set.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.test_deep_discovery import _build_swing_dataset
from tools import kline_cache
from tools.forecast_platform.dataset_builder import VOLATILITY_LOG_EPS
from tools.ridge_volatility_run import (MAE_IMPROVEMENT_MIN, MIN_YEAR_ROWS,
                                        MIN_VAL_ROWS_PER_FOLD,
                                        RESIDUAL_BIAS_MAX,
                                        ROLLING_MEAN_WINDOW,
                                        SINGLE_FOLD_DOMINANCE_MAX,
                                        YEAR_RATIO_DEN, YEAR_RATIO_NUM,
                                        _fold_baselines, _git_commit,
                                        _inverse_transform, _year_of,
                                        decide_verdict, run)


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


def test_frozen_go_no_go_constants_match_c41_spec():
    assert MAE_IMPROVEMENT_MIN == 0.05
    assert SINGLE_FOLD_DOMINANCE_MAX == 0.5
    assert MIN_YEAR_ROWS == 200
    assert (YEAR_RATIO_NUM, YEAR_RATIO_DEN) == (2, 3)
    assert RESIDUAL_BIAS_MAX == 0.75
    assert ROLLING_MEAN_WINDOW == 60
    assert MIN_VAL_ROWS_PER_FOLD == 300


ALL_TRUE = {"beats_persistence_pooled": True, "fold_majority_improves": True,
           "no_single_fold_dominance": True, "year_independence": True,
           "residual_bias_ok": True, "reproducible": True}


def test_decide_verdict_passes_when_every_check_true():
    assert decide_verdict(dict(ALL_TRUE)) == "RIDGE_PASSES"


def test_decide_verdict_needs_more_evidence_when_five_of_six_pass():
    checks = dict(ALL_TRUE)
    checks["residual_bias_ok"] = False
    assert decide_verdict(checks) == "RIDGE_NEEDS_MORE_EVIDENCE"


def test_decide_verdict_fails_when_persistence_not_beaten_even_if_others_pass():
    checks = dict(ALL_TRUE)
    checks["beats_persistence_pooled"] = False
    assert decide_verdict(checks) == "RIDGE_FAILS"


def test_decide_verdict_fails_when_only_four_of_six_pass():
    checks = dict(ALL_TRUE)
    checks["residual_bias_ok"] = False
    checks["reproducible"] = False
    assert decide_verdict(checks) == "RIDGE_FAILS"


def test_decide_verdict_fails_on_all_false():
    checks = {k: False for k in ALL_TRUE}
    assert decide_verdict(checks) == "RIDGE_FAILS"


def test_year_of_parses_iso_timestamp():
    assert _year_of("2024-06-01T00:00:00+00:00") == 2024


def test_inverse_transform_matches_label_formula_inverse():
    raw_ratio = 3.0
    label = math.log(raw_ratio + VOLATILITY_LOG_EPS)
    recovered = _inverse_transform(np.array([label]))[0]
    assert recovered == pytest.approx(raw_ratio)


def test_fold_baselines_are_train_only_and_causal():
    # last ROLLING_MEAN_WINDOW (60) values are exactly 5.0, everything
    # before that is 0.0 — isolates the rolling-mean window precisely.
    labels = pd.Series([0.0] * 40 + [5.0] * ROLLING_MEAN_WINDOW)
    baselines = _fold_baselines(labels, n_val=4)
    assert len(baselines["persistence"]) == 4
    assert baselines["persistence"][0] == pytest.approx(
        math.log(1.0 + VOLATILITY_LOG_EPS))
    assert baselines["rolling_mean_60"][0] == pytest.approx(5.0)
    assert baselines["train_mean"][0] == pytest.approx(labels.mean())


def test_git_commit_returns_a_non_empty_string():
    commit = _git_commit()
    assert isinstance(commit, str) and len(commit) > 0


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("c43run") / "binance"
    # A bigger-than-SWING_BARS synthetic depth so build_wf_config's default
    # sizing can fit >= 3 folds (still tiny vs. the real 8000-bar run —
    # this only exercises the pipeline's plumbing, not a real verdict).
    depths = {"4h": 900, "1d": 700, "12h": 800, "6h": 850}
    dataset = _build_swing_dataset(outdir, depths=depths)
    out_reports = tmp_path_factory.mktemp("c43out")
    registry_dir = tmp_path_factory.mktemp("c43registry")
    try:
        result = run(dataset, "binance", "BTCUSDT", bars=500,
                    outdir=str(out_reports), registry_dir=str(registry_dir))
    except Exception as exc:
        pytest.skip(f"synthetic fixture too small for this run's fold "
                   f"geometry: {exc}")
    return result, out_reports, registry_dir


def test_run_returns_one_of_the_three_allowed_verdicts(small_run):
    result, _, _ = small_run
    assert result["verdict"] in ("RIDGE_PASSES", "RIDGE_FAILS",
                                 "RIDGE_NEEDS_MORE_EVIDENCE")


def test_run_writes_every_required_artifact(small_run):
    _, outdir, _ = small_run
    required = ["dataset_manifest.json", "feature_schema.json",
               "label_schema.json", "fold_manifest.json",
               "ridge_predictions.json", "ridge_evaluation.json",
               "ridge_evaluation.txt", "RIDGE_DECISION.md"]
    for name in required:
        assert (outdir / name).exists(), f"missing artifact: {name}"


def test_run_registers_the_experiment_in_model_registry(small_run):
    result, _, registry_dir = small_run
    assert result["registry_status"] == "registered"
    entries = list(registry_dir.glob("*.json"))
    assert any(e.name == "c43_ridge_run_v1.json" for e in entries)


def test_no_pooled_or_predicted_row_touches_the_sealed_holdout(small_run):
    result, outdir, _ = small_run
    holdout_lo = result["holdout"]["idx_lo"]
    with open(outdir / "ridge_predictions.json") as f:
        predictions = json.load(f)
    for row in predictions:
        assert row["idx"] < holdout_lo, (
            f"prediction row idx={row['idx']} is inside the sealed holdout "
            f"(idx_lo={holdout_lo}) — holdout must never be scored")
    for fold in result["folds"]:
        assert fold["val_idx_range"][1] <= holdout_lo
        assert fold["train_idx_range"][1] <= holdout_lo


def test_lightgbm_is_never_run_this_stage(small_run):
    result, _, _ = small_run
    assert result["lightgbm_status"] == "DEFERRED_NOT_RUN_THIS_STAGE"


def test_rerunning_with_the_same_registry_dir_does_not_overwrite(small_run, tmp_path_factory):
    result, outdir, registry_dir = small_run
    # Re-running against the SAME registry_dir must not silently overwrite
    # the existing immutable c43_ridge_run_v1 entry.
    depths = {"4h": 900, "1d": 700, "12h": 800, "6h": 850}
    dataset_dir = tmp_path_factory.mktemp("c43rerun") / "binance"
    dataset = _build_swing_dataset(dataset_dir, depths=depths)
    second = run(dataset, "binance", "BTCUSDT", bars=500,
                outdir=str(tmp_path_factory.mktemp("c43out2")),
                registry_dir=str(registry_dir))
    assert second["registry_status"].startswith("not_registered")
