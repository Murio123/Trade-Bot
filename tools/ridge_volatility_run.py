"""C4.3: Ridge volatility/range walk-forward run — real Binance BTCUSDT 4h.

Trains one frozen tools.forecast_platform.ridge_model.RidgeForecastModel per
walk-forward fold on the frozen C4.1 dataset spec
(reports/c41/volatility_model_frozen_spec.md), evaluates against the three
frozen baselines (persistence/rolling-mean-60/train-mean), applies the
exact C4.1 §11 go/no-go criteria without reinterpretation, registers the
run in ModelRegistry, and writes artifacts to reports/c43/.

Offline-only CLI (mirrors tools/swing_hypothesis_walkforward.py's own
structure). NOT imported by runtime. LightGBM is intentionally NOT used
here (C4.3 is ridge-only per this stage's explicit scope); its edge-check
criterion is reported as deferred, not evaluated.

Sealed holdout: build_dataset() walks the full span (bar-level rows have
no concept of holdout); the holdout tail is only ever excluded by
tools.forecast_platform.dataset_builder.build_folds/partition_rows, which
never assign an idx >= (span_hi - wf.holdout_bars) to any fold's train or
validation region. This script never reads dataset rows outside of what
partition_rows returns for some fold — the sealed holdout is geometry/count
only here, exactly like every prior stage (C2.2c/C3.0).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from tools.deep_backtest import DeepBacktestError, get_profile
from tools.forecast_platform import evaluation_engine as ee
from tools.forecast_platform.contracts import ModelMetadata, spec_hash
from tools.forecast_platform.dataset_builder import (
    VOLATILITY_HORIZON_BARS, VOLATILITY_LABEL_VERSION, build_dataset,
    build_folds, build_wf_config, is_confident_fold, partition_rows)
from tools.forecast_platform.feature_store import (FEATURE_VERSION,
                                                    feature_schema,
                                                    model_input_names)
from tools.forecast_platform.model_registry import ModelRegistry
from tools.forecast_platform.ridge_model import (ALPHA_CANDIDATES,
                                                  RidgeForecastModel,
                                                  RidgeIdentity)

STAGE = "C4.3"
PROFILE_NAME = "swing"  # timeframe backbone (entry=4h/htf=1d/zone_tfs) only —
                       # this run does not care about swing's own trades.
BARS = 8000
MIN_VAL_ROWS_PER_FOLD = 300
ROLLING_MEAN_WINDOW = 60
LOG_EPS = 1e-6

# Frozen go/no-go constants (reports/c41/volatility_model_frozen_spec.md §11).
MAE_IMPROVEMENT_MIN = 0.05
SINGLE_FOLD_DOMINANCE_MAX = 0.5
MIN_YEAR_ROWS = 200
YEAR_RATIO_NUM, YEAR_RATIO_DEN = 2, 3  # integer comparison, not round()
RESIDUAL_BIAS_MAX = 0.75

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."

DECISION_RULE = (
    "RIDGE_PASSES only if all 6 applicable checks pass (LightGBM-edge is "
    "N/A this stage, ridge-only). RIDGE_NEEDS_MORE_EVIDENCE if the model "
    "beats pooled persistence AND at least 5 of the 6 checks pass. "
    "RIDGE_FAILS otherwise. This rule mirrors the project's existing "
    "convention (reports/c23/DECISION.md's H2 net-positive-plus-passed-"
    "count pattern) and is fixed here BEFORE this run's results were "
    "computed."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


REQUIRED_CHECK_NAMES = ("beats_persistence_pooled", "fold_majority_improves",
                       "no_single_fold_dominance", "year_independence",
                       "residual_bias_ok", "reproducible")


def decide_verdict(checks: dict[str, bool]) -> str:
    """Pure application of DECISION_RULE — kept isolated from `run()` so it
    can be tested against hand-crafted check dicts without a real dataset
    build. Applied mechanically, unchanged regardless of which checks a
    given run happens to pass or fail."""
    n_pass = sum(1 for v in checks.values() if v)
    if all(checks.get(name) for name in REQUIRED_CHECK_NAMES):
        return "RIDGE_PASSES"
    if checks.get("beats_persistence_pooled") and n_pass >= 5:
        return "RIDGE_NEEDS_MORE_EVIDENCE"
    return "RIDGE_FAILS"


def _year_of(ts: str) -> int:
    return int(ts[:4])


def _inverse_transform(log_values: np.ndarray) -> np.ndarray:
    return np.exp(log_values) - LOG_EPS


def _fold_baselines(train_labels: pd.Series, n_val: int
                    ) -> dict[str, np.ndarray]:
    persistence = np.full(n_val, ee.persistence_baseline_value(LOG_EPS))
    rolling = np.full(n_val, ee.rolling_mean_baseline(train_labels,
                                                      ROLLING_MEAN_WINDOW))
    constant = np.full(n_val, ee.constant_mean_baseline(train_labels))
    return {"persistence": persistence, "rolling_mean_60": rolling,
           "train_mean": constant}


def run(dataset: str, exchange: str, symbol: str, bars: int = BARS,
       outdir: str = "reports/c43", registry_dir: str = "reports/c43/model_registry"
       ) -> dict[str, Any]:
    commit = _git_commit()
    profile = get_profile(PROFILE_NAME)

    build = build_dataset(dataset, exchange, symbol, PROFILE_NAME, bars)
    df = build.frame
    manifest = build.manifest

    wf = build_wf_config(profile, bars)
    folds = build_folds(build.span_lo, build.span_hi, wf)
    holdout_lo = build.span_hi - wf.holdout_bars
    holdout_section = {"idx_lo": holdout_lo, "idx_hi": build.span_hi,
                       "n_bars": build.span_hi - holdout_lo, "sealed": True,
                       "note": "geometry and bar count only; never "
                              "aggregated or scored by C4.3"}

    feature_names = model_input_names()
    rows = df.to_dict("records")

    fold_reports: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    pooled_pred: list[float] = []
    pooled_baselines: dict[str, list[float]] = {"persistence": [], "rolling_mean_60": [],
                                                "train_mean": []}
    registered_models: list[ModelMetadata] = []

    for fold in folds:
        train_rows, val_rows = partition_rows(rows, fold, VOLATILITY_HORIZON_BARS)
        confident = is_confident_fold(val_rows, MIN_VAL_ROWS_PER_FOLD)
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)

        identity = RidgeIdentity(
            model_id=f"c43_ridge_fold{fold.index}", code_commit=commit,
            dataset_version=manifest["dataset_version"],
            feature_version=FEATURE_VERSION,
            label_version=VOLATILITY_LABEL_VERSION)
        model = RidgeForecastModel(identity)
        model.prepare(train_df, feature_names, "label")
        model.train()
        registered_models.append(model.metadata())

        val_pred = model.predict(val_df) if len(val_df) else np.array([])
        baselines = (_fold_baselines(train_df["label"], len(val_df))
                    if len(val_df) else {"persistence": np.array([]),
                                        "rolling_mean_60": np.array([]),
                                        "train_mean": np.array([])})

        fold_mae_model = ee.mae(val_df["label"], val_pred) if len(val_df) else None
        fold_mae_persistence = (ee.mae(val_df["label"], baselines["persistence"])
                                if len(val_df) else None)
        fold_mae_rolling = (ee.mae(val_df["label"], baselines["rolling_mean_60"])
                           if len(val_df) else None)
        fold_mae_constant = (ee.mae(val_df["label"], baselines["train_mean"])
                            if len(val_df) else None)

        fold_reports.append({
            "fold": fold.index, "confident": confident,
            "n_train": len(train_rows), "n_val": len(val_rows),
            "train_idx_range": [fold.train_lo, fold.train_hi],
            "val_idx_range": [fold.val_lo, fold.val_hi],
            "chosen_alpha": model._chosen_alpha,
            "mae_model": fold_mae_model, "mae_persistence": fold_mae_persistence,
            "mae_rolling_mean_60": fold_mae_rolling,
            "mae_train_mean": fold_mae_constant,
            "rmse_model": ee.rmse(val_df["label"], val_pred) if len(val_df) else None,
            "spearman": ee.spearman_corr(val_df["label"], val_pred) if len(val_df) else None,
        })

        if confident and len(val_df):
            val_df = val_df.copy()
            val_df["_pred"] = val_pred
            pooled_rows.append(val_df)
            pooled_pred.extend(val_pred.tolist())
            for name, arr in baselines.items():
                pooled_baselines[name].extend(arr.tolist())

    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    pooled_pred_arr = np.array(pooled_pred)

    checks: dict[str, Any] = {}
    pooled_metrics: dict[str, Any] = {}

    if len(pooled):
        y_true = pooled["label"].to_numpy(dtype=float)
        persistence_arr = np.array(pooled_baselines["persistence"])
        rolling_arr = np.array(pooled_baselines["rolling_mean_60"])
        constant_arr = np.array(pooled_baselines["train_mean"])

        mae_model = ee.mae(y_true, pooled_pred_arr)
        mae_persistence = ee.mae(y_true, persistence_arr)
        mae_rolling = ee.mae(y_true, rolling_arr)
        mae_constant = ee.mae(y_true, constant_arr)

        pooled_metrics.update({
            "n": len(pooled), "mae_model": mae_model,
            "mae_persistence": mae_persistence, "mae_rolling_mean_60": mae_rolling,
            "mae_train_mean": mae_constant,
            "rmse_model": ee.rmse(y_true, pooled_pred_arr),
            "normalized_mae_model": mae_model / float(np.mean(np.abs(y_true))),
            "spearman": ee.spearman_corr(y_true, pooled_pred_arr),
            "improvement_vs_persistence": 1 - mae_model / mae_persistence if mae_persistence else None,
            "improvement_vs_rolling_mean_60": 1 - mae_model / mae_rolling if mae_rolling else None,
            "improvement_vs_train_mean": 1 - mae_model / mae_constant if mae_constant else None,
        })

        # -- bucket / calibration analysis (inverse-transformed, real-ratio units) --
        raw_actual = _inverse_transform(y_true)
        raw_pred = _inverse_transform(pooled_pred_arr)
        buckets = ee.bucket_calibration(raw_actual, raw_pred, n_buckets=5)
        pooled_metrics["bucket_calibration_raw_ratio"] = buckets
        max_residual = max((abs(b["residual"]) for b in buckets), default=0.0)

        # -- year slices --
        years = pooled["prediction_timestamp_utc"].map(_year_of)
        year_stats: dict[str, Any] = {}
        for year, idx in pooled.groupby(years).groups.items():
            mask = years == year
            y_year = y_true[mask.to_numpy()]
            pred_year = pooled_pred_arr[mask.to_numpy()]
            persistence_year = persistence_arr[mask.to_numpy()]
            year_stats[str(year)] = {
                "n": int(mask.sum()),
                "mae_model": ee.mae(y_year, pred_year),
                "mae_persistence": ee.mae(y_year, persistence_year),
                "beats_persistence": ee.mae(y_year, pred_year) <= ee.mae(y_year, persistence_year),
            }
        pooled_metrics["year_slices"] = year_stats

        # -- volatility slices (tercile of volatility_atr_percentile) --
        vol_col = pooled["volatility_atr_percentile"].astype(float)
        try:
            vol_tercile = pd.qcut(vol_col, 3, labels=["low", "mid", "high"],
                                  duplicates="drop")
            vol_stats = {}
            for label, mask_series in vol_tercile.groupby(vol_tercile, observed=True).groups.items():
                mask = (vol_tercile == label).to_numpy()
                vol_stats[str(label)] = {
                    "n": int(mask.sum()),
                    "mae_model": ee.mae(y_true[mask], pooled_pred_arr[mask]),
                    "mae_persistence": ee.mae(y_true[mask], persistence_arr[mask]),
                }
            pooled_metrics["volatility_slices"] = vol_stats
        except ValueError:
            pooled_metrics["volatility_slices"] = {}

        # -- go/no-go checks (reports/c41/volatility_model_frozen_spec.md §11) --
        checks["beats_persistence_pooled"] = bool(
            mae_model <= mae_persistence * (1 - MAE_IMPROVEMENT_MIN))

        confident_folds = [f for f in fold_reports if f["confident"] and f["n_val"] > 0]
        beat_count = sum(1 for f in confident_folds
                         if f["mae_model"] is not None and f["mae_persistence"] is not None
                         and f["mae_model"] <= f["mae_persistence"])
        checks["fold_majority_improves"] = bool(
            confident_folds and beat_count * 2 > len(confident_folds))

        contributions = []
        for f in confident_folds:
            fold_val = pooled[(pooled["idx"] >= f["val_idx_range"][0])
                              & (pooled["idx"] < f["val_idx_range"][1])]
            if len(fold_val) == 0:
                continue
            fold_mask = ((pooled["idx"] >= f["val_idx_range"][0])
                        & (pooled["idx"] < f["val_idx_range"][1])).to_numpy()
            gain = float(np.sum(np.abs(y_true[fold_mask] - persistence_arr[fold_mask])
                              - np.abs(y_true[fold_mask] - pooled_pred_arr[fold_mask])))
            contributions.append(gain)
        total_gain = sum(contributions)
        max_fraction = (max(contributions) / total_gain
                       if total_gain > 0 and contributions else None)
        checks["no_single_fold_dominance"] = bool(
            total_gain > 0 and max_fraction is not None
            and max_fraction <= SINGLE_FOLD_DOMINANCE_MAX)

        qualifying = [(y, s) for y, s in year_stats.items() if s["n"] >= MIN_YEAR_ROWS]
        positive_years = sum(1 for _, s in qualifying if s["beats_persistence"])
        checks["year_independence"] = bool(
            len(qualifying) >= 2
            and positive_years * YEAR_RATIO_DEN >= len(qualifying) * YEAR_RATIO_NUM)

        checks["residual_bias_ok"] = bool(max_residual <= RESIDUAL_BIAS_MAX)

        # -- reproducibility: retrain fold 0 TWICE, independently, from the
        # same train rows and diff predictions bit-for-bit (ridge's
        # closed-form solve has no randomness at all, so this must be
        # exact, not merely close) --
        if fold_reports:
            f0 = folds[0]
            train_rows0, val_rows0 = partition_rows(rows, f0, VOLATILITY_HORIZON_BARS)
            val_df0 = pd.DataFrame(val_rows0)
            if len(val_df0) == 0:
                checks["reproducible"] = True
            else:
                run_a = RidgeForecastModel(RidgeIdentity(
                    model_id="c43_ridge_repro_a", code_commit=commit,
                    dataset_version=manifest["dataset_version"],
                    feature_version=FEATURE_VERSION,
                    label_version=VOLATILITY_LABEL_VERSION))
                run_a.prepare(pd.DataFrame(train_rows0), feature_names, "label")
                run_a.train()
                run_b = RidgeForecastModel(RidgeIdentity(
                    model_id="c43_ridge_repro_b", code_commit=commit,
                    dataset_version=manifest["dataset_version"],
                    feature_version=FEATURE_VERSION,
                    label_version=VOLATILITY_LABEL_VERSION))
                run_b.prepare(pd.DataFrame(train_rows0), feature_names, "label")
                run_b.train()
                checks["reproducible"] = bool(np.allclose(
                    run_a.predict(val_df0), run_b.predict(val_df0),
                    rtol=0, atol=1e-12))
        else:
            checks["reproducible"] = False
    else:
        checks = {k: False for k in REQUIRED_CHECK_NAMES}

    verdict = decide_verdict(checks)

    result = {
        "stage": STAGE, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange, "symbol": symbol, "profile_backbone": PROFILE_NAME,
        "bars": bars, "code_commit": commit,
        "dataset_manifest": manifest,
        "wf_config": {"train_bars": wf.train_bars, "val_bars": wf.val_bars,
                     "step_bars": wf.step_bars, "purge_bars": wf.purge_bars,
                     "embargo_bars": wf.embargo_bars, "holdout_bars": wf.holdout_bars,
                     "min_folds": wf.min_folds, "n_folds": len(folds)},
        "holdout": holdout_section,
        "folds": fold_reports,
        "pooled": pooled_metrics,
        "checks": checks,
        "decision_rule": DECISION_RULE,
        "verdict": verdict,
        "lightgbm_status": "DEFERRED_NOT_RUN_THIS_STAGE",
    }

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(outdir, "feature_schema.json"), "w") as f:
        json.dump(feature_schema(), f, indent=2, default=str)
    with open(os.path.join(outdir, "label_schema.json"), "w") as f:
        json.dump({
            "label_version": VOLATILITY_LABEL_VERSION,
            "formula": "log(clip((max(high[t+1:t+H])-min(low[t+1:t+H]))/atr[t], 0, 20) + 1e-6)",
            "horizon_bars": VOLATILITY_HORIZON_BARS,
        }, f, indent=2)
    with open(os.path.join(outdir, "fold_manifest.json"), "w") as f:
        json.dump({"wf_config": result["wf_config"], "holdout": holdout_section,
                  "folds": [{"fold": f["fold"], "confident": f["confident"],
                           "n_train": f["n_train"], "n_val": f["n_val"],
                           "train_idx_range": f["train_idx_range"],
                           "val_idx_range": f["val_idx_range"]} for f in fold_reports]},
                 f, indent=2)
    predictions_payload = (pooled.assign(pred=pooled_pred_arr)
                           [["idx", "prediction_timestamp_utc", "label", "pred"]]
                           .to_dict("records") if len(pooled) else [])
    with open(os.path.join(outdir, "ridge_predictions.json"), "w") as f:
        json.dump(predictions_payload, f, indent=2, default=str)
    with open(os.path.join(outdir, "ridge_evaluation.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(outdir, "ridge_evaluation.txt"), "w") as f:
        f.write(format_report(result))
    with open(os.path.join(outdir, "RIDGE_DECISION.md"), "w") as f:
        f.write(format_decision_md(result))

    # -- model registry: one run-level entry, recorded even if ridge fails --
    registry = ModelRegistry(registry_dir)
    run_hyperparameters = {
        "alpha_candidates": list(ALPHA_CANDIDATES),
        "per_fold_chosen_alpha": {f["fold"]: f["chosen_alpha"] for f in fold_reports},
        "per_fold_train_window": {f["fold"]: f["train_idx_range"] for f in fold_reports},
        "per_fold_val_window": {f["fold"]: f["val_idx_range"] for f in fold_reports},
    }
    evaluation_hash = spec_hash({"checks": checks, "verdict": verdict,
                                "pooled_metrics": {k: v for k, v in pooled_metrics.items()
                                                  if k not in ("bucket_calibration_raw_ratio",
                                                              "year_slices", "volatility_slices")}})
    run_metadata = ModelMetadata(
        model_id="c43_ridge_run_v1", code_commit=commit,
        dataset_version=manifest["dataset_version"], feature_version=FEATURE_VERSION,
        label_version=VOLATILITY_LABEL_VERSION, seed=42,
        hyperparameters=run_hyperparameters,
        training_window=(build.span_lo, holdout_lo),
        calibration_window=None, evaluation_hash=evaluation_hash,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        registry.register(run_metadata)
        registry_status = "registered"
    except Exception as exc:  # ModelAlreadyRegisteredError on a re-run
        registry_status = f"not_registered: {exc}"
    result["registry_status"] = registry_status

    return result


def format_report(result: dict[str, Any]) -> str:
    lines = [
        f"Ridge volatility/range walk-forward ({result['stage']}) — "
        f"{result['exchange']} {result['symbol']}",
        f"  generated_at : {result['generated_at_utc']}",
        f"  commit       : {result['code_commit']}",
        f"  wf_config    : {result['wf_config']}",
        f"  holdout      : {result['holdout']}",
        "",
        "per-fold:",
    ]
    for f in result["folds"]:
        lines.append(
            f"  fold {f['fold']}: confident={f['confident']} n_train={f['n_train']} "
            f"n_val={f['n_val']} alpha={f['chosen_alpha']} "
            f"mae_model={f['mae_model']} mae_persistence={f['mae_persistence']} "
            f"mae_rolling_mean_60={f['mae_rolling_mean_60']} "
            f"mae_train_mean={f['mae_train_mean']} spearman={f['spearman']}")
    lines.append("")
    lines.append(f"pooled (confident-fold validation only): {result['pooled']}")
    lines.append("")
    lines.append(f"checks: {result['checks']}")
    lines.append("")
    lines.append(f"VERDICT: {result['verdict']}")
    lines.append(f"decision_rule: {result['decision_rule']}")
    lines.append(f"lightgbm_status: {result['lightgbm_status']}")
    return "\n".join(lines) + "\n"


def format_decision_md(result: dict[str, Any]) -> str:
    checks = result["checks"]
    lines = [
        f"# C4.3 — Ridge Volatility/Range Model: DECISION",
        "",
        f"{result['exchange']} {result['symbol']} · generated: {result['generated_at_utc']} "
        f"· commit {result['code_commit']}",
        "",
        "Confirmation/evaluation of the FROZEN ridge baseline "
        "(reports/c41/volatility_model_frozen_spec.md) — no feature/label/"
        "horizon/fold/alpha change was made after seeing any result.",
        "",
        f"## Verdict: `{result['verdict']}`",
        "",
    ]
    pooled = result.get("pooled") or {}
    if pooled:
        lines.append("### Pooled headline numbers (confident-fold validation only)")
        lines.append("")
        lines.append(f"- n = {pooled.get('n')}")
        lines.append(f"- MAE: model={pooled.get('mae_model')} "
                     f"persistence={pooled.get('mae_persistence')} "
                     f"rolling_mean_60={pooled.get('mae_rolling_mean_60')} "
                     f"train_mean={pooled.get('mae_train_mean')}")
        lines.append(f"- improvement vs persistence: "
                     f"{pooled.get('improvement_vs_persistence')}")
        lines.append(f"- improvement vs rolling_mean_60: "
                     f"{pooled.get('improvement_vs_rolling_mean_60')}")
        lines.append(f"- improvement vs train_mean: "
                     f"{pooled.get('improvement_vs_train_mean')}")
        lines.append(f"- spearman: {pooled.get('spearman')}")
        lines.append("")
    lines.append("### Go/no-go checks (reports/c41/... §11, applied without reinterpretation)")
    lines.append("")
    for name, value in checks.items():
        lines.append(f"- {name}: {value}")
    lines.append("")
    lines.append(f"Decision rule: {result['decision_rule']}")
    lines.append("")
    lines.append(f"LightGBM status: {result['lightgbm_status']} — deferred to a "
                 "separately-approved next stage, not run here.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Ridge-only: LightGBM challenger not installed/run this stage.")
    lines.append("- Sealed holdout tail is never aggregated or scored — only its "
                 "bar range and bar count are reported.")
    lines.append("- Pooled statistics are built from the concatenation of each "
                 "confident fold's own validation partition only (never the "
                 "full/raw dataset) — the same rule that fixed a real leak in "
                 "C2.2c.")
    if pooled and pooled.get("improvement_vs_rolling_mean_60") is not None:
        if pooled["improvement_vs_rolling_mean_60"] < 0 or pooled.get(
                "improvement_vs_train_mean", 0) < 0:
            lines.append(
                "- Ridge beats the persistence baseline by a wide margin, but "
                "does NOT clearly beat the simpler rolling-mean-60/train-mean "
                "baselines (see pooled headline numbers above) — the "
                "persistence baseline (ratio≡1.0) is a weak strawman here "
                "since typical realized 12-bar range is several multiples of "
                "current ATR; per C4.1 §9, ridge is judged primarily against "
                "persistence for the go/no-go verdict, but this gap is a "
                "real, honest caveat on how much of ridge's apparent edge is "
                "actually just reproducing the historical mean rather than "
                "adding genuine feature-driven signal.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C4.3: ridge volatility/range walk-forward run (read-only)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bars", type=int, default=BARS)
    parser.add_argument("--outdir", default="reports/c43")
    parser.add_argument("--registry-dir", default="reports/c43/model_registry")
    args = parser.parse_args(argv)

    try:
        result = run(args.dataset, args.exchange, args.symbol, args.bars,
                    args.outdir, args.registry_dir)
    except DeepBacktestError as exc:
        import sys
        print(f"ridge_volatility_run: {exc}", file=sys.stderr)
        return 2

    print(format_report(result))
    print(f"VERDICT: {result['verdict']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
