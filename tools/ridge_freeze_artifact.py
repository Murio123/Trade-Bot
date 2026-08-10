"""C4.4a: freeze ONE Ridge artifact, plus the reference distribution.

Why this exists. C4.3 trained a model per walk-forward fold, read the metrics
off them and threw the weights away — RidgeForecastModel.save() was written
but never called. So there is no artifact to load, and the three fold models
are not candidates: each saw only two thirds of history and they disagree on
alpha by a factor of 100 (10.0, 10.0, 0.1).

What this does NOT do. It makes no new modelling decision. Same 19 frozen
features, same frozen label, same frozen alpha grid, same alpha-selection
procedure, same data source, trained on the training region that ends where
the sealed holdout begins. Nothing is tuned to any observed result. It is a
one-time fixation of an already-specified model, not new research, and the
artifact is hashed so a later run can prove it did not drift.

What the artifact predicts. The frozen label is
log(clip(range(t+1..t+12)/atr[t], 0, 20) + 1e-6) — the SIZE of the coming
range, with no direction in it: a 3% rise and a 3% fall score identically.
C4.3c rejected this model as a point predictor of that size. Only its
RANKING survives (C4.3d/C4.3e), which is why the shipped product ranks and
the artifact rides along as a recorded shadow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from tools.deep_backtest import DeepBacktestError, get_profile
from tools.forecast_platform.contracts import ModelMetadata, spec_hash
from tools.forecast_platform.dataset_builder import (
    VOLATILITY_HORIZON_BARS, VOLATILITY_LABEL_VERSION, build_dataset,
    build_wf_config)
from tools.forecast_platform.feature_store import (FEATURE_VERSION,
                                                    model_input_names)
from tools.forecast_platform.model_registry import ModelRegistry
from tools.forecast_platform.ridge_model import (ALPHA_CANDIDATES,
                                                  RidgeForecastModel,
                                                  RidgeIdentity)
from tools.ridge_volatility_run import BARS, PROFILE_NAME, _git_commit
from volatility import percentile as pct
from volatility.ranker import HORIZON_BARS, TRAILING_WINDOW

STAGE = "C4.4a"
ARTIFACT_ID = "c44_ridge_frozen_v1"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def training_region(span_lo: int, holdout_lo: int, horizon_bars: int
                    ) -> tuple[int, int]:
    """[lo, hi) usable for training.

    The upper bound is pulled back by the horizon: a bar within `horizon_bars`
    of the holdout has a label built from bars inside it, which would leak the
    sealed region into the weights.
    """
    return span_lo, holdout_lo - horizon_bars


def run(dataset: str, exchange: str, symbol: str, bars: int = BARS,
        outdir: str = "reports/c44") -> dict[str, Any]:
    commit = _git_commit()
    profile = get_profile(PROFILE_NAME)

    build = build_dataset(dataset, exchange, symbol, PROFILE_NAME, bars)
    df = build.frame
    wf = build_wf_config(profile, bars)
    holdout_lo = build.span_hi - wf.holdout_bars
    lo, hi = training_region(build.span_lo, holdout_lo, VOLATILITY_HORIZON_BARS)

    train_df = df[(df["idx"] >= lo) & (df["idx"] < hi)].copy()
    if train_df.empty:
        raise DeepBacktestError("training region is empty")
    if int(train_df["idx"].max()) >= holdout_lo - VOLATILITY_HORIZON_BARS:
        raise DeepBacktestError("training region reaches into the purge band")

    feature_names = model_input_names()
    model = RidgeForecastModel(RidgeIdentity(
        model_id=ARTIFACT_ID, code_commit=commit,
        dataset_version=build.manifest["dataset_version"],
        feature_version=FEATURE_VERSION,
        label_version=VOLATILITY_LABEL_VERSION))
    model.prepare(train_df, feature_names, "label")
    model.train()

    os.makedirs(outdir, exist_ok=True)
    artifact_path = os.path.join(outdir, f"{ARTIFACT_ID}.json")
    model.save(artifact_path)
    artifact_sha = _sha256_file(artifact_path)

    # --- reference distributions, both from the TRAINING region only --------
    ridge_scores = model.predict(train_df)
    labels = train_df["label"].to_numpy(dtype=float)
    idx = train_df["idx"].to_numpy(dtype=np.int64)

    # The shipped ranker's own reference: inverted trailing mean, computed at
    # every training bar that has enough observable history.
    from tools.forecast_platform.evaluation_engine import \
        rolling_mean_series_baseline
    trailing = rolling_mean_series_baseline(
        idx, labels, idx, window=TRAILING_WINDOW,
        horizon_bars=HORIZON_BARS, max_source_idx=holdout_lo)
    ok = ~np.isnan(trailing)
    ranker_ref = pct.build_reference(
        -trailing[ok], {"stage": STAGE, "kind": "inverted_trailing_mean",
                        "n": int(ok.sum()), "window": TRAILING_WINDOW,
                        "train_idx_range": [lo, hi]})
    ridge_ref = pct.build_reference(
        ridge_scores, {"stage": STAGE, "kind": "ridge_frozen",
                       "n": int(len(ridge_scores)),
                       "train_idx_range": [lo, hi]})
    pct.save_reference(ranker_ref, os.path.join(outdir, "reference_ranker.json"))
    pct.save_reference(ridge_ref, os.path.join(outdir, "reference_ridge.json"))

    # --- honest "scenarios": realized range per predicted rank bucket -------
    scenarios = realized_range_by_category(ranker_ref, -trailing[ok], labels[ok])

    registry = ModelRegistry(os.path.join(outdir, "model_registry"))
    metadata = ModelMetadata(
        model_id=ARTIFACT_ID, code_commit=commit,
        dataset_version=build.manifest["dataset_version"],
        feature_version=FEATURE_VERSION,
        label_version=VOLATILITY_LABEL_VERSION, seed=model.seed,
        hyperparameters={"alpha_candidates": list(ALPHA_CANDIDATES),
                         "chosen_alpha": model._chosen_alpha,
                         "train_idx_range": [lo, hi]},
        training_window=(lo, hi), calibration_window=None,
        evaluation_hash=artifact_sha,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        registry.register(metadata)
        registry_status = "registered"
    except Exception as exc:  # already registered on a re-run
        registry_status = f"not_registered: {exc}"

    result = {
        "stage": STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit, "exchange": exchange, "symbol": symbol,
        "artifact_path": artifact_path, "artifact_sha256": artifact_sha,
        "chosen_alpha": model._chosen_alpha,
        "n_train_rows": int(len(train_df)),
        "train_idx_range": [lo, hi],
        "holdout_idx_lo": holdout_lo,
        "dataset_version": build.manifest["dataset_version"],
        "feature_version": FEATURE_VERSION,
        "label_version": VOLATILITY_LABEL_VERSION,
        "registry_status": registry_status,
        "scenarios_by_category": scenarios,
        "spec_hash": spec_hash({"features": feature_names,
                                "alpha_candidates": list(ALPHA_CANDIDATES),
                                "train_idx_range": [lo, hi]}),
        "note": ("Frozen fixation of an already-specified model. No new "
                 "modelling decision, no tuning to any observed result. The "
                 "label carries range SIZE only — there is no direction in "
                 "it, so this artifact cannot express an up/down probability."),
    }
    with open(os.path.join(outdir, "freeze_report.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def realized_range_by_category(ref: pct.ReferenceDistribution,
                               scores: np.ndarray, labels: np.ndarray
                               ) -> dict[str, Any]:
    """What actually happened, historically, per predicted category.

    This is the only honest form of "scenario" available from this model: an
    empirical frequency, not a forecast. Ranges are reported back in raw
    ratio-of-ATR units, which is what a reader can picture.
    """
    raw = np.exp(labels) - 1e-6  # undo the frozen log transform
    out: dict[str, Any] = {}
    for name in pct.CATEGORIES:
        mask = np.array([pct.categorize(ref.percentile_of(s)) == name
                         for s in scores])
        if not mask.any():
            out[name] = {"n": 0}
            continue
        sel = raw[mask]
        q10, q50, q90 = np.quantile(sel, [0.10, 0.50, 0.90])
        out[name] = {
            "n": int(mask.sum()),
            "realized_atr_ratio_p10": float(q10),
            "realized_atr_ratio_median": float(q50),
            "realized_atr_ratio_p90": float(q90),
        }
    return out


def format_report(r: dict[str, Any]) -> str:
    lines = [
        f"Frozen Ridge artifact ({r['stage']}) — {r['exchange']} {r['symbol']}",
        f"  commit        : {r['code_commit']}",
        f"  artifact      : {r['artifact_path']}",
        f"  sha256        : {r['artifact_sha256']}",
        f"  chosen_alpha  : {r['chosen_alpha']}",
        f"  train rows    : {r['n_train_rows']} over idx {r['train_idx_range']}",
        f"  holdout starts: {r['holdout_idx_lo']} (never touched)",
        f"  registry      : {r['registry_status']}",
        "",
        "historical realized 48h range per predicted category "
        "(multiples of ATR at the prediction bar):",
    ]
    for name, s in r["scenarios_by_category"].items():
        if not s.get("n"):
            lines.append(f"  {name:<9} no rows")
            continue
        lines.append(f"  {name:<9} n={s['n']:<5} "
                     f"p10={s['realized_atr_ratio_p10']:.2f}  "
                     f"median={s['realized_atr_ratio_median']:.2f}  "
                     f"p90={s['realized_atr_ratio_p90']:.2f}")
    lines.append("")
    lines.append(r["note"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C4.4a: freeze the Ridge artifact")
    p.add_argument("--dataset", required=True)
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--bars", type=int, default=BARS)
    p.add_argument("--outdir", default="reports/c44")
    args = p.parse_args(argv)
    try:
        result = run(args.dataset, args.exchange, args.symbol, args.bars,
                     args.outdir)
    except DeepBacktestError as exc:
        import sys
        print(f"ridge_freeze_artifact: {exc}", file=sys.stderr)
        return 2
    print(format_report(result))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
