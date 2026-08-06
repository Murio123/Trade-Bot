"""C4.2 §6: Evaluation Engine — reusable metrics, no model-specific code.

Computes MAE/RMSE/Spearman, fold/year summaries, regression-style
calibration-bucket analysis, and baseline comparisons (persistence/
rolling-mean/train-mean/random — reports/c41/volatility_model_frozen_spec.md
§9/§10). Every function operates on plain arrays/Series/DataFrames —
nothing here knows what a ridge or LightGBM model is, so future
direction/magnitude/regime models reuse it unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tools.regime_gate_confirmation import percentile_rank


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman_corr(y_true, y_pred) -> float | None:
    """Rank-based correlation via pandas' own .rank() + numpy corrcoef —
    no scipy dependency needed for this."""
    s_true = pd.Series(y_true, dtype=float)
    s_pred = pd.Series(y_pred, dtype=float)
    if len(s_true) < 2 or s_true.nunique() < 2 or s_pred.nunique() < 2:
        return None
    corr = np.corrcoef(s_true.rank(), s_pred.rank())[0, 1]
    return float(corr) if not math.isnan(corr) else None


def fold_summary(fold_index: int, y_true, y_pred) -> dict[str, Any]:
    return {"fold": fold_index, "n": len(y_true), "mae": mae(y_true, y_pred),
           "rmse": rmse(y_true, y_pred),
           "spearman": spearman_corr(y_true, y_pred)}


def year_summary(df: pd.DataFrame, pred_col: str, label_col: str = "label",
                 ts_col: str = "prediction_timestamp_utc"
                 ) -> dict[str, dict[str, Any]]:
    if len(df) == 0:
        return {}
    years = pd.to_datetime(df[ts_col]).dt.year
    out: dict[str, dict[str, Any]] = {}
    for year, group in df.groupby(years):
        out[str(int(year))] = {
            "n": len(group), "mae": mae(group[label_col], group[pred_col]),
            "rmse": rmse(group[label_col], group[pred_col]),
        }
    return out


def bucket_calibration(y_true, y_pred, n_buckets: int = 5
                       ) -> list[dict[str, Any]]:
    """Regression-style bucket/reliability analysis (C4.1 §10): bucket rows
    by predicted value into quantiles, report mean actual per bucket. This
    function is edge-agnostic — it computes bucket edges from whatever
    `y_pred` it is given. Callers evaluating a real fold must pass
    validation-region predictions only, with bucket EDGES conceptually
    anchored on the fold's train region per the frozen spec's isolation
    requirement (enforced by the caller, not by this pure function)."""
    frame = pd.DataFrame({"y_true": np.asarray(y_true, dtype=float),
                         "y_pred": np.asarray(y_pred, dtype=float)})
    if len(frame) == 0:
        return []
    try:
        frame["bucket"] = pd.qcut(frame["y_pred"], n_buckets, duplicates="drop")
    except ValueError:
        return []
    out = []
    for bucket, group in frame.groupby("bucket", observed=True):
        out.append({"bucket": str(bucket), "n": len(group),
                   "mean_predicted": float(group["y_pred"].mean()),
                   "mean_actual": float(group["y_true"].mean()),
                   "residual": float(group["y_true"].mean() - group["y_pred"].mean())})
    return out


# ---------------------------------------------------------------------------
# Baselines (reports/c41/volatility_model_frozen_spec.md §9).
# ---------------------------------------------------------------------------

def persistence_baseline_value(eps: float = 1e-6) -> float:
    """predicted_ratio == 1.0 in raw-ratio space -> log(1+eps) in the
    log-transformed label space this model trains in (§2/§9.A)."""
    return math.log(1.0 + eps)


def rolling_mean_baseline(train_labels: pd.Series, window: int = 60) -> float:
    """Trailing-window mean of TRAIN-region labels only, refit per fold
    (§9.B) — a single scalar used as the constant prediction for that
    fold's validation region."""
    if len(train_labels) == 0:
        return float("nan")
    tail = train_labels.iloc[-window:] if len(train_labels) > window else train_labels
    return float(tail.mean())


def constant_mean_baseline(train_labels: pd.Series) -> float:
    """Fold-constant mean over the ENTIRE train region (§9.C)."""
    return float(train_labels.mean()) if len(train_labels) else float("nan")


def rolling_mean_series_baseline(source_idx, source_labels, query_idx, *,
                                 window: int = 60, horizon_bars: int = 12,
                                 max_source_idx: int | None = None
                                 ) -> np.ndarray:
    """Point-in-time trailing mean: ONE prediction per query row (C4.3d).

    `rolling_mean_baseline` above collapses to a single scalar per fold, so
    inside a fold it is constant and has no rank-order power by construction.
    Comparing a model's Spearman against it is therefore vacuous — any
    non-degenerate model "wins". This is the fair alternative: a trailing
    average an observer could actually have tracked bar to bar.

    Observability is the whole point. The label of bar u is computed from
    bars u+1..u+horizon_bars, so it is only fully known at bar u+horizon_bars.
    At query bar t the observer therefore knows label[u] exactly when
    u <= t - horizon_bars, and this function admits no later label into the
    window. Labels from the query row's own validation region are eligible
    when they satisfy that inequality — that is legitimate point-in-time
    knowledge, not leakage, and it is what makes the baseline time-varying.

    `max_source_idx` hard-excludes anything at or beyond it (used to keep the
    sealed holdout out even though fold geometry already places it out of
    reach). Returns NaN for a query row with no observable history yet.

    Both arrays must be ordered by idx; `source_idx` must be unique.
    """
    src_idx = np.asarray(source_idx, dtype=np.int64)
    src_lab = np.asarray(source_labels, dtype=float)
    q_idx = np.asarray(query_idx, dtype=np.int64)
    if src_idx.shape != src_lab.shape:
        raise ValueError("source_idx and source_labels must be the same length")
    if src_idx.size and np.any(np.diff(src_idx) <= 0):
        raise ValueError("source_idx must be strictly increasing and unique")
    if window <= 0:
        raise ValueError("window must be positive")

    if max_source_idx is not None:
        keep = src_idx < max_source_idx
        src_idx, src_lab = src_idx[keep], src_lab[keep]

    # Prefix sums make each query O(1) after one searchsorted, and keep the
    # result independent of how the queries are batched.
    csum = np.concatenate(([0.0], np.cumsum(src_lab)))
    # 'right' => count of source rows with idx <= (t - horizon_bars).
    cutoff = q_idx - horizon_bars
    end = np.searchsorted(src_idx, cutoff, side="right")
    start = np.maximum(end - window, 0)
    count = end - start
    out = np.full(q_idx.shape, np.nan, dtype=float)
    ok = count > 0
    out[ok] = (csum[end[ok]] - csum[start[ok]]) / count[ok]
    return out


def random_constant_baseline(y_true, train_labels, n_draws: int = 500,
                             seed: int = 42) -> dict[str, Any]:
    """Generic "random baseline": draw `n_draws` values from the TRAIN
    label distribution, score each as a constant prediction against the
    actual validation `y_true`, and report the distribution of resulting
    MAE plus (via tools.regime_gate_confirmation.percentile_rank, reused
    unchanged) where a given model's actual MAE falls within it."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=float)
    train_labels = np.asarray(train_labels, dtype=float)
    if len(y_true) == 0 or len(train_labels) == 0:
        return {"n_draws": n_draws, "mean": None, "std": None, "draws": []}
    draws = rng.choice(train_labels, size=n_draws, replace=True)
    maes = [mae(y_true, np.full(len(y_true), d)) for d in draws]
    return {"n_draws": n_draws, "mean": float(np.mean(maes)),
           "std": float(np.std(maes)), "draws": maes}


def rank_against_random(model_mae: float, random_maes: list[float]) -> float:
    """Where the model's MAE ranks against the random-baseline MAE
    distribution — lower is better for an error metric, so a LOW
    percentile_rank here (few random draws did at least as well) is the
    good outcome, the mirror image of how percentile_rank is read for an
    expectancy metric elsewhere in this project."""
    return percentile_rank(model_mae, random_maes)
