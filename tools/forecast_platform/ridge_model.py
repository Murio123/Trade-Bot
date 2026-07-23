"""C4.3: Ridge ForecastModel — the frozen baseline
(reports/c41/volatility_model_frozen_spec.md §6/§7).

Implements only the tools.forecast_platform.model_interface.ForecastModel
contract; contains no dataset/fold logic of its own (reuses
tools.forecast_platform.dataset_builder/training_engine unchanged). Every
constant below is copied verbatim from the frozen C4.1 spec and must not
change after seeing any result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from tools.forecast_platform.contracts import ModelMetadata
from tools.forecast_platform.model_interface import ForecastModel

# --- Frozen (reports/c41/volatility_model_frozen_spec.md §6/§7) ---
ALPHA_CANDIDATES: tuple[float, ...] = (0.1, 1.0, 10.0)
INNER_SPLIT_FRACTION = 0.2  # last 20% of TRAIN region, time-ordered, train-only
FIT_INTERCEPT = True
SEED = 42

# Frozen category lists (C4.1 §6) — one-hot columns are fixed regardless of
# what any single fold's train region happens to contain; an unseen
# category at inference time maps to an all-zero row, never a new column.
CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "htf_bias": ("bullish", "bearish", "neutral"),
    "regime_current": ("trend_up", "trend_down", "range", "high_volatility"),
}


class RidgeTrainingError(Exception):
    """A ridge fold failed closed (singular normal-equations matrix, or
    train()/predict() called out of order)."""


def _one_hot(df: pd.DataFrame, column: str, levels: tuple[str, ...]
            ) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{column}__{level}": (df[column] == level).astype(float)
         for level in levels}, index=df.index)


def _design_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Deterministic numeric design matrix: frozen-category one-hot for
    categoricals, 0/1 for booleans, float pass-through otherwise. Column
    order follows `feature_names` (categoricals expand in place)."""
    parts = []
    for name in feature_names:
        col = df[name]
        if name in CATEGORICAL_LEVELS:
            parts.append(_one_hot(df, name, CATEGORICAL_LEVELS[name]))
        elif col.dtype == bool:
            parts.append(col.astype(float).to_frame(name))
        else:
            parts.append(pd.to_numeric(col, errors="coerce").to_frame(name))
    return pd.concat(parts, axis=1)


def _drop_missing(design: pd.DataFrame, label: pd.Series
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """C4.1 §6: rows with any missing required feature are EXCLUDED, never
    imputed."""
    mask = design.notna().all(axis=1) & label.notna()
    return design[mask], label[mask]


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float
              ) -> tuple[np.ndarray, float]:
    """Closed-form ridge via the normal equations, solved with
    np.linalg.solve (LU-based — deterministic, never a stochastic
    SAG/SGD solver, per C4.1 §7). The intercept column is NOT penalized
    (its row/column of the penalty matrix is zero)."""
    n, p = x.shape
    x_design = np.hstack([np.ones((n, 1)), x])
    penalty = np.eye(p + 1) * alpha
    penalty[0, 0] = 0.0
    a = x_design.T @ x_design + penalty
    b = x_design.T @ y
    try:
        coef = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise RidgeTrainingError(
            f"ridge normal-equations matrix is singular for alpha={alpha}: "
            f"{exc}") from exc
    return coef[1:], float(coef[0])


def _ridge_predict(x: np.ndarray, weights: np.ndarray, intercept: float
                   ) -> np.ndarray:
    return x @ weights + intercept


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@dataclass(frozen=True)
class RidgeIdentity:
    """Identity fields the caller supplies — everything a bare model
    instance can't know on its own (C4.1 §12 model-identity requirement)."""
    model_id: str
    code_commit: str
    dataset_version: str
    feature_version: str
    label_version: str


class RidgeForecastModel(ForecastModel):
    """Frozen ridge baseline: StandardScaler (train-fit only, §6) + one-hot
    (frozen category list, §6) + closed-form ridge with 3 frozen alpha
    candidates, selected via a nested, TRAIN-ONLY, time-ordered 80/20 split
    (§7). One fresh instance per walk-forward fold — see
    tools.forecast_platform.training_engine.run_training."""

    def __init__(self, identity: RidgeIdentity, seed: int = SEED):
        self.identity = identity
        self.seed = seed
        self._label_col = "label"
        self._feature_names: list[str] | None = None
        self._design_columns: list[str] | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._intercept: float | None = None
        self._chosen_alpha: float | None = None
        self._train_idx_range: tuple[int, int] | None = None
        self._inner_val_idx_range: tuple[int, int] | None = None
        self._train_df: pd.DataFrame | None = None

    # -- ForecastModel contract --------------------------------------------

    def prepare(self, train_df: pd.DataFrame, feature_names: list[str],
               label_col: str = "label") -> None:
        self._label_col = label_col
        self._feature_names = list(feature_names)
        ordered = train_df.sort_values("idx").reset_index(drop=True)
        self._train_df = ordered
        if len(ordered):
            self._train_idx_range = (int(ordered["idx"].min()),
                                     int(ordered["idx"].max()))
        else:
            self._train_idx_range = None

    def train(self) -> None:
        df = self._train_df
        if df is None or len(df) == 0:
            raise RidgeTrainingError(
                "prepare() must be called with a non-empty train_df before "
                "train()")
        design = _design_matrix(df, self._feature_names)
        label = df[self._label_col]
        design, label = _drop_missing(design, label)
        if len(design) == 0:
            raise RidgeTrainingError(
                "every train row was excluded by the missing-value policy "
                "(§6) — cannot fit ridge on zero rows")
        self._design_columns = list(design.columns)

        self._chosen_alpha, self._inner_val_idx_range = self._select_alpha(
            df, design, label)

        mean = design.mean(axis=0).to_numpy()
        std = design.std(axis=0, ddof=0).to_numpy()
        std_safe = np.where(std == 0, 1.0, std)
        self._scaler_mean, self._scaler_std = mean, std_safe
        x_full = (design.to_numpy(dtype=float) - mean) / std_safe
        self._weights, self._intercept = _ridge_fit(
            x_full, label.to_numpy(dtype=float), self._chosen_alpha)

    def _select_alpha(self, df: pd.DataFrame, design: pd.DataFrame,
                      label: pd.Series
                      ) -> tuple[float, tuple[int, int] | None]:
        """Nested, TRAIN-ONLY alpha selection (§7): last 20% of the train
        region (time-ordered, since `df` is already idx-sorted) is an
        inner validation split; the outer fold's actual validation region
        is never touched here."""
        split = int(round(len(design) * (1 - INNER_SPLIT_FRACTION)))
        split = max(1, min(split, len(design) - 1)) if len(design) > 1 else 0
        if split <= 0 or split >= len(design):
            # Too little data for a genuine inner split — use the first
            # frozen candidate rather than silently skipping selection.
            return ALPHA_CANDIDATES[0], None

        inner_train_x = design.iloc[:split].to_numpy(dtype=float)
        inner_val_x = design.iloc[split:].to_numpy(dtype=float)
        inner_train_y = label.iloc[:split].to_numpy(dtype=float)
        inner_val_y = label.iloc[split:].to_numpy(dtype=float)

        mean = inner_train_x.mean(axis=0)
        std = inner_train_x.std(axis=0)
        std_safe = np.where(std == 0, 1.0, std)
        x_train_scaled = (inner_train_x - mean) / std_safe
        x_val_scaled = (inner_val_x - mean) / std_safe

        best_alpha, best_mae = ALPHA_CANDIDATES[0], None
        for alpha in ALPHA_CANDIDATES:
            w, b = _ridge_fit(x_train_scaled, inner_train_y, alpha)
            preds = _ridge_predict(x_val_scaled, w, b)
            m = _mae(inner_val_y, preds)
            if best_mae is None or m < best_mae:
                best_mae, best_alpha = m, alpha

        inner_val_idx_range = (int(df.iloc[split]["idx"]),
                               int(df.iloc[-1]["idx"]))
        return best_alpha, inner_val_idx_range

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._weights is None:
            raise RidgeTrainingError("train() must be called before predict()")
        design = _design_matrix(df, self._feature_names)
        design = design.reindex(columns=self._design_columns)
        x = design.to_numpy(dtype=float)
        x = (x - self._scaler_mean) / self._scaler_std
        return _ridge_predict(x, self._weights, self._intercept)

    def evaluate(self, df: pd.DataFrame, label_col: str = "label"
                ) -> dict[str, Any]:
        if len(df) == 0:
            return {"n": 0, "mae": None}
        preds = self.predict(df)
        y_true = df[label_col].to_numpy(dtype=float)
        valid = ~np.isnan(preds) & ~np.isnan(y_true)
        return {"n": int(valid.sum()),
               "mae": _mae(y_true[valid], preds[valid]) if valid.any() else None}

    def save(self, path: str) -> None:
        payload = {
            "feature_names": self._feature_names,
            "design_columns": self._design_columns,
            "scaler_mean": self._scaler_mean.tolist(),
            "scaler_std": self._scaler_std.tolist(),
            "weights": self._weights.tolist(),
            "intercept": self._intercept,
            "chosen_alpha": self._chosen_alpha,
            "train_idx_range": list(self._train_idx_range) if self._train_idx_range else None,
            "inner_val_idx_range": list(self._inner_val_idx_range) if self._inner_val_idx_range else None,
            "seed": self.seed,
            "label_col": self._label_col,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        self._feature_names = payload["feature_names"]
        self._design_columns = payload["design_columns"]
        self._scaler_mean = np.array(payload["scaler_mean"])
        self._scaler_std = np.array(payload["scaler_std"])
        self._weights = np.array(payload["weights"])
        self._intercept = payload["intercept"]
        self._chosen_alpha = payload["chosen_alpha"]
        self._train_idx_range = (tuple(payload["train_idx_range"])
                                 if payload["train_idx_range"] else None)
        self._inner_val_idx_range = (tuple(payload["inner_val_idx_range"])
                                     if payload["inner_val_idx_range"] else None)
        self.seed = payload["seed"]
        self._label_col = payload["label_col"]

    def metadata(self) -> ModelMetadata:
        fingerprint = json.dumps({
            "chosen_alpha": self._chosen_alpha,
            "design_columns": self._design_columns,
            "weights": self._weights.tolist() if self._weights is not None else None,
            "intercept": self._intercept,
        }, sort_keys=True, default=str)
        evaluation_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return ModelMetadata(
            model_id=self.identity.model_id,
            code_commit=self.identity.code_commit,
            dataset_version=self.identity.dataset_version,
            feature_version=self.identity.feature_version,
            label_version=self.identity.label_version,
            seed=self.seed,
            hyperparameters={"alpha_candidates": list(ALPHA_CANDIDATES),
                            "chosen_alpha": self._chosen_alpha,
                            "fit_intercept": FIT_INTERCEPT,
                            "inner_split_fraction": INNER_SPLIT_FRACTION},
            training_window=self._train_idx_range or (0, 0),
            calibration_window=self._inner_val_idx_range,
            evaluation_hash=evaluation_hash,
            created_at_utc=datetime.now(timezone.utc).isoformat())
