"""C4.2 §4: Training Engine — reusable walk-forward training loop.

Contains NO model-specific code: it only calls the
tools.forecast_platform.model_interface.ForecastModel contract and reuses
tools.deep_backtest.Fold/WFConfig (via tools.forecast_platform.
dataset_builder.partition_rows, unchanged) for fold geometry. Any future
model family (direction/magnitude/regime, ridge/LightGBM/other) plugs in
through `model_factory` without this module changing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from tools.deep_backtest import Fold
from tools.forecast_platform.dataset_builder import (Row, is_confident_fold,
                                                       partition_rows)
from tools.forecast_platform.model_interface import ForecastModel

ModelFactory = Callable[[], ForecastModel]


@dataclass
class FoldResult:
    fold_index: int
    confident: bool
    n_train: int
    n_val: int
    model: ForecastModel
    val_metrics: dict[str, Any]


@dataclass
class TrainingResult:
    fold_results: list[FoldResult] = field(default_factory=list)

    @property
    def confident_fold_results(self) -> list[FoldResult]:
        return [r for r in self.fold_results if r.confident]


def run_training(df: pd.DataFrame, folds: list[Fold],
                 model_factory: ModelFactory, feature_names: list[str],
                 horizon_bars: int, *, label_col: str = "label",
                 min_val_rows: int = 300) -> TrainingResult:
    """Train one independent model instance per fold — never reused across
    folds, so a fold's model can only ever have seen that fold's own train
    region (no cross-fold state leak)."""
    rows: list[Row] = df.to_dict("records")
    result = TrainingResult()
    for fold in folds:
        train_rows, val_rows = partition_rows(rows, fold, horizon_bars)
        confident = is_confident_fold(val_rows, min_val_rows)
        model = model_factory()
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)
        model.prepare(train_df, feature_names, label_col)
        model.train()
        val_metrics = model.evaluate(val_df, label_col) if len(val_df) else {}
        result.fold_results.append(FoldResult(
            fold_index=fold.index, confident=confident,
            n_train=len(train_rows), n_val=len(val_rows), model=model,
            val_metrics=val_metrics))
    return result


def pooled_validation_rows(df: pd.DataFrame, folds: list[Fold],
                          horizon_bars: int, *, confident_only: bool = True,
                          min_val_rows: int = 300) -> pd.DataFrame:
    """Validation-only pooling across every (confident) fold — the exact
    rule fixed as a real bug in C2.2c (reports/c23/DECISION.md): any pooled
    statistic must be built from the concatenation of each fold's OWN
    validation partition, never from the raw full dataset."""
    rows: list[Row] = df.to_dict("records")
    pooled: list[Row] = []
    for fold in folds:
        _, val_rows = partition_rows(rows, fold, horizon_bars)
        if confident_only and not is_confident_fold(val_rows, min_val_rows):
            continue
        pooled.extend(val_rows)
    return pd.DataFrame(pooled)
