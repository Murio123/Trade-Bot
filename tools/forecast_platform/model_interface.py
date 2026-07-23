"""C4.2 §3: Forecast Model interface — the one contract every model
implements (ridge/LightGBM in C4.3; direction/magnitude/regime models
later, C4.0). The Training Engine calls these methods in exactly this
order, once per walk-forward fold: prepare() -> train() -> predict() ->
evaluate() -> metadata(). save()/load() are used by the Model Registry, not
the training loop.

No concrete model is implemented in this module — C4.2 is infrastructure
only (per its own stop condition); C4.3 implements the first concrete
subclasses (ridge, LightGBM) against this same interface, unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from tools.forecast_platform.contracts import ModelMetadata


class ForecastModel(ABC):
    @abstractmethod
    def prepare(self, train_df: pd.DataFrame, feature_names: list[str],
               label_col: str = "label") -> None:
        """Fit any preprocessing (scaler, frozen category list, ...) on
        TRAIN rows only. Must never read validation/holdout rows."""

    @abstractmethod
    def train(self) -> None:
        """Fit the model itself using whatever prepare() set up."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """One prediction per row of `df`, in the model's own label space
        (e.g. log-space for the frozen C4.1 volatility target)."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, label_col: str = "label"
                ) -> dict[str, Any]:
        """Metrics for `df` (typically one fold's validation rows). Models
        may add their own diagnostics here; the standard metrics (MAE/
        RMSE/Spearman/...) are computed by tools.forecast_platform.
        evaluation_engine, not duplicated per model."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model artifact to `path`."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a trained model artifact from `path`."""

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """This trained instance's immutable ModelMetadata (C4.2 §7)."""
