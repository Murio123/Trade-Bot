"""C4.2 §7/§8: Forecast Contract + Model Registry identity schema.

Offline-only, no network/DB/runtime import. Every forecasting model (C4.0's
four candidate families) emits predictions through ForecastRecord and
registers trained artifacts through ModelMetadata — the future Decision
Engine (not built in C4) is specified to consume ONLY ForecastRecord, never
a model-specific structure (research memo `reports/research/
quant_architecture_lessons.md` §6: forecast and decision are separate
layers).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


def spec_hash(payload: dict[str, Any]) -> str:
    """Deterministic sha256 hex digest of a JSON-serializable spec dict.

    Used for feature_version/label_version/dataset_version/evaluation_hash:
    two specs that hash equal are guaranteed byte-identical (sorted keys, no
    whitespace ambiguity); two that differ are guaranteed to differ. Never
    used for anything security-sensitive — this is a version fingerprint,
    not an authentication token.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ForecastRecord:
    """The one structure every forecasting model emits (C4.2 §8).

    Every field is required so the Decision Engine (future stage) never has
    to special-case a model — `confidence`/`prediction_interval` may be
    `None` for a model that doesn't produce them, but the field must always
    be present.
    """
    timestamp: str  # ISO-8601 UTC prediction timestamp
    model_id: str
    prediction: float
    confidence: float | None
    prediction_interval: tuple[float, float] | None
    calibrated: bool
    dataset_version: str
    feature_version: str
    commit_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForecastRecord":
        data = dict(payload)
        interval = data.get("prediction_interval")
        if interval is not None:
            data["prediction_interval"] = tuple(interval)
        return cls(**data)


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable identity of one trained model artifact (C4.2 §7 / C4.1 §12).

    Two artifacts are only comparable if every field here matches (or the
    difference is explicitly noted elsewhere) — see the model-identity
    requirement frozen in reports/c40/ml_data_feature_inventory.md §8 and
    reports/c41/volatility_model_frozen_spec.md §12.
    """
    model_id: str
    code_commit: str
    dataset_version: str
    feature_version: str
    label_version: str
    seed: int
    hyperparameters: dict[str, Any]
    training_window: tuple[int, int]
    calibration_window: tuple[int, int] | None
    evaluation_hash: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelMetadata":
        data = dict(payload)
        data["training_window"] = tuple(data["training_window"])
        if data.get("calibration_window") is not None:
            data["calibration_window"] = tuple(data["calibration_window"])
        return cls(**data)
