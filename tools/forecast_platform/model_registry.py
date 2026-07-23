"""C4.2 §7: Model Registry — immutable trained-model metadata store.

Offline, filesystem-backed. Nothing is ever overwritten: register() raises
if a model_id already has an entry. This stores METADATA only (C4.2's
ModelMetadata contract); actual model artifact bytes (ridge coefficients,
a LightGBM booster) are persisted via the ForecastModel.save()/load()
contract, not here — the two are deliberately separate so a metadata
lookup never needs to deserialize a full model.
"""
from __future__ import annotations

import json
import os

from tools.forecast_platform.contracts import ModelMetadata


class ModelAlreadyRegisteredError(Exception):
    """A model_id already has an immutable registry entry."""


class ModelRegistry:
    def __init__(self, registry_dir: str) -> None:
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    def _path(self, model_id: str) -> str:
        return os.path.join(self.registry_dir, f"{model_id}.json")

    def register(self, metadata: ModelMetadata) -> str:
        path = self._path(metadata.model_id)
        if os.path.exists(path):
            raise ModelAlreadyRegisteredError(
                f"model_id {metadata.model_id!r} is already registered at "
                f"{path!r} — model versions are immutable, use a new model_id")
        with open(path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        return path

    def get(self, model_id: str) -> ModelMetadata:
        with open(self._path(model_id)) as f:
            payload = json.load(f)
        return ModelMetadata.from_dict(payload)

    def list_ids(self) -> list[str]:
        if not os.path.isdir(self.registry_dir):
            return []
        return sorted(fname[:-5] for fname in os.listdir(self.registry_dir)
                     if fname.endswith(".json"))
