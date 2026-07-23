"""C4.2: tests for tools/forecast_platform/model_registry.py."""
from __future__ import annotations

import pytest

from tools.forecast_platform.contracts import ModelMetadata
from tools.forecast_platform.model_registry import (
    ModelAlreadyRegisteredError, ModelRegistry)


def _metadata(model_id="vol_ridge_v1"):
    return ModelMetadata(
        model_id=model_id, code_commit="deadbeef", dataset_version="ds1",
        feature_version="c41_v1", label_version="c41_volatility_range_v1",
        seed=42, hyperparameters={"alpha": 1.0}, training_window=(300, 6000),
        calibration_window=(5500, 6000), evaluation_hash="eval1",
        created_at_utc="2026-01-01T00:00:00+00:00")


def test_register_then_get_round_trips(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    meta = _metadata()
    registry.register(meta)
    assert registry.get(meta.model_id) == meta


def test_register_refuses_to_overwrite_an_existing_model_id(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    registry.register(_metadata("vol_ridge_v1"))
    with pytest.raises(ModelAlreadyRegisteredError):
        registry.register(_metadata("vol_ridge_v1"))


def test_list_ids_returns_every_registered_model_sorted(tmp_path):
    registry = ModelRegistry(str(tmp_path))
    registry.register(_metadata("vol_ridge_v2"))
    registry.register(_metadata("vol_ridge_v1"))
    assert registry.list_ids() == ["vol_ridge_v1", "vol_ridge_v2"]


def test_list_ids_empty_for_a_fresh_registry(tmp_path):
    registry = ModelRegistry(str(tmp_path / "fresh"))
    assert registry.list_ids() == []
