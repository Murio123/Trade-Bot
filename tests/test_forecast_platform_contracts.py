"""C4.2: tests for tools/forecast_platform/contracts.py."""
from __future__ import annotations

from tools.forecast_platform.contracts import (ForecastRecord, ModelMetadata,
                                                spec_hash)


def test_spec_hash_deterministic_regardless_of_key_order():
    a = {"b": 2, "a": 1, "c": [1, 2, 3]}
    b = {"a": 1, "c": [1, 2, 3], "b": 2}
    assert spec_hash(a) == spec_hash(b)


def test_spec_hash_differs_on_value_change():
    a = {"alpha": 0.1}
    b = {"alpha": 0.2}
    assert spec_hash(a) != spec_hash(b)


def test_spec_hash_is_a_hex_sha256_digest():
    h = spec_hash({"x": 1})
    assert len(h) == 64
    int(h, 16)  # raises ValueError if not valid hex


def test_forecast_record_round_trips_through_dict():
    rec = ForecastRecord(
        timestamp="2026-01-01T00:00:00+00:00", model_id="vol_ridge_v1",
        prediction=0.42, confidence=None, prediction_interval=(0.1, 0.9),
        calibrated=False, dataset_version="ds1", feature_version="c41_v1",
        commit_hash="abc123")
    payload = rec.to_dict()
    restored = ForecastRecord.from_dict(payload)
    assert restored == rec
    assert restored.prediction_interval == (0.1, 0.9)


def test_forecast_record_allows_none_confidence_and_interval():
    rec = ForecastRecord(
        timestamp="2026-01-01T00:00:00+00:00", model_id="vol_ridge_v1",
        prediction=0.1, confidence=None, prediction_interval=None,
        calibrated=False, dataset_version="ds1", feature_version="c41_v1",
        commit_hash="abc123")
    assert ForecastRecord.from_dict(rec.to_dict()) == rec


def test_model_metadata_round_trips_through_dict():
    meta = ModelMetadata(
        model_id="vol_ridge_v1", code_commit="deadbeef",
        dataset_version="ds1", feature_version="c41_v1",
        label_version="c41_volatility_range_v1", seed=42,
        hyperparameters={"alpha": 1.0}, training_window=(300, 6000),
        calibration_window=(5500, 6000), evaluation_hash="eval1",
        created_at_utc="2026-01-01T00:00:00+00:00")
    payload = meta.to_dict()
    restored = ModelMetadata.from_dict(payload)
    assert restored == meta
    assert restored.training_window == (300, 6000)
    assert restored.calibration_window == (5500, 6000)


def test_model_metadata_allows_none_calibration_window():
    meta = ModelMetadata(
        model_id="vol_ridge_v1", code_commit="deadbeef",
        dataset_version="ds1", feature_version="c41_v1",
        label_version="c41_volatility_range_v1", seed=42,
        hyperparameters={}, training_window=(300, 6000),
        calibration_window=None, evaluation_hash="eval1",
        created_at_utc="2026-01-01T00:00:00+00:00")
    assert ModelMetadata.from_dict(meta.to_dict()) == meta
