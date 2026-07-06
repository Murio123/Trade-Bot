"""Stage 15B2: failure-safe DB wrapper enrich_forecast_with_lifecycle.

Proves the wrapper reads the previous comparable forecast with the right keys,
returns the 7 persist-ready lifecycle fields, degrades to {} on ANY failure
(so insert_forecast still runs), never mutates the caller's record, and carries
no exchange-execution / order / API code.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from analyzer.setup_lifecycle import LIFECYCLE_FIELD_KEYS, NEW_SETUP
from forecast_lifecycle import enrich_forecast_with_lifecycle

UTC = timezone.utc
T0 = datetime(2026, 7, 5, 12, 0, tzinfo=UTC)


def _fc(**kw) -> dict:
    base = {
        "id": 2,
        "symbol": "BTCUSDT",
        "analysis_type": "SWING",
        "strategy_version": "v1",
        "context_version": "v1",
        "analysis_status": "WAIT",
        "candidate_direction": "long",
        "raw_confidence": 0.65,
        "long_score": 7.0,
        "short_score": 2.0,
        "signal_candle_close_time": T0 + timedelta(hours=4),
    }
    base.update(kw)
    return base


class _FakeDB:
    """Records the previous_forecast call and returns a canned previous row."""

    def __init__(self, previous=None, raises=False):
        self._previous = previous
        self._raises = raises
        self.calls: list[dict] = []

    async def previous_forecast(self, *, symbol, analysis_type, before):
        self.calls.append(
            {"symbol": symbol, "analysis_type": analysis_type, "before": before})
        if self._raises:
            raise RuntimeError("db boom")
        return self._previous


def _run(coro):
    return asyncio.run(coro)


def test_wrapper_calls_previous_forecast_with_expected_keys():
    db = _FakeDB(previous=None)
    fc = _fc()
    _run(enrich_forecast_with_lifecycle(db, fc))
    assert db.calls == [{
        "symbol": "BTCUSDT",
        "analysis_type": "SWING",
        "before": fc["signal_candle_close_time"],
    }]


def test_previous_found_returns_seven_fields():
    prev = _fc(id=41, long_score=5.0, signal_candle_close_time=T0)
    out = _run(enrich_forecast_with_lifecycle(_FakeDB(previous=prev), _fc()))
    assert set(out) == set(LIFECYCLE_FIELD_KEYS)
    assert out["previous_forecast_id"] == 41
    assert out["setup_lifecycle_comparable"] is True
    assert out["setup_score_delta"] == 2.0  # 7.0 - 5.0 (long side)


def test_previous_none_returns_new_setup_fields():
    out = _run(enrich_forecast_with_lifecycle(_FakeDB(previous=None), _fc()))
    assert out["setup_lifecycle_status"] == NEW_SETUP
    assert out["setup_lifecycle_comparable"] is False
    assert out["previous_forecast_id"] is None
    assert set(out) == set(LIFECYCLE_FIELD_KEYS)


def test_previous_forecast_raises_returns_empty():
    out = _run(enrich_forecast_with_lifecycle(_FakeDB(raises=True), _fc()))
    assert out == {}


def test_classification_raises_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise ValueError("classify boom")

    monkeypatch.setattr("forecast_lifecycle.build_lifecycle_fields", _boom)
    out = _run(enrich_forecast_with_lifecycle(_FakeDB(previous=_fc()), _fc()))
    assert out == {}


def test_wrapper_does_not_mutate_fc():
    fc = _fc()
    fc_copy = dict(fc)
    _run(enrich_forecast_with_lifecycle(_FakeDB(previous=_fc(id=9)), fc))
    assert fc == fc_copy


def test_no_exchange_execution_tokens_in_module():
    import forecast_lifecycle
    with open(forecast_lifecycle.__file__, encoding="utf-8") as fh:
        src = fh.read()
    forbidden = ("create_order", "market_order", "limit_order",
                 "api_key", "api_secret", "exchange.create")
    hits = [tok for tok in forbidden if tok in src]
    assert not hits, f"forecast_lifecycle carries execution tokens {hits}"
