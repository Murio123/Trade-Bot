"""Stage 9: analytics metadata on the forecast record (additive, nullable).

Доказывает: build_forecast_record проставляет market_regime/volatility_regime/
strategy_version/context_version из уже существующих result/ctx/config, а при
отсутствии источника пишет None (не падает). Поля — чисто ledger/analytics;
торговые решения не затрагиваются.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import config
from database import Database
from signal_engine.forecast_record import build_forecast_record
from signal_engine.profiles import get_profile

UTC = timezone.utc


def _base(**extra):
    return {
        "status": "blocked", "blocked_at": "htf_filter",
        "signal_candle_close_time": datetime(2026, 7, 1, 4, tzinfo=UTC),
        "decision_time": datetime(2026, 7, 1, 4, 1, tzinfo=UTC),
        **extra,
    }


def test_market_regime_written_from_result():
    fc = build_forecast_record(
        _base(market_regime="trend_up"),
        {"volatility": {"regime": "expansion"}},
        get_profile("swing"))
    assert fc["market_regime"] == "trend_up"


def test_volatility_regime_written_from_ctx():
    fc = build_forecast_record(
        _base(), {"volatility": {"regime": "compression"}}, get_profile("swing"))
    assert fc["volatility_regime"] == "compression"


def test_versions_written_from_config():
    fc = build_forecast_record(_base(), {}, get_profile("swing"))
    assert fc["strategy_version"] == config.STRATEGY_VERSION
    assert fc["context_version"] == config.CONTEXT_VERSION


def test_missing_regime_and_volatility_degrade_to_none():
    # Ранний data-quality блок: market_regime ещё не посчитан, ctx без volatility.
    fc = build_forecast_record(
        _base(blocked_at="stale_data"), {}, get_profile("swing"))
    assert fc is not None
    assert fc["market_regime"] is None
    assert fc["volatility_regime"] is None
    # Версии всё равно проставлены (источник — config, всегда доступен).
    assert fc["strategy_version"] == config.STRATEGY_VERSION


def test_volatility_present_but_regime_key_absent_is_none():
    fc = build_forecast_record(_base(), {"volatility": {}}, get_profile("swing"))
    assert fc["volatility_regime"] is None


def test_record_with_new_fields_inserts_via_memory_store():
    fc = build_forecast_record(
        _base(market_regime="range"),
        {"volatility": {"regime": "normal"}},
        get_profile("swing"))
    db = Database(dsn=None)
    fid = asyncio.run(db.insert_forecast(fc))
    assert fid == 1
    stored = db._mem.forecasts[0]
    assert stored["market_regime"] == "range"
    assert stored["volatility_regime"] == "normal"
    assert stored["context_version"] == config.CONTEXT_VERSION


def test_all_persistable_fields_are_covered_by_forecast_cols():
    """Drift guardrail: every field build_forecast_record emits must be listed
    in database._FORECAST_COLS.

    insert_forecast builds its INSERT from `[c for c in _FORECAST_COLS if c in
    fc]`, so any emitted key NOT on the whitelist is silently dropped (data loss
    with no error). A new forecast field added without updating the whitelist
    would fail this test instead of vanishing at write time.
    """
    from database import _FORECAST_COLS

    fc = build_forecast_record(
        _base(direction="long", long_score=4.0, short_score=1.0,
              market_regime="trend_up"),
        {"volatility": {"regime": "expansion"}}, get_profile("swing"))

    dropped = set(fc) - set(_FORECAST_COLS)
    assert not dropped, (
        "build_forecast_record emits fields absent from _FORECAST_COLS; "
        f"insert_forecast would silently drop them: {sorted(dropped)}"
    )


def test_decision_fields_unchanged_by_stage9():
    # Направление/статус/скор — как раньше; новые поля ничего не переопределяют.
    fc = build_forecast_record(
        _base(direction="long", long_score=4.0, short_score=1.0,
              market_regime="trend_up"),
        {"volatility": {"regime": "expansion"}}, get_profile("swing"))
    assert fc["candidate_direction"] == "long"
    assert fc["analysis_status"] == "NO_TRADE"
    assert fc["blocked_gate"] == "htf_filter"
    assert fc["long_score"] == 4.0
