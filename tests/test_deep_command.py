"""Stage 14 Option B / Step 2.2 — /deep объясняет СОХРАНЁННЫЙ forecast.

Доказывает: forecast_to_deep_input только переносит уже посчитанные поля и не
приписывает ложный blocked_gate; deep_cmd читает latest_forecast, форматирует
объяснение и НЕ трогает старый decision-engine (gather_swing_context /
compute_quality_score / swing_analysis). Источник истины — сохранённый forecast.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import bot.handlers as h
from bot.handlers import forecast_to_deep_input


def _forecast(**kw) -> dict:
    """Реалистичная forecasts-строка (как build_forecast_record/insert)."""
    base = {
        "symbol": "BTCUSDT", "analysis_type": "SWING", "timeframe": "4h",
        "analysis_status": "ENTER", "candidate_direction": "long",
        "final_bias": "LONG", "long_score": 74.0, "short_score": 31.0,
        "raw_confidence": 0.62, "blocked_gate": None,
        "expected_move_points": 1800.0, "expected_move_percent": 2.7,
        "expected_move_atr": 1.4, "stop_loss": 63000.0,
        "take_profit_levels": [67000.0, 69000.0], "risk_reward": 2.3,
        "no_trade_reasons": None, "market_regime": "trend_up",
        "volatility_regime": "expansion", "strategy_version": "s1",
        "context_version": "c1", "entry_zone": {"low": 64500.0, "high": 65500.0},
    }
    base.update(kw)
    return base


# --- forecast_to_deep_input (pure) ------------------------------------------

# 1
def test_preserves_decision_fields_exactly() -> None:
    fc = _forecast()
    deep = forecast_to_deep_input(fc)
    for key in ("analysis_status", "candidate_direction", "final_bias",
                "long_score", "short_score", "raw_confidence", "stop_loss",
                "take_profit_levels", "risk_reward", "market_regime",
                "volatility_regime", "strategy_version", "context_version",
                "entry_zone"):
        assert deep[key] == fc[key], key


# 2
def test_enter_has_no_fake_blocked_gate() -> None:
    deep = forecast_to_deep_input(_forecast(analysis_status="ENTER",
                                            blocked_gate=None))
    assert deep["status"] == "alert"
    assert deep["blocked_gate"] is None


# 3
def test_wait_has_no_fake_blocked_gate() -> None:
    deep = forecast_to_deep_input(_forecast(analysis_status="WAIT",
                                            candidate_direction="long",
                                            blocked_gate=None))
    assert deep["status"] == "journal"
    assert deep["blocked_gate"] is None


# 4
def test_no_trade_with_gate_preserved() -> None:
    deep = forecast_to_deep_input(_forecast(analysis_status="NO_TRADE",
                                            blocked_gate="htf_filter"))
    assert deep["status"] == "blocked"
    assert deep["blocked_gate"] == "htf_filter"


# 5
def test_no_trade_without_gate_stays_none() -> None:
    deep = forecast_to_deep_input(_forecast(analysis_status="NO_TRADE",
                                            blocked_gate=None))
    assert deep["status"] == "ignored"
    assert deep["blocked_gate"] is None


# 6
def test_does_not_mutate_input() -> None:
    fc = _forecast()
    before = copy.deepcopy(fc)
    forecast_to_deep_input(fc)
    assert fc == before
    assert "status" not in fc  # синтетический status не протёк во вход


# --- deep_cmd wiring --------------------------------------------------------

def _fake_update() -> MagicMock:
    upd = MagicMock()
    upd.effective_message.reply_text = AsyncMock()
    return upd


def _run_deep(forecast) -> str:
    """Прогнать deep_cmd с замоканным db.latest_forecast; вернуть текст ответа."""
    upd = _fake_update()
    ctx = MagicMock()
    fake_db = MagicMock()
    fake_db.latest_forecast = AsyncMock(return_value=forecast)
    with patch.object(h, "db", fake_db):
        asyncio.run(h.deep_cmd(upd, ctx))
    assert fake_db.latest_forecast.await_count == 1
    return upd.effective_message.reply_text.call_args[0][0]


# 7
def test_deep_cmd_with_saved_forecast() -> None:
    text = _run_deep(_forecast())
    assert "ENTER" in text        # decision из forecast
    assert "LONG" in text         # bias из forecast
    assert "62" in text           # confidence = round(0.62*100)


# 8
def test_deep_cmd_fallback_when_no_forecast() -> None:
    text = _run_deep(None)
    assert text == ("📭 Пока нет сохранённого прогноза. "
                    "Дождитесь ближайшего анализа.")


# 9
def test_deep_cmd_does_not_call_old_path() -> None:
    # Статически: исходник deep_cmd не упоминает старый движок.
    src = inspect.getsource(h.deep_cmd)
    for banned in ("gather_swing_context", "compute_quality_score",
                   "swing_analysis", "generate_report"):
        assert banned not in src, f"deep_cmd must not reference {banned!r}"

    # Поведенчески: даже под моками старые точки входа не вызываются.
    import ai.swing_analysis as sa
    import pipeline as pl
    import signal_engine.quality_score as qs
    with patch.object(pl, "gather_swing_context",
                      MagicMock(), create=True) as m_gather, \
         patch.object(qs, "compute_quality_score", MagicMock()) as m_quality, \
         patch.object(sa, "generate_report", AsyncMock()) as m_report:
        _run_deep(_forecast())
        m_gather.assert_not_called()
        m_quality.assert_not_called()
        m_report.assert_not_called()


# 10
def test_other_handlers_unchanged() -> None:
    # /deep по-прежнему зарегистрирован, а базовые команды на месте.
    assert h.COMMAND_DISPATCH["deep"] is h.deep_cmd
    assert h.COMMAND_DISPATCH["help"] is h.help_cmd  # help — через dispatch
    assert callable(h.start_cmd)                     # start — через CommandHandler
