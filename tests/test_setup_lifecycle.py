"""Stage 15A: чистый lifecycle-классификатор.

Доказывает: classify_transition детерминирован, покрывает все 7 статусов и
объясняющие флаги, сравнивает только внутри одной стратегии/mode, отсекает
микро-дельты порогами, честно деградирует на None-полях и не трогает
decision-path (leaf-модуль без runtime-импортов).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analyzer.setup_lifecycle import (
    CONTINUATION, DOWNGRADED, EXPIRED, INVALIDATED, NEW_SETUP, RESOLVED,
    STATUSES, UPGRADED, Thresholds, classify_transition,
)

T0 = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)


def _fc(**kw) -> dict:
    """Сопоставимый ряд forecasts с разумными дефолтами; поля перекрываются kw."""
    base = {
        "id": 1,
        "symbol": "BTCUSDT",
        "analysis_type": "SWING",
        "strategy_version": "v1",
        "context_version": "v1",
        "analysis_status": "WAIT",
        "candidate_direction": "long",
        "raw_confidence": 0.60,
        "long_score": 6.0,
        "short_score": 2.0,
        "blocked_gate": None,
        "no_trade_reasons": None,
        "signal_candle_close_time": T0,
    }
    base.update(kw)
    return base


# --- NEW_SETUP --------------------------------------------------------------

def test_no_previous_is_new_setup():
    res = classify_transition(None, _fc())
    assert res.status == NEW_SETUP
    assert res.previous_id is None
    assert res.reasons == ()


def test_incomparable_symbol_is_new_setup():
    prev = _fc(symbol="ETHUSDT")
    res = classify_transition(prev, _fc(symbol="BTCUSDT"))
    assert res.status == NEW_SETUP
    assert res.comparable is False


def test_incomparable_strategy_version_is_new_setup():
    prev = _fc(strategy_version="v0")
    res = classify_transition(prev, _fc(strategy_version="v1"))
    assert res.status == NEW_SETUP
    assert res.comparable is False


def test_different_analysis_type_is_new_setup():
    prev = _fc(analysis_type="INTRADAY")
    res = classify_transition(prev, _fc(analysis_type="SWING"))
    assert res.status == NEW_SETUP


# --- CONTINUATION -----------------------------------------------------------

def test_same_state_subthreshold_is_continuation():
    prev = _fc(raw_confidence=0.60, long_score=6.0,
               signal_candle_close_time=T0)
    cur = _fc(raw_confidence=0.62, long_score=6.4,
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == CONTINUATION
    assert res.previous_id == 1


def test_mixed_up_and_down_is_continuation():
    # confidence вырос, но relevant long_score упал -> неоднозначно.
    prev = _fc(raw_confidence=0.60, long_score=8.0,
               signal_candle_close_time=T0)
    cur = _fc(raw_confidence=0.70, long_score=6.0,
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == CONTINUATION
    assert "confidence_up" in res.reasons
    assert "long_score_down" in res.reasons


# --- UPGRADED / DOWNGRADED (по оси состояния) -------------------------------

def test_wait_to_enter_is_upgraded():
    prev = _fc(analysis_status="WAIT", signal_candle_close_time=T0)
    cur = _fc(analysis_status="ENTER",
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == UPGRADED
    assert "state_up" in res.reasons


def test_no_trade_to_wait_is_upgraded():
    prev = _fc(analysis_status="NO_TRADE", signal_candle_close_time=T0)
    cur = _fc(analysis_status="WAIT",
              signal_candle_close_time=T0 + timedelta(hours=4))
    assert classify_transition(prev, cur).status == UPGRADED


def test_enter_to_wait_is_downgraded():
    prev = _fc(analysis_status="ENTER", signal_candle_close_time=T0)
    cur = _fc(analysis_status="WAIT",
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == DOWNGRADED
    assert "state_down" in res.reasons


# --- UPGRADED / DOWNGRADED (по силе, та же ось) -----------------------------

def test_same_state_confidence_up_is_upgraded():
    prev = _fc(raw_confidence=0.55, signal_candle_close_time=T0)
    cur = _fc(raw_confidence=0.75,
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == UPGRADED
    assert "confidence_up" in res.reasons


def test_same_state_relevant_score_down_is_downgraded():
    prev = _fc(candidate_direction="long", long_score=8.0,
               signal_candle_close_time=T0)
    cur = _fc(candidate_direction="long", long_score=5.0,
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == DOWNGRADED
    assert "long_score_down" in res.reasons


def test_irrelevant_side_score_move_is_continuation():
    # long-сетап: рост short_score не должен апгрейдить.
    prev = _fc(candidate_direction="long", short_score=2.0,
               signal_candle_close_time=T0)
    cur = _fc(candidate_direction="long", short_score=6.0,
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == CONTINUATION
    assert "short_score_up" in res.reasons  # флаг есть, но на статус не влияет


# --- INVALIDATED ------------------------------------------------------------

def test_direction_flip_is_invalidated():
    prev = _fc(candidate_direction="long", signal_candle_close_time=T0)
    cur = _fc(candidate_direction="short",
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == INVALIDATED
    assert "bias_flip" in res.reasons


def test_active_killed_by_new_gate_is_invalidated():
    prev = _fc(analysis_status="WAIT", blocked_gate=None,
               signal_candle_close_time=T0)
    cur = _fc(analysis_status="NO_TRADE", candidate_direction="long",
              blocked_gate="htf_veto",
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == INVALIDATED
    assert "gate_appeared" in res.reasons


def test_active_to_no_trade_new_reason_is_invalidated():
    prev = _fc(analysis_status="ENTER", no_trade_reasons=None,
               signal_candle_close_time=T0)
    cur = _fc(analysis_status="NO_TRADE", candidate_direction="long",
              no_trade_reasons=["spread_too_wide"],
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == INVALIDATED
    assert "reasons_added" in res.reasons


# --- EXPIRED ----------------------------------------------------------------

def test_active_with_large_gap_is_expired():
    prev = _fc(analysis_status="WAIT", signal_candle_close_time=T0)
    cur = _fc(analysis_status="WAIT",
              signal_candle_close_time=T0 + timedelta(hours=12))
    res = classify_transition(prev, cur)
    assert res.status == EXPIRED


def test_no_trade_with_large_gap_is_not_expired():
    # Нечему истекать: предыдущий не был активным сетапом.
    prev = _fc(analysis_status="NO_TRADE", signal_candle_close_time=T0)
    cur = _fc(analysis_status="NO_TRADE", candidate_direction="long",
              signal_candle_close_time=T0 + timedelta(hours=12))
    res = classify_transition(prev, cur)
    assert res.status != EXPIRED


def test_custom_gap_window_marks_intraday_expired():
    prev = _fc(analysis_status="ENTER", signal_candle_close_time=T0)
    cur = _fc(analysis_status="ENTER",
              signal_candle_close_time=T0 + timedelta(minutes=45))
    thr = Thresholds(max_gap_seconds=30 * 60)  # 30m окно для 15m mode
    assert classify_transition(prev, cur, thresholds=thr).status == EXPIRED


# --- RESOLVED ---------------------------------------------------------------

def test_resolved_previous_enter_outcome():
    prev = _fc(analysis_status="ENTER", signal_candle_close_time=T0)
    cur = _fc(analysis_status="WAIT",
              signal_candle_close_time=T0 + timedelta(hours=4))
    out = {"forecast_id": 1, "resolved": True}
    res = classify_transition(prev, cur, previous_outcome=out)
    assert res.status == RESOLVED


def test_unresolved_outcome_does_not_resolve():
    prev = _fc(analysis_status="ENTER", signal_candle_close_time=T0)
    cur = _fc(analysis_status="ENTER",
              signal_candle_close_time=T0 + timedelta(hours=4))
    out = {"forecast_id": 1, "resolved": False}
    assert classify_transition(prev, cur, previous_outcome=out).status != RESOLVED


def test_resolved_only_for_previous_enter():
    # Предыдущий WAIT с resolved-исходом — не RESOLVED (входа не было).
    prev = _fc(analysis_status="WAIT", signal_candle_close_time=T0)
    cur = _fc(analysis_status="WAIT",
              signal_candle_close_time=T0 + timedelta(hours=4))
    out = {"forecast_id": 1, "resolved": True}
    assert classify_transition(prev, cur, previous_outcome=out).status != RESOLVED


# --- пороги и None-поля -----------------------------------------------------

def test_confidence_threshold_boundary():
    prev = _fc(raw_confidence=0.60, signal_candle_close_time=T0)
    cur = _fc(raw_confidence=0.65,
              signal_candle_close_time=T0 + timedelta(hours=4))
    # ровно на пороге 0.05 -> считается движением вверх
    res = classify_transition(prev, cur, thresholds=Thresholds(confidence_delta=0.05))
    assert res.status == UPGRADED
    # выше порога -> шум, CONTINUATION
    res2 = classify_transition(prev, cur, thresholds=Thresholds(confidence_delta=0.06))
    assert res2.status == CONTINUATION


def test_none_scores_do_not_crash_and_continue():
    prev = _fc(raw_confidence=None, long_score=None, short_score=None,
               signal_candle_close_time=T0)
    cur = _fc(raw_confidence=None, long_score=None, short_score=None,
              signal_candle_close_time=T0 + timedelta(hours=4))
    res = classify_transition(prev, cur)
    assert res.status == CONTINUATION
    assert res.reasons == ()


def test_missing_timestamps_disable_expiry():
    prev = _fc(analysis_status="WAIT", signal_candle_close_time=None,
               decision_time=None, created_at=None)
    cur = _fc(analysis_status="WAIT", signal_candle_close_time=None,
              decision_time=None, created_at=None)
    # без якорей времени gap неизвестен -> не EXPIRED
    assert classify_transition(prev, cur).status != EXPIRED


def test_iso_string_timestamps_supported():
    prev = _fc(analysis_status="WAIT",
               signal_candle_close_time="2026-07-05T12:00:00+00:00")
    cur = _fc(analysis_status="WAIT",
              signal_candle_close_time="2026-07-05T23:00:00Z")  # +11h
    assert classify_transition(prev, cur).status == EXPIRED


# --- инварианты -------------------------------------------------------------

def test_status_always_in_known_set():
    prev = _fc(signal_candle_close_time=T0)
    cur = _fc(signal_candle_close_time=T0 + timedelta(hours=4))
    assert classify_transition(prev, cur).status in STATUSES


def test_pure_no_input_mutation():
    prev = _fc(signal_candle_close_time=T0)
    cur = _fc(analysis_status="ENTER",
              signal_candle_close_time=T0 + timedelta(hours=4))
    prev_copy, cur_copy = dict(prev), dict(cur)
    classify_transition(prev, cur)
    assert prev == prev_copy and cur == cur_copy
