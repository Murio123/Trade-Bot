"""Stage 14: Deep Analysis Renderer v2 — pure explanation-слой.

Доказывает: render_deep ОБЪЯСНЯЕТ готовый cascade-result, но НЕ меняет решение —
decision/bias/scores/confidence/expected_move/stop/TP/RR идентичны входу; decision
выводится тем же STATUS_MAP, что и persist-слой; отсутствующие обогащения
(bollinger/volatility/historical) деградируют в None без падения; blocked-результат
рендерится с gate и «что должно измениться».
"""
from __future__ import annotations

from signal_engine.deep_renderer import render_deep
from signal_engine.schema import STATUS_MAP


def _enter_signal(**extra):
    """Полный доставляемый сигнал run_cascade (ENTER)."""
    base = {
        "status": "alert",
        "analysis_status": "ENTER",
        "analysis_type": "SWING",
        "candidate_direction": "long",
        "final_bias": "LONG",
        "long_score": 74.0,
        "short_score": 31.0,
        "raw_confidence": 0.62,
        "confidence_score": 62,
        "confidence_modifier": 1.1,
        "conflict_factor": 0.9,
        "mtf_agreement": {"agree_ratio": 0.75},
        "expected_move_points": 1800.0,
        "expected_move_percent": 2.7,
        "expected_move_atr": 1.4,
        "stop_loss": 63000.0,
        "invalidation_level": 63000.0,
        "target_1": 67000.0,
        "target_2": 69000.0,
        "take_profit_levels": [67000.0, 69000.0],
        "tp2_source": "structure",
        "targets_structure": True,
        "risk_reward": 2.3,
        "market_regime": "trend_up",
        "volatility_regime": "expansion",
        "reasons": ["BOS_up", "bullish_ob"],
        "confirming_factors": ["BOS_up", "bullish_ob"],
        "contradicting_factors": ["rsi_overbought"],
        "mtf": {"4H": "up", "1D": "up"},
        "timeframe_alignment": {"trends": {"4H": "up", "1D": "up"},
                                "agree_ratio": 0.75},
        "cancel_conditions": ["закрытие свечи за ATR-стопом"],
        "confirmation_conditions": ["свеча entry-ТФ закрывается в направлении сделки"],
    }
    base.update(extra)
    return base


def _blocked(stage: str, **extra):
    base = {"status": "blocked", "blocked_at": stage, "deliverable": False,
            "candidate_direction": "short", "long_score": 40.0, "short_score": 55.0,
            "market_regime": "range"}
    base.update(extra)
    return base


# --- Инвариант: renderer не меняет решение (passthrough) ---------------------

def test_decision_and_levels_passthrough_unchanged():
    sig = _enter_signal()
    out = render_deep(sig)
    assert out["decision"] == "ENTER"
    assert out["bias"] == "LONG"
    assert out["scores"] == {"long": 74.0, "short": 31.0}
    assert out["confidence"]["score"] == 62
    assert out["expected_move"] == {"points": 1800.0, "percent": 2.7, "atr": 1.4}
    assert out["stop"]["stop_loss"] == 63000.0
    assert out["targets"]["levels"] == [67000.0, 69000.0]
    assert out["risk_reward"] == 2.3


def test_renderer_never_invents_decision_for_blocked():
    # blocked -> NO_TRADE через тот же STATUS_MAP, никакого BUY/LONG «от себя».
    out = render_deep(_blocked("htf_filter"))
    assert out["decision"] == STATUS_MAP["blocked"] == "NO_TRADE"
    assert out["blocked_gate"] == "htf_filter"


def test_wait_status_mapped_from_journal():
    out = render_deep(_enter_signal(status="journal", analysis_status="WAIT"))
    assert out["decision"] == "WAIT"


# --- 22 секции присутствуют ---------------------------------------------------

def test_all_stage14_sections_present():
    out = render_deep(_enter_signal())
    expected = {
        "decision", "bias", "analysis_type", "confidence", "scores",
        "expected_move", "stop", "targets", "risk_reward", "blocked_gate",
        "no_trade_reasons", "supporting_factors", "contradicting_factors",
        "mtf_structure", "volatility", "bollinger", "regime",
        "historical_quality", "similar_setup", "what_must_change",
        "what_invalidates", "what_to_watch",
    }
    assert expected <= set(out)


def test_similar_setup_is_placeholder():
    assert render_deep(_enter_signal())["similar_setup"] is None


# --- what_must_change: пересказ гейтов, не новое решение -----------------------

def test_what_must_change_lists_gate_and_reasons_for_blocked():
    out = render_deep(_blocked(
        "no_trade", no_trade_reasons=["R:R ниже минимума", "низкая confidence"]))
    wmc = out["what_must_change"]
    assert "снятие гейта: no_trade" in wmc
    assert "R:R ниже минимума" in wmc
    assert "низкая confidence" in wmc


def test_what_must_change_empty_for_enter():
    assert render_deep(_enter_signal())["what_must_change"] is None


# --- Опциональные обогащения деградируют в None -------------------------------

def test_optional_enrichments_default_to_none():
    out = render_deep(_enter_signal())
    assert out["bollinger"] is None
    assert out["historical_quality"] is None


def test_optional_enrichments_passthrough_when_provided():
    out = render_deep(
        _enter_signal(),
        bollinger={"bb_squeeze": True, "regime_clue": "expansion"},
        historical={"realized_r": 1.4, "matches": 12})
    assert out["bollinger"] == {"bb_squeeze": True, "regime_clue": "expansion"}
    assert out["historical_quality"] == {"realized_r": 1.4, "matches": 12}


def test_volatility_uses_detailed_context_when_given():
    out = render_deep(_enter_signal(),
                      volatility={"regime": "expansion", "atr_percentile": 82})
    assert out["volatility"]["atr_percentile"] == 82
    assert out["regime"]["volatility_regime"] == "expansion"


# --- Скудный blocked-result не роняет renderer --------------------------------

def test_minimal_blocked_result_degrades_gracefully():
    out = render_deep({"status": "blocked", "blocked_at": "stale_data"})
    assert out["decision"] == "NO_TRADE"
    assert out["bias"] == "NEUTRAL"
    assert out["scores"] == {"long": None, "short": None}
    assert out["expected_move"] == {"points": None, "percent": None, "atr": None}
    assert out["mtf_structure"] is None
    assert out["what_invalidates"] is None
