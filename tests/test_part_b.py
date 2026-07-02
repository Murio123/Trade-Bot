"""Part B: regime weighting, mandatory NO_TRADE gates, result schema."""
from __future__ import annotations

import config
from signal_engine.no_trade_gate import (bad_risk_reward, low_confidence,
                                         missing_invalidation,
                                         position_conflict, tf_conflict)
from signal_engine.regime import detect_regime, weighted_total
from signal_engine.schema import build_result


# --- regime -------------------------------------------------------------------

def test_detect_regime_hierarchy():
    up = {"price": 110, "ema20": 105, "ema50": 100, "ema200": 90}
    down = {"price": 80, "ema20": 85, "ema50": 90, "ema200": 100}
    flat = {"price": 100, "ema20": 101, "ema50": 99, "ema200": 100}
    assert detect_regime(up) == "trend_up"
    assert detect_regime(down) == "trend_down"
    assert detect_regime(flat) == "range"
    assert detect_regime({}) == "range"  # missing data -> conservative
    assert detect_regime(up, {"atr_percentile": 95}) == "high_volatility"


def test_regime_weights_change_the_verdict():
    scores = {"trend": 2, "structure": 3, "momentum": 2, "volume": 1, "macro": 0}
    in_trend = weighted_total(scores, "trend_up")
    in_range = weighted_total(scores, "range")
    # Same evidence weighs more in a trend (trend x1.5) than in a range (x0.5).
    assert in_trend > in_range
    assert weighted_total(scores, "unknown") == sum(scores.values())


# --- NO_TRADE gates -------------------------------------------------------------

def test_min_risk_reward_gate(monkeypatch):
    monkeypatch.setattr(config, "MIN_RISK_REWARD", 1.5)
    assert bad_risk_reward(100, 95, 104) is not None    # RR 0.8 -> blocked
    assert bad_risk_reward(100, 95, 110) is None        # RR 2.0 -> ok
    assert bad_risk_reward(100, 100, 110) is not None   # no risk = no stop


def test_invalidation_required():
    assert missing_invalidation(None, None) is not None
    assert missing_invalidation(95.0, 95.0) is None


def test_low_confidence_gate(monkeypatch):
    monkeypatch.setattr(config, "MIN_CONFIDENCE", 0.3)
    assert low_confidence(0.1) is not None
    assert low_confidence(0.5) is None


def test_tf_conflict_gate():
    assert tf_conflict({"agree": 0, "total": 3, "ratio": 0.0}) is not None
    assert tf_conflict({"agree": 2, "total": 3, "ratio": 0.67}) is None
    assert tf_conflict({"agree": 0, "total": 0, "ratio": 0.0}) is None  # unrated


def test_position_conflict_gate():
    open_long = [{"id": 7, "symbol": "BTCUSDT", "direction": "long"}]
    assert position_conflict(open_long, "short", "BTCUSDT") is not None
    assert position_conflict(open_long, "long", "BTCUSDT") is None
    assert position_conflict([], "short", "BTCUSDT") is None
    legacy = [{"id": 1, "symbol": None, "direction": "long"}]  # pre-column rows
    assert position_conflict(legacy, "short", "BTCUSDT") is not None


# --- schema ---------------------------------------------------------------------

def test_build_result_has_all_mandatory_fields():
    signal = {"entry_price": 100.0, "atr": 2.0, "stop_loss": 95.0,
              "target_1": 105.0, "target_2": 110.0, "direction": "long",
              "confidence": 0.42, "reasons": ["Реакция от бычьего Order Block"],
              "mtf": {"4h": "bullish"}, "stop_basis": "structure"}
    ctx = {"last_close_time": "2026-07-02 08:00:00+00:00"}
    res = build_result(signal, ctx, "trend_up", ["RSI перекуплен"],
                       {"agree": 3, "total": 3, "ratio": 1.0})
    fields = res.to_signal_fields()
    for key in ("market_regime", "primary_bias", "timeframe_alignment",
                "setup_type", "entry_zone", "invalidation_level", "stop_loss",
                "take_profit_levels", "expected_risk_reward", "confidence_score",
                "confirming_factors", "contradicting_factors",
                "confirmation_conditions", "cancel_conditions",
                "no_trade_reasons", "freshness_timestamp"):
        assert key in fields, key
    assert fields["primary_bias"] == "LONG"
    assert fields["expected_risk_reward"] == 2.0
    assert fields["confidence_score"] == 42
    assert fields["contradicting_factors"] == ["RSI перекуплен"]
    assert fields["entry_zone"]["low"] < 100 < fields["entry_zone"]["high"]
