"""Core signal-engine invariants: the money-critical logic."""
from datetime import datetime, timedelta, timezone

from signal_engine.confluence import (calculate_confluence_score,
                                      has_diverse_confirmation)
from signal_engine.cooldown import should_send_signal
from signal_engine.daily_limiter import within_daily_limit
from signal_engine.htf_filter import filter_by_htf, get_htf_bias
from signal_engine.mtf_confidence import mtf_confidence_factor
from signal_engine.profiles import PROFILES, get_profile
from risk.position_sizing import calculate_position


# --- HTF filter (level 1): counter-trend must be blocked -------------------

def test_htf_bias_detection():
    assert get_htf_bias({"price": 100, "ema20": 95, "ema50": 90, "ema200": 80}) == "bullish"
    assert get_htf_bias({"price": 80, "ema20": 85, "ema50": 90, "ema200": 100}) == "bearish"
    assert get_htf_bias({"price": 100, "ema20": 105, "ema50": 90, "ema200": 80}) == "neutral"


def test_htf_filter_blocks_counter_trend():
    assert filter_by_htf("short", "bullish") is None
    assert filter_by_htf("long", "bearish") is None
    assert filter_by_htf("long", "bullish") == "long"
    assert filter_by_htf("short", "neutral") == "short"


# --- Confluence (levels 2-3) ------------------------------------------------

def test_confluence_is_directional():
    bull_data = {"rsi": 25, "macd_bullish_cross": True, "in_discount": True}
    long_total, _, _ = calculate_confluence_score(bull_data, "long")
    short_total, _, _ = calculate_confluence_score(bull_data, "short")
    assert long_total > short_total


def test_diversity_gate():
    assert not has_diverse_confirmation({"momentum": 5, "trend": 0,
                                         "volume": 0, "structure": 0, "macro": 0}, 3)
    assert has_diverse_confirmation({"momentum": 2, "trend": 1,
                                     "volume": 0, "structure": 3, "macro": 0}, 3)


# --- MTF agreement (level 5) ------------------------------------------------

def test_mtf_factor_scales_with_agreement():
    full, _ = mtf_confidence_factor(["bullish"] * 4, "long")
    split, _ = mtf_confidence_factor(["bullish", "bullish", "bearish", "bearish"], "long")
    against, _ = mtf_confidence_factor(["bearish"] * 4, "long")
    assert full > split > against
    assert full == 1.2 and against == 0.7


# --- Cooldown (level 6) -----------------------------------------------------

def test_cooldown_blocks_recent_duplicate():
    now = datetime.now(timezone.utc)
    last = {"timestamp": now - timedelta(hours=1), "direction": "long",
            "entry_price": 60000}
    new = {"direction": "long", "price": 60050}
    assert not should_send_signal(new, last, atr=1000, cooldown_hours=4)
    old = {"timestamp": now - timedelta(hours=10), "direction": "long",
           "entry_price": 50000}
    assert should_send_signal(new, old, atr=1000, cooldown_hours=4)


# --- Daily limit (level 7) ----------------------------------------------------

def test_daily_limit_is_a_hard_cap():
    day = [{"score": 8}, {"score": 9}, {"score": 10}]
    assert not within_daily_limit(day, max_per_day=3)
    # No escalation escape hatch: a full day stays full regardless of score.
    assert within_daily_limit(day[:2], max_per_day=3)


# --- Position sizing ----------------------------------------------------------

def test_position_sizing_long_short_symmetry():
    long = calculate_position(60000, 800, "long", account_balance=10000,
                              risk_percent=1, atr_multiplier=1.5)
    short = calculate_position(60000, 800, "short", account_balance=10000,
                               risk_percent=1, atr_multiplier=1.5)
    assert long["stop_loss"] < 60000 < short["stop_loss"]
    assert long["risk_amount"] == short["risk_amount"] == 100.0
    # risk amount / stop distance == size (position_size is rounded to 4 dp)
    assert abs(long["position_size"] * long["stop_distance"] - 100.0) < 0.1


def test_profiles_are_consistent():
    for name, p in PROFILES.items():
        assert p["entry"] in p["mtf"] or name == "swing"
        assert p["targets"][0] < p["targets"][1]
        assert p["cooldown_hours"] > 0
    assert get_profile("nonsense")["label"] == PROFILES["swing"]["label"]
