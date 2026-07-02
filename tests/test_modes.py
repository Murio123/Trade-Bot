"""Per-mode requirements: minimum expected move, TF grids, status mapping."""
from __future__ import annotations

import pytest

from risk.position_sizing import calculate_position
from signal_engine.no_trade_gate import (effective_expected_move,
                                         insufficient_expected_move)
from signal_engine.profiles import PROFILES
from signal_engine.schema import STATUS_MAP

THRESHOLDS = {"intraday": 500, "swing": 1500, "position": 3000}
TYPES = {"intraday": "INTRADAY", "swing": "SWING", "position": "POSITIONAL"}


def test_mode_configs_are_separate_and_correct():
    for name, min_pts in THRESHOLDS.items():
        p = PROFILES[name]
        assert p["minimum_expected_move_points"] == min_pts, name
        assert p["analysis_type"] == TYPES[name]
        assert p["minimum_confidence"] > 0 and p["minimum_risk_reward"] > 0
        assert p["forecast_horizon"] and p["expected_holding_period"]


@pytest.mark.parametrize("name,min_pts", THRESHOLDS.items())
def test_threshold_gate_per_mode(name, min_pts):
    at = TYPES[name]
    assert insufficient_expected_move(min_pts - 1, min_pts, at) is not None
    assert insufficient_expected_move(min_pts, min_pts, at) is None
    assert insufficient_expected_move(min_pts + 1, min_pts, at) is None


def test_expected_move_capped_by_volatility():
    # Structural target 2000 pts away, but ATR only supports ~800 over the
    # horizon -> the effective move is the volatility ceiling, not the target.
    move = effective_expected_move(entry=100_000, tp2=102_000, atr=200,
                                   horizon_hours=64, tf_hours=4)  # 200*sqrt(16)=800
    assert move == 800.0
    # Volatility supports more than the structure offers -> structural wins.
    move2 = effective_expected_move(100_000, 102_000, atr=1000,
                                    horizon_hours=64, tf_hours=4)
    assert move2 == 2000.0
    # No ATR -> structural distance, no crash.
    assert effective_expected_move(100_000, 100_500, 0.0, 24, 1) == 500.0


def test_threshold_is_filter_not_take_profit():
    # Targets come from position sizing / structure and must be identical
    # whether or not the gate would pass — the gate only reads them.
    pos = calculate_position(100_000, 400, "long", atr_multiplier=1.5,
                             targets_r=(1.5, 3.0))
    before = (pos["target_1"], pos["target_2"])
    insufficient_expected_move(
        effective_expected_move(100_000, pos["target_2"], 400, 96, 4),
        1500, "SWING")
    assert (pos["target_1"], pos["target_2"]) == before
    # And the target is NOT the threshold value.
    assert pos["target_2"] - 100_000 != 1500


def test_status_mapping():
    assert STATUS_MAP["alert"] == "ENTER"
    assert STATUS_MAP["journal"] == "WAIT"
    assert STATUS_MAP["cooldown"] == "WAIT"
    assert STATUS_MAP["blocked"] == "NO_TRADE"


def test_tf_grids_are_native_and_known():
    from signal_engine.vetoes import TF_HOURS
    for p in PROFILES.values():
        for tf in [p["entry"], p["htf"], *p["mtf"], *p["zone_tfs"]]:
            assert tf in TF_HOURS, tf  # every TF is fetchable and mapped
