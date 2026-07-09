"""Trade style profiles: swing vs intraday.

Each profile defines the entry timeframe, the higher timeframe used for the
bias filter, the three timeframes checked for multi-timeframe agreement, and
risk/cooldown parameters tuned to that horizon.
"""
from __future__ import annotations

import os
from typing import Any


def _targets(name: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = os.getenv(name)
    if raw:
        try:
            a, b = [float(x) for x in raw.split(",")[:2]]
            return (a, b)
        except (ValueError, TypeError):
            pass
    return default


def _fnum(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (ValueError, TypeError):
        return default

PROFILES: dict[str, dict[str, Any]] = {
    # Timeframe grids use NATIVE exchange intervals only. The requested
    # 3H/5H/8H are not served by Bybit/Binance; their neighbours (2H/4H/6H/12H)
    # carry the same information (adjacent TFs are correlated by design), so
    # each role samples ONE representative TF instead of stacking neighbours.
    "swing": {
        # Hierarchy: 1D/12H = global regime, 6H = structure & zones,
        # 4H = trading setup and entry, 1H = confirmation (via mtf).
        "label": "СВИНГ",
        "emoji": "📊",
        "analysis_type": "SWING",
        "entry": "4h",
        "htf": "1d",
        "htf_policy": "block_counter_trend",
        "mtf": ["1h", "4h", "12h", "1d"],  # confirm / setup / structure / regime
        "zone_tfs": ["12h", "6h"],         # key zones: 12H regime + 6H structure
        "structural_stop": True,
        "stop_tf": "12h",
        "cooldown_hours": 8,
        "atr_mult": _fnum("SWING_ATR_MULT", 1.5),
        "targets": _targets("SWING_TARGETS", (1.5, 3.0)),
        "interval_minutes": 240,
        "minimum_expected_move_points": _fnum("SWING_MIN_MOVE_PTS", 1500),
        "minimum_confidence": _fnum("SWING_MIN_CONFIDENCE", 0.3),
        "minimum_risk_reward": _fnum("SWING_MIN_RR", 1.5),
        "forecast_horizon": "1-14 дней",
        "forecast_horizon_hours": 96,      # ATR ceiling window for the move gate
        "expected_holding_period": "1 день – 3 недели",
        # Max seconds between the entry-candle close and the decision for an
        # ENTER to stay executable; beyond it the run is downgraded to WAIT.
        "max_decision_delay_seconds": _fnum("SWING_MAX_DELAY_SEC", 900),
    },
    "position": {
        # Hierarchy: 1D = global regime, 12H = direction & major zones,
        # 6H/4H = positional scenario (4H entry), 1H = entry refinement.
        "label": "ПОЗИЦИОННЫЙ",
        "emoji": "🌊",
        "analysis_type": "POSITIONAL",
        "entry": "4h",
        "htf": "1d",
        "htf_policy": "block_counter_trend",
        "mtf": ["4h", "12h", "1d"],
        "zone_tfs": ["1d", "12h"],
        "structural_stop": True,
        "stop_tf": "1d",
        "cooldown_hours": 24,
        "atr_mult": _fnum("POSITION_ATR_MULT", 1.2),
        "targets": _targets("POSITION_TARGETS", (1.2, 2.5)),
        "interval_minutes": 240,
        "minimum_expected_move_points": _fnum("POSITION_MIN_MOVE_PTS", 3000),
        "minimum_confidence": _fnum("POSITION_MIN_CONFIDENCE", 0.3),
        "minimum_risk_reward": _fnum("POSITION_MIN_RR", 1.5),
        "forecast_horizon": "1-6 недель",
        "forecast_horizon_hours": 336,
        "expected_holding_period": "от нескольких дней до недель",
        "max_decision_delay_seconds": _fnum("POSITION_MAX_DELAY_SEC", 1800),
    },
    "intraday": {
        # Hierarchy: 4H/2H = context & direction, 1H/30M = setup,
        # 15M = confirmation and entry timing.
        "label": "ИНТРАДЕЙ",
        "emoji": "⚡",
        "analysis_type": "INTRADAY",
        "entry": "15m",
        "htf": "4h",                 # direction from the 4H context (was 1H)
        "htf_policy": "block_counter_trend",
        "mtf": ["15m", "1h", "4h"],  # confirm / setup / context
        "zone_tfs": ["4h", "2h", "1h"],  # context+setup zones
        "stop_tf": "1h",             # 15m-ATR stops are eaten by fees
        "cooldown_hours": 2,
        "atr_mult": _fnum("INTRADAY_ATR_MULT", 1.2),
        "targets": _targets("INTRADAY_TARGETS", (1.0, 2.0)),
        "interval_minutes": 15,
        "minimum_expected_move_points": _fnum("INTRADAY_MIN_MOVE_PTS", 500),
        "minimum_confidence": _fnum("INTRADAY_MIN_CONFIDENCE", 0.3),
        "minimum_risk_reward": _fnum("INTRADAY_MIN_RR", 1.5),
        "forecast_horizon": "до 24 часов",
        "forecast_horizon_hours": 24,
        "expected_holding_period": "15 минут – 24 часа",
        "max_decision_delay_seconds": _fnum("INTRADAY_MAX_DELAY_SEC", 180),
    },
}

DEFAULT_PROFILE = "swing"


def get_profile(name: str | None) -> dict[str, Any]:
    return PROFILES.get(name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
