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
    "swing": {
        "label": "СВИНГ",
        "emoji": "📊",
        "entry": "1h",
        "htf": "1d",
        "mtf": ["1h", "4h", "12h", "1d"],  # 1H entry / 4H+12H zones / 1D trend
        "zone_tfs": ["12h", "4h"],          # OB/FVG/targets from higher timeframes
        "structural_stop": True,            # stop behind HTF structure, not 1H-ATR
        "cooldown_hours": 8,
        "atr_mult": _fnum("SWING_ATR_MULT", 1.5),
        "targets": _targets("SWING_TARGETS", (1.5, 3.0)),
        "interval_minutes": 60,    # scheduled hourly (matches 1H entry)
    },
    "intraday": {
        "label": "ИНТРАДЕЙ",
        "emoji": "⚡",
        "entry": "15m",
        "htf": "1h",
        "mtf": ["15m", "1h", "4h"],  # 15m entry / 1H trend / 4H context
        "zone_tfs": ["4h", "1h"],     # OB/FVG/targets from higher timeframes
        "stop_tf": "1h",              # stop from 1H ATR: a 1.2x15m-ATR stop is so
                                      # tight that fees eat 0.5-1R per round trip
        "cooldown_hours": 2,
        "atr_mult": _fnum("INTRADAY_ATR_MULT", 1.2),
        "targets": _targets("INTRADAY_TARGETS", (1.0, 2.0)),
        "interval_minutes": 15,    # scheduled every 15m
    },
}

DEFAULT_PROFILE = "swing"


def get_profile(name: str | None) -> dict[str, Any]:
    return PROFILES.get(name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
