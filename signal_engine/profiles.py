"""Trade style profiles: swing vs intraday.

Each profile defines the entry timeframe, the higher timeframe used for the
bias filter, the three timeframes checked for multi-timeframe agreement, and
risk/cooldown parameters tuned to that horizon.
"""
from __future__ import annotations

from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "swing": {
        "label": "СВИНГ",
        "emoji": "📊",
        "entry": "4h",
        "htf": "1d",
        "mtf": ["4h", "12h", "1d"],  # 1D trend / 12H zones / 4H entry
        "cooldown_hours": 8,
        "atr_mult": 1.5,
        "targets": (1.5, 3.0),
        "interval_minutes": 240,   # scheduled every 4h
    },
    "intraday": {
        "label": "ИНТРАДЕЙ",
        "emoji": "⚡",
        "entry": "15m",
        "htf": "4h",
        "mtf": ["15m", "1h", "4h"],
        "cooldown_hours": 2,
        "atr_mult": 1.2,
        "targets": (1.0, 2.0),
        "interval_minutes": 15,    # scheduled every 15m
    },
}

DEFAULT_PROFILE = "swing"


def get_profile(name: str | None) -> dict[str, Any]:
    return PROFILES.get(name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
