"""Session Volatility Profile.

Computes realised volatility per trading session (Asia / London / New York)
from historical hourly candles. Nothing is hard-coded — the ranges are derived
from the supplied data so the "best session" reflects current regime.

Session UTC hour ranges (approximate):
    Asia   : 00:00 - 08:00
    London : 07:00 - 16:00
    NewYork: 12:00 - 21:00
"""
from __future__ import annotations

from typing import Any

import pandas as pd

SESSIONS = {
    "Asia": range(0, 8),
    "London": range(7, 16),
    "NewYork": range(12, 21),
}


def session_for_hour(hour: int) -> str:
    # Pick the session whose window contains the hour; overlaps resolve to the
    # later-opening one for simplicity.
    if hour in SESSIONS["NewYork"]:
        return "NewYork"
    if hour in SESSIONS["London"]:
        return "London"
    if hour in SESSIONS["Asia"]:
        return "Asia"
    return "Off"


def compute_session_stats(df_1h: pd.DataFrame) -> dict[str, Any]:
    """df_1h must contain open_time (tz-aware), high, low, close."""
    if df_1h is None or len(df_1h) < 24:
        return {"sessions": {}, "best_session": None}

    df = df_1h.copy()
    df["hour"] = df["open_time"].dt.hour
    df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df["session"] = df["hour"].map(session_for_hour)

    stats: dict[str, Any] = {}
    for name in SESSIONS:
        subset = df[df["session"] == name]
        if len(subset):
            stats[name] = {
                "avg_range_pct": round(float(subset["range_pct"].mean()), 3),
                "samples": int(len(subset)),
            }

    best = None
    if stats:
        best = max(stats, key=lambda s: stats[s]["avg_range_pct"])

    # Current session relative to last candle.
    current_hour = int(df["hour"].iloc[-1])
    current_session = session_for_hour(current_hour)

    return {
        "sessions": stats,
        "best_session": best,
        "current_session": current_session,
    }
