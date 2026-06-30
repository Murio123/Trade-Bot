"""Level 7: Daily limit (blocking).

Even if more valid signals form in a day, only the top-N by final score are
delivered.
"""
from __future__ import annotations

from typing import Any

import config


def within_daily_limit(delivered_today: list[dict[str, Any]],
                       max_per_day: int | None = None) -> bool:
    max_per_day = max_per_day if max_per_day is not None else config.MAX_SIGNALS_PER_DAY
    return len(delivered_today) < max_per_day


def beats_weakest(candidate_score: int, delivered_today: list[dict[str, Any]],
                  max_per_day: int | None = None) -> bool:
    """If the day is full, the candidate must outscore the weakest delivered
    signal to (conceptually) take its slot."""
    max_per_day = max_per_day if max_per_day is not None else config.MAX_SIGNALS_PER_DAY
    if len(delivered_today) < max_per_day:
        return True
    weakest = min((s.get("score", 0) for s in delivered_today), default=0)
    return candidate_score > weakest
