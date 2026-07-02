"""Level 7: Daily limit (blocking).

A hard cap: once MAX_SIGNALS_PER_DAY alerts have been delivered for the
stream, further valid signals are stored as journal entries (visible via
/signal) instead of being pushed. A delivered Telegram alert cannot be
recalled, so "replacing the weakest" is not enforceable — the previous
beats_weakest escape hatch let the cap creep upward on strong days.
"""
from __future__ import annotations

from typing import Any

import config


def within_daily_limit(delivered_today: list[dict[str, Any]],
                       max_per_day: int | None = None) -> bool:
    max_per_day = max_per_day if max_per_day is not None else config.MAX_SIGNALS_PER_DAY
    return len(delivered_today) < max_per_day
