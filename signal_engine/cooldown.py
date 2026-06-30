"""Level 6: Cooldown and de-duplication (blocking)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config


def _now() -> datetime:
    return datetime.now(timezone.utc)


def should_send_signal(new_signal: dict[str, Any], last_signal: Optional[dict[str, Any]],
                       atr: float, cooldown_hours: int | None = None) -> bool:
    cooldown_hours = cooldown_hours if cooldown_hours is not None else config.COOLDOWN_HOURS
    if last_signal is None:
        return True

    last_ts = last_signal.get("timestamp") or last_signal.get("created_at")
    if last_ts is not None:
        if isinstance(last_ts, str):
            try:
                last_ts = datetime.fromisoformat(last_ts)
            except ValueError:
                last_ts = None
    if last_ts is not None:
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        if (_now() - last_ts) < timedelta(hours=cooldown_hours):
            return False

    same_dir = new_signal.get("direction") == last_signal.get("direction")
    new_price = new_signal.get("price") or new_signal.get("entry_price")
    last_price = last_signal.get("price") or last_signal.get("entry_price")
    if same_dir and new_price is not None and last_price is not None:
        if abs(new_price - last_price) < atr * 0.5:
            return False
    return True
