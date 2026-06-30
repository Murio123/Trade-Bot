"""Trade journal: record delivered signals as trades and resolve their outcome.

Resolution walks forward through klines from the trade's open time and decides
whether the stop or the first target was hit first, recording win/loss and the
realised R multiple. A trade left open past an expiry is closed at market.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pandas as pd

from database import db, utcnow

log = logging.getLogger(__name__)

EXPIRY_DAYS = 21


async def record_signal_as_trade(signal_id: int, signal: dict[str, Any]) -> int | None:
    """Open a journal entry tied to a delivered signal."""
    try:
        return await db.insert_trade({
            "signal_id": signal_id,
            "direction": signal.get("direction"),
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target_1"),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("record_signal_as_trade failed: %s", exc)
        return None


def _r_multiple(entry: float, stop: float, exit_price: float, direction: str) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    if direction == "long":
        return round((exit_price - entry) / risk, 2)
    return round((entry - exit_price) / risk, 2)


def resolve_trade(trade: dict[str, Any], df: pd.DataFrame,
                  current_price: float) -> dict[str, Any] | None:
    """Return {'outcome','exit_price','pnl_r'} if resolved, else None."""
    direction = trade.get("direction")
    entry = trade.get("entry_price")
    stop = trade.get("stop_loss")
    target = trade.get("target")
    opened_at = trade.get("opened_at")
    if None in (direction, entry, stop, target):
        return None

    # Only consider candles after the trade opened.
    candles = df
    if opened_at is not None and "open_time" in df.columns:
        ts = pd.Timestamp(opened_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        candles = df[df["open_time"] >= ts]

    for _, c in candles.iterrows():
        high, low = float(c["high"]), float(c["low"])
        if direction == "long":
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            hit_stop = high >= stop
            hit_target = low <= target
        # Conservative: if a candle hits both, assume the stop was hit first.
        if hit_stop:
            return {"outcome": "loss", "exit_price": stop,
                    "pnl_r": _r_multiple(entry, stop, stop, direction)}
        if hit_target:
            return {"outcome": "win", "exit_price": target,
                    "pnl_r": _r_multiple(entry, stop, target, direction)}

    # Expiry: close stale trades at market.
    if opened_at is not None:
        opened_ts = pd.Timestamp(opened_at)
        if opened_ts.tzinfo is None:
            opened_ts = opened_ts.tz_localize("UTC")
        if utcnow() - opened_ts.to_pydatetime() > timedelta(days=EXPIRY_DAYS):
            r = _r_multiple(entry, stop, current_price, direction)
            outcome = "win" if r > 0 else "loss" if r < 0 else "breakeven"
            return {"outcome": outcome, "exit_price": current_price, "pnl_r": r}

    return None


async def get_stats(symbol: str) -> dict[str, Any]:
    return await db.journal_stats(symbol)
