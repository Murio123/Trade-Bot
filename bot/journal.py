"""Trade journal: record delivered signals and manage their lifecycle.

Lifecycle (single position, stop management — no partial sizing):
    open  -> price hits TP1  -> stop moved to breakeven (stage = tp1)
    tp1   -> price hits TP2  -> closed win
    tp1   -> price falls back to breakeven stop -> closed breakeven (0R)
    open  -> price hits original stop -> closed loss (-1R)

Resolution replays klines from the trade's open time each run, so it is
stateless except for the stored ``stage`` which is used to emit each milestone
notification exactly once. Stale trades expire at market after 21 days.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pandas as pd

import config
from database import db, utcnow

log = logging.getLogger(__name__)

EXPIRY_DAYS = 21
_STAGE_RANK = {"open": 0, "tp1": 1, "closed": 2}


async def can_open_new_trade(symbol: str) -> bool:
    """Aggregate risk gate: the three profiles trade the same symbol, and
    each open trade risks RISK_PERCENT — cap their number across profiles.

    Rows with symbol NULL predate the column and are counted against the cap.
    """
    try:
        open_now = await db.open_trades()
    except Exception as exc:  # noqa: BLE001
        log.warning("can_open_new_trade failed (%s) — allowing", exc)
        return True
    same = [t for t in open_now if t.get("symbol") in (None, symbol)]
    return len(same) < config.MAX_OPEN_TRADES


async def has_active_trade(symbol: str, analysis_type: str | None,
                           timeframe: str | None = None) -> bool:
    """Delivery guard: an unresolved journal trade for the same symbol and
    profile means the setup is already taken — a new alert would only add a
    duplicate manual trade idea before the first resolves.

    Pre-migration rows with analysis_type NULL are matched by timeframe
    instead. Fail-open: a DB error must never silence real alerts.
    """
    try:
        open_now = await db.open_trades()
    except Exception as exc:  # noqa: BLE001
        log.warning("has_active_trade failed (%s) — not suppressing", exc)
        return False
    for t in open_now:
        if t.get("symbol") not in (None, symbol):
            continue
        t_type = t.get("analysis_type")
        if t_type is not None:
            if t_type == analysis_type:
                return True
        elif timeframe is not None and t.get("timeframe") == timeframe:
            return True
    return False


async def record_signal_as_trade(signal_id: int, signal: dict[str, Any]) -> int | None:
    # Re-check right before the insert: the scheduler and manual /signal run
    # their guards independently, and both could observe "no active trade"
    # before either records one. Not fully atomic, but it shrinks the race
    # window to this single call without a schema migration.
    if await has_active_trade(signal.get("symbol") or config.SYMBOL,
                              signal.get("analysis_type"),
                              signal.get("timeframe")):
        log.info("signal alert suppressed: active signal already open "
                 "(journal insert skipped for signal #%s)", signal_id)
        return None
    try:
        return await db.insert_trade({
            "signal_id": signal_id,
            "direction": signal.get("direction"),
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target_1"),
            "tp1": signal.get("target_1"),
            "tp2": signal.get("target_2"),
            "timeframe": signal.get("timeframe"),
            "symbol": signal.get("symbol"),
            "analysis_type": signal.get("analysis_type"),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("record_signal_as_trade failed: %s", exc)
        return None


def pick_frame(trade: dict[str, Any], frames: dict[str, Any]):
    """Choose the klines frame to replay a trade on.

    Prefer the trade's own timeframe, but only if that history still covers
    the trade's open time (15m frames span ~10 days); otherwise fall back to
    1H, which reaches ~41 days back.
    """
    tf = trade.get("timeframe") or "1h"
    df = frames.get(tf)
    fallback = frames.get("1h")
    if df is None or len(df) == 0:
        return fallback
    opened_at = trade.get("opened_at")
    if opened_at is not None and "open_time" in df.columns:
        ts = pd.Timestamp(opened_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if df["open_time"].iloc[0] > ts and fallback is not None:
            return fallback
    return df


def _r(entry: float, stop: float, exit_price: float, direction: str) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    sign = 1 if direction == "long" else -1
    return round(sign * (exit_price - entry) / risk, 2)


def evaluate_trade(trade: dict[str, Any], df: pd.DataFrame,
                   current_price: float) -> dict[str, Any] | None:
    """Replay the trade and return the new events/state, or None if unchanged."""
    direction = trade.get("direction")
    entry = trade.get("entry_price")
    stop = trade.get("stop_loss")
    tp1 = trade.get("tp1") or trade.get("target")
    tp2 = trade.get("tp2")
    opened_at = trade.get("opened_at")
    stored_stage = trade.get("stage") or "open"
    if None in (direction, entry, stop, tp1):
        return None

    candles = df
    if opened_at is not None and "open_time" in df.columns:
        ts = pd.Timestamp(opened_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        candles = df[df["open_time"] >= ts]

    long = direction == "long"
    hit_tp1 = False
    cur_stop = stop
    closed: dict[str, Any] | None = None

    for _, c in candles.iterrows():
        high, low = float(c["high"]), float(c["low"])
        if not hit_tp1:
            stop_hit = (low <= stop) if long else (high >= stop)
            tp1_hit = (high >= tp1) if long else (low <= tp1)
            if stop_hit:  # conservative: stop before target within a candle
                closed = {"outcome": "loss", "exit_price": stop,
                          "pnl_r": _r(entry, stop, stop, direction)}
                break
            if tp1_hit:
                hit_tp1 = True
                cur_stop = entry  # move to breakeven
                if tp2 is None:   # single-target signal -> TP1 closes the trade
                    closed = {"outcome": "win", "exit_price": tp1,
                              "pnl_r": _r(entry, stop, tp1, direction)}
                    break
                # Intra-candle order is unknown: the breakeven stop takes
                # effect from the NEXT candle. Otherwise the very candle that
                # paid +1R at TP1 would close the trade at 0R whenever its low
                # also touched the entry (typical for 1H/4H bars).
                continue
        if hit_tp1 and tp2 is not None:
            be_hit = (low <= cur_stop) if long else (high >= cur_stop)
            tp2_hit = (high >= tp2) if long else (low <= tp2)
            if be_hit:
                closed = {"outcome": "breakeven", "exit_price": cur_stop, "pnl_r": 0.0}
                break
            if tp2_hit:
                closed = {"outcome": "win", "exit_price": tp2,
                          "pnl_r": _r(entry, stop, tp2, direction)}
                break

    # Expiry for trades that never resolved.
    if closed is None and opened_at is not None:
        opened_ts = pd.Timestamp(opened_at)
        if opened_ts.tzinfo is None:
            opened_ts = opened_ts.tz_localize("UTC")
        if utcnow() - opened_ts.to_pydatetime() > timedelta(days=EXPIRY_DAYS):
            r = _r(entry, stop, current_price, direction)
            closed = {"outcome": "win" if r > 0 else "loss" if r < 0 else "breakeven",
                      "exit_price": current_price, "pnl_r": r, "expired": True}

    new_stage = "closed" if closed else ("tp1" if hit_tp1 else "open")
    if _STAGE_RANK[new_stage] <= _STAGE_RANK.get(stored_stage, 0):
        return None  # nothing new since last check

    events = []
    if hit_tp1 and _STAGE_RANK.get(stored_stage, 0) < 1:
        events.append({"type": "tp1", "stop": entry})
    if closed:
        events.append({"type": "closed", **closed})

    return {
        "new_stage": new_stage,
        "current_stop": cur_stop,
        "closed": closed,
        "events": events,
    }


async def get_stats(symbol: str) -> dict[str, Any]:
    return await db.journal_stats(symbol)
