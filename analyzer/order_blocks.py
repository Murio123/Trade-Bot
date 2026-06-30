"""Order Block detection with Break-of-Structure (BOS) logic.

An order block is the last opposite-coloured candle before an impulsive move
that breaks market structure. A bullish OB is the last down-candle before a
strong rally that takes out a prior swing high; a bearish OB is the last
up-candle before a strong drop through a prior swing low.

A zone stays "live" until price returns to test it.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _swing_high(df: pd.DataFrame, i: int, left: int, right: int) -> bool:
    h = df["high"].iloc[i]
    lo = max(0, i - left)
    hi = min(len(df), i + right + 1)
    return h == df["high"].iloc[lo:hi].max()


def _swing_low(df: pd.DataFrame, i: int, left: int, right: int) -> bool:
    l = df["low"].iloc[i]
    lo = max(0, i - left)
    hi = min(len(df), i + right + 1)
    return l == df["low"].iloc[lo:hi].min()


def detect_order_blocks(df: pd.DataFrame, swing: int = 5,
                        impulse_mult: float = 1.3, max_blocks: int = 5) -> dict[str, Any]:
    """Detect recent live bullish / bearish order blocks.

    Returns the latest live OB of each type plus interaction flags with the
    current price (reaction / rejection wick).
    """
    n = len(df)
    out: dict[str, Any] = {
        "bullish_ob": None,
        "bearish_ob": None,
        "price_in_bullish_ob": False,
        "price_in_bearish_ob": False,
        "rejection_wick": False,
        "blocks": [],
    }
    if n < swing * 2 + 5:
        return out

    body = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(20).mean()
    price = float(df["close"].iloc[-1])
    last = df.iloc[-1]

    bull_blocks: list[dict[str, Any]] = []
    bear_blocks: list[dict[str, Any]] = []

    for i in range(swing, n - 2):
        impulse = body.iloc[i + 1]
        ref = avg_body.iloc[i + 1]
        if ref is None or pd.isna(ref) or impulse < ref * impulse_mult:
            continue
        candle = df.iloc[i]
        nxt = df.iloc[i + 1]

        # Bullish OB: down candle, followed by up impulse breaking prior swing high
        if candle["close"] < candle["open"] and nxt["close"] > nxt["open"]:
            prior_high = df["high"].iloc[max(0, i - swing):i].max()
            if nxt["close"] > prior_high:  # BOS up
                bull_blocks.append({
                    "type": "bullish",
                    "low": float(candle["low"]),
                    "high": float(candle["high"]),
                    "index": i,
                })

        # Bearish OB: up candle, followed by down impulse breaking prior swing low
        if candle["close"] > candle["open"] and nxt["close"] < nxt["open"]:
            prior_low = df["low"].iloc[max(0, i - swing):i].min()
            if nxt["close"] < prior_low:  # BOS down
                bear_blocks.append({
                    "type": "bearish",
                    "low": float(candle["low"]),
                    "high": float(candle["high"]),
                    "index": i,
                })

    # Keep only "live" blocks: not yet retested after creation by a close inside.
    def _live(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        live = []
        for blk in blocks:
            after = df.iloc[blk["index"] + 2:]
            tested = ((after["low"] <= blk["high"]) & (after["high"] >= blk["low"])).any()
            blk = {**blk, "tested": bool(tested)}
            live.append(blk)
        return live

    bull_blocks = _live(bull_blocks)[-max_blocks:]
    bear_blocks = _live(bear_blocks)[-max_blocks:]
    out["blocks"] = bull_blocks + bear_blocks

    if bull_blocks:
        ob = bull_blocks[-1]
        out["bullish_ob"] = ob
        in_zone = ob["low"] <= price <= ob["high"]
        out["price_in_bullish_ob"] = bool(in_zone)
        if in_zone:
            # bullish rejection wick: long lower wick on the current candle
            lower_wick = min(last["open"], last["close"]) - last["low"]
            candle_range = max(last["high"] - last["low"], 1e-9)
            out["rejection_wick"] = bool(lower_wick / candle_range > 0.4)

    if bear_blocks:
        ob = bear_blocks[-1]
        out["bearish_ob"] = ob
        in_zone = ob["low"] <= price <= ob["high"]
        out["price_in_bearish_ob"] = bool(in_zone)
        if in_zone and not out["rejection_wick"]:
            upper_wick = last["high"] - max(last["open"], last["close"])
            candle_range = max(last["high"] - last["low"], 1e-9)
            out["rejection_wick"] = bool(upper_wick / candle_range > 0.4)

    return out
