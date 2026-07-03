"""Forecast outcome measurement: what price actually did after a forecast.

Pure, deterministic recomputation from klines — the tracking job can rerun
any number of times and produce the same row (idempotent upsert). Horizon
returns stay None until the horizon has elapsed: censoring is explicit,
unresolved forecasts are never dropped from the statistics.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

# Measured horizons (hours). 72h closes the tracking window: after it the
# outcome row is final even if neither TP nor stop was ever touched.
HORIZONS_H = (1, 4, 12, 24, 72)
FINAL_HORIZON_H = 72
REACH_POINTS = (500, 1500, 3000)


def measure_outcome(forecast: dict[str, Any], df: pd.DataFrame,
                    now: datetime, taker_fee_pct: float,
                    slippage_pct: float) -> dict[str, Any] | None:
    """Compute the outcome row for one forecast from 1H klines.

    Returns None when the measurement cannot anchor (no reference price /
    no candles after the decision yet).
    """
    anchor = _ts(forecast.get("decision_time"))
    ref = (forecast.get("executable_price_at_decision")
           or forecast.get("signal_close_price"))
    direction = forecast.get("candidate_direction")
    if anchor is None or not ref or direction not in ("long", "short"):
        return None

    # Lower bound on close_time so the candle CONTAINING the anchor is
    # included: cron decisions land at ~HH:01 while 1H bars open on the hour,
    # and excluding the spanning bar would censor return_1h forever.
    window = df[(df["close_time"] > anchor)
                & (df["open_time"] < anchor + timedelta(hours=FINAL_HORIZON_H))]
    if len(window) == 0:
        return None

    sign = 1 if direction == "long" else -1
    highs, lows = window["high"].astype(float), window["low"].astype(float)
    mfe = float((highs.max() - ref) if sign == 1 else (ref - lows.min()))
    mae = float((ref - lows.min()) if sign == 1 else (highs.max() - ref))
    mfe, mae = max(mfe, 0.0), max(mae, 0.0)

    out: dict[str, Any] = {
        "forecast_id": forecast["id"],
        "anchor_time": anchor.to_pydatetime(),
        "reference_price": float(ref),
        "mfe_points": round(mfe, 2),
        "mae_points": round(mae, 2),
    }
    for pts, col in zip(REACH_POINTS, ("reached_500", "reached_1500", "reached_3000")):
        out[col] = mfe >= pts

    # Directional % return at each ELAPSED horizon (None = censored).
    now_ts = _ts(now)
    for h in HORIZONS_H:
        col = f"return_{h}h"
        cutoff = anchor + timedelta(hours=h)
        if now_ts < cutoff:
            out[col] = None
            continue
        upto = window[window["close_time"] <= cutoff]
        if len(upto) == 0:
            out[col] = None
            continue
        price_h = float(upto["close"].iloc[-1])
        out[col] = round(sign * (price_h - ref) / ref * 100, 4)

    tp1, tp2, stop = _levels(forecast)
    tp1_hit = tp2_hit = stop_hit = False
    for _, c in window.iterrows():
        high, low = float(c["high"]), float(c["low"])
        # Conservative intra-candle order: the stop counts first, matching
        # the journal's resolution rule.
        if stop is not None and not stop_hit:
            if (low <= stop) if sign == 1 else (high >= stop):
                stop_hit = True
                break
        if tp1 is not None and not tp1_hit:
            tp1_hit = (high >= tp1) if sign == 1 else (low <= tp1)
        if tp2 is not None and not tp2_hit:
            tp2_hit = (high >= tp2) if sign == 1 else (low <= tp2)
        if tp2_hit:
            break
    out["tp1_hit"], out["tp2_hit"], out["stop_hit"] = tp1_hit, tp2_hit, stop_hit

    final_elapsed = now_ts >= anchor + timedelta(hours=FINAL_HORIZON_H)
    out["resolved"] = bool(tp2_hit or stop_hit or final_elapsed)
    r72 = out.get("return_72h")
    out["net_after_costs"] = (
        round(r72 - (2 * taker_fee_pct + slippage_pct), 4)
        if r72 is not None else None)
    return out


def _levels(forecast: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    tps = forecast.get("take_profit_levels") or []
    tp1 = float(tps[0]) if len(tps) > 0 and tps[0] else None
    tp2 = float(tps[1]) if len(tps) > 1 and tps[1] else None
    stop = forecast.get("stop_loss")
    return tp1, tp2, (float(stop) if stop else None)


def _ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts
