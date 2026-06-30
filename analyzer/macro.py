"""Macro correlation: DXY (US dollar index) and US10Y (10-year yield).

Uses FRED if FRED_API_KEY is set, otherwise Stooq's free CSV endpoint (no key).
A falling DXY is generally a bullish backdrop for BTC.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import httpx

import config

log = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
STOOQ_BASE = "https://stooq.com/q/d/l/"

FRED_SERIES = {"dxy": "DTWEXBGS", "us10y": "DGS10"}
STOOQ_SYMBOLS = {"dxy": "dx.f", "us10y": "10usy.b"}


async def _fred_series(series_id: str) -> list[float]:
    params = {
        "series_id": series_id,
        "api_key": config.FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 30,
    }
    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
        resp = await client.get(FRED_BASE, params=params)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
    values = []
    for o in obs:
        try:
            values.append(float(o["value"]))
        except (TypeError, ValueError):
            continue
    return values[::-1]  # chronological


async def _stooq_series(symbol: str) -> list[float]:
    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
        resp = await client.get(STOOQ_BASE, params={"s": symbol, "i": "d"})
        resp.raise_for_status()
        text = resp.text
    values = []
    for line in io.StringIO(text).read().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 5:
            try:
                values.append(float(parts[4]))  # close
            except ValueError:
                continue
    return values[-30:]


async def _series(name: str) -> list[float]:
    try:
        if config.FRED_API_KEY:
            return await _fred_series(FRED_SERIES[name])
        return await _stooq_series(STOOQ_SYMBOLS[name])
    except httpx.HTTPError as exc:
        log.warning("macro series %s failed: %s", name, exc)
        return []


def _trend(values: list[float]) -> str:
    if len(values) < 5:
        return "unknown"
    recent = values[-5:]
    return "down" if recent[-1] < recent[0] else "up" if recent[-1] > recent[0] else "flat"


async def get_macro() -> dict[str, Any]:
    dxy = await _series("dxy")
    us10y = await _series("us10y")
    dxy_trend = _trend(dxy)
    us10y_trend = _trend(us10y)
    return {
        "dxy": dxy[-1] if dxy else None,
        "dxy_trend": dxy_trend,
        "us10y": us10y[-1] if us10y else None,
        "us10y_trend": us10y_trend,
        # Falling dollar = bullish context for BTC.
        "dxy_bearish_correlation": dxy_trend == "down",
        "dxy_bullish_correlation": dxy_trend == "up",
    }
