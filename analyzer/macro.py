"""Macro context: US10Y (10-year Treasury yield).

Uses FRED if FRED_API_KEY is set, otherwise Stooq's free CSV endpoint (no key).
DXY was removed from the analysis on request; US10Y is kept for context only
(it does not contribute to the confluence score).
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

FRED_SERIES = {"us10y": "DGS10"}
STOOQ_SYMBOLS = {"us10y": "10usy.b"}


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
    us10y = await _series("us10y")
    us10y_trend = _trend(us10y)
    return {
        "us10y": us10y[-1] if us10y else None,
        "us10y_trend": us10y_trend,
    }
