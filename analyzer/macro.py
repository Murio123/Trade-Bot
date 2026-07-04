"""Macro context: US10Y (10-year Treasury yield).

Provider chain (first valid wins), each failure degrades gracefully:
1. FRED API (only when FRED_API_KEY is set) — keyed, most reliable;
2. Stooq free CSV endpoint (no key), symbol 10yusy.b;
3. FRED plain data endpoint (no key): https://fred.stlouisfed.org/data/DGS10.

US10Y is context only (it does not contribute to the confluence score), so a
total failure returns {"us10y": None, ...} and never raises.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

import httpx

import config

log = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
STOOQ_BASE = "https://stooq.com/q/d/l/"
FRED_DATA_BASE = "https://fred.stlouisfed.org/data"

FRED_SERIES = {"us10y": "DGS10"}
# Stooq ticker for the US 10Y yield is 10YUSY.B ("10usy.b" is a 404).
STOOQ_SYMBOLS = {"us10y": "10yusy.b"}

# FRED plain-data rows: "2026-07-02  4.25" (or CSV "2026-07-02,4.25");
# missing observations are ".".
_FRED_ROW = re.compile(r"^(\d{4}-\d{2}-\d{2})[,\s]+(\S+)\s*$")


async def _fred_series(series_id: str) -> list[float]:
    """FRED JSON API (needs FRED_API_KEY)."""
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
        v = _valid_float(o.get("value"))
        if v is not None:
            values.append(v)
    return values[::-1]  # chronological


async def _stooq_series(symbol: str) -> list[float]:
    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
        resp = await client.get(STOOQ_BASE, params={"s": symbol, "i": "d"})
        resp.raise_for_status()
        return _parse_stooq_csv(resp.text)


def _parse_stooq_csv(text: str) -> list[float]:
    """Stooq daily CSV: Date,Open,High,Low,Close,Volume. An HTML page
    (anti-bot challenge / error page served with 200) is invalid data."""
    if text.lstrip()[:1] == "<":
        return []
    values = []
    for line in text.splitlines()[1:]:  # skip header
        parts = line.split(",")
        if len(parts) >= 5:
            v = _valid_float(parts[4])  # close
            if v is not None:
                values.append(v)
    return values[-30:]


async def _fred_data_series(series_id: str) -> list[float]:
    """Keyless fallback: FRED plain data endpoint (text table)."""
    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT,
                                 follow_redirects=True) as client:
        resp = await client.get(f"{FRED_DATA_BASE}/{series_id}")
        resp.raise_for_status()
        return _parse_fred_data(resp.text)


def _parse_fred_data(text: str) -> list[float]:
    """Parse 'DATE VALUE' rows; skip header block, '.' placeholders and
    blank/invalid rows; keep the last valid observations (chronological)."""
    values = []
    for line in text.splitlines():
        m = _FRED_ROW.match(line.strip())
        if not m:
            continue
        v = _valid_float(m.group(2))
        if v is not None:
            values.append(v)
    return values[-30:]


def _valid_float(raw: Any) -> float | None:
    """A number, not NaN/Inf, not the '.' FRED placeholder — else None."""
    if raw is None or raw == "." or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


async def _series(name: str) -> list[float]:
    """Provider cascade; every step degrades with a concise warning
    (no stacktrace: an unavailable macro source is a routine event)."""
    if config.FRED_API_KEY:
        try:
            values = await _fred_series(FRED_SERIES[name])
            if values:
                return values
            log.warning("macro %s: FRED API returned no data, trying Stooq", name)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("macro %s: FRED API failed (%s), trying Stooq", name, exc)

    try:
        values = await _stooq_series(STOOQ_SYMBOLS[name])
        if values:
            return values
        log.warning("macro %s: Stooq returned invalid data, "
                    "falling back to FRED %s", name, FRED_SERIES[name])
    except httpx.HTTPError as exc:
        log.warning("macro %s: Stooq failed (%s), falling back to FRED %s",
                    name, exc, FRED_SERIES[name])

    try:
        values = await _fred_data_series(FRED_SERIES[name])
        if values:
            log.info("macro %s: using FRED %s fallback", name, FRED_SERIES[name])
            return values
        log.warning("macro %s: FRED fallback returned no data — degrading to None",
                    name)
    except httpx.HTTPError as exc:
        log.warning("macro %s: FRED fallback failed (%s) — degrading to None",
                    name, exc)
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
