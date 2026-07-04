"""Correlation engine: BTC vs ETH and traditional macro assets.

ETH comes from the exchange; macro series come from Stooq's free CSV endpoint
(no key). Correlations use daily returns over a recent window. DXY lives here
(advanced/context only) and intentionally does NOT feed the core confluence
score.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import httpx
import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

STOOQ_BASE = "https://stooq.com/q/d/l/"
# Asset -> (stooq symbol, "is BTC-bullish when this rises?")
MACRO_ASSETS = {
    "DXY": ("dx.f", False),
    "US10Y": ("10yusy.b", False),  # тикер Stooq для US 10Y ("10usy.b" — 404)
    "Gold": ("xauusd", True),
    "SPX": ("^spx", True),
    "NASDAQ": ("^ndq", True),
    "VIX": ("^vix", False),
}


async def _stooq_closes(symbol: str, limit: int = 40) -> list[float]:
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            resp = await client.get(STOOQ_BASE, params={"s": symbol, "i": "d"})
            resp.raise_for_status()
            text = resp.text
    except httpx.HTTPError as exc:
        log.warning("stooq %s failed: %s", symbol, exc)
        return []
    closes = []
    for line in io.StringIO(text).read().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 5:
            try:
                closes.append(float(parts[4]))
            except ValueError:
                continue
    return closes[-limit:]


def _returns(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return np.array([])
    return np.diff(arr) / arr[:-1]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    m = min(len(a), len(b))
    if m < 5:
        return None
    a, b = a[-m:], b[-m:]
    if a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 2)


def _trend(values: list[float]) -> str:
    if len(values) < 5:
        return "flat"
    return "up" if values[-1] > values[-5] else "down" if values[-1] < values[-5] else "flat"


async def get_correlations(binance, df_1d_btc: pd.DataFrame) -> dict[str, Any]:
    btc_closes = df_1d_btc["close"].tolist()[-40:]
    btc_ret = _returns(btc_closes)

    assets: dict[str, Any] = {}
    supportive = 0
    counted = 0

    # ETH from the exchange.
    try:
        eth_df = await binance.klines("1d", limit=60, symbol="ETHUSDT")
        eth_closes = eth_df["close"].tolist()[-40:]
        assets["ETH"] = {
            "correlation": _corr(btc_ret, _returns(eth_closes)),
            "trend": _trend(eth_closes),
        }
        counted += 1
        if assets["ETH"]["trend"] == "up":
            supportive += 1
    except Exception as exc:  # noqa: BLE001
        log.warning("ETH correlation failed: %s", exc)

    # Macro assets from Stooq.
    for name, (symbol, bullish_when_up) in MACRO_ASSETS.items():
        closes = await _stooq_closes(symbol)
        if not closes:
            continue
        trend = _trend(closes)
        assets[name] = {
            "correlation": _corr(btc_ret, _returns(closes)),
            "trend": trend,
        }
        counted += 1
        if trend != "flat":
            is_support = (trend == "up") == bullish_when_up
            if is_support:
                supportive += 1

    support_ratio = (supportive / counted) if counted else 0.0
    if support_ratio >= 0.6:
        verdict = "bullish"
    elif support_ratio <= 0.35:
        verdict = "bearish"
    else:
        verdict = "neutral"

    return {
        "assets": assets,
        "supportive": supportive,
        "counted": counted,
        "support_ratio": round(support_ratio, 2),
        "verdict": verdict,
    }
