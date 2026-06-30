"""Binance USDⓈ-M Futures public market data.

Only public, read-only endpoints are used (no signing required). A read-only
API key may be supplied via env but is not necessary for these endpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
import pandas as pd

import config

log = logging.getLogger(__name__)

# Map our timeframe labels to Binance interval strings.
INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}


class BinanceClient:
    def __init__(self, base: str | None = None, timeout: float | None = None):
        self.base = base or config.BINANCE_FAPI_BASE
        self.timeout = timeout or config.HTTP_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        client = await self._http()
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    # --- klines -----------------------------------------------------------
    async def klines(self, interval: str, limit: int = 300,
                     symbol: str | None = None) -> pd.DataFrame:
        symbol = symbol or config.SYMBOL
        binance_interval = INTERVALS.get(interval, interval)
        raw = await self._get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": binance_interval, "limit": limit},
        )
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        for c in ("open", "high", "low", "close", "volume", "quote_volume",
                  "taker_buy_base", "taker_buy_quote"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    async def mark_price(self, symbol: str | None = None) -> dict[str, Any]:
        symbol = symbol or config.SYMBOL
        return await self._get("/fapi/v1/premiumIndex", {"symbol": symbol})

    async def current_price(self, symbol: str | None = None) -> float:
        symbol = symbol or config.SYMBOL
        data = await self._get("/fapi/v1/ticker/price", {"symbol": symbol})
        return float(data["price"])

    async def funding_rate(self, symbol: str | None = None, limit: int = 30) -> dict[str, Any]:
        """Current funding (from premiumIndex) + history for anomaly detection."""
        symbol = symbol or config.SYMBOL
        premium = await self.mark_price(symbol)
        history = await self._get(
            "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}
        )
        rates = [float(h["fundingRate"]) for h in history]
        avg = sum(rates) / len(rates) if rates else 0.0
        current = float(premium.get("lastFundingRate", 0.0))
        # population std-dev
        if len(rates) > 1:
            var = sum((r - avg) ** 2 for r in rates) / len(rates)
            std = var ** 0.5
        else:
            std = 0.0
        z = (current - avg) / std if std else 0.0
        return {
            "current": current,
            "avg": avg,
            "std": std,
            "zscore": z,
            "anomalous": abs(z) >= 2.0,
            "next_funding_time": premium.get("nextFundingTime"),
        }

    async def open_interest(self, symbol: str | None = None) -> float:
        symbol = symbol or config.SYMBOL
        data = await self._get("/fapi/v1/openInterest", {"symbol": symbol})
        return float(data["openInterest"])

    async def open_interest_hist(self, period: str = "1h", limit: int = 30,
                                 symbol: str | None = None) -> list[dict[str, Any]]:
        symbol = symbol or config.SYMBOL
        try:
            return await self._get(
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit},
            )
        except httpx.HTTPError as exc:  # data endpoint occasionally rate-limited
            log.warning("openInterestHist failed: %s", exc)
            return []

    async def long_short_ratio(self, period: str = "1h", limit: int = 1,
                               symbol: str | None = None) -> dict[str, Any]:
        symbol = symbol or config.SYMBOL
        try:
            data = await self._get(
                "/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": period, "limit": limit},
            )
            if data:
                latest = data[-1]
                return {
                    "long_account": float(latest["longAccount"]),
                    "short_account": float(latest["shortAccount"]),
                    "ratio": float(latest["longShortRatio"]),
                }
        except httpx.HTTPError as exc:
            log.warning("longShortRatio failed: %s", exc)
        return {"long_account": None, "short_account": None, "ratio": None}

    async def agg_trades(self, limit: int = 1000,
                         symbol: str | None = None) -> list[dict[str, Any]]:
        """Recent aggregated trades — used to compute CVD."""
        symbol = symbol or config.SYMBOL
        return await self._get(
            "/fapi/v1/aggTrades", {"symbol": symbol, "limit": limit}
        )
