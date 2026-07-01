"""Bybit V5 public market data — drop-in alternative to BinanceClient.

Exposes the same async method surface (klines, current_price, funding_rate,
open_interest, long_short_ratio, agg_trades) returning the same shapes, so the
signal pipeline does not care which exchange is behind it. Used as a fallback
when Binance geo-blocks the deployment region (HTTP 451).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
import pandas as pd

import config

log = logging.getLogger(__name__)

BASE = "https://api.bybit.com"
INTERVALS = {"15m": "15", "1h": "60", "4h": "240", "12h": "720", "1d": "D"}
INTERVAL_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000,
               "12h": 43_200_000, "1d": 86_400_000}


class BybitClient:
    def __init__(self, base: str = BASE, timeout: float | None = None):
        self.base = base
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
        data = resp.json()
        if data.get("retCode") != 0:
            raise httpx.HTTPError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
        return data.get("result", {})

    def _symbol(self, symbol: str | None) -> str:
        return symbol or config.SYMBOL

    # --- klines -----------------------------------------------------------
    async def klines(self, interval: str, limit: int = 300,
                     symbol: str | None = None,
                     end_time: int | None = None) -> pd.DataFrame:
        symbol = self._symbol(symbol)
        bybit_interval = INTERVALS.get(interval, interval)
        params = {"category": "linear", "symbol": symbol,
                  "interval": bybit_interval, "limit": min(limit, 1000)}
        if end_time is not None:
            params["end"] = int(end_time)
        result = await self._get("/v5/market/kline", params)
        rows = result.get("list", [])
        # Bybit returns newest-first; reverse to chronological order.
        rows = list(reversed(rows))
        cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
        df = pd.DataFrame(rows, columns=cols)
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
        dur = INTERVAL_MS.get(interval, 0)
        df["close_time"] = df["open_time"] + pd.to_timedelta(dur, unit="ms")
        return df

    async def _ticker(self, symbol: str | None = None) -> dict[str, Any]:
        symbol = self._symbol(symbol)
        result = await self._get("/v5/market/tickers",
                                 {"category": "linear", "symbol": symbol})
        lst = result.get("list", [])
        return lst[0] if lst else {}

    async def current_price(self, symbol: str | None = None) -> float:
        ticker = await self._ticker(symbol)
        return float(ticker.get("lastPrice", 0.0))

    async def funding_rate(self, symbol: str | None = None, limit: int = 30) -> dict[str, Any]:
        symbol = self._symbol(symbol)
        ticker = await self._ticker(symbol)
        current = float(ticker.get("fundingRate", 0.0) or 0.0)
        try:
            hist_res = await self._get("/v5/market/funding/history", {
                "category": "linear", "symbol": symbol, "limit": limit,
            })
            rates = [float(h["fundingRate"]) for h in hist_res.get("list", [])]
        except httpx.HTTPError as exc:
            log.warning("bybit funding history failed: %s", exc)
            rates = []
        avg = sum(rates) / len(rates) if rates else 0.0
        if len(rates) > 1:
            std = (sum((r - avg) ** 2 for r in rates) / len(rates)) ** 0.5
        else:
            std = 0.0
        z = (current - avg) / std if std else 0.0
        return {
            "current": current, "avg": avg, "std": std, "zscore": z,
            "anomalous": abs(z) >= 2.0,
            "next_funding_time": ticker.get("nextFundingTime"),
        }

    async def open_interest(self, symbol: str | None = None) -> float:
        ticker = await self._ticker(symbol)
        return float(ticker.get("openInterest", 0.0) or 0.0)

    async def open_interest_hist(self, period: str = "1h", limit: int = 30,
                                 symbol: str | None = None) -> list[dict[str, Any]]:
        symbol = self._symbol(symbol)
        try:
            result = await self._get("/v5/market/open-interest", {
                "category": "linear", "symbol": symbol,
                "intervalTime": period, "limit": limit,
            })
            return result.get("list", [])
        except httpx.HTTPError as exc:
            log.warning("bybit open-interest hist failed: %s", exc)
            return []

    async def long_short_ratio(self, period: str = "1h", limit: int = 1,
                               symbol: str | None = None) -> dict[str, Any]:
        symbol = self._symbol(symbol)
        try:
            result = await self._get("/v5/market/account-ratio", {
                "category": "linear", "symbol": symbol,
                "period": period, "limit": limit,
            })
            lst = result.get("list", [])
            if lst:
                latest = lst[0]
                buy = float(latest["buyRatio"])
                sell = float(latest["sellRatio"])
                return {
                    "long_account": buy,
                    "short_account": sell,
                    "ratio": (buy / sell) if sell else None,
                }
        except httpx.HTTPError as exc:
            log.warning("bybit long/short ratio failed: %s", exc)
        return {"long_account": None, "short_account": None, "ratio": None}

    async def agg_trades(self, limit: int = 1000,
                         symbol: str | None = None) -> list[dict[str, Any]]:
        """Recent trades mapped to Binance aggTrade shape for compute_cvd().

        Bybit `side` is the taker side: 'Buy' -> aggressive buy (positive CVD),
        'Sell' -> aggressive sell. compute_cvd reads `m` (isBuyerMaker): a sell
        means the buyer was the maker, so m=True yields negative delta.
        """
        symbol = self._symbol(symbol)
        result = await self._get("/v5/market/recent-trade", {
            "category": "linear", "symbol": symbol, "limit": min(limit, 1000),
        })
        trades = []
        for t in result.get("list", []):
            trades.append({"q": t.get("size", "0"), "m": t.get("side") == "Sell"})
        return trades
