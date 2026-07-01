"""Market-data client with automatic failover between exchanges.

Presents the same async surface as a single exchange client but routes each
call through an ordered list of backends. If the primary backend geo-blocks
the request (HTTP 451 / 403) or errors, it transparently retries on the next
backend and remembers the one that worked.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

import config
from analyzer.binance import BinanceClient
from analyzer.bybit import BybitClient

log = logging.getLogger(__name__)

# Errors that mean "this exchange won't serve us from here" -> try the next one.
_GEO_STATUS = {451, 403}


class MarketClient:
    def __init__(self) -> None:
        self.binance = BinanceClient()
        self.bybit = BybitClient()
        if config.EXCHANGE == "binance":
            self.backends = [("binance", self.binance), ("bybit", self.bybit)]
        elif config.EXCHANGE == "bybit":
            self.backends = [("bybit", self.bybit), ("binance", self.binance)]
        else:  # auto -> Bybit first (more reliable from cloud regions)
            self.backends = [("bybit", self.bybit), ("binance", self.binance)]
        self.active_name = self.backends[0][0]

    async def close(self) -> None:
        await self.binance.close()
        await self.bybit.close()

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # Try the last-known-good backend first, then the rest.
        ordered = sorted(self.backends, key=lambda b: b[0] != self.active_name)
        last_exc: Exception | None = None
        for name, backend in ordered:
            try:
                result = await getattr(backend, method)(*args, **kwargs)
                if name != self.active_name:
                    log.info("Market data source switched to %s", name)
                    self.active_name = name
                return result
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if status in _GEO_STATUS:
                    log.warning("%s blocked (HTTP %s) on %s -> trying next backend",
                                name, status, method)
                    continue
                log.warning("%s HTTP error on %s: %s -> trying next backend",
                            name, method, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("%s failed on %s: %s -> trying next backend",
                            name, method, exc)
                continue
        raise last_exc if last_exc else RuntimeError(f"No backend served {method}")

    # --- delegated methods (identical signatures across both clients) -----
    async def klines(self, interval: str, limit: int = 300, symbol: str | None = None,
                     end_time: int | None = None):
        return await self._call("klines", interval, limit, symbol, end_time=end_time)

    async def current_price(self, symbol: str | None = None) -> float:
        return await self._call("current_price", symbol)

    async def funding_rate(self, symbol: str | None = None, limit: int = 30) -> dict[str, Any]:
        return await self._call("funding_rate", symbol, limit)

    async def open_interest(self, symbol: str | None = None) -> float:
        return await self._call("open_interest", symbol)

    async def open_interest_hist(self, period: str = "1h", limit: int = 30,
                                 symbol: str | None = None) -> list[dict[str, Any]]:
        return await self._call("open_interest_hist", period, limit, symbol)

    async def long_short_ratio(self, period: str = "1h", limit: int = 1,
                               symbol: str | None = None) -> dict[str, Any]:
        return await self._call("long_short_ratio", period, limit, symbol)

    async def agg_trades(self, limit: int = 1000, symbol: str | None = None) -> list[dict[str, Any]]:
        return await self._call("agg_trades", limit, symbol)


def create_market_client() -> MarketClient:
    return MarketClient()
