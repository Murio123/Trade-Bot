"""M1 fix: failover with 429 handling and exponential backoff."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from analyzer import exchange
from analyzer.exchange import MarketClient


def _http_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://exchange.test/klines")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


class _Backend:
    """Scripted backend: each call pops the next behaviour (value or exc)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def current_price(self, symbol=None):
        self.calls += 1
        step = self.script.pop(0) if self.script else RuntimeError("exhausted")
        if isinstance(step, Exception):
            raise step
        return step


def _client(primary: _Backend, secondary: _Backend) -> MarketClient:
    mc = MarketClient.__new__(MarketClient)  # skip real Binance/Bybit init
    mc.backends = [("bybit", primary), ("binance", secondary)]
    mc.active_name = "bybit"
    return mc


def test_429_fails_over_to_other_backend_without_sleep(monkeypatch):
    async def _no_sleep(_):
        raise AssertionError("should not back off when the other backend works")
    monkeypatch.setattr(exchange.asyncio, "sleep", _no_sleep)

    primary = _Backend([_http_error(429, {"retry-after": "7"})])
    secondary = _Backend([61000.0])
    mc = _client(primary, secondary)

    assert asyncio.run(mc.current_price()) == 61000.0
    assert mc.active_name == "binance"  # remembers the backend that worked


def test_backoff_between_rounds_honours_retry_after(monkeypatch):
    delays = []

    async def _record_sleep(seconds):
        delays.append(seconds)
    monkeypatch.setattr(exchange.asyncio, "sleep", _record_sleep)

    # Round 1: both fail (429 with retry-after 3). Round 2: primary succeeds.
    primary = _Backend([_http_error(429, {"retry-after": "3"}), 60500.0])
    secondary = _Backend([_http_error(429)])
    mc = _client(primary, secondary)

    assert asyncio.run(mc.current_price()) == 60500.0
    assert delays == [3.0]  # waited the advertised retry-after, not the 0.5s base


def test_raises_after_all_rounds_exhausted(monkeypatch):
    async def _fast_sleep(_):
        pass
    monkeypatch.setattr(exchange.asyncio, "sleep", _fast_sleep)

    primary = _Backend([RuntimeError("down")] * 3)
    secondary = _Backend([RuntimeError("down")] * 3)
    mc = _client(primary, secondary)

    with pytest.raises(RuntimeError):
        asyncio.run(mc.current_price())
    # 2 backends x 3 rounds
    assert primary.calls == 3
    assert secondary.calls == 3
