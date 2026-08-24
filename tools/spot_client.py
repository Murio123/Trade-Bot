"""S1: offline spot market-data client, pinned to one venue.

Read-only research tool. It does NOT trade, does NOT write to the database,
does NOT touch scheduler, pipeline, signal_engine or any futures logic, and is
NOT imported by runtime code.

Why this is separate from `analyzer/binance.py` rather than an extension of it.
That client is live-runtime code on the futures API (`/fapi/*`), reached
through a failover wrapper that may silently switch venues mid-run. Both
properties are wrong here:

  * spot needs `/api/v3/*`, a different API with different semantics;
  * a research panel assembled from two venues is not one panel. The C1.2
    lesson (drift point D15) was that a failover client can mix two data
    semantics inside a single dataset and nothing downstream can tell. So the
    venue is pinned here exactly as `tools/kline_cache.py` pins it, and
    failover is deliberately absent.

Keeping it in `tools/` also means S1 changes no runtime file at all.

The one non-obvious capability, and the reason the whole spot track is
feasible: **`/api/v3/klines` serves history for delisted symbols.** A pair that
no longer appears in `exchangeInfo` still answers with its full price path up
to the day it stopped trading. Survivorship control therefore does not depend
on bulk dump downloads — it depends on knowing which symbols ever existed
(`dump_symbols`), which the public data-dump index enumerates.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Iterable

import httpx

SPOT_BASE = "https://api.binance.com"
# The public index of Binance's own data dumps. Used ONLY to enumerate symbol
# names — no dump file is downloaded. It is the only source that still lists
# pairs which have been fully delisted, and it is therefore the spine of the
# survivorship-free universe.
DUMP_INDEX = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DUMP_PREFIX = "data/spot/monthly/klines/"

# Binance returns at most 1000 klines per request regardless of what is asked.
MAX_LIMIT = 1000

INTERVAL_MS: dict[str, int] = {
    "1d": 86_400_000,
    "12h": 43_200_000,
    "8h": 28_800_000,
    "4h": 14_400_000,
    "1h": 3_600_000,
}

KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume",
                 "close_time", "quote_volume", "trades",
                 "taker_buy_base", "taker_buy_quote", "ignore")


class SpotClientError(Exception):
    """The venue could not answer, or answered something unusable."""


class SpotClient:
    """Binance spot, pinned. No failover, no venue mixing, no writes."""

    def __init__(self, base: str = SPOT_BASE, timeout: float = 30.0,
                 max_retries: int = 4) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET with backoff on rate limits and transient failures.

        429 and 418 are Binance's rate-limit answers and are retried with
        growing delay rather than treated as errors; a panel of several hundred
        symbols will meet them, and giving up on one symbol would silently
        shrink the universe — the exact failure this stage exists to prevent.
        """
        client = await self._http()
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.get(url, params=params)
                if resp.status_code in (429, 418, 503, 504):
                    retry_after = float(resp.headers.get("Retry-After", delay))
                    await asyncio.sleep(min(retry_after, 60.0))
                    delay = min(delay * 2, 60.0)
                    last = SpotClientError(f"HTTP {resp.status_code} from {url}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise SpotClientError(f"{url} failed after {self.max_retries} attempts: "
                              f"{last}")

    # -- symbols ---------------------------------------------------------

    async def exchange_info(self) -> list[dict[str, Any]]:
        """Currently known symbols, with status and quote asset.

        This is the *present tense* view: it contains no symbol that has been
        fully delisted. On its own it is a survivorship trap.
        """
        data = await self._get(f"{self.base}/api/v3/exchangeInfo")
        symbols = data.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise SpotClientError("exchangeInfo returned no symbols")
        return symbols

    async def dump_symbols(self) -> list[str]:
        """Every symbol that ever had daily spot data published.

        Paginated: the index answers 1000 prefixes per page and marks itself
        truncated. Stopping at page one would return an alphabetical slice
        (everything up to roughly '1000C...') and look like a complete
        universe, so truncation is followed to the end and an unterminated
        listing raises rather than returning a short answer.
        """
        client = await self._http()
        found: list[str] = []
        marker: str | None = None
        for _ in range(64):
            params: dict[str, Any] = {"delimiter": "/", "prefix": DUMP_PREFIX}
            if marker:
                params["marker"] = marker
            resp = await client.get(DUMP_INDEX, params=params)
            resp.raise_for_status()
            body = resp.text
            page = re.findall(
                r"<Prefix>" + re.escape(DUMP_PREFIX) + r"([^/]+)/</Prefix>",
                body)
            found.extend(page)
            if not re.search(r"<IsTruncated>true</IsTruncated>", body):
                return found
            nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", body)
            if nxt:
                marker = nxt.group(1)
            elif page:
                marker = f"{DUMP_PREFIX}{page[-1]}/"
            else:
                raise SpotClientError(
                    "dump index reports truncation but returned no prefixes")
        raise SpotClientError("dump index pagination did not terminate")

    # -- klines ----------------------------------------------------------

    async def klines(self, symbol: str, interval: str, *,
                     start_ms: int | None = None,
                     end_ms: int | None = None,
                     limit: int = MAX_LIMIT) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval,
                                  "limit": min(limit, MAX_LIMIT)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        raw = await self._get(f"{self.base}/api/v3/klines", params)
        if not isinstance(raw, list):
            raise SpotClientError(f"klines for {symbol} returned {type(raw)}")
        return raw

    async def full_history(self, symbol: str, interval: str = "1d", *,
                           start_ms: int = 0,
                           max_pages: int = 200) -> list[list[Any]]:
        """Every bar from `start_ms` forward, paginated to exhaustion.

        Pagination walks forward from the first bar rather than backward from
        now, because "backward until it looks like enough" is how a dataset
        acquires a silently truncated head. There is no safety cap on depth
        beyond `max_pages`, which exists only to bound a pathological loop and
        raises rather than returning a short history.
        """
        step = INTERVAL_MS.get(interval)
        if step is None:
            raise SpotClientError(f"unsupported interval {interval!r}")
        out: list[list[Any]] = []
        cursor = int(start_ms)
        for _ in range(max_pages):
            page = await self.klines(symbol, interval, start_ms=cursor)
            if not page:
                return out
            if out and page[0][0] <= out[-1][0]:
                page = [row for row in page if row[0] > out[-1][0]]
                if not page:
                    return out
            out.extend(page)
            if len(page) < MAX_LIMIT:
                return out
            cursor = int(page[-1][0]) + step
        raise SpotClientError(
            f"{symbol} {interval}: pagination hit {max_pages} pages without "
            f"reaching the end; refusing to return a truncated history")


def usdt_pairs(symbols: Iterable[str]) -> list[str]:
    """USDT-quoted names, deduplicated and ordered."""
    return sorted({s for s in symbols if s.endswith("USDT")})
