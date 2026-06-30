"""Liquidation map / heatmap.

Primary source: Coinglass free tier (if COINGLASS_API_KEY is set). Fallback:
an OI-based estimate that projects likely long/short liquidation clusters from
common leverage buckets around the current price.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

import config

log = logging.getLogger(__name__)

COINGLASS_BASE = "https://open-api-v3.coinglass.com"
# Typical retail leverage tiers -> liquidation distance from entry (~1/lev).
LEVERAGE_TIERS = [10, 25, 50, 100]


async def fetch_coinglass(symbol: str = "BTC") -> dict[str, Any] | None:
    if not config.COINGLASS_API_KEY:
        return None
    headers = {"CG-API-KEY": config.COINGLASS_API_KEY, "accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            resp = await client.get(
                f"{COINGLASS_BASE}/api/futures/liquidation/v2/heatmap",
                params={"symbol": symbol, "interval": "4h"},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        log.warning("Coinglass liquidation fetch failed: %s", exc)
        return None


def estimate_from_oi(price: float, open_interest: float,
                     long_short_ratio: float | None) -> dict[str, Any]:
    """Estimate liquidation clusters around price from leverage tiers."""
    long_clusters = []
    short_clusters = []
    ls = long_short_ratio if long_short_ratio else 1.0
    long_weight = ls / (1 + ls)
    short_weight = 1 / (1 + ls)
    for lev in LEVERAGE_TIERS:
        dist = price / lev
        # Long liquidations sit below price, short liquidations above.
        long_clusters.append({
            "price": round(price - dist, 2),
            "leverage": lev,
            "notional": round(open_interest * long_weight / len(LEVERAGE_TIERS), 2),
        })
        short_clusters.append({
            "price": round(price + dist, 2),
            "leverage": lev,
            "notional": round(open_interest * short_weight / len(LEVERAGE_TIERS), 2),
        })
    return {
        "source": "oi_estimate",
        "long_liquidations": long_clusters,
        "short_liquidations": short_clusters,
    }


async def get_liquidation_map(price: float, open_interest: float,
                              long_short_ratio: float | None,
                              symbol: str = "BTC") -> dict[str, Any]:
    cg = await fetch_coinglass(symbol)
    if cg and cg.get("data"):
        return {"source": "coinglass", "raw": cg["data"]}
    return estimate_from_oi(price, open_interest, long_short_ratio)


def nearest_sweep_signal(liq_map: dict[str, Any], price: float,
                         atr_value: float) -> str | None:
    """Detect whether a big liquidation cluster sits just above/below price.

    Returns 'sweep_imminent_above' / 'sweep_imminent_below' / None.
    """
    if liq_map.get("source") != "oi_estimate":
        return None
    threshold = atr_value * 1.5 if atr_value else price * 0.01
    above = [c for c in liq_map.get("short_liquidations", [])
             if 0 < c["price"] - price <= threshold]
    below = [c for c in liq_map.get("long_liquidations", [])
             if 0 < price - c["price"] <= threshold]
    sum_above = sum(c["notional"] for c in above)
    sum_below = sum(c["notional"] for c in below)
    if sum_above > sum_below and above:
        return "sweep_imminent_above"
    if sum_below > sum_above and below:
        return "sweep_imminent_below"
    return None
