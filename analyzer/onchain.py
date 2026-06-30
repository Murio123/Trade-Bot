"""On-chain metrics via Glassnode / CryptoQuant free tier.

Exchange netflow, whale transactions, MVRV Z-Score, SOPR. All optional —
returns Nones / neutral flags when no API key is configured so the signal
engine simply doesn't award on-chain points.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

import config

log = logging.getLogger(__name__)

GLASSNODE_BASE = "https://api.glassnode.com/v1/metrics"


async def _glassnode(path: str, params: dict[str, Any] | None = None) -> Any:
    if not config.GLASSNODE_API_KEY:
        return None
    q = {"a": "BTC", "api_key": config.GLASSNODE_API_KEY, "i": "24h"}
    if params:
        q.update(params)
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            resp = await client.get(f"{GLASSNODE_BASE}/{path}", params=q)
            resp.raise_for_status()
            data = resp.json()
            return data[-1] if isinstance(data, list) and data else None
    except httpx.HTTPError as exc:
        log.warning("glassnode %s failed: %s", path, exc)
        return None


async def get_onchain() -> dict[str, Any]:
    out: dict[str, Any] = {
        "exchange_netflow": None,
        "whale_transactions": None,
        "mvrv_z": None,
        "sopr": None,
        "available": False,
    }
    if not config.GLASSNODE_API_KEY:
        return out

    netflow = await _glassnode("transactions/transfers_volume_exchanges_net")
    mvrv = await _glassnode("market/mvrv_z_score")
    sopr = await _glassnode("indicators/sopr")
    whales = await _glassnode("transactions/transfers_volume_to_exchanges_count",
                              {"i": "1h"})

    if netflow:
        out["exchange_netflow"] = netflow.get("v")
    if mvrv:
        out["mvrv_z"] = mvrv.get("v")
    if sopr:
        out["sopr"] = sopr.get("v")
    if whales:
        out["whale_transactions"] = whales.get("v")
    out["available"] = any(
        out[k] is not None for k in ("exchange_netflow", "mvrv_z", "sopr")
    )
    return out
