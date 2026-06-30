"""Crypto news headlines + lightweight sentiment.

NewsAPI free tier for headlines; a small keyword lexicon scores sentiment
without external NLP dependencies. Also exposes the Fear & Greed index
(alternative.me, no key) used by the /fear command.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

import config

log = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FNG_URL = "https://api.alternative.me/fng/"

_POSITIVE = {
    "surge", "rally", "bullish", "soar", "gain", "adoption", "approval",
    "inflow", "record", "breakout", "upgrade", "institutional", "etf",
}
_NEGATIVE = {
    "crash", "plunge", "bearish", "dump", "hack", "ban", "lawsuit", "sell-off",
    "selloff", "outflow", "liquidation", "fraud", "fear", "decline", "drop",
}


def _score_headline(title: str) -> int:
    words = set(title.lower().replace(",", " ").replace(".", " ").split())
    return len(words & _POSITIVE) - len(words & _NEGATIVE)


async def get_news_sentiment(query: str = "bitcoin OR btc OR crypto",
                             limit: int = 20) -> dict[str, Any]:
    out: dict[str, Any] = {
        "headlines": [],
        "sentiment_score": 0,
        "sentiment": "neutral",
        "available": False,
    }
    if not config.NEWSAPI_KEY:
        return out
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": config.NEWSAPI_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            resp = await client.get(NEWSAPI_URL, params=params)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
    except httpx.HTTPError as exc:
        log.warning("newsapi failed: %s", exc)
        return out

    total = 0
    headlines = []
    for a in articles:
        title = a.get("title") or ""
        s = _score_headline(title)
        total += s
        headlines.append({"title": title, "score": s, "url": a.get("url")})

    out["headlines"] = headlines[:10]
    out["sentiment_score"] = total
    out["sentiment"] = "bullish" if total > 1 else "bearish" if total < -1 else "neutral"
    out["available"] = True
    return out


async def get_fear_greed() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            resp = await client.get(FNG_URL, params={"limit": 1})
            resp.raise_for_status()
            data = resp.json().get("data", [])
    except httpx.HTTPError as exc:
        log.warning("fear&greed failed: %s", exc)
        return {"value": None, "classification": None}
    if not data:
        return {"value": None, "classification": None}
    item = data[0]
    return {
        "value": int(item.get("value", 0)),
        "classification": item.get("value_classification"),
    }
