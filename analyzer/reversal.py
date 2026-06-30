"""Reversal / exhaustion detection — catching bottoms and tops.

Combines several independent exhaustion signals at a swing extreme:
  - CVD divergence (price new low, CVD higher low = buyer absorption, and mirror)
  - exhaustion candle (hammer / shooting star: long rejection wick)
  - volume climax (capitulation / blow-off spike at the extreme)
  - Bollinger reversion (pierce the band, close back inside)
  - RSI extreme + turn (out of 20/80)

A bullish reversal (bottom) needs >= 2 confirming factors; >= 3 is "strong".
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from analyzer.divergence import detect_divergence


def detect_reversal(df: pd.DataFrame, ind: dict[str, Any],
                    cvd_series: pd.Series | None = None,
                    swing_n: int = 10) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bullish_reversal": False, "bearish_reversal": False,
        "bull_strong": False, "bear_strong": False,
        "bull_score": 0, "bear_score": 0,
        "factors_bull": [], "factors_bear": [],
    }
    if df is None or len(df) < swing_n + 2:
        return out

    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    recent = df.iloc[-swing_n:]
    at_low = l <= recent["low"].min() * 1.001
    at_high = h >= recent["high"].max() * 0.999

    bull: list[str] = []
    bear: list[str] = []

    # 1. Exhaustion candle (rejection wick) at the extreme.
    if at_low and lower_wick >= body * 2 and lower_wick / rng > 0.5:
        bull.append("Молот — отвержение снизу")
    if at_high and upper_wick >= body * 2 and upper_wick / rng > 0.5:
        bear.append("Shooting star — отвержение сверху")

    # 2. Volume climax at the extreme.
    vol = float(last["volume"])
    avg = ind.get("avg_volume")
    if avg and vol > avg * 2.5:
        if at_low:
            bull.append("Климакс объёма — капитуляция")
        if at_high:
            bear.append("Климакс объёма — blow-off")

    # 3. Bollinger reversion (pierce + close back inside).
    bl, bu = ind.get("bb_lower"), ind.get("bb_upper")
    if bl is not None and l < bl and c > bl:
        bull.append("Возврат внутрь нижней BB")
    if bu is not None and h > bu and c < bu:
        bear.append("Возврат внутрь верхней BB")

    # 4. RSI extreme + turn (20/80 region).
    rsi, rprev = ind.get("rsi"), ind.get("rsi_prev")
    if rsi is not None and rprev is not None:
        if rprev < 30 and rsi > rprev:
            bull.append("RSI разворот из перепроданности")
        if rprev > 70 and rsi < rprev:
            bear.append("RSI разворот из перекупленности")

    # 5. CVD divergence (absorption).
    if cvd_series is not None and len(cvd_series) == len(df):
        div = detect_divergence(df, cvd_series)
        if div.get("bullish_divergence"):
            bull.append("CVD-дивергенция — поглощение покупателем")
        if div.get("bearish_divergence"):
            bear.append("CVD-дивергенция — поглощение продавцом")

    out["factors_bull"] = bull
    out["factors_bear"] = bear
    out["bull_score"] = len(bull)
    out["bear_score"] = len(bear)
    out["bullish_reversal"] = len(bull) >= 2
    out["bearish_reversal"] = len(bear) >= 2
    out["bull_strong"] = len(bull) >= 3
    out["bear_strong"] = len(bear) >= 3
    return out
