"""Reversal / exhaustion detection — catching bottoms and tops.

Combines up to nine independent signals at a swing extreme:
  - exhaustion candle (hammer / shooting star: long rejection wick)
  - volume climax (capitulation / blow-off spike at the extreme)
  - Bollinger reversion (pierce the band, close back inside)
  - RSI extreme + turn (out of 20/80) and StochRSI extreme + turn
  - CVD divergence (price new low, CVD higher low = absorption, and mirror)
  - RSI divergence (extreme not confirmed by momentum)
  - engulfing confirmation candle
  - stop-hunt sweep (pierce the prior extreme, close back beyond it)
  - at a key HTF level (amplifier, only with other factors present)

A reversal needs >= 3 confirming factors; >= 4 is "strong".
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from analyzer.divergence import detect_divergence


def detect_reversal(df: pd.DataFrame, ind: dict[str, Any],
                    cvd_series: pd.Series | None = None,
                    swing_n: int = 10, at_key_level: bool = False) -> dict[str, Any]:
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

    # 4b. Stochastic RSI extreme + turn.
    if ind.get("stochrsi_bull_turn"):
        bull.append("StochRSI разворот из перепроданности")
    if ind.get("stochrsi_bear_turn"):
        bear.append("StochRSI разворот из перекупленности")

    # 5. CVD divergence (absorption).
    if cvd_series is not None and len(cvd_series) == len(df):
        div = detect_divergence(df, cvd_series)
        if div.get("bullish_divergence"):
            bull.append("CVD-дивергенция — поглощение покупателем")
        if div.get("bearish_divergence"):
            bear.append("CVD-дивергенция — поглощение продавцом")

    # 6. RSI divergence (price extreme not confirmed by momentum).
    rsi_series = (ind.get("_series") or {}).get("rsi")
    if rsi_series is not None and len(rsi_series) == len(df):
        div = detect_divergence(df, rsi_series)
        if div.get("bullish_divergence"):
            bull.append("RSI-дивергенция — минимум без импульса")
        if div.get("bearish_divergence"):
            bear.append("RSI-дивергенция — максимум без импульса")

    # 7. Engulfing confirmation candle.
    if len(df) >= 2:
        prev = df.iloc[-2]
        po, pc = float(prev["open"]), float(prev["close"])
        if at_low and pc < po and c > o and c > po and o < pc:
            bull.append("Бычье поглощение — подтверждающая свеча")
        if at_high and pc > po and c < o and c < po and o > pc:
            bear.append("Медвежье поглощение — подтверждающая свеча")

    # 8. Stop-hunt sweep: pierce the prior extreme, close back beyond it.
    if len(df) >= swing_n + 1:
        prior = df.iloc[-(swing_n + 1):-1]
        prior_low = float(prior["low"].min())
        prior_high = float(prior["high"].max())
        if l < prior_low and c > prior_low:
            bull.append("Свип минимума — стоп-хант и возврат")
        if h > prior_high and c < prior_high:
            bear.append("Свип максимума — стоп-хант и возврат")

    # 6. Reversal at a key HTF level amplifies an existing directional read.
    if at_key_level:
        if bull:
            bull.append("У ключевого уровня (HTF)")
        if bear:
            bear.append("У ключевого уровня (HTF)")

    out["factors_bull"] = bull
    out["factors_bear"] = bear
    out["bull_score"] = len(bull)
    out["bear_score"] = len(bear)
    # With 9 possible factors the bars are higher than the original 6-factor
    # detector: 3+ to flag a reversal, 4+ to call it strong.
    out["bullish_reversal"] = len(bull) >= 3
    out["bearish_reversal"] = len(bear) >= 3
    out["bull_strong"] = len(bull) >= 4
    out["bear_strong"] = len(bear) >= 4
    return out
