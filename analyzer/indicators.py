"""Standard technical indicators computed from klines.

Uses pandas-ta when available, with pure-pandas fallbacks so the bot keeps
working even if the optional dependency is missing.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import pandas_ta as ta  # type: ignore
    _HAS_TA = True
except Exception:  # pragma: no cover
    ta = None  # type: ignore
    _HAS_TA = False


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    if _HAS_TA:
        return ta.rsi(series, length=length)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(length).mean()
    sd = series.rolling(length).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bbw": width})


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    """Return the latest indicator snapshot for a klines DataFrame."""
    close = df["close"]
    out: dict[str, Any] = {}

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema100 = ema(close, 100)
    ema200 = ema(close, 200)

    rsi14 = rsi(close, 14)
    macd_df = macd(close)
    bb = bollinger(close)
    atr14 = atr(df, 14)

    avg_volume = df["volume"].rolling(20).mean()

    last = -1
    out.update({
        "price": float(close.iloc[last]),
        "ema20": _f(ema20.iloc[last]),
        "ema50": _f(ema50.iloc[last]),
        "ema100": _f(ema100.iloc[last]),
        "ema200": _f(ema200.iloc[last]),
        "rsi": _f(rsi14.iloc[last]),
        "rsi_prev": _f(rsi14.iloc[last - 1]) if len(rsi14) > 1 else None,
        "macd": _f(macd_df["macd"].iloc[last]),
        "macd_signal": _f(macd_df["signal"].iloc[last]),
        "macd_hist": _f(macd_df["hist"].iloc[last]),
        "macd_hist_prev": _f(macd_df["hist"].iloc[last - 1]) if len(macd_df) > 1 else None,
        "bb_upper": _f(bb["bb_upper"].iloc[last]),
        "bb_lower": _f(bb["bb_lower"].iloc[last]),
        "bb_mid": _f(bb["bb_mid"].iloc[last]),
        "bbw": _f(bb["bbw"].iloc[last]),
        "bbw_avg": _f(bb["bbw"].rolling(50).mean().iloc[last]),
        "atr": _f(atr14.iloc[last]),
        "volume": float(df["volume"].iloc[last]),
        "avg_volume": _f(avg_volume.iloc[last]),
    })

    # Derived booleans used by the confluence engine.
    macd_hist = macd_df["hist"]
    out["macd_bullish_cross"] = bool(
        len(macd_hist) > 1 and macd_hist.iloc[last - 1] <= 0 < macd_hist.iloc[last]
    )
    out["macd_bearish_cross"] = bool(
        len(macd_hist) > 1 and macd_hist.iloc[last - 1] >= 0 > macd_hist.iloc[last]
    )

    e20, e50, e100, e200 = out["ema20"], out["ema50"], out["ema100"], out["ema200"]
    price = out["price"]
    out["ema_aligned_bullish"] = bool(
        None not in (e20, e50, e200) and price > e20 > e50 > e200
    )
    out["ema_aligned_bearish"] = bool(
        None not in (e20, e50, e200) and price < e20 < e50 < e200
    )

    bbw = out["bbw"]
    bbw_avg = out["bbw_avg"]
    # BBW squeeze: bands compressed vs their own average -> volatility coiling.
    out["bb_squeeze"] = bool(bbw is not None and bbw_avg is not None and bbw < bbw_avg * 0.6)
    # BBW expansion: bands wider than usual -> volatility releasing.
    out["bb_expansion"] = bool(bbw is not None and bbw_avg is not None and bbw > bbw_avg * 1.3)
    # Breakout of a Bollinger band (directional).
    bb_upper = out["bb_upper"]
    bb_lower = out["bb_lower"]
    out["bb_breakout_up"] = bool(bb_upper is not None and price > bb_upper)
    out["bb_breakout_down"] = bool(bb_lower is not None and price < bb_lower)

    # Attach full series for downstream modules (divergence etc).
    out["_series"] = {
        "rsi": rsi14,
        "macd": macd_df["macd"],
        "macd_hist": macd_hist,
    }
    return out


def _f(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
