"""Historical pattern recognition.

Builds a feature vector describing the current market window and finds the
closest historical analogues in BTC's own kline history, then measures what
happened next. This produces REAL forward-return statistics (no fabricated
numbers) — the quality of the estimate depends on how much history is supplied.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analyzer.indicators import atr, ema, rsi


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["ret10"] = close / close.shift(10) - 1
    feats["ret20"] = close / close.shift(20) - 1
    feats["rsi"] = rsi(close, 14)
    feats["atr_pct"] = atr(df, 14) / close * 100
    feats["d_ema50"] = (close - ema(close, 50)) / close * 100
    feats["d_ema200"] = (close - ema(close, 200)) / close * 100
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats.values, feats


def find_analogues(df: pd.DataFrame, horizon: int = 42, top_k: int = 20,
                   tf_hours: float = 4.0) -> dict[str, Any]:
    out: dict[str, Any] = {
        "matches": 0, "bullish_pct": None, "bearish_pct": None,
        "avg_return": None, "median_return": None, "best": None, "worst": None,
        "avg_holding_hours": None, "confidence": None, "top5": [],
    }
    n = len(df)
    if n < 220 + horizon:
        return out

    mat, feats = _feature_matrix(df)
    # Standardise features across history.
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd[sd == 0] = 1.0
    norm = (mat - mu) / sd

    current = norm[-1]
    if np.isnan(current).any():
        return out

    # Candidate windows: enough warmup and room for the forward horizon.
    candidates = range(200, n - horizon - 1)
    dists = []
    for j in candidates:
        v = norm[j]
        if np.isnan(v).any():
            continue
        d = float(np.sqrt(np.sum((current - v) ** 2)))
        dists.append((d, j))
    if not dists:
        return out

    dists.sort(key=lambda x: x[0])
    nearest = dists[:top_k]

    close = df["close"].values
    returns = []
    top5 = []
    for d, j in nearest:
        fwd = close[j + horizon] / close[j] - 1
        returns.append(fwd)
        if len(top5) < 5:
            top5.append({
                "time": str(df["open_time"].iloc[j].date()) if "open_time" in df else j,
                "forward_return_pct": round(fwd * 100, 2),
                "distance": round(d, 2),
            })

    arr = np.array(returns)
    out["matches"] = len(arr)
    out["bullish_pct"] = round(float((arr > 0).mean() * 100), 1)
    out["bearish_pct"] = round(float((arr <= 0).mean() * 100), 1)
    out["avg_return"] = round(float(arr.mean() * 100), 2)
    out["median_return"] = round(float(np.median(arr) * 100), 2)
    out["best"] = round(float(arr.max() * 100), 2)
    out["worst"] = round(float(arr.min() * 100), 2)
    out["avg_holding_hours"] = round(horizon * tf_hours, 1)
    # Confidence: closer average distance -> higher (0..100).
    avg_dist = float(np.mean([d for d, _ in nearest]))
    out["confidence"] = round(max(0.0, 100.0 - avg_dist * 12), 1)
    out["top5"] = top5
    return out
