"""Volume Profile: POC, VAH, VAL.

Buckets traded volume by price level over a lookback window and derives the
Point of Control (highest-volume price) and the Value Area (70% of volume).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_volume_profile(df: pd.DataFrame, bins: int = 50,
                           lookback: int = 200, value_area: float = 0.70) -> dict[str, Any]:
    out: dict[str, Any] = {"poc": None, "vah": None, "val": None, "bins": []}
    if len(df) < 10:
        return out

    window = df.iloc[-lookback:]
    low = float(window["low"].min())
    high = float(window["high"].max())
    if high <= low:
        return out

    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)

    # Distribute each candle's volume across the price range it covered.
    for _, row in window.iterrows():
        c_low, c_high, c_vol = row["low"], row["high"], row["volume"]
        if c_high <= c_low:
            lo_idx = np.searchsorted(edges, c_low) - 1
            lo_idx = min(max(lo_idx, 0), bins - 1)
            vol[lo_idx] += c_vol
            continue
        lo_idx = max(np.searchsorted(edges, c_low) - 1, 0)
        hi_idx = min(np.searchsorted(edges, c_high) - 1, bins - 1)
        span = hi_idx - lo_idx + 1
        vol[lo_idx:hi_idx + 1] += c_vol / span

    poc_idx = int(np.argmax(vol))
    poc = float(centers[poc_idx])

    total = vol.sum()
    target = total * value_area
    included = {poc_idx}
    acc = vol[poc_idx]
    lo, hi = poc_idx, poc_idx
    while acc < target and (lo > 0 or hi < bins - 1):
        left = vol[lo - 1] if lo > 0 else -1
        right = vol[hi + 1] if hi < bins - 1 else -1
        if right >= left:
            hi += 1
            included.add(hi)
            acc += max(vol[hi], 0)
        else:
            lo -= 1
            included.add(lo)
            acc += max(vol[lo], 0)

    vah = float(centers[max(included)])
    val = float(centers[min(included)])

    out.update({
        "poc": round(poc, 2),
        "vah": round(vah, 2),
        "val": round(val, 2),
        "bins": [{"price": round(float(c), 2), "volume": float(v)}
                 for c, v in zip(centers, vol)],
    })
    return out
