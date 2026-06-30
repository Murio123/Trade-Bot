"""Premium / Discount (equilibrium) of the dealing range.

The 50% of the recent swing range is "equilibrium". Below it is discount
(favoured for longs), above it is premium (favoured for shorts). Computed on
the higher timeframe that defines the swing's dealing range.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_equilibrium(df: pd.DataFrame, lookback: int = 50) -> dict[str, Any]:
    out: dict[str, Any] = {"high": None, "low": None, "eq": None,
                           "zone": "equilibrium", "pos": 0.5}
    if df is None or len(df) < 5:
        return out
    window = df.iloc[-lookback:]
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    price = float(df["close"].iloc[-1])
    out.update({"high": round(hi, 2), "low": round(lo, 2), "eq": round((hi + lo) / 2, 2)})
    if hi <= lo:
        return out
    pos = (price - lo) / (hi - lo)
    out["pos"] = round(pos, 2)
    if pos < 0.45:
        out["zone"] = "discount"
    elif pos > 0.55:
        out["zone"] = "premium"
    else:
        out["zone"] = "equilibrium"
    return out
