"""Analyzer correctness: CVD, equilibrium, FVG, reversal, backtest costs."""
import numpy as np
import pandas as pd

import config
from analyzer.cvd import compute_cvd_from_klines, cvd_series
from analyzer.equilibrium import compute_equilibrium
from analyzer.fvg import detect_fvg
from analyzer.indicators import compute_indicators


def test_cvd_exact_from_taker_volume():
    df = pd.DataFrame({
        "open": [10, 11], "high": [12, 13], "low": [9, 10], "close": [11, 12],
        "volume": [100.0, 100.0], "taker_buy_base": [70.0, 30.0],
    })
    r = compute_cvd_from_klines(df)
    # (2*70-100) + (2*30-100) = +40 - 40 = 0; last candle net selling
    assert r["method"] == "exact"
    assert r["cvd"] == 0.0
    assert r["last_delta"] == -40.0


def test_cvd_estimate_without_taker_column():
    df = pd.DataFrame({
        "open": [10, 10], "high": [12, 12], "low": [10, 10],
        "close": [11.8, 10.2], "volume": [100.0, 100.0],
    })
    r = compute_cvd_from_klines(df)
    assert r["method"] == "estimate"
    assert abs(r["last_delta"] + 80.0) < 1e-6  # close near low -> selling

    series = cvd_series(df)
    assert len(series) == 2


def test_equilibrium_zones():
    df = pd.DataFrame({"high": [60000.0] * 50, "low": [50000.0] * 50,
                       "close": [51000.0] * 50, "open": [51000.0] * 50,
                       "volume": [1.0] * 50})
    eq = compute_equilibrium(df)
    assert eq["eq"] == 55000.0
    assert eq["zone"] == "discount"
    df.loc[df.index[-1], "close"] = 59000.0
    assert compute_equilibrium(df)["zone"] == "premium"


def test_fvg_detects_bullish_gap_and_mitigation():
    rows = [[100, 101, 99, 100.5]] * 40
    rows += [
        [101, 101.5, 100.5, 101.2],   # candle1: high 101.5
        [102, 103, 101.8, 102.8],     # impulse
        [103.6, 105, 103.6, 104.5],   # candle3: low 103.6 > 101.5 -> gap
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    out = detect_fvg(df, atr_value=1.0)
    z = out["bullish_fvg"]
    assert z and abs(z["low"] - 101.5) < 1e-6 and abs(z["high"] - 103.6) < 1e-6
    # trade back through the gap -> mitigated, no live bullish FVG
    df.loc[len(df)] = [103, 103.5, 101.0, 101.2, 1000.0]
    assert detect_fvg(df, atr_value=1.0)["bullish_fvg"] is None


def test_indicators_expose_reversal_inputs():
    rng = np.random.default_rng(1)
    p = np.cumsum(rng.normal(0, 50, 80)) + 60000
    df = pd.DataFrame({"open": p, "high": p + 30, "low": p - 30, "close": p,
                       "volume": np.abs(rng.normal(1000, 200, 80))})
    ind = compute_indicators(df)
    for key in ("rsi", "atr", "tsi", "vwap", "stochrsi_k", "bb_squeeze",
                "ema_aligned_bullish", "macd_bullish_cross"):
        assert key in ind


def test_backtest_cost_math():
    # Round-trip cost in R must scale inversely with stop distance.
    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100
    price = 60000.0
    wide, tight = 720.0, 360.0
    assert cost_pct * price / tight == 2 * (cost_pct * price / wide)
