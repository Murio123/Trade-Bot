"""Детерминированный синтетический рынок для контрактных и golden-тестов.

Генерирует klines сидированным random-walk и собирает legacy-контекст той же
формы, что gather_market_context, но БЕЗ сети: чистые анализаторы на
синтетических свечах, внешние источники (funding/macro/onchain/liq map)
подставляются пустыми. Цены зависят только от seed; времена свечей
привязаны к реальному "сейчас" (иначе live-гейт stale_data всё заблокирует),
поэтому golden-снапшоты маскируют временные и сессионные поля.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from analyzer.cvd import compute_cvd_from_klines, cvd_series
from analyzer.divergence import detect_divergence
from analyzer.equilibrium import compute_equilibrium
from analyzer.indicators import compute_indicators
from analyzer.liquidity import detect_liquidity
from analyzer.reversal import detect_reversal
from analyzer.session_stats import compute_session_stats
from analyzer.volatility import analyze_volatility
from analyzer.volume_profile import compute_volume_profile
from pipeline import _build_htf_zones, _near_key_level, _reversal_mtf
from signal_engine.profiles import get_profile
from signal_engine.vetoes import TF_HOURS

TF_MINUTES = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "6h": 360,
              "12h": 720, "1d": 1440, "1w": 10080}


def _grid_now(tf: str) -> datetime:
    """Последнее закрытие свечи ТФ не позже текущего момента (UTC)."""
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    step = TF_MINUTES[tf]
    minutes = now.hour * 60 + now.minute
    floored = (minutes // step) * step
    return now.replace(hour=0, minute=0) + timedelta(minutes=floored)


def make_klines(tf: str, bars: int = 300, seed: int = 1, drift: float = 0.0,
                vol: float = 0.02, start_price: float = 100_000.0) -> pd.DataFrame:
    """Сидированный random-walk. drift/vol — в долях цены В СУТКИ."""
    frac = TF_MINUTES[tf] / 1440.0
    rng = np.random.default_rng([seed, TF_MINUTES[tf]])
    rets = rng.normal(drift * frac, vol * np.sqrt(frac), bars)
    closes = start_price * np.exp(np.cumsum(rets))
    opens = np.concatenate([[start_price], closes[:-1]])
    spread = np.abs(rng.normal(0.0, vol * 0.4 * np.sqrt(frac), bars))
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    volume = rng.uniform(50.0, 150.0, bars)

    end = _grid_now(tf)
    step = timedelta(minutes=TF_MINUTES[tf])
    close_times = [end - step * (bars - 1 - i) for i in range(bars)]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volume,
        "open_time": pd.to_datetime([t - step for t in close_times], utc=True),
        "close_time": pd.to_datetime(close_times, utc=True),
    })


def build_ctx(profile_name: str = "swing", seed: int = 1, drift: float = 0.0,
              vol: float = 0.02) -> dict[str, Any]:
    """Зеркало gather_market_context на синтетических свечах (без сети)."""
    profile = get_profile(profile_name)
    signal_timeframe = profile["entry"]
    needed = {"1h", "4h", "12h", "1d", signal_timeframe} | set(profile["mtf"])
    dfs = {tf: make_klines(tf, seed=seed, drift=drift, vol=vol)
           for tf in sorted(needed)}
    inds = {tf: compute_indicators(df) for tf, df in dfs.items()}

    df_signal, ind_signal = dfs[signal_timeframe], inds[signal_timeframe]
    atr_value = ind_signal.get("atr") or 0.0

    zone_tfs = [t for t in (profile.get("zone_tfs") or [signal_timeframe])
                if t in dfs] or [signal_timeframe]
    zones = _build_htf_zones(dfs, inds, zone_tfs, ind_signal["price"], df_signal)
    order_blocks, fvg, htf_levels = (zones["order_blocks"], zones["fvg"],
                                     zones["levels"])
    liquidity = detect_liquidity(df_signal, atr_value=atr_value)
    volume_profile = compute_volume_profile(dfs.get(zone_tfs[0], df_signal))
    at_level = _near_key_level(ind_signal["price"], htf_levels, order_blocks,
                               fvg, max(atr_value * 0.3,
                                        ind_signal["price"] * 0.002))
    reversal = detect_reversal(df_signal, ind_signal, cvd_series(df_signal),
                               at_key_level=at_level)
    reversal_mtf = _reversal_mtf(dfs, inds, ind_signal["price"], htf_levels,
                                 order_blocks, fvg)

    range_tf = profile["htf"] if profile["htf"] in dfs else zone_tfs[0]
    series = ind_signal.get("_series", {})
    div_rsi = detect_divergence(df_signal, series.get("rsi"))
    div_macd = detect_divergence(df_signal, series.get("macd"))

    tf_hours = TF_HOURS.get(signal_timeframe, 1.0)
    last_close = df_signal["close_time"].iloc[-1]

    return {
        "symbol": "BTCUSDT",
        "timeframe": signal_timeframe,
        "timestamp": last_close.to_pydatetime() + timedelta(seconds=60),
        "price": ind_signal["price"],
        "atr": atr_value,
        "ind_1h": inds["1h"], "ind_4h": inds["4h"], "ind_1d": inds["1d"],
        "ind_signal": ind_signal,
        "df_signal": df_signal,
        "inds_by_tf": inds,
        "funding": {}, "open_interest": 0.0, "long_short_ratio": {},
        "cvd": compute_cvd_from_klines(df_signal),
        "macro": {}, "onchain": {},
        "order_blocks": order_blocks,
        "liquidity": liquidity,
        "volume_profile": volume_profile,
        "fvg": fvg,
        "htf_levels": htf_levels,
        "zone_tfs": zone_tfs,
        "equilibrium": compute_equilibrium(dfs.get(range_tf, df_signal)),
        "reversal": reversal,
        "reversal_mtf": reversal_mtf,
        "divergence": {
            "bullish_divergence": bool(div_rsi.get("bullish_divergence")
                                       or div_macd.get("bullish_divergence")),
            "bearish_divergence": bool(div_rsi.get("bearish_divergence")
                                       or div_macd.get("bearish_divergence")),
            "rsi": div_rsi, "macd": div_macd,
        },
        "liquidation_map": None,
        "sweep_signal": None,
        "sessions": compute_session_stats(dfs["1h"]),
        "volatility": analyze_volatility(df_signal, tf_per_day=24.0 / tf_hours),
        "volatility_1d": analyze_volatility(dfs["1d"], tf_per_day=1.0),
        "last_close_time": last_close,
        # Фиксированная свежесть (120 с) — детерминизм golden-снапшотов.
        "decision_time": last_close.to_pydatetime() + timedelta(seconds=120),
        "executable_price": round(ind_signal["price"] * 1.0002, 2),
        "executable_price_degraded": False,
    }


def _synthetic_oi_hist(seed: int, drift: float, bars: int = 30) -> list[dict]:
    """Детерминированный ряд открытого интереса (сонаправлен дрейфу цены)."""
    rng = np.random.default_rng([seed, 999])
    steps = rng.normal(drift * 0.5, 0.01, bars)
    values = 1_000_000.0 * np.exp(np.cumsum(steps))
    return [{"sumOpenInterest": round(float(v), 2)} for v in values]


def build_unified_ctx(seed: int = 1, drift: float = 0.0,
                      vol: float = 0.02) -> dict[str, Any]:
    """Offline-зеркало gather_unified_swing_context (без сети).

    Собирает production-подобный legacy ctx (build_ctx) и дособирает те же
    enrichment-данные, что live-builder, через общие чистые функции
    unified_context (детерминизм golden-снапшота).
    """
    from unified_context import build_swing_enrichment, enrich_swing_context

    ctx = build_ctx("swing", seed=seed, drift=drift, vol=vol)
    dfs = {tf: make_klines(tf, seed=seed, drift=drift, vol=vol)
           for tf in ("1w", "1d", "12h", "4h", "1h")}
    oi_hist = _synthetic_oi_hist(seed, drift)
    correlation = {
        "verdict": "risk_on" if drift >= 0 else "risk_off",
        "assets": {"ETH": {"correlation": 0.8,
                           "trend": "up" if drift >= 0 else "down"}},
        "supportive": 1, "counted": 1,
    }
    enrichment = build_swing_enrichment(dfs, ctx.get("atr"), oi_hist,
                                        correlation)
    return enrich_swing_context(ctx, enrichment)
