"""Unified Context Builder для Swing (этап 2, shadow/compatibility-слой).

Собирает production-контекст без изменений (``gather_market_context``), затем
ДОсобирает только shadow/enrichment-данные и отдаёт типизированный
``UnifiedMarketContext``. Ничего в legacy-контексте не мутируется, скоринг и
Telegram-вывод не затрагиваются, production runtime этот модуль не вызывает.

Иерархия таймфреймов Swing:
    1W  — global regime;
    1D  — directional bias;
    12H — market phase;
    4H  — setup confirmation;
    1H  — execution refinement.

Enrichment-данные (все опциональные, читаются адаптером ``from_legacy`` только
при наличии соответствующих ключей):
    - structure_by_tf (BOS/CHoCH по 1W/1D/12H/4H/1H);
    - volatility_stage2 (realized_volatility, range_width_atr,
      candle_expansion_ratio, compression_score, abnormal_candle);
    - oi_change_pct / price_oi_relation / oi_rising;
    - correlation.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from analyzer.correlation import get_correlations
from analyzer.indicators import compute_indicators
from analyzer.structure import analyze_structure
from contracts import UnifiedMarketContext
from pipeline import (_drop_unclosed, _oi_rising, _safe,
                      gather_market_context)
from signal_engine.profiles import get_profile

log = logging.getLogger(__name__)

# Иерархия ТФ Swing от глобального режима к уточнению исполнения.
SWING_TF_HIERARCHY = ("1w", "1d", "12h", "4h", "1h")

# Порог «аномальной» свечи: диапазон текущей свечи относительно среднего.
_ABNORMAL_CANDLE_RATIO = 2.0
# Окно для stage-2 volatility (свечей).
_VOL_WINDOW = 20


# ---------------------------------------------------------------------------
# Публичный builder
# ---------------------------------------------------------------------------

async def gather_unified_swing_context(binance: Any) -> UnifiedMarketContext:
    """Собрать unified swing context в shadow-режиме.

    1. ``gather_market_context(profile_name="swing")`` — production-контекст,
       не мутируется;
    2. дособрать deep/enrichment-данные (1W-фрейм, structure по иерархии,
       stage-2 volatility, OI-динамика, корреляции) — с терпимостью к сбоям;
    3. вернуть ``UnifiedMarketContext.from_legacy(enriched_ctx, profile)``.
    """
    profile = get_profile("swing")
    ctx = await gather_market_context(binance, profile_name="swing")

    # Сырые фреймы иерархии (production-ctx хранит только df_signal).
    raw = await asyncio.gather(
        *[binance.klines(tf, limit=300) for tf in SWING_TF_HIERARCHY],
        return_exceptions=True)
    dfs: dict[str, Any] = {}
    for tf, df in zip(SWING_TF_HIERARCHY, raw):
        df = _safe(df, None)
        dfs[tf] = _drop_unclosed(df) if df is not None else None

    df_1d = dfs.get("1d")
    oi_hist, corr = await asyncio.gather(
        binance.open_interest_hist(period="4h", limit=30),
        get_correlations(binance, df_1d) if df_1d is not None else _none(),
        return_exceptions=True)
    oi_hist = _safe(oi_hist, [])
    corr = _safe(corr, {})

    enrichment = build_swing_enrichment(dfs, ctx.get("atr"), oi_hist, corr)
    enriched = enrich_swing_context(ctx, enrichment)
    return UnifiedMarketContext.from_legacy(enriched, profile)


async def _none() -> None:
    return None


# ---------------------------------------------------------------------------
# Чистые функции enrichment (переиспользуются offline-тестами)
# ---------------------------------------------------------------------------

def enrich_swing_context(ctx: dict[str, Any],
                         extra: dict[str, Any]) -> dict[str, Any]:
    """Вернуть НОВЫЙ ctx-словарь с enrichment-ключами, не трогая исходный.

    Legacy ctx не мутируется: делается поверхностная копия, а ``inds_by_tf``
    копируется отдельно (в него аддитивно кладётся 1W).
    """
    enriched = dict(ctx)

    inds_by_tf = dict(ctx.get("inds_by_tf") or {})
    inds_1w = extra.get("inds_1w")
    if inds_1w:
        inds_by_tf["1w"] = inds_1w
    enriched["inds_by_tf"] = inds_by_tf

    enriched["structure_by_tf"] = extra.get("structure_by_tf") or {}

    vol2 = extra.get("volatility_stage2")
    if vol2 is not None:
        enriched["volatility_stage2"] = vol2

    for key in ("oi_change_pct", "price_oi_relation", "oi_rising",
                "correlation"):
        val = extra.get(key)
        if val is not None:
            enriched[key] = val
    return enriched


def build_swing_enrichment(dfs: dict[str, Any], atr_value: float | None,
                           oi_hist: list[dict[str, Any]] | None,
                           correlation: dict[str, Any] | None) -> dict[str, Any]:
    """Посчитать enrichment из сырых фреймов иерархии и деривативов.

    Чистая: ни сети, ни мутаций. Одинаково используется live-builder'ом и
    синтетическим offline-builder'ом (детерминизм golden-снапшота).
    """
    structure_by_tf = {
        tf: analyze_structure(dfs[tf])
        for tf in SWING_TF_HIERARCHY
        if dfs.get(tf) is not None and len(dfs[tf])
    }
    df_1w = dfs.get("1w")
    inds_1w = (compute_indicators(df_1w)
               if df_1w is not None and len(df_1w) else None)

    df_entry = dfs.get("4h")
    vol2 = stage2_volatility(df_entry, atr_value)
    oi_change_pct, price_oi_relation, oi_rising = oi_enrichment(oi_hist,
                                                                df_entry)
    return {
        "structure_by_tf": structure_by_tf,
        "inds_1w": inds_1w,
        "volatility_stage2": vol2,
        "oi_change_pct": oi_change_pct,
        "price_oi_relation": price_oi_relation,
        "oi_rising": oi_rising,
        "correlation": correlation or None,
    }


# ---------------------------------------------------------------------------
# Stage-2 volatility helpers
# ---------------------------------------------------------------------------

def stage2_volatility(df: Any, atr_value: float | None) -> dict[str, Any]:
    """Детерминированные stage-2 метрики волатильности (shadow-only).

    - realized_volatility — std лог-доходностей окна (сырой RV, не годовой);
    - range_width_atr — ширина диапазона окна в единицах ATR;
    - candle_expansion_ratio — диапазон последней свечи / средний диапазон;
    - compression_score — перцентиль ширины полос Боллинджера (0..100,
      низкий = сжатие);
    - abnormal_candle — candle_expansion_ratio >= порога.
    """
    out: dict[str, Any] = {
        "realized_volatility": None, "range_width_atr": None,
        "candle_expansion_ratio": None, "compression_score": None,
        "abnormal_candle": None, "liquidation_event": None,
    }
    if df is None or len(df) < _VOL_WINDOW + 1:
        return out

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) >= _VOL_WINDOW:
        rv = float(log_ret.iloc[-_VOL_WINDOW:].std())
        out["realized_volatility"] = round(rv, 6) if not math.isnan(rv) else None

    window_high = float(high.iloc[-_VOL_WINDOW:].max())
    window_low = float(low.iloc[-_VOL_WINDOW:].min())
    if atr_value:
        out["range_width_atr"] = round((window_high - window_low) / atr_value, 4)

    candle_range = (high - low)
    mean_range = float(candle_range.iloc[-_VOL_WINDOW:].mean())
    last_range = float(candle_range.iloc[-1])
    if mean_range > 0:
        ratio = last_range / mean_range
        out["candle_expansion_ratio"] = round(ratio, 4)
        out["abnormal_candle"] = bool(ratio >= _ABNORMAL_CANDLE_RATIO)

    out["compression_score"] = _bbw_percentile(close)
    return out


def _bbw_percentile(close: pd.Series, window: int = _VOL_WINDOW) -> float | None:
    """Перцентиль текущей ширины полос Боллинджера в её собственной истории."""
    if len(close) < window + 5:
        return None
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    bbw = (2 * 2 * std) / mid  # (upper - lower) / mid, где полосы = mid ± 2*std
    bbw = bbw.dropna()
    if len(bbw) < 5:
        return None
    current = float(bbw.iloc[-1])
    rank = float((bbw <= current).mean() * 100)
    return round(rank, 1)


# ---------------------------------------------------------------------------
# OI enrichment helpers
# ---------------------------------------------------------------------------

def oi_enrichment(oi_hist: list[dict[str, Any]] | None,
                  df: Any) -> tuple[float | None, str | None, bool | None]:
    """Динамика открытого интереса и её связь с ценой.

    Возвращает (oi_change_pct, price_oi_relation, oi_rising). Отношение
    цена/OI классифицируется знаками изменений на общем окне:
      - rising_price_rising_oi   — приток новых денег (подтверждение тренда);
      - rising_price_falling_oi  — закрытие шортов (short covering);
      - falling_price_rising_oi  — набор новых шортов;
      - falling_price_falling_oi — делеверидж / ликвидация лонгов.
    """
    oi_change_pct = _oi_change_pct(oi_hist)
    rising = _oi_rising(oi_hist or [])
    relation = None
    if oi_change_pct is not None and df is not None and len(df):
        n = min(len(oi_hist), len(df)) if oi_hist else 0
        if n >= 2:
            price_first = float(df["close"].iloc[-n])
            price_last = float(df["close"].iloc[-1])
            price_up = price_last >= price_first
            oi_up = oi_change_pct >= 0
            relation = (
                f"{'rising' if price_up else 'falling'}_price_"
                f"{'rising' if oi_up else 'falling'}_oi")
    return oi_change_pct, relation, rising


def _oi_change_pct(oi_hist: list[dict[str, Any]] | None) -> float | None:
    if not oi_hist or len(oi_hist) < 2:
        return None
    first, last = _oi_val(oi_hist[0]), _oi_val(oi_hist[-1])
    if first is None or last is None or first == 0:
        return None
    return round((last - first) / first * 100, 4)


def _oi_val(item: dict[str, Any]) -> float | None:
    for key in ("sumOpenInterest", "openInterest", "sumOpenInterestValue"):
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                return None
    return None
