"""Bollinger Bands as volatility/regime CONTEXT — Stage 12 (context-only).

Чистый leaf-модуль: считает Bollinger-контекст из klines для будущей аналитики
(journal, deep analysis, strategy report, возможный persist). НЕ торговое правило.

ЯВНО НЕ делает:
  * не эмитит LONG / SHORT / NEUTRAL / ENTER / WAIT / NO_TRADE;
  * не влияет на confidence, realized_r, scoring, gating, Telegram;
  * не импортирует database / tools / signal_engine / bot / ai / risk /
    contracts / pipeline / scheduler — только stdlib + numpy/pandas.

Bollinger используется как КОНТЕКСТ волатильности/режима, а не как «upper=SHORT,
lower=LONG». Поля-подсказки (squeeze / expansion / band-walk / mean-reversion
risk / regime_clue) описывают состояние волатильности; интерпретация и любые
торговые решения остаются вне этого модуля.

Все значения детерминированы и деградируют в None при нехватке истории.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Параметры полос и классификации (context-only пороги, НЕ пороги стратегии).
LENGTH = 20
STD = 2.0
WIDTH_HISTORY_MIN = 5          # минимум точек ширины для честного перцентиля
WALK_BARS = 3                  # сколько последних баров формируют band-walk
SQUEEZE_PCT = 20.0             # перцентиль ширины <= squeeze -> сжатие
EXPANSION_PCT = 80.0           # перцентиль ширины >= expansion -> расширение
WALK_UP = 0.8                  # %b >= => цена «идёт» по верхней полосе
WALK_DOWN = 0.2                # %b <= => по нижней полосе

OUTPUT_KEYS = (
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_width_percentile",
    "bb_percent_b", "bb_squeeze", "bb_expansion",
    "close_outside_upper", "close_outside_lower",
    "band_walk_direction", "mean_reversion_risk", "regime_clue",
)


def _empty() -> dict[str, Any]:
    """Graceful-деградация: все поля None (недостаточно данных)."""
    return {k: None for k in OUTPUT_KEYS}


def _percentile_rank(series: pd.Series, value: float) -> float | None:
    s = series.dropna()
    if len(s) == 0 or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float((s <= value).mean() * 100), 1)


def bollinger_context(df: Any, length: int = LENGTH, std: float = STD) -> dict[str, Any]:
    """Bollinger-контекст последней свечи (context-only). None-safe.

    Аргумент — klines DataFrame с колонкой ``close``. Возвращает dict с ключами
    OUTPUT_KEYS; при нехватке истории — все None.
    """
    if df is None or "close" not in getattr(df, "columns", []):
        return _empty()
    close = df["close"].astype(float)
    if len(close.dropna()) < length + WIDTH_HISTORY_MIN:
        return _empty()

    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)

    out = _empty()
    m, u, low = _f(mid.iloc[-1]), _f(upper.iloc[-1]), _f(lower.iloc[-1])
    w, pb = _f(width.iloc[-1]), _f(percent_b.iloc[-1])
    c = _f(close.iloc[-1])
    if None in (m, u, low) or u <= low:
        # Полосы схлопнулись (нулевая волатильность окна) — считать нечего.
        return out

    out["bb_middle"] = round(m, 2)
    out["bb_upper"] = round(u, 2)
    out["bb_lower"] = round(low, 2)
    out["bb_width"] = round(w, 6) if w is not None else None
    out["bb_percent_b"] = round(pb, 4) if pb is not None else None
    out["bb_width_percentile"] = _percentile_rank(width, w)

    wp = out["bb_width_percentile"]
    out["bb_squeeze"] = bool(wp is not None and wp <= SQUEEZE_PCT)
    out["bb_expansion"] = bool(wp is not None and wp >= EXPANSION_PCT)
    out["close_outside_upper"] = bool(c is not None and c > u)
    out["close_outside_lower"] = bool(c is not None and c < low)

    out["band_walk_direction"] = _band_walk(percent_b)
    out["mean_reversion_risk"] = _mean_reversion_risk(
        pb, out["close_outside_upper"], out["close_outside_lower"],
        out["band_walk_direction"])
    out["regime_clue"] = _regime_clue(out)
    return out


def _band_walk(percent_b: pd.Series) -> str | None:
    """Тренд «идёт» по полосе: последние WALK_BARS %b подряд у одной границы."""
    pbs = percent_b.dropna()
    if len(pbs) < WALK_BARS:
        return None
    last = pbs.iloc[-WALK_BARS:]
    if bool((last >= WALK_UP).all()):
        return "up"
    if bool((last <= WALK_DOWN).all()):
        return "down"
    return None


def _mean_reversion_risk(pb: float | None, outside_up: bool, outside_low: bool,
                         walk: str | None) -> str | None:
    """Риск возврата к средней. Высокий — цена ВНЕ полосы без band-walk-тренда
    (растяжение без импульса); при band-walk растяжение может продолжаться."""
    if pb is None:
        return None
    outside = outside_up or outside_low
    if outside and walk is None:
        return "high"
    if outside and walk is not None:
        return "medium"
    if (pb >= 0.9 or pb <= 0.1) and walk is None:
        return "medium"
    return "low"


def _regime_clue(o: dict[str, Any]) -> str:
    """Подсказка о режиме волатильности (context-only, НЕ торговый сигнал)."""
    if o["bb_squeeze"]:
        return "squeeze_coiling"
    if o["bb_expansion"] and o["band_walk_direction"]:
        return "expansion_trend_walk"
    if o["bb_expansion"]:
        return "expansion_breakout_risk"
    if o["band_walk_direction"]:
        return "trend_band_walk"
    if o["mean_reversion_risk"] == "high":
        return "range_mean_reversion"
    return "neutral_range"


def _f(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
