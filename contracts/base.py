"""Базовые типы контрактов unified-движка (этап 1: только контракты).

Контракты — чистые данные: никакой аналитической логики и никаких импортов
production-модулей. Вся адаптерная логика живёт в contracts/legacy.py
(односторонняя зависимость: contracts -> production, но не наоборот).

Соглашения:
- обязательные поля адаптер заполняет всегда; их отсутствие в legacy-контексте
  — ContractError (в живом пайплайне невозможно);
- опциональное поле без источника данных = None, а имя блока/поля попадает в
  DataQualityContext.missing_blocks;
- поля класса [E] (explanation-only, см. дизайн) не должны читаться скорингом;
  программный enforcement появится вместе с analyze()/final_gate() (этап 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any


class ContractError(ValueError):
    """Обязательное поле контракта не может быть заполнено из legacy-данных."""


class VolatilityRegime(str, Enum):
    LOW_VOLATILITY = "low"
    COMPRESSION = "compression"
    NORMAL_TREND = "normal"
    HIGH_VOLATILITY = "high"
    LIQUIDATION_EVENT = "liquidation"


class ReversalEffect(str, Enum):
    """Влияние reversal-модуля на сценарий режима (никогда не сам сигнал)."""
    AMPLIFY = "amplify"
    DAMPEN = "dampen"
    BLOCK = "block"      # на этапе 1 не выдаётся (нужен MSS, этап 6)
    NEUTRAL = "neutral"


class Decision(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class BlockMeta:
    """Происхождение и свежесть контекст-блока."""
    source: str                              # откуда данные ("legacy_ctx", "binance", ...)
    as_of: datetime | None = None            # момент актуальности данных
    freshness_seconds: float | None = None   # возраст на decision_time; None = неизвестен
    degraded: bool = False                   # источник упал / использован fallback


@dataclass(frozen=True)
class PricePoint:
    """Цена с volatility-нормировкой и происхождением."""
    price: float
    provenance: str            # "liquidity"|"structural"|"htf_zone"|"range_edge"|
                               # "volume_profile"|"r_multiple_fallback"|"ema"|"ob_edge"|"atr"
    atr_distance: float | None = None   # |price - ref| / ATR
    pct_distance: float | None = None   # |price - ref| / ref * 100
    tf: str | None = None


def price_point(price: float, provenance: str, ref_price: float | None = None,
                atr: float | None = None, tf: str | None = None) -> PricePoint:
    """Фабрика PricePoint: считает нормировки, когда есть якорная цена/ATR."""
    atr_d = pct_d = None
    if ref_price:
        dist = abs(price - ref_price)
        pct_d = round(dist / ref_price * 100, 4)
        if atr:
            atr_d = round(dist / atr, 3)
    return PricePoint(price=float(price), provenance=provenance,
                      atr_distance=atr_d, pct_distance=pct_d, tf=tf)


def to_jsonable(obj: Any) -> Any:
    """Детерминированная сериализация контрактов для ledger и golden-тестов.

    dataclass -> dict (в порядке объявления полей), Enum -> value,
    datetime/pandas.Timestamp -> ISO-строка, numpy-скаляры -> питоновские,
    NaN/Inf -> None. Незнакомые объекты -> str(obj).
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Enum):
        return obj.value
    # pandas.Timestamp / любые datetime-подобные
    to_py = getattr(obj, "to_pydatetime", None)
    if callable(to_py):
        obj = to_py()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # numpy-скаляры
    item = getattr(obj, "item", None)
    if callable(item) and not isinstance(obj, (dict, list, tuple, set)):
        try:
            return to_jsonable(item())
        except (TypeError, ValueError):
            pass
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)
