"""Stage 17A: pure offline "similar setups history" core.

По «текущему» сетапу находит похожие в прошлом (по атрибутам решения, а не по
ценовым feature-векторам — этим занимается отдельный analyzer.historical) и
показывает, чем они закончились: средний/медианный realized_r, лучший/худший,
разбивка win/loss, счётчики пропущенных/избегнутых и примеры прошлых сетапов.
Analytics / manual-only, НЕ runtime, НЕ auto-trading.

Полностью offline, read-only, leaf-модуль:
  * только stdlib/typing + чистые analyzer.trade_patterns и
    analyzer.journal_classify (их НЕ меняем и НЕ дублируем);
  * НЕ импортирует database, scheduler, pipeline, signal_engine, risk, bot,
    ai, tools; НЕ ходит в сеть/БД/Binance; НЕ пишет и НЕ мигрирует;
  * НЕ содержит exchange/API/order-кода; НЕ трогает scoring, final_gate,
    run_cascade, decision-path, DRY_RUN или Telegram;
  * НЕ импортируется runtime-кодом.

Сходство — объяснимое взвешенное перекрытие: совпадение категориальных
признаков (setup_type / analysis_type / direction / market_regime /
volatility_regime / setup_lifecycle_status / confidence_bucket) и
Jaccard-пересечение списков (setup_lifecycle_reasons / confirming_factors /
contradicting_factors), каждый вклад ограничен своим весом. Score нормируется
на 0..1 по признакам, которые ЕСТЬ у target, поэтому отсутствие признака у
target не штрафует кандидатов.

Честность исхода — жёсткое правило: realized_r берётся из УЖЕ измеренного
outcome (формула R не пересчитывается и не дублируется); в avg/median/best/worst
попадают только строки с числовым realized_r. unresolved / no_levels / none НЕ
считаются win/loss.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from analyzer.journal_classify import (
    AVOIDED_LOSS, BAD_TRADE, GOOD_TRADE, MISSED_OPPORTUNITY, classify,
)
from analyzer.trade_patterns import CONFIDENCE_MISSING, confidence_bucket

# Измерения сходства и их веса по умолчанию (объяснимое взвешенное перекрытие).
DEFAULT_WEIGHTS: dict[str, float] = {
    "setup_type": 2.0,
    "analysis_type": 1.5,
    "direction": 1.5,
    "market_regime": 1.0,
    "volatility_regime": 1.0,
    "confidence_bucket": 1.0,
    "setup_lifecycle_status": 1.5,
    "setup_lifecycle_reasons": 1.0,   # Jaccard, вклад ограничен весом
    "confirming_factors": 2.0,        # Jaccard, вклад ограничен весом
    "contradicting_factors": 2.0,     # Jaccard, вклад ограничен весом
}

# Категориальные измерения (точное совпадение) и списочные (Jaccard).
_CATEGORICAL = ("setup_type", "analysis_type", "direction", "market_regime",
                "volatility_regime", "setup_lifecycle_status",
                "confidence_bucket")
_JACCARD = ("setup_lifecycle_reasons", "confirming_factors",
            "contradicting_factors")

# Списки факторов, из которых собирается shared/different для примера.
_FACTOR_FIELDS = ("confirming_factors", "contradicting_factors")

# journal_classify.classify label -> человекочитаемый bucket для счётчиков.
WIN = "win"
LOSS = "loss"
MISSED = "missed"
AVOIDED = "avoided"
UNRESOLVED = "unresolved"

_LABEL_TO_BUCKET = {
    GOOD_TRADE: WIN,
    BAD_TRADE: LOSS,
    MISSED_OPPORTUNITY: MISSED,
    AVOIDED_LOSS: AVOIDED,
}


# ---------------------------------------------------------------------------
# Доступ к признакам (чистые помощники, входы не мутируются)
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _direction(row: dict[str, Any]) -> str:
    return _text(row.get("direction") or row.get("candidate_direction"))


def _categorical_value(row: dict[str, Any], dim: str) -> str:
    """Строковое значение категориального измерения (нормализованное)."""
    if dim == "direction":
        return _direction(row)
    if dim == "confidence_bucket":
        bucket = confidence_bucket(row)
        return "" if bucket == CONFIDENCE_MISSING else bucket
    return _text(row.get(dim))


def _factor_set(value: Any) -> set[str]:
    """Множество непустых строковых элементов; не-списки -> пустое множество."""
    if not isinstance(value, (list, tuple)):
        return set()
    return {_text(item) for item in value if _text(item)}


def _combined_factor_set(row: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for field in _FACTOR_FIELDS:
        result |= _factor_set(row.get(field))
    return result


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Similarity (чистые функции)
# ---------------------------------------------------------------------------

def resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Слить переданные веса поверх дефолтных (неизвестные ключи игнорируются)."""
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        for key, val in weights.items():
            if key in merged and isinstance(val, (int, float)) \
                    and not isinstance(val, bool):
                merged[key] = float(val)
    return merged


def similarity(target: dict[str, Any], candidate: dict[str, Any],
               weights: dict[str, float] | None = None) -> float:
    """Оценка сходства 0..1 по признакам, присутствующим у target.

    Чистая функция: входы не мутируются. Знаменатель — сумма весов измерений,
    которые есть у target; если сравнивать не по чему -> 0.0.
    """
    w = resolve_weights(weights)
    num = 0.0
    den = 0.0

    for dim in _CATEGORICAL:
        tv = _categorical_value(target, dim)
        if not tv:
            continue
        den += w[dim]
        if _categorical_value(candidate, dim) == tv:
            num += w[dim]

    for dim in _JACCARD:
        tset = _factor_set(target.get(dim))
        if not tset:
            continue
        den += w[dim]
        num += w[dim] * _jaccard(tset, _factor_set(candidate.get(dim)))

    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Исход строки (read-only через journal_classify; R не пересчитывается)
# ---------------------------------------------------------------------------

def _outcome(row: dict[str, Any]) -> dict[str, Any] | None:
    o = row.get("outcome")
    return o if isinstance(o, dict) else None


def _outcome_label(row: dict[str, Any]) -> str:
    return classify(row, _outcome(row))


def _realized_r(row: dict[str, Any]) -> float | None:
    """realized_r из уже измеренного outcome (только настоящее число)."""
    o = _outcome(row)
    if o is None:
        return None
    r = o.get("realized_r")
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        return None
    return float(r)


# ---------------------------------------------------------------------------
# Время (локальный парсер для детерминированной сортировки примеров)
# ---------------------------------------------------------------------------

def _anchor_epoch(row: dict[str, Any]) -> float:
    for key in ("signal_candle_close_time", "decision_time", "created_at"):
        dt = _to_datetime(row.get(key))
        if dt is not None:
            return dt.timestamp()
    return float("-inf")


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _timestamp(row: dict[str, Any]) -> str | None:
    for key in ("signal_candle_close_time", "decision_time", "created_at"):
        value = row.get(key)
        if value:
            return str(value)
    return None


# ---------------------------------------------------------------------------
# Поиск похожих сетапов
# ---------------------------------------------------------------------------

def _target_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "timestamp": _timestamp(row),
        "symbol": row.get("symbol"),
        "analysis_type": row.get("analysis_type"),
        "direction": _direction(row) or None,
        "setup_type": row.get("setup_type"),
        "market_regime": row.get("market_regime"),
        "volatility_regime": row.get("volatility_regime"),
        "confidence_bucket": confidence_bucket(row),
        "setup_lifecycle_status": row.get("setup_lifecycle_status"),
    }


def _is_same_setup(target: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Кандидат — это сам target (по идентичности объекта или по id)."""
    if candidate is target:
        return True
    tid = target.get("id")
    return tid is not None and candidate.get("id") == tid


def find_similar_setups(target: dict[str, Any], rows: list[dict[str, Any]],
                        *, limit: int = 10, min_score: float = 0.0,
                        weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Найти похожие прошлые сетапы и агрегировать их исход.

    Чистая функция над target-строкой и историей (row = forecast + опциональный
    вложенный ``outcome``). Возвращает JSON-serializable dict вида
    ``{target, summary, matches}``.

    Похожие = строки со ``similarity >= min_score`` (сам target исключается).
    summary агрегирует по всей выборке; ``matches`` показывает до ``limit``
    примеров, отсортированных по убыванию сходства, затем по времени и id.
    ``avg/median/best/worst_realized_r`` — только по строкам с числовым R;
    unresolved не считается win/loss.
    """
    w = resolve_weights(weights)
    rows = rows or []
    target_factors = _combined_factor_set(target)

    candidates = 0
    qualifying: list[dict[str, Any]] = []
    for cand in rows:
        if not isinstance(cand, dict) or _is_same_setup(target, cand):
            continue
        candidates += 1
        score = similarity(target, cand, w)
        if score < min_score:
            continue
        label = _outcome_label(cand)
        cand_factors = _combined_factor_set(cand)
        qualifying.append({
            "id": cand.get("id"),
            "timestamp": _timestamp(cand),
            "symbol": cand.get("symbol"),
            "analysis_type": cand.get("analysis_type"),
            "direction": _direction(cand) or None,
            "similarity_score": round(score, 4),
            "shared_factors": sorted(target_factors & cand_factors),
            "different_factors": sorted(target_factors ^ cand_factors),
            "realized_r": _realized_r(cand),
            "outcome_label": label,
            "setup_type": cand.get("setup_type"),
            "market_regime": cand.get("market_regime"),
            "volatility_regime": cand.get("volatility_regime"),
            "confidence_bucket": confidence_bucket(cand),
            "setup_lifecycle_status": cand.get("setup_lifecycle_status"),
            "_bucket": _LABEL_TO_BUCKET.get(label, UNRESOLVED),
            "_anchor": _anchor_epoch(cand),
        })

    qualifying.sort(
        key=lambda r: (-r["similarity_score"], -r["_anchor"],
                       -(r["id"] if isinstance(r["id"], (int, float)) else 0)))

    bucket_counts = {WIN: 0, LOSS: 0, MISSED: 0, AVOIDED: 0, UNRESOLVED: 0}
    realized: list[float] = []
    for r in qualifying:
        bucket_counts[r["_bucket"]] += 1
        if r["realized_r"] is not None:
            realized.append(r["realized_r"])

    wins, losses = bucket_counts[WIN], bucket_counts[LOSS]
    matches = [{k: v for k, v in r.items() if not k.startswith("_")}
               for r in qualifying[:max(limit, 0)]]

    return {
        "target": _target_view(target),
        "summary": {
            "candidates": candidates,
            "matches": len(qualifying),
            "with_realized_r": len(realized),
            "wins": wins,
            "losses": losses,
            "missed": bucket_counts[MISSED],
            "avoided": bucket_counts[AVOIDED],
            "unresolved": bucket_counts[UNRESOLVED],
            "avg_realized_r": (round(statistics.fmean(realized), 4)
                               if realized else None),
            "median_realized_r": (round(statistics.median(realized), 4)
                                  if realized else None),
            "best_realized_r": max(realized) if realized else None,
            "worst_realized_r": min(realized) if realized else None,
            "win_rate": (round(wins / (wins + losses), 4)
                         if (wins + losses) else None),
        },
        "matches": matches,
    }
