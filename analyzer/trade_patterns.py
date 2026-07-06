"""Stage 16A: pure offline trade pattern miner (core only).

Отвечает на вопрос «какие повторяющиеся факторы систематически связаны с
плохими сделками, пропущенными сетапами, хорошими сделками и корректными
пропусками» — чтобы человек учитывал их в РУЧНОМ трейдинге. Analytics /
manual-only, НЕ runtime, НЕ auto-trading.

Полностью offline, read-only, leaf-модуль:
  * только stdlib/typing + чистый analyzer.journal_classify (его НЕ меняем и НЕ
    дублируем: метки bad/good/missed/avoided берутся из classify());
  * НЕ импортирует database, scheduler, pipeline, signal_engine, risk, bot,
    ai, tools; НЕ ходит в сеть/БД/Binance; НЕ пишет и НЕ мигрирует;
  * НЕ содержит exchange/API/order-кода; НЕ трогает scoring, final_gate,
    run_cascade, decision-path, DRY_RUN или Telegram;
  * НЕ импортируется runtime-кодом.

Честность пропусков — жёсткое правило: unresolved / no_levels / none НЕ
считаются истиной исхода и НЕ участвуют в факторной статистике. Отсутствие
данных никогда не превращается в bad/good/missed/avoided.

Конвенция строки (row): плоский forecast-dict с полями решения (как в
JSON-экспорте forecasts) и опциональным вложенным ключом ``outcome`` — уже
измеренной строкой forecast_outcomes (dict) либо None. Та же пара
(forecast, outcome), что потребляет analyzer.journal_classify.classify.
"""
from __future__ import annotations

from typing import Any

from analyzer.journal_classify import (
    AVOIDED_LOSS, BAD_TRADE, GOOD_TRADE, MISSED_OPPORTUNITY,
    NO_LEVELS, NONE, UNRESOLVED, classify,
)

# --- Внутренние bucket'ы истины исхода (стабильные строковые константы) ------
BAD = "bad"
GOOD = "good"
MISSED = "missed"
AVOIDED = "avoided"
INSUFFICIENT = "insufficient"

# journal_classify.classify -> bucket. unresolved / no_levels / none — НЕ истина.
_LABEL_TO_BUCKET = {
    BAD_TRADE: BAD,
    GOOD_TRADE: GOOD,
    MISSED_OPPORTUNITY: MISSED,
    AVOIDED_LOSS: AVOIDED,
    UNRESOLVED: INSUFFICIENT,
    NO_LEVELS: INSUFFICIENT,
    NONE: INSUFFICIENT,
}

# Bucket'ы, которые считаются достоверным исходом (участвуют в статистике).
_TRUTH_BUCKETS = (BAD, GOOD, MISSED, AVOIDED)


# ---------------------------------------------------------------------------
# Confidence bucket
# ---------------------------------------------------------------------------

CONFIDENCE_MISSING = "confidence_missing"
_CONFIDENCE_SOURCES = ("calibrated_confidence", "raw_confidence", "confidence")


def _is_number(value: Any) -> bool:
    """True только для настоящих чисел (bool исключён — он подкласс int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def confidence_bucket(row: dict[str, Any]) -> str:
    """Дискретный бакет уверенности из первого числового источника.

    Приоритет источников: calibrated_confidence -> raw_confidence ->
    confidence. Значения в диапазоне [0, 1] трактуются как доля и приводятся к
    шкале 0-100 (так 0.85 == 85). Если ни один источник не задан числом —
    ``confidence_missing`` (пропуск не выдаётся за низкую/высокую уверенность).
    """
    if not isinstance(row, dict):
        return CONFIDENCE_MISSING
    for key in _CONFIDENCE_SOURCES:
        value = row.get(key)
        if not _is_number(value):
            continue
        val = float(value)
        if 0.0 <= val <= 1.0:
            val *= 100.0
        if val < 50.0:
            return "confidence_<50"
        if val < 60.0:
            return "confidence_50_60"
        if val < 70.0:
            return "confidence_60_70"
        if val < 80.0:
            return "confidence_70_80"
        return "confidence_80_plus"
    return CONFIDENCE_MISSING


# ---------------------------------------------------------------------------
# Извлечение факторов
# ---------------------------------------------------------------------------

def _factor(dimension: str, value: Any) -> dict[str, str] | None:
    """Собрать запись фактора; None для пустых/непечатаемых значений."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return {"dimension": dimension, "value": text, "key": f"{dimension}:{text}"}


def extract_pattern_factors(row: dict[str, Any]) -> list[dict[str, str]]:
    """Достать факторы прогноза как список записей ``{dimension, value, key}``.

    Чистая функция: row НЕ мутируется. Все отсутствующие/битые поля
    обрабатываются безопасно (без исключений). В пределах одной строки
    одинаковые ключи дедуплицируются — фактор учитывается один раз на строку.

    Измерения: confirming_factors, contradicting_factors, setup_type,
    market_regime, volatility_regime, confidence_bucket,
    setup_lifecycle_status, setup_lifecycle_reasons, analysis_type, direction.
    """
    if not isinstance(row, dict):
        return []

    factors: list[dict[str, str]] = []

    # Списочные измерения: одна запись на элемент (не-списки игнорируются).
    for dimension, field in (("confirming_factors", "confirming_factors"),
                             ("contradicting_factors", "contradicting_factors"),
                             ("setup_lifecycle_reasons", "setup_lifecycle_reasons")):
        raw = row.get(field)
        if isinstance(raw, (list, tuple)):
            for item in raw:
                factors.append(_factor(dimension, item))

    # Скалярные измерения.
    factors.append(_factor("setup_type", row.get("setup_type")))
    factors.append(_factor("market_regime", row.get("market_regime")))
    factors.append(_factor("volatility_regime", row.get("volatility_regime")))
    factors.append(_factor("setup_lifecycle_status",
                           row.get("setup_lifecycle_status")))
    factors.append(_factor("analysis_type", row.get("analysis_type")))
    factors.append(_factor("direction",
                           row.get("direction") or row.get("candidate_direction")))

    # confidence_bucket derivable всегда (включая confidence_missing).
    factors.append(_factor("confidence_bucket", confidence_bucket(row)))

    # Отфильтровать None и дедуплицировать по ключу, сохраняя порядок.
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for f in factors:
        if f is None or f["key"] in seen:
            continue
        seen.add(f["key"])
        result.append(f)
    return result


# ---------------------------------------------------------------------------
# Классификация исхода строки
# ---------------------------------------------------------------------------

def _row_bucket(row: dict[str, Any]) -> str:
    """Bucket истины исхода строки через journal_classify.classify (read-only)."""
    outcome = row.get("outcome")
    if not isinstance(outcome, dict):
        outcome = None
    label = classify(row, outcome)
    return _LABEL_TO_BUCKET.get(label, INSUFFICIENT)


# ---------------------------------------------------------------------------
# Майнинг паттернов
# ---------------------------------------------------------------------------

def _rate(part: int, total: int) -> float:
    return part / total if total else 0.0


def _lift(factor_rate: float, baseline_rate: float) -> float | None:
    """Склонность фактора к исходу vs базовая частота. None при baseline == 0."""
    if baseline_rate <= 0.0:
        return None
    return factor_rate / baseline_rate


def _sort_key(p: dict[str, Any]) -> tuple:
    """Порядок: нормальные раньше low_sample; выше lift; больше total;
    затем ключ по алфавиту (детерминированно). None-lift тонет внутри группы."""
    lifts = [x for x in (p["bad_lift"], p["missed_lift"]) if x is not None]
    top_lift = max(lifts) if lifts else float("-inf")
    return (
        p["low_sample"],        # False(0) раньше True(1)
        -top_lift,              # выше lift раньше
        -p["total"],            # больше total раньше
        p["key"],               # стабильный tiebreak
    )


def mine_trade_patterns(rows: list[dict[str, Any]], *,
                        min_count: int = 3) -> dict[str, Any]:
    """Собрать факторную статистику плохих/хороших/пропущенных/избегнутых исходов.

    Чистая функция над списком строк (row = forecast + опциональный вложенный
    ``outcome``). Возвращает JSON-serializable dict со сводкой и отсортированным
    списком паттернов. Строки без достоверного исхода (insufficient) считаются
    в сводке, но НЕ участвуют в факторной статистике — отсутствие данных не
    выдаётся за исход.

    ``min_count`` — порог выборки: паттерны с total < min_count помечаются
    ``low_sample`` и опускаются в конец сортировки.
    """
    rows = rows or []

    bucket_counts = {BAD: 0, GOOD: 0, MISSED: 0, AVOIDED: 0, INSUFFICIENT: 0}
    # key -> {dimension, value, count_bad, count_good, count_missed, count_avoided}
    factor_stats: dict[str, dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            bucket_counts[INSUFFICIENT] += 1
            continue
        bucket = _row_bucket(row)
        bucket_counts[bucket] += 1
        if bucket not in _TRUTH_BUCKETS:
            continue  # unresolved / no_levels / none — не истина исхода
        for f in extract_pattern_factors(row):
            stat = factor_stats.get(f["key"])
            if stat is None:
                stat = {"dimension": f["dimension"], "value": f["value"],
                        "key": f["key"], "count_bad": 0, "count_good": 0,
                        "count_missed": 0, "count_avoided": 0}
                factor_stats[f["key"]] = stat
            stat[f"count_{bucket}"] += 1

    classified_rows = sum(bucket_counts[b] for b in _TRUTH_BUCKETS)
    baseline_bad = _rate(bucket_counts[BAD], classified_rows)
    baseline_missed = _rate(bucket_counts[MISSED], classified_rows)

    patterns: list[dict[str, Any]] = []
    for stat in factor_stats.values():
        total = (stat["count_bad"] + stat["count_good"]
                 + stat["count_missed"] + stat["count_avoided"])
        bad_rate = _rate(stat["count_bad"], total)
        missed_rate = _rate(stat["count_missed"], total)
        good_rate = _rate(stat["count_good"], total)
        avoided_rate = _rate(stat["count_avoided"], total)
        patterns.append({
            "dimension": stat["dimension"],
            "value": stat["value"],
            "key": stat["key"],
            "count_bad": stat["count_bad"],
            "count_good": stat["count_good"],
            "count_missed": stat["count_missed"],
            "count_avoided": stat["count_avoided"],
            "total": total,
            "bad_rate": bad_rate,
            "missed_rate": missed_rate,
            "good_rate": good_rate,
            "avoided_rate": avoided_rate,
            "bad_lift": _lift(bad_rate, baseline_bad),
            "missed_lift": _lift(missed_rate, baseline_missed),
            "low_sample": total < min_count,
        })

    patterns.sort(key=_sort_key)

    return {
        "summary": {
            "rows": len(rows),
            "classified_rows": classified_rows,
            "bad_trade_count": bucket_counts[BAD],
            "good_trade_count": bucket_counts[GOOD],
            "missed_opportunity_count": bucket_counts[MISSED],
            "correct_skip_count": bucket_counts[AVOIDED],
            "insufficient_data_count": bucket_counts[INSUFFICIENT],
        },
        "patterns": patterns,
    }
