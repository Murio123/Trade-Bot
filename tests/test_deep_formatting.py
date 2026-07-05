"""Stage 14 Option B / Step 1 — presentation for render_deep sections.

Доказывает: format_deep_sections ТОЛЬКО отображает готовые значения секций —
не считает решений, не выводит LONG/SHORT/ENTER/WAIT/NO_TRADE, деградирует в
«н/д», не мутирует вход и не тянет БД/scheduler/pipeline/ai/risk. Старый
format_deep не затронут.
"""
from __future__ import annotations

import copy
import inspect

from bot import formatting
from bot.formatting import format_deep_sections
from signal_engine.deep_renderer import render_deep


def _enter_sections() -> dict:
    """Реальный ENTER-выход render_deep из доставляемого сигнала."""
    result = {
        "status": "alert",
        "analysis_status": "ENTER",
        "analysis_type": "SWING",
        "candidate_direction": "long",
        "final_bias": "LONG",
        "long_score": 74.0,
        "short_score": 31.0,
        "raw_confidence": 0.62,
        "confidence_score": 62,
        "expected_move_points": 1800.0,
        "expected_move_percent": 2.7,
        "expected_move_atr": 1.4,
        "stop_loss": 63000.0,
        "invalidation_level": 63000.0,
        "take_profit_levels": [67000.0, 69000.0],
        "risk_reward": 2.3,
        "market_regime": "trend_up",
        "volatility_regime": "expansion",
        "confirming_factors": ["BOS_up", "bullish_ob"],
        "contradicting_factors": ["rsi_overbought"],
        "cancel_conditions": ["закрытие свечи за ATR-стопом"],
        "confirmation_conditions": ["свеча закрывается в направлении сделки"],
    }
    return render_deep(result)


def _blocked_sections() -> dict:
    result = {
        "status": "blocked",
        "analysis_status": "NO_TRADE",
        "candidate_direction": "short",
        "blocked_at": "htf_filter",
        "no_trade_reasons": ["против старшего тренда", "низкая уверенность"],
    }
    return render_deep(result)


def _wait_sections() -> dict:
    result = {
        "status": "journal",
        "analysis_status": "WAIT",
        "candidate_direction": "long",
        "long_score": 55.0,
        "short_score": 40.0,
        "no_trade_reasons": ["ждём подтверждения структуры"],
    }
    return render_deep(result)


# 1. ENTER
def test_enter_formatting() -> None:
    text = format_deep_sections(_enter_sections())
    assert "Решение: ENTER" in text
    assert "Смещение: LONG" in text
    assert "SWING" in text
    assert "LONG 74" in text and "SHORT 31" in text
    assert "63 000" in text          # stop
    assert "67 000" in text          # TP1
    assert "69 000" in text          # TP2
    assert "2.3" in text             # R:R
    assert "BOS_up" in text          # supporting
    assert "rsi_overbought" in text  # contradicting


# 2. WAIT
def test_wait_formatting() -> None:
    text = format_deep_sections(_wait_sections())
    assert "Решение: WAIT" in text
    assert "ждём подтверждения структуры" in text


# 3. NO_TRADE
def test_no_trade_formatting() -> None:
    text = format_deep_sections(_blocked_sections())
    assert "Решение: NO_TRADE" in text
    assert "против старшего тренда" in text


# 4. blocked_gate
def test_blocked_gate_formatting() -> None:
    text = format_deep_sections(_blocked_sections())
    assert "htf_filter" in text
    assert "гейт" in text.lower()


# 5. None/missing -> «н/д»
def test_missing_values_degrade() -> None:
    text = format_deep_sections(render_deep({"status": "ignored"}))
    assert "н/д" in text
    # ни одного «None»/пустого места на месте значений
    assert "None" not in text


def test_partial_sections_do_not_crash() -> None:
    # Пустой dict — даже без status.
    text = format_deep_sections({})
    assert isinstance(text, str) and text
    assert "н/д" in text


# 6. Не мутирует вход
def test_does_not_mutate_input() -> None:
    sections = _enter_sections()
    before = copy.deepcopy(sections)
    format_deep_sections(sections)
    assert sections == before


# 7. Не тянет БД/scheduler/pipeline/ai/risk
def test_formatter_has_no_forbidden_dependencies() -> None:
    src = inspect.getsource(format_deep_sections)
    # Точные паттерны импорта/вызова, не подстроки (иначе «risk» ловит
    # «risk_reward»). Форматтер должен быть чистым presentation-слоем.
    for banned in ("import database", "database.", "import scheduler",
                   "scheduler.", "import pipeline", "pipeline.", "import ai",
                   "from ai", " risk.", "import risk", "compute_quality_score",
                   "run_cascade", "final_gate"):
        assert banned not in src, f"formatter must not reference {banned!r}"


# 8. Старый format_deep не изменён (по-прежнему работает и даёт свой заголовок)
def test_legacy_format_deep_unchanged() -> None:
    quality = {"decision": "NO TRADE", "overall": 0, "direction": None,
               "scores": {}, "plan": {}, "breakdown": []}
    ctx = {"price": 65000, "historical": {}, "volatility": {}, "reversal": {}}
    text = formatting.format_deep(quality, ctx, "")
    assert "ГЛУБОКИЙ АНАЛИЗ" in text
    assert "NO TRADE" in text


# 9. В форматтере нет пересчёта решения — decision берётся из sections как есть
def test_no_decision_recalculation() -> None:
    # Явно противоречивый вход: bias LONG, но decision NO_TRADE.
    # Форматтер обязан показать ровно то, что дано, ничего не «исправляя».
    sections = _enter_sections()
    sections["decision"] = "NO_TRADE"
    text = format_deep_sections(sections)
    assert "Решение: NO_TRADE" in text
    assert "Смещение: LONG" in text
