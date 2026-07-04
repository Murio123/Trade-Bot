"""Контракты решений: SwingAnalysisResult, SynthesisResult, FinalDecision.

Ключевой принцип — разделение направления и качества сделки:
direction_probability / setup_quality / execution_quality /
calibrated_confidence / contradiction_score — пять независимых полей.
Допустимый исход: direction=LONG, вероятность высокая, decision=WAIT.

На этапе 1 количественные поля качества заполняются ПРОВИЗОРНЫМИ
детерминированными формулами из чисел legacy-каскада (см. contracts/legacy.py);
они помечены как shadow-only и не участвуют в production-решении.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.base import Decision, PricePoint
from contracts.context import MetaContext


@dataclass(frozen=True)
class Scenario:
    """Торговый сценарий с provenance каждой цены."""
    direction: str                             # "long"|"short"
    entry_zone: tuple[float, float] | None     # <- build_result.entry_zone (low, high)
    stop: PricePoint | None                    # <- stop_loss + stop_basis
    tp1: PricePoint | None                     # <- target_1 (+ targets_structure)
    tp2: PricePoint | None                     # <- target_2 + tp2_source
    risk_reward: float | None                  # <- risk_reward (по TP2)
    expected_move_points: float | None
    expected_move_atr: float | None
    setup_type: str | None = None              # "continuation"|"reversal"|"range"|None
                                               # (этап 1: эвристика; полноценно — этап 8)


@dataclass(frozen=True)
class SwingAnalysisResult:
    """Выход analyze(context, mode) — детерминированный, до AI и до гейта."""
    mode: str                                  # <- profile["analysis_type"]
    direction: str | None                      # <- signal["direction"] / candidate_direction
    direction_probability: float | None       # провизорно: score_dir / (long+short)
    setup_quality: float | None               # провизорно: min(weighted_total / 10, 1)
    execution_quality: float | None           # провизорно: f(freshness, displacement/ATR)
    contradiction_score: float | None         # провизорно: counter / (total + counter)
    evidence: dict[str, Any]                  # category_scores + long/short totals
    reasons: list[str]                        # <- reasons (confirming)
    counter_reasons: list[str]                # <- contradicting_factors
    mtf_alignment: dict[str, Any]             # <- mtf + mtf_agreement
    regime_hierarchy: dict[str, str | None]   # {"1w": None, "1d": regime, ...}; 1W/фазы — этап 8
    scenario: Scenario | None = None
    alternative: Scenario | None = None       # этап 8: пока None


@dataclass(frozen=True)
class SynthesisResult:
    """Выход AI-synthesis: строго структурный, чисел и цен не меняет.

    AI может только УЖЕСТОЧИТЬ решение (recommendation_to_wait: ENTER->WAIT),
    никогда не ослабить и не обойти final gate.
    """
    primary_scenario: str                      # нарратив (legacy: ai_text)
    confirmations: list[str]
    contradictions: list[str]
    concise_reasons: list[str]                 # <= 3
    alternative_scenario: str | None = None
    main_risk: str | None = None
    event_risk: str | None = None
    news_risk: str | None = None
    entry_condition: str | None = None
    invalidation: str | None = None            # текст, НЕ цена
    recommendation_to_wait: bool | None = None
    model_version: str | None = None
    degraded: bool = False                     # AI недоступен -> детерминированный fallback
    # legacy ai_confidence сознательно отброшен: числовая уверенность
    # остаётся детерминированной функцией доказательств.


@dataclass(frozen=True)
class GateCheck:
    name: str                                  # "stale_data"|"htf_filter"|"rr"|"event_veto"|...
    passed: bool
    detail: str | None = None


@dataclass(frozen=True)
class FinalDecision:
    """Единственный публикуемый результат; всё остальное — его представления."""
    decision: Decision                         # <- status: alert->ENTER_*, journal/cooldown->WAIT,
                                               #    ignored/blocked->NO_TRADE
    mode: str
    direction: str | None
    meta: MetaContext                          # snapshot воспроизводимости (engine="legacy_v1" у адаптера)
    scenario: Scenario | None
    direction_probability: float | None
    setup_quality: float | None
    execution_quality: float | None
    contradiction_score: float | None
    gates: list[GateCheck]                     # след гейта; полный 25-гейт трейс — этап 3
    vetoes_triggered: list[str]                # имена непройденных гейтов
    reasons: list[str]                         # <= 3, <- reasons[:3]
    calibrated_confidence: float | None = None # этап 10: пока None
    main_risk: str | None = None               # <- synthesis (этап 1: None)
    invalidation_note: str | None = None       # <- cancel_conditions[0]
    warnings: list[str] = field(default_factory=list)  # freshness/degraded/event/news
    forecast_id: int | None = None             # связь с forecasts ledger

    def to_json(self) -> dict[str, Any]:
        from contracts.base import to_jsonable
        return to_jsonable(self)
