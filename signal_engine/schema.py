"""Structured analysis result (Part B): every signal carries the full,
verifiable picture — regime, bias, levels, both sides of the evidence,
confirmation/cancel conditions and freshness — while staying a superset of
the legacy signal dict (Execution/Risk consume the old keys untouched).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    market_regime: str
    primary_bias: str                      # LONG / SHORT / NEUTRAL
    timeframe_alignment: dict[str, Any]    # per-TF trends + agree ratio
    setup_type: str
    entry_zone: dict[str, float]           # low / high around the entry
    invalidation_level: float | None
    stop_loss: float | None
    take_profit_levels: list[float]
    expected_risk_reward: float | None
    confidence_score: int                  # 0-100
    confirming_factors: list[str]
    contradicting_factors: list[str]
    confirmation_conditions: list[str]
    cancel_conditions: list[str]
    no_trade_reasons: list[str] = field(default_factory=list)
    freshness_timestamp: str | None = None

    def to_signal_fields(self) -> dict[str, Any]:
        """New keys merged into the legacy signal dict (old keys untouched)."""
        return asdict(self)


def build_result(signal: dict[str, Any], ctx: dict[str, Any],
                 regime: str, counter_reasons: list[str],
                 mtf_info: dict[str, Any]) -> AnalysisResult:
    """Assemble the structured result from a cascade signal + context."""
    entry = signal.get("entry_price") or 0.0
    atr = signal.get("atr") or 0.0
    stop = signal.get("stop_loss")
    tp1, tp2 = signal.get("target_1"), signal.get("target_2")
    risk = abs(entry - stop) if (entry and stop) else None
    rr = round(abs(tp2 - entry) / risk, 2) if (risk and tp2) else None
    direction = signal.get("direction")
    bias = {"long": "LONG", "short": "SHORT"}.get(direction, "NEUTRAL")
    last_close = ctx.get("last_close_time")

    inv_note = ("закрытие свечи за уровнем стопа"
                if signal.get("stop_basis") == "structure"
                else "закрытие свечи за ATR-стопом")
    return AnalysisResult(
        market_regime=regime,
        primary_bias=bias,
        timeframe_alignment={"trends": signal.get("mtf", {}),
                             **(mtf_info or {})},
        setup_type=(signal.get("reasons") or ["confluence"])[0],
        entry_zone={"low": round(entry - atr * 0.25, 2),
                    "high": round(entry + atr * 0.25, 2)},
        invalidation_level=stop,
        stop_loss=stop,
        take_profit_levels=[t for t in (tp1, tp2) if t],
        expected_risk_reward=rr,
        confidence_score=int(round((signal.get("confidence") or 0) * 100)),
        confirming_factors=list(signal.get("reasons") or []),
        contradicting_factors=list(counter_reasons or []),
        confirmation_conditions=[
            f"вход в зоне {round(entry - atr * 0.25, 2)}–{round(entry + atr * 0.25, 2)}",
            "свеча entry-ТФ закрывается в направлении сделки",
        ],
        cancel_conditions=[inv_note,
                           "сценарий отменяется при смене рыночного режима"],
        freshness_timestamp=str(last_close) if last_close is not None else None,
    )
