"""Map a cascade result (any status, blocked included) to a forecasts row.

Pure function: the scheduler stays a thin caller, and the mapping is unit
testable without a database or network.
"""
from __future__ import annotations

from typing import Any

import config
from signal_engine.schema import STATUS_MAP

_BIAS = {"long": "LONG", "short": "SHORT"}


def build_forecast_record(result: dict[str, Any], ctx: dict[str, Any],
                          profile: dict[str, Any]) -> dict[str, Any] | None:
    """Build the row for db.insert_forecast; None when the run carries no
    identifiable signal candle (nothing to anchor the forecast to)."""
    candle_close = (result.get("signal_candle_close_time")
                    or ctx.get("last_close_time"))
    decision_time = result.get("decision_time") or ctx.get("decision_time")
    if candle_close is None or decision_time is None:
        return None

    status = result.get("status", "blocked")
    analysis_status = (result.get("analysis_status")
                       or STATUS_MAP.get(status, "NO_TRADE"))
    direction = result.get("direction") or result.get("candidate_direction")

    blocked_gate = None
    if status == "blocked":
        blocked_gate = result.get("blocked_at") or "unknown"

    tps = result.get("take_profit_levels")
    if not tps:
        tps = [t for t in (result.get("target_1"), result.get("target_2")) if t]

    return {
        "symbol": result.get("symbol") or ctx.get("symbol"),
        "analysis_type": (result.get("analysis_type")
                          or profile.get("analysis_type", "SWING")),
        "timeframe": result.get("timeframe") or ctx.get("timeframe"),
        "signal_candle_close_time": _as_datetime(candle_close),
        "decision_time": _as_datetime(decision_time),
        "data_freshness_seconds": result.get("data_freshness_seconds"),
        "decision_latency_seconds": result.get("decision_latency_seconds"),
        "candidate_direction": direction,
        "final_bias": result.get("final_bias") or _BIAS.get(direction, "NEUTRAL"),
        "analysis_status": analysis_status,
        "blocked_gate": blocked_gate,
        "long_score": result.get("long_score"),
        "short_score": result.get("short_score"),
        "raw_confidence": result.get("raw_confidence") or result.get("confidence"),
        "calibrated_confidence": None,  # filled by a future calibration stage
        "expected_move_points": result.get("expected_move_points"),
        "expected_move_percent": result.get("expected_move_percent"),
        "expected_move_atr": result.get("expected_move_atr"),
        "entry_zone": result.get("entry_zone"),
        "signal_close_price": result.get("signal_close_price") or ctx.get("price"),
        "executable_price_at_decision": (
            result.get("executable_price_at_decision")
            or ctx.get("executable_price") or ctx.get("price")),
        "stop_loss": result.get("stop_loss"),
        "take_profit_levels": tps or None,
        "tp2_source": result.get("tp2_source"),
        "risk_reward": result.get("risk_reward"),
        "no_trade_reasons": result.get("no_trade_reasons") or None,
        "prompt_version": config.PROMPT_VERSION,
        "model_version": config.MODEL_VERSION,
    }


def _as_datetime(value: Any) -> Any:
    """pandas.Timestamp -> stdlib datetime (asyncpg-friendly); passthrough else."""
    to_py = getattr(value, "to_pydatetime", None)
    return to_py() if callable(to_py) else value
