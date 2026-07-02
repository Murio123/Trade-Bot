"""Quality vetoes — the "when NOT to trade" rules from the methodology.

These run after the confluence score and block otherwise-valid setups in
conditions where the edge statistically degrades:

- dead zone: price in the middle of the dealing range with no structural
  backing — chop territory, no side has the advantage;
- crowded funding: funding at a statistical extreme WITH the trade's crowd
  (going long when longs already pay heavily) — squeeze fuel points against us.
"""
from __future__ import annotations

from typing import Any

# |z-score| of the current funding vs its history to call it extreme.
FUNDING_Z_EXTREME = 2.0
# Equilibrium band treated as "middle of the range".
DEAD_ZONE_LOW, DEAD_ZONE_HIGH = 0.45, 0.55


def dead_zone(structure_score: float, eq: dict[str, Any] | None) -> bool:
    """Mid-range + zero structure points = no man's land, skip the trade."""
    if not eq or structure_score > 0:
        return False
    pos = eq.get("pos")
    if pos is None:
        return False
    return DEAD_ZONE_LOW <= pos <= DEAD_ZONE_HIGH


def reversal_trend_alignment(direction: str, htf_bias: str) -> str:
    """How a reversal relates to the HTF trend.

    'aligned'  — bottom in an uptrend / top in a downtrend: a pullback entry,
                 the highest-quality setup;
    'counter'  — fading the HTF trend: needs extra confirmation;
    'neutral'  — no clear trend.
    """
    if direction == "bull":
        if htf_bias == "bullish":
            return "aligned"
        if htf_bias == "bearish":
            return "counter"
    elif direction == "bear":
        if htf_bias == "bearish":
            return "aligned"
        if htf_bias == "bullish":
            return "counter"
    return "neutral"


def reversal_alert_min_tfs(base_min: int, alignment: str) -> int:
    """Counter-trend reversals must clear a higher multi-TF bar."""
    return base_min + 1 if alignment == "counter" else base_min


def reversal_alert_allowed(last: dict[str, Any] | None, direction: str,
                           tf_count: int, now, cooldown_hours: float) -> bool:
    """Throttle reversal alerts.

    Within the cooldown the same direction may fire again ONLY as an
    escalation — confirmation spread to MORE timeframes than the previous
    alert. Price movement alone never re-arms it (that was the spam: every
    new low re-triggered "дно" as the knife kept falling).
    """
    if last is None or last.get("direction") != direction:
        return True
    from datetime import timedelta
    if now - last["time"] >= timedelta(hours=cooldown_hours):
        return True
    return tf_count > last.get("tf_count", 0)


def crowded_funding(direction: str, funding: dict[str, Any] | None) -> bool:
    """Extreme funding with the crowd on our side -> squeeze risk against us.

    Positive funding = longs pay (crowded longs): bad time to join the longs.
    Negative funding = shorts pay (crowded shorts): bad time to join the shorts.
    """
    if not funding:
        return False
    current = funding.get("current")
    z = funding.get("zscore")
    if current is None or z is None or abs(z) < FUNDING_Z_EXTREME:
        return False
    if direction == "long" and current > 0:
        return True
    if direction == "short" and current < 0:
        return True
    return False
