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
