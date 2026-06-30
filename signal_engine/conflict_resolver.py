"""Level 4: Conflict resolution.

When structure points one way but a liquidity sweep is imminent in the
opposite direction, wait for the sweep rather than entering immediately.
"""
from __future__ import annotations

from typing import Any


def resolve_conflicts(signals: dict[str, Any]) -> str:
    liq = signals.get("liquidation_map")
    ob = signals.get("order_block")
    primary = signals.get("primary_direction")

    if liq == "sweep_imminent_above" and ob == "bullish":
        return "wait_for_sweep"
    if liq == "sweep_imminent_below" and ob == "bearish":
        return "wait_for_sweep"
    return primary
