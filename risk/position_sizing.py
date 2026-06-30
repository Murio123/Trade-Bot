"""ATR-based position sizing.

Stop-loss distance and position size scale with current ATR rather than a
fixed percentage. Targets are placed at risk-multiples (R) of the stop
distance.
"""
from __future__ import annotations

from typing import Any

import config


def calculate_position(entry_price: float, atr: float, direction: str = "long",
                       account_balance: float | None = None,
                       risk_percent: float | None = None,
                       atr_multiplier: float | None = None,
                       targets_r: tuple[float, float] = (1.5, 3.0)) -> dict[str, Any]:
    account_balance = account_balance if account_balance is not None else config.ACCOUNT_BALANCE
    risk_percent = risk_percent if risk_percent is not None else config.RISK_PERCENT
    atr_multiplier = atr_multiplier if atr_multiplier is not None else config.ATR_MULTIPLIER

    stop_distance = max(atr * atr_multiplier, 1e-9)
    if direction == "long":
        stop_loss = entry_price - stop_distance
        target_1 = entry_price + stop_distance * targets_r[0]
        target_2 = entry_price + stop_distance * targets_r[1]
    else:
        stop_loss = entry_price + stop_distance
        target_1 = entry_price - stop_distance * targets_r[0]
        target_2 = entry_price - stop_distance * targets_r[1]

    risk_amount = account_balance * (risk_percent / 100)
    position_size = risk_amount / stop_distance

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "position_size": round(position_size, 4),
        "risk_amount": round(risk_amount, 2),
        "stop_distance": round(stop_distance, 2),
        "atr_multiplier_used": atr_multiplier,
        "risk_percent": risk_percent,
    }
