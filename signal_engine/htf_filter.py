"""Level 1: Higher-Timeframe Bias Filter (blocking).

Signals against the daily trend are blocked entirely — not sent, not recorded
as "weak". This is the protection against entering against the higher trend.

The filter is applied through a POLICY declared by the trade-style profile
(``htf_policy``) rather than hard-wired into the cascade. Today the only
policy is ``block_counter_trend`` — the historical behaviour — so every
existing profile keeps its exact semantics. A profile that omits the key, or
names a policy that does not exist, falls back to that same behaviour: the
gate can only ever be made stricter by accident, never looser.
"""
from __future__ import annotations

from typing import Any, Optional


def get_htf_bias(data_1d: dict[str, Any]) -> str:
    price = data_1d.get("price")
    ema20 = data_1d.get("ema20")
    ema50 = data_1d.get("ema50")
    ema200 = data_1d.get("ema200")
    if None in (price, ema20, ema50, ema200):
        return "neutral"
    if price > ema20 > ema50 > ema200:
        return "bullish"
    if price < ema20 < ema50 < ema200:
        return "bearish"
    return "neutral"


def filter_by_htf(signal_direction: Optional[str], htf_bias: str) -> Optional[str]:
    if htf_bias == "bullish" and signal_direction == "short":
        return None
    if htf_bias == "bearish" and signal_direction == "long":
        return None
    return signal_direction


# Policy name -> implementation. Each takes (direction, htf_bias, ctx) and
# returns the admitted direction, or None to block.
DEFAULT_HTF_POLICY = "block_counter_trend"


def _block_counter_trend(direction: Optional[str], htf_bias: str,
                         ctx: dict[str, Any] | None) -> Optional[str]:
    return filter_by_htf(direction, htf_bias)


HTF_POLICIES = {
    DEFAULT_HTF_POLICY: _block_counter_trend,
}


def apply_htf_policy(direction: Optional[str], htf_bias: str,
                     profile: dict[str, Any] | None = None,
                     ctx: dict[str, Any] | None = None) -> Optional[str]:
    """Apply the HTF gate declared by the profile's ``htf_policy``.

    ctx carries the market context for policies that need more than the bias
    (none do yet); it is accepted now so call sites stay stable.
    """
    name = (profile or {}).get("htf_policy") or DEFAULT_HTF_POLICY
    policy = HTF_POLICIES.get(name, _block_counter_trend)
    return policy(direction, htf_bias, ctx)
