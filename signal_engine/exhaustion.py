"""Exhaustion predicates for counter-trend (bounce / mean-reversion) setups.

Pure leaf-module: reads the market context and answers ONE question — does the
tape show enough exhaustion to justify trading AGAINST the higher-timeframe
trend? Nothing here scores, sizes, or decides; the only consumer is the
``require_exhaustion`` HTF policy in :mod:`signal_engine.htf_filter`.

The bar is deliberately high. A counter-trend entry is the lowest-quality
setup in the methodology, so every mandatory condition must hold AND at least
``BOUNCE_MIN_OPTIONAL_CONFIRMATIONS`` of the corroborating ones:

mandatory (bullish; the bearish set is the exact mirror)
  1. exhaustion confirmed on >= BOUNCE_MIN_REVERSAL_TFS timeframes;
  2. the entry timeframe's own reversal is STRONG (>= 4 of its 9 factors);
  3. the last closed 1H candle already turned up (no knife-catching);
  4. price is in DISCOUNT — never buy a bounce in premium;
  5. liquidity swept below AND reclaimed (sweep + reversal candle);
  6. funding is not crowded WITH us (no squeeze fuel pointing our way).

optional (>= BOUNCE_MIN_OPTIONAL_CONFIRMATIONS of)
  bullish RSI/MACD divergence; funding stretched against the crowd we join;
  price reacting inside a bullish OB/FVG; CVD on the buyers' side.

FAIL CLOSED: a missing, malformed or partial context yields ``False`` — and
that holds for NESTED values too, not just the top-level shape. A non-numeric
``bull_tf_count`` or ``funding.zscore`` is a symptom of a corrupted data
source, so it blocks rather than being coerced into a number. This module
never raises: an exception here would surface as a blocked cascade at best and
an admitted counter-trend trade at worst.
"""
from __future__ import annotations

import math
from typing import Any

import config
from signal_engine.vetoes import crowded_funding

# Context keys backing the MANDATORY conditions. Absence is not "no evidence",
# it is "we cannot tell" — and we do not fade a trend on what we cannot tell.
REQUIRED_CTX_KEYS = ("reversal", "reversal_mtf", "equilibrium", "liquidity",
                     "funding")


def _sub(ctx: dict[str, Any], key: str) -> dict[str, Any]:
    """Sub-dict of ctx, or {} when absent/malformed (optional sources only)."""
    value = ctx.get(key)
    return value if isinstance(value, dict) else {}


def safe_float(value: Any) -> float | None:
    """A real, finite number, or None. STRICT: never coerces.

    A numeric string is not a number here — ``"2"`` is a symptom of a broken
    data source, and silently parsing it would let corrupted context flow into
    a counter-trend admission. bool is rejected too: ``True >= 2`` is legal
    Python and nonsense as a timeframe count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _usable(ctx: Any) -> bool:
    """Every mandatory source present and dict-shaped."""
    if not isinstance(ctx, dict):
        return False
    return all(isinstance(ctx.get(k), dict) for k in REQUIRED_CTX_KEYS)


def _funding_numbers(funding: dict[str, Any]) -> tuple[bool, float | None, float | None]:
    """Validate the funding pair before anything compares or abs()-es it.

    Returns (usable, current, zscore). A key that is ABSENT degrades to None —
    that is ordinary (``pipeline`` substitutes ``{}`` when the exchange call
    fails) and simply means "no funding evidence". A key that is PRESENT but
    not a finite number is malformed, and the whole predicate must fail closed
    rather than hand a string to ``crowded_funding``'s ``abs()``.
    """
    numbers: list[float | None] = []
    for key in ("current", "zscore"):
        if key not in funding or funding[key] is None:
            numbers.append(None)
            continue
        number = safe_float(funding[key])
        if number is None:
            return False, None, None
        numbers.append(number)
    return True, numbers[0], numbers[1]


def _funding_stretched(current: float | None, zscore: float | None,
                       bullish: bool) -> bool:
    """Funding at a contrarian extreme: the crowd we FADE is paying.

    Bullish bounce wants crowded shorts (negative funding); mirror for bearish.
    Distinct from the crowded_funding veto, which fires at |z| >= 2.0. Both
    arguments are pre-validated numbers or None.
    """
    if current is None or zscore is None:
        return False
    if bullish:
        return current < 0 and zscore <= -config.FUNDING_Z_SIGNAL
    return current > 0 and zscore >= config.FUNDING_Z_SIGNAL


def bullish_exhaustion(ctx: Any) -> tuple[bool, list[str]]:
    """Is a counter-trend LONG justified in a bearish HTF regime?

    Returns (verdict, reasons_that_fired). The reasons are diagnostic only —
    they explain an admitted setup, they never feed scoring.
    """
    if not _usable(ctx):
        return False, []

    rev = ctx["reversal"]
    rev_mtf = ctx["reversal_mtf"]
    liq = ctx["liquidity"]

    funding_ok, current, zscore = _funding_numbers(ctx["funding"])
    if not funding_ok:
        return False, []
    funding = {"current": current, "zscore": zscore}

    tf_count = safe_float(rev_mtf.get("bull_tf_count"))
    mandatory = (
        (tf_count is not None and tf_count >= config.BOUNCE_MIN_REVERSAL_TFS,
         f"Разворот подтверждён на {tf_count:.0f} ТФ" if tf_count else ""),
        (bool(rev.get("bull_strong")),
         "Сильный разворотный сетап — признаки дна"),
        (bool(rev_mtf.get("bull_candle_confirm")),
         "1H свеча подтвердила разворот вверх"),
        (ctx["equilibrium"].get("zone") == "discount",
         "Цена в дисконте (ниже равновесия)"),
        (bool(liq.get("liquidity_swept_below") and liq.get("reversal_candle")),
         "Снятие ликвидности снизу + возврат"),
        (not crowded_funding("long", funding),
         "Funding не перегрет в лонг"),
    )
    if not all(ok for ok, _ in mandatory):
        return False, []

    optional = (
        (bool(_sub(ctx, "divergence").get("bullish_divergence")),
         "Бычья дивергенция RSI/MACD"),
        (_funding_stretched(current, zscore, bullish=True),
         "Funding заметно отрицательный — перегрет шорт"),
        (bool(_sub(ctx, "order_blocks").get("price_in_bullish_ob")
              or _sub(ctx, "fvg").get("price_in_bullish_fvg")),
         "Реакция в бычьей зоне (OB/FVG)"),
        (bool(_sub(ctx, "cvd").get("cvd_bullish")),
         "CVD на стороне покупателей"),
    )
    fired = [reason for ok, reason in optional if ok]
    if len(fired) < config.BOUNCE_MIN_OPTIONAL_CONFIRMATIONS:
        return False, []

    return True, [reason for _, reason in mandatory] + fired


def bearish_exhaustion(ctx: Any) -> tuple[bool, list[str]]:
    """Is a counter-trend SHORT justified in a bullish HTF regime?

    Exact mirror of :func:`bullish_exhaustion`.
    """
    if not _usable(ctx):
        return False, []

    rev = ctx["reversal"]
    rev_mtf = ctx["reversal_mtf"]
    liq = ctx["liquidity"]

    funding_ok, current, zscore = _funding_numbers(ctx["funding"])
    if not funding_ok:
        return False, []
    funding = {"current": current, "zscore": zscore}

    tf_count = safe_float(rev_mtf.get("bear_tf_count"))
    mandatory = (
        (tf_count is not None and tf_count >= config.BOUNCE_MIN_REVERSAL_TFS,
         f"Разворот подтверждён на {tf_count:.0f} ТФ" if tf_count else ""),
        (bool(rev.get("bear_strong")),
         "Сильный разворотный сетап — признаки пика"),
        (bool(rev_mtf.get("bear_candle_confirm")),
         "1H свеча подтвердила разворот вниз"),
        (ctx["equilibrium"].get("zone") == "premium",
         "Цена в премиуме (выше равновесия)"),
        (bool(liq.get("liquidity_swept_above") and liq.get("reversal_candle")),
         "Снятие ликвидности сверху + возврат"),
        (not crowded_funding("short", funding),
         "Funding не перегрет в шорт"),
    )
    if not all(ok for ok, _ in mandatory):
        return False, []

    optional = (
        (bool(_sub(ctx, "divergence").get("bearish_divergence")),
         "Медвежья дивергенция RSI/MACD"),
        (_funding_stretched(current, zscore, bullish=False),
         "Funding заметно положительный — перегрет лонг"),
        (bool(_sub(ctx, "order_blocks").get("price_in_bearish_ob")
              or _sub(ctx, "fvg").get("price_in_bearish_fvg")),
         "Реакция в медвежьей зоне (OB/FVG)"),
        (bool(_sub(ctx, "cvd").get("cvd_bearish")),
         "CVD на стороне продавцов"),
    )
    fired = [reason for ok, reason in optional if ok]
    if len(fired) < config.BOUNCE_MIN_OPTIONAL_CONFIRMATIONS:
        return False, []

    return True, [reason for _, reason in mandatory] + fired
