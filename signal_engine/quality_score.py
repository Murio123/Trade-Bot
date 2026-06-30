"""Weighted Trade Quality Score (/100) for the advanced swing module.

Scoring follows the user's institutional weighting (trend-heavy, structure and
history matter, oscillators are confirmation) rather than equal-weight sections,
so a strong directional setup is not killed by a single weak section. Direction
comes from the 1D bias; the score expresses conviction in that direction.

Decision tiers:
    >= 88  STRONG BUY/SELL
    >= 75  BUY/SELL
    >= 62  WEAK BUY/SELL  (consider, with caution)
    <  62  NO TRADE       (cash is a position)
"""
from __future__ import annotations

from typing import Any

# (key, label, max points) — weights sum to 100.
FACTORS = [
    ("trend", "Тренд 1D", 20),
    ("structure", "Структура/OB", 15),
    ("fvg", "FVG", 8),
    ("rsi", "RSI", 8),
    ("macd", "MACD", 8),
    ("tsi", "TSI", 8),
    ("volume", "Объём/CVD", 8),
    ("derivatives", "Деривативы", 12),
    ("correlation", "Корреляции", 5),
    ("historical", "История", 8),
]
MAXES = {k: m for k, _, m in FACTORS}


def _bias(ind: dict[str, Any]) -> str:
    price, e50, e200 = ind.get("price"), ind.get("ema50"), ind.get("ema200")
    if None in (price, e50, e200):
        return "range"
    if price > e200 and e50 > e200:
        return "bullish"
    if price < e200 and e50 < e200:
        return "bearish"
    return "range"


def compute_quality_score(ctx: dict[str, Any]) -> dict[str, Any]:
    ind_1d = ctx["ind_1d"]
    ind_12h = ctx.get("ind_12h", ctx.get("ind_4h", {}))
    ind_4h = ctx["ind_4h"]
    price = ctx["price"]
    atr = ctx.get("atr") or (price * 0.01)

    bias_1d = _bias(ind_1d)
    direction = "long" if bias_1d == "bullish" else "short" if bias_1d == "bearish" else None
    bull = direction == "long"
    want = "bullish" if bull else "bearish"

    scores: dict[str, float] = {k: 0 for k, _, _ in FACTORS}
    notes: dict[str, str] = {}

    if direction is None:
        # No tradable bias -> everything zero, NO TRADE.
        plan = {"rr": None}
        return {"direction": None, "scores": scores, "maxes": MAXES, "notes": notes,
                "breakdown": _breakdown(scores), "overall": 0, "decision": "NO TRADE",
                "plan": plan, "htf_bias": bias_1d}

    st_1d = ctx.get("structure_1d", {})
    st_12h = ctx.get("structure_12h", {})
    ob = ctx.get("order_blocks", {})
    fvg = ctx.get("fvg", {})

    # 1. Trend (20): 1D EMA bias + structure sequence + 12H agreement.
    t = 0
    if _bias(ind_1d) == want:
        t += 10
    if st_1d.get("trend") == want:
        t += 6
    if _bias(ind_12h) == want or st_12h.get("trend") == want:
        t += 4
    scores["trend"] = min(t, 20)
    notes["trend"] = f"1D {bias_1d}, структура {st_1d.get('sequence')}"

    # 2. Structure / Order Block (15): BOS in direction + OB presence/reaction.
    s = 0
    ev1, ev12 = st_1d.get("last_event") or "", st_12h.get("last_event") or ""
    if (bull and "BOS_up" in (ev1, ev12)) or (not bull and "BOS_down" in (ev1, ev12)):
        s += 7
    elif (bull and "CHoCH_up" in (ev1, ev12)) or (not bull and "CHoCH_down" in (ev1, ev12)):
        s += 4
    has_ob = ob.get("bullish_ob") if bull else ob.get("bearish_ob")
    if has_ob:
        s += 5
    in_ob = ob.get("price_in_bullish_ob") if bull else ob.get("price_in_bearish_ob")
    if in_ob:
        s += 3
    # Reversal/exhaustion at an extreme adds structural conviction.
    rev = ctx.get("reversal", {})
    rev_ok = rev.get("bullish_reversal") if bull else rev.get("bearish_reversal")
    if rev_ok:
        s += 3 if (rev.get("bull_strong") if bull else rev.get("bear_strong")) else 2
    scores["structure"] = min(s, 15)
    rev_factors = (rev.get("factors_bull") if bull else rev.get("factors_bear")) or []
    notes["structure"] = (f"event {ev1 or ev12 or '—'}, OB {'есть' if has_ob else 'нет'}"
                          + (f", разворот: {', '.join(rev_factors)}" if rev_factors else ""))

    # 3. FVG (8).
    in_fvg = fvg.get("price_in_bullish_fvg") if bull else fvg.get("price_in_bearish_fvg")
    has_fvg = fvg.get("bullish_fvg") if bull else fvg.get("bearish_fvg")
    scores["fvg"] = 8 if in_fvg else (3 if has_fvg else 0)

    # 4-6. Oscillators on 4H (confirmation).
    rsi = ind_4h.get("rsi")
    if rsi is not None:
        if bull:
            scores["rsi"] = 8 if 40 <= rsi <= 65 else 6 if rsi < 40 else 2
        else:
            scores["rsi"] = 8 if 35 <= rsi <= 60 else 6 if rsi > 60 else 2
    scores["macd"] = 8 if ((bull and ind_4h.get("macd_bullish_cross")) or
                           (not bull and ind_4h.get("macd_bearish_cross"))) else (
        5 if ((bull and (ind_4h.get("macd_hist") or 0) > 0) or
              (not bull and (ind_4h.get("macd_hist") or 0) < 0)) else 0)
    scores["tsi"] = 8 if ((bull and ind_4h.get("tsi_bullish")) or
                          (not bull and ind_4h.get("tsi_bearish"))) else 0

    # 7. Volume / CVD (8).
    v = 0
    vol, avg = ind_4h.get("volume"), ind_4h.get("avg_volume")
    if vol and avg and vol > avg * 1.5:
        v += 4
    elif vol and avg and vol > avg:
        v += 2
    cvd = ctx.get("cvd", {})
    if (bull and cvd.get("cvd_bullish")) or (not bull and cvd.get("cvd_bearish")):
        v += 4
    scores["volume"] = min(v, 8)

    # 8. Derivatives (12): OI rising, funding not extreme/contrarian, crowd fuel.
    d = 6
    funding = (ctx.get("funding") or {}).get("current")
    if funding is not None:
        if abs(funding) > 0.0005:
            d -= 3
        if (bull and funding < 0) or (not bull and funding > 0):
            d += 3
    if ctx.get("oi_rising"):
        d += 3
    lsr = (ctx.get("long_short_ratio") or {}).get("ratio")
    if lsr is not None:
        # Crowded opposite side = liquidation fuel in our direction.
        if (bull and lsr < 0.8) or (not bull and lsr > 1.5):
            d += 2
        elif (bull and lsr > 2.0) or (not bull and lsr < 0.5):
            d -= 1
    scores["derivatives"] = max(0, min(d, 12))
    notes["derivatives"] = f"funding {funding}, OI {'рост' if ctx.get('oi_rising') else 'плоско'}, L/S {lsr}"

    # 9. Correlation (5).
    verdict = (ctx.get("correlation") or {}).get("verdict", "neutral")
    scores["correlation"] = 5 if verdict == want else 2 if verdict == "neutral" else 0

    # 10. Historical alignment (8).
    hist = ctx.get("historical", {})
    bp, conf = hist.get("bullish_pct"), hist.get("confidence") or 0
    if bp is not None:
        skew = bp if bull else (100 - bp)
        scores["historical"] = round(skew / 100 * 5 + conf / 100 * 3, 1)
        notes["historical"] = f"{hist.get('matches')} аналогов, {skew:.0f}% в сторону сделки, conf {conf}"

    plan = _build_plan(ctx, direction, atr, price)
    if plan.get("rr") is not None:
        notes["plan"] = f"R:R ≈ {plan['rr']}"

    overall = round(sum(scores.values()))
    decision = _decide(overall, direction)
    return {
        "direction": direction,
        "scores": scores,
        "maxes": MAXES,
        "breakdown": _breakdown(scores),
        "notes": notes,
        "overall": overall,
        "decision": decision,
        "plan": plan,
        "htf_bias": bias_1d,
    }


def _breakdown(scores: dict[str, float]) -> list[dict[str, Any]]:
    return [{"key": k, "label": label, "earned": round(scores.get(k, 0), 1), "max": m}
            for k, label, m in FACTORS]


def _build_plan(ctx: dict[str, Any], direction: str | None, atr: float, price: float) -> dict[str, Any]:
    if direction is None:
        return {"rr": None}
    bull = direction == "long"
    st4 = ctx.get("structure_4h", {})
    st12 = ctx.get("structure_12h", {})
    buffer = atr * 0.5

    if bull:
        swing = st4.get("last_swing_low") or st12.get("last_swing_low") or (price - atr * 2)
        stop = swing - buffer
        risk = max(price - stop, atr * 0.5)
        target_struct = st12.get("last_swing_high")
    else:
        swing = st4.get("last_swing_high") or st12.get("last_swing_high") or (price + atr * 2)
        stop = swing + buffer
        risk = max(stop - price, atr * 0.5)
        target_struct = st12.get("last_swing_low")

    sign = 1 if bull else -1
    tp1 = price + risk * 1.5 * sign
    tp2 = price + risk * 3.0 * sign
    tp3 = price + risk * 5.0 * sign

    valid_struct = target_struct and (
        (bull and target_struct > price) or (not bull and target_struct < price))
    rr = abs(target_struct - price) / risk if valid_struct else 3.0
    return {
        "entry": round(price, 2), "stop": round(stop, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2),
        "risk": round(risk, 2), "rr": round(rr, 2),
        "max_drawdown_pct": round(risk / price * 100, 2),
    }


def _decide(overall: int, direction: str | None) -> str:
    if direction is None:
        return "NO TRADE"
    side = "BUY" if direction == "long" else "SELL"
    if overall >= 88:
        return f"STRONG {side}"
    if overall >= 75:
        return side
    if overall >= 62:
        return f"WEAK {side}"
    return "NO TRADE"
