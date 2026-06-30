"""Deterministic Trade Quality Score (/100) for the advanced swing module.

Scores ten sections 0-10 from the gathered swing context (real data only — no
LLM), derives a direction, structure-based stop/targets and R:R, and maps the
total to a final decision. Below 85 -> NO TRADE (never force a trade).
"""
from __future__ import annotations

from typing import Any

SECTIONS = [
    "trend_alignment", "market_structure", "liquidity", "volume", "momentum",
    "derivatives", "macro", "historical", "risk_profile", "execution",
]


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
    ind_12h = ctx["ind_12h"]
    ind_4h = ctx["ind_4h"]
    price = ctx["price"]
    atr = ctx.get("atr") or (price * 0.01)

    bias_1d = _bias(ind_1d)
    direction = "long" if bias_1d == "bullish" else "short" if bias_1d == "bearish" else None
    bull = direction == "long"

    scores: dict[str, int] = {s: 0 for s in SECTIONS}
    notes: dict[str, str] = {}

    # 1. Trend alignment across 1D / 12H / 4H.
    biases = [bias_1d, _bias(ind_12h), _bias(ind_4h)]
    want = "bullish" if bull else "bearish"
    agree = sum(1 for b in biases if b == want)
    struct_1d = ctx.get("structure_1d", {})
    s = agree * 3  # 0,3,6,9
    if struct_1d.get("trend") == want:
        s += 1
    scores["trend_alignment"] = min(s, 10)
    notes["trend_alignment"] = f"1D/12H/4H совпадают: {agree}/3; структура 1D: {struct_1d.get('sequence')}"

    # 2. Market structure (BOS/CHoCH + clean sequence + OB on 12H).
    st12 = ctx.get("structure_12h", {})
    ob = ctx.get("order_blocks", {})
    s = 0
    ev = st12.get("last_event") or ""
    if (bull and ev == "BOS_up") or (not bull and ev == "BOS_down"):
        s += 5
    elif (bull and ev == "CHoCH_up") or (not bull and ev == "CHoCH_down"):
        s += 3
    if st12.get("trend") == want:
        s += 2
    if (bull and ob.get("bullish_ob")) or (not bull and ob.get("bearish_ob")):
        s += 3
    scores["market_structure"] = min(s, 10)
    notes["market_structure"] = f"12H event: {ev or '—'}, OB: {'да' if (ob.get('bullish_ob') or ob.get('bearish_ob')) else 'нет'}"

    # 3. Liquidity (sweep + pools).
    liq = ctx.get("liquidity", {})
    s = 0
    if (bull and liq.get("liquidity_swept_below")) or (not bull and liq.get("liquidity_swept_above")):
        s += 5
    if liq.get("reversal_candle"):
        s += 2
    if liq.get("equal_highs") or liq.get("equal_lows"):
        s += 3
    scores["liquidity"] = min(s, 10)
    notes["liquidity"] = f"свип: {'да' if s >= 5 else 'нет'}"

    # 4. Volume (vs average + CVD direction).
    s = 0
    vol, avg = ind_4h.get("volume"), ind_4h.get("avg_volume")
    if vol and avg and vol > avg * 1.5:
        s += 5
    elif vol and avg and vol > avg:
        s += 3
    cvd = ctx.get("cvd", {})
    if (bull and cvd.get("cvd_bullish")) or (not bull and cvd.get("cvd_bearish")):
        s += 5
    scores["volume"] = min(s, 10)
    notes["volume"] = f"объём>{'1.5x' if (vol and avg and vol>avg*1.5) else 'avg' if (vol and avg and vol>avg) else 'norm'}; CVD {cvd.get('method')}"

    # 5. Momentum on 4H (RSI/MACD/TSI/BBW).
    s = 0
    rsi = ind_4h.get("rsi")
    if rsi is not None:
        if bull and 40 <= rsi <= 70:
            s += 2
        if not bull and 30 <= rsi <= 60:
            s += 2
    if (bull and ind_4h.get("macd_bullish_cross")) or (not bull and ind_4h.get("macd_bearish_cross")):
        s += 3
    if (bull and ind_4h.get("tsi_bullish")) or (not bull and ind_4h.get("tsi_bearish")):
        s += 3
    if ind_4h.get("bb_squeeze") or (bull and ind_4h.get("bb_breakout_up")) or (not bull and ind_4h.get("bb_breakout_down")):
        s += 2
    scores["momentum"] = min(s, 10)
    notes["momentum"] = f"RSI {round(rsi,1) if rsi else '—'}, TSI {'+' if ind_4h.get('tsi_bullish') else '-'}"

    # 6. Derivatives (funding sane, OI rising, L/S not extreme).
    s = 5
    funding = (ctx.get("funding") or {}).get("current")
    if funding is not None:
        if abs(funding) > 0.0005:  # extreme funding -> crowded
            s -= 3
        if (bull and funding < 0) or (not bull and funding > 0):
            s += 2  # contrarian funding supports the trade
    oi_rising = ctx.get("oi_rising")
    if oi_rising:
        s += 2
    lsr = (ctx.get("long_short_ratio") or {}).get("ratio")
    if lsr is not None and (lsr > 2.0 or lsr < 0.5):
        s -= 1  # crowded positioning
    scores["derivatives"] = max(0, min(s, 10))
    notes["derivatives"] = f"funding {funding}, OI {'рост' if oi_rising else 'плоско'}, L/S {lsr}"

    # 7. Macro / correlation.
    corr = ctx.get("correlation", {})
    verdict = corr.get("verdict", "neutral")
    if verdict == want:
        scores["macro"] = 8
    elif verdict == "neutral":
        scores["macro"] = 5
    else:
        scores["macro"] = 2
    notes["macro"] = f"корреляции: {verdict} ({corr.get('supportive')}/{corr.get('counted')})"

    # 8. Historical similarity skew.
    hist = ctx.get("historical", {})
    bull_pct = hist.get("bullish_pct")
    conf = hist.get("confidence") or 0
    if bull_pct is not None:
        skew = bull_pct if bull else (100 - bull_pct)
        s = (skew / 100) * 7 + (conf / 100) * 3
        scores["historical"] = int(round(min(s, 10)))
        notes["historical"] = f"{hist.get('matches')} аналогов, {skew:.0f}% в сторону сделки, conf {conf}"
    else:
        scores["historical"] = 0
        notes["historical"] = "недостаточно истории"

    # 9 & 10 need a structure-based stop/target.
    plan = _build_plan(ctx, direction, atr, price)

    rr = plan.get("rr")
    if rr is None:
        scores["risk_profile"] = 0
    elif rr >= 4:
        scores["risk_profile"] = 10
    elif rr >= 3:
        scores["risk_profile"] = 8
    elif rr >= 2:
        scores["risk_profile"] = 5
    else:
        scores["risk_profile"] = 2
    notes["risk_profile"] = f"R:R ≈ {rr}"

    # 10. Execution quality: price inside a 12H zone of interest.
    fvg = ctx.get("fvg", {})
    in_zone = (
        (bull and (ob.get("price_in_bullish_ob") or fvg.get("price_in_bullish_fvg")))
        or (not bull and (ob.get("price_in_bearish_ob") or fvg.get("price_in_bearish_fvg")))
    )
    vp = ctx.get("volume_profile", {})
    poc = vp.get("poc")
    mid_range = bool(poc and abs(price - poc) / price < 0.005)
    s = 0
    if in_zone:
        s += 7
    if not mid_range:
        s += 3
    scores["execution"] = min(s, 10)
    notes["execution"] = f"в зоне интереса: {'да' if in_zone else 'нет'}"

    overall = sum(scores.values()) if direction else 0

    decision = _decide(overall, direction)
    return {
        "direction": direction,
        "scores": scores,
        "notes": notes,
        "overall": overall,
        "decision": decision,
        "plan": plan,
        "htf_bias": bias_1d,
    }


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

    tp1 = price + risk * 1.5 * (1 if bull else -1)
    tp2 = price + risk * 3.0 * (1 if bull else -1)
    tp3 = price + risk * 5.0 * (1 if bull else -1)

    # Use the structural target only if it sits in the trade's direction;
    # otherwise fall back to the synthetic 3R (TP2) target.
    valid_struct = target_struct and (
        (bull and target_struct > price) or (not bull and target_struct < price))
    if valid_struct:
        rr = abs(target_struct - price) / risk
    else:
        rr = 3.0
    return {
        "entry": round(price, 2),
        "stop": round(stop, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "risk": round(risk, 2),
        "rr": round(rr, 2),
        "max_drawdown_pct": round(risk / price * 100, 2),
    }


def _decide(overall: int, direction: str | None) -> str:
    if direction is None or overall < 85:
        return "NO TRADE"
    if direction == "long":
        return "STRONG BUY" if overall >= 92 else "BUY"
    return "STRONG SELL" if overall >= 92 else "SELL"
