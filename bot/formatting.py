"""Render signals and analysis into the Telegram message format from the ТЗ."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import config

DIRECTION_LABEL = {"long": "🟢 ЛОНГ", "short": "🔴 ШОРТ"}
BIAS_LABEL = {"bullish": "Бычий ✅", "bearish": "Медвежий ✅", "neutral": "Нейтральный ⚪"}


def _fmt_price(value: Any) -> str:
    if value is None:
        return "н/д"
    return f"{value:,.0f}".replace(",", " ")


def _stop_pct(signal: dict[str, Any]) -> str:
    entry = signal.get("entry_price")
    stop = signal.get("stop_loss")
    if not entry or stop is None:
        return ""
    return f", {abs(stop - entry) / entry * 100:.1f}%"


def _fmt_duration(hours: Any) -> str:
    if not hours or hours <= 0:
        return "н/д"
    if hours < 24:
        return f"{round(hours)} ч"
    return f"{round(hours / 24, 1)} дн"


def to_display_tz(dt: datetime) -> datetime:
    """Convert a stored UTC timestamp to the user's display timezone.

    Naive datetimes are treated as UTC (how the DB stores them).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(config.DISPLAY_TZ)


def fmt_display_time(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return f"{to_display_tz(dt):{fmt}} {config.DISPLAY_TZ_LABEL}"


def format_risk_cap_note() -> str:
    """Warning appended to an alert delivered over the open-positions cap."""
    return (f"⚠️ Лимит одновременных позиций ({config.MAX_OPEN_TRADES}) исчерпан — "
            "вход сверх риск-бюджета. Сделка НЕ добавлена в журнал.")


def format_signal(signal: dict[str, Any]) -> str:
    ts = signal.get("timestamp")
    if isinstance(ts, datetime):
        ts_str = fmt_display_time(ts)
    else:
        ts_str = str(ts or "")

    direction = signal.get("direction", "")
    style = signal.get("style_label", "СИГНАЛ")
    emoji = signal.get("style_emoji", "🤖")
    lines = [
        f"{emoji} {config.SYMBOL_DISPLAY} {style} | {signal.get('timeframe', '').upper()} | {ts_str}",
        f"📊 Направление: {DIRECTION_LABEL.get(direction, direction)}",
    ]
    if signal.get("display_price") is not None:
        lines.append(f"💰 Текущая цена: {_fmt_price(signal.get('display_price'))}")
    lines += [
        f"🕯 Цена свечи / вход: {_fmt_price(signal.get('entry_price'))}",
        "",
        "📐 Risk Management:",
        f"🛑 Стоп-лосс: {_fmt_price(signal.get('stop_loss'))} "
        f"({'за структурой HTF' if signal.get('stop_basis') == 'structure' else str(signal.get('atr_multiplier_used', config.ATR_MULTIPLIER)) + '×ATR ' + str(signal.get('stop_atr_tf', '')).upper()}"
        f"{_stop_pct(signal)})",
        f"🎯 Цель 1: {_fmt_price(signal.get('target_1'))}"
        f"{' (HTF-структура)' if signal.get('targets_structure') else ''}",
        f"🎯 Цель 2: {_fmt_price(signal.get('target_2'))}",
        f"💼 Размер позиции: {signal.get('position_size')} {config.SYMBOL_DISPLAY} "
        f"({signal.get('risk_percent', config.RISK_PERCENT)}% риска)",
        f"⏳ Удержание: ~{_fmt_duration(signal.get('hold_tp1_hours'))}–"
        f"{_fmt_duration(signal.get('hold_tp2_hours'))} (до TP1–TP2)",
        "",
        f"• HTF Bias ({signal.get('htf_tf', '1d').upper()}): "
        f"{BIAS_LABEL.get(signal.get('htf_bias', 'neutral'))} (фильтр пройден)",
    ]

    for reason in signal.get("reasons", [])[:8]:
        lines.append(f"• {reason}")

    mtf = signal.get("mtf") or {}
    agr = signal.get("mtf_agreement") or {}
    if mtf and agr.get("total"):
        tf_str = " ".join(
            f"{tf}:{'🔼' if t == 'bullish' else '🔽' if t == 'bearish' else '⚪'}"
            for tf, t in mtf.items())
        lines.append(f"• Согласие ТФ {agr.get('agree')}/{agr.get('total')} → {tf_str}")

    best = signal.get("session_best")
    if best:
        sessions = signal.get("sessions") or {}
        vol = sessions.get(best, {}).get("avg_range_pct")
        vol_str = f" (исторически +{vol}% волатильность)" if vol else ""
        lines.append(f"⏰ Лучшая сессия: {best}{vol_str}")

    ai_text = signal.get("ai_text")
    if ai_text:
        lines.append("")
        lines.append(f"🧠 {ai_text}")

    if signal.get("status") == "journal":
        lines.append("")
        lines.append("📒 Слабый сигнал (score 5-7) — записан в журнал, доступен по /signal")

    return "\n".join(lines)


def _level_context(ctx: dict[str, Any], price: float | None) -> list[str]:
    """Where the price sits relative to key levels: the zone it's reacting at
    (OB/FVG with its timeframe) plus the nearest support/resistance around."""
    if not price:
        return []
    out: list[str] = ["📐 Уровни:"]

    ob = ctx.get("order_blocks", {}) or {}
    fvg = ctx.get("fvg", {}) or {}
    at = None
    z = ob.get("bullish_ob")
    if z and ob.get("price_in_bullish_ob"):
        tf = f" [{z['tf'].upper()}]" if z.get("tf") else ""
        at = f"бычий OB{tf} {_fmt_price(z['low'])}–{_fmt_price(z['high'])}"
    if at is None:
        z = ob.get("bearish_ob")
        if z and ob.get("price_in_bearish_ob"):
            tf = f" [{z['tf'].upper()}]" if z.get("tf") else ""
            at = f"медвежий OB{tf} {_fmt_price(z['low'])}–{_fmt_price(z['high'])}"
    if at is None:
        z = fvg.get("bullish_fvg")
        if z and fvg.get("price_in_bullish_fvg"):
            tf = f" [{z['tf'].upper()}]" if z.get("tf") else ""
            at = f"бычий FVG{tf} {_fmt_price(z['low'])}–{_fmt_price(z['high'])}"
    if at is None:
        z = fvg.get("bearish_fvg")
        if z and fvg.get("price_in_bearish_fvg"):
            tf = f" [{z['tf'].upper()}]" if z.get("tf") else ""
            at = f"медвежий FVG{tf} {_fmt_price(z['low'])}–{_fmt_price(z['high'])}"
    if at:
        out.append(f"  Реакция в зоне: {at}")

    support, resistance = _nearest_sr(ctx, price)
    if resistance:
        out.append(f"  🔺 Сопротивление: {_fmt_price(resistance)} "
                   f"(+{(resistance - price) / price * 100:.1f}%)")
    if support:
        out.append(f"  🔻 Поддержка: {_fmt_price(support)} "
                   f"(−{(price - support) / price * 100:.1f}%)")
    return out if len(out) > 1 else []


def build_reversal_plan(ctx: dict[str, Any], direction: str) -> dict[str, Any] | None:
    """Concrete entry/stop/targets for a confirmed reversal."""
    price = ctx.get("price")
    if not price:
        return None
    inds = ctx.get("inds_by_tf", {})
    atr_1h = (inds.get("1h") or {}).get("atr") or ctx.get("atr") or price * 0.01
    support, resistance = _nearest_sr(ctx, price)

    if direction == "bull":
        stop = (support - atr_1h * 0.5) if (support and support < price) else price - atr_1h * 2
        risk = max(price - stop, atr_1h * 0.5)
        tp1 = resistance if (resistance and resistance - price >= risk) else price + risk * 1.5
        tp2 = price + risk * 3
    else:
        stop = (resistance + atr_1h * 0.5) if (resistance and resistance > price) else price + atr_1h * 2
        risk = max(stop - price, atr_1h * 0.5)
        tp1 = support if (support and price - support >= risk) else price - risk * 1.5
        tp2 = price - risk * 3
    return {
        "entry": round(price, 2), "stop": round(stop, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2),
        "rr": round(abs(tp1 - price) / risk, 2),
        "risk_pct": round(risk / price * 100, 2),
    }


def format_reversal_alert(ctx: dict[str, Any], direction: str,
                          factors: list[str], strong: bool,
                          tfs: list[str] | None = None,
                          alignment: str = "neutral") -> str:
    bull = direction == "bull"
    if alignment == "aligned":
        head = ("🎯 ДНО ОТКАТА ПО ТРЕНДУ — точка входа в ЛОНГ" if bull
                else "🎯 ПИК ОТСКОКА ПО ТРЕНДУ — точка входа в ШОРТ")
    elif alignment == "counter":
        head = ("🟡 ДНО против тренда 1D" if bull
                else "🟡 ПИК против тренда 1D")
    else:
        head = ("🟢 ДНО: разворот ВВЕРХ" if bull else "🔴 ПИК: разворот ВНИЗ")
    tf_str = ", ".join(_TF_LABEL.get(t, t.upper()) for t in (tfs or []))
    lines = [
        f"🔔 {head}{' — сильный сигнал' if strong else ''} | {config.SYMBOL_DISPLAY}",
        f"Подтверждено: {len(tfs or [])} ТФ ({tf_str}) + разворотная свеча 1H",
        "",
        "Почему:",
    ]
    lines += [f"  • {f}" for f in factors[:5]]

    level_ctx = _level_context(ctx, ctx.get("price"))
    if level_ctx:
        lines += [""] + level_ctx

    plan = build_reversal_plan(ctx, direction)
    if plan:
        d = "ЛОНГ" if bull else "ШОРТ"
        lines += [
            "",
            f"🎯 План ({d}):",
            f"  Вход: {_fmt_price(plan['entry'])} (по рынку)",
            f"  🛑 Стоп: {_fmt_price(plan['stop'])} ({plan['risk_pct']}%)",
            f"  🎯 TP1: {_fmt_price(plan['tp1'])}",
            f"  🎯 TP2: {_fmt_price(plan['tp2'])}",
        ]

    if alignment == "aligned":
        lines += ["", "✅ Разворот в сторону тренда 1D."]
    elif alignment == "counter":
        lines += ["", "⚠️ ПРОТИВ тренда 1D — контртренд: уменьшенный объём, "
                      "быстрая фиксация, стоп неприкосновенен."]
    else:
        lines += ["", "⚠️ Тренд 1D не определён — торгуй от уровней, риск ≤1%."]
    return "\n".join(lines)


_TF_LABEL = {"1h": "1H", "4h": "4H", "12h": "12H", "1d": "1D"}


def _nearest_sr(ctx: dict[str, Any], price: float | None) -> tuple[float | None, float | None]:
    """Nearest support (below) and resistance (above) across all level sources."""
    if not price:
        return None, None
    pts: list[float] = []
    levels = ctx.get("htf_levels", {})
    pts += list(levels.get("highs", [])) + list(levels.get("lows", []))
    vp = ctx.get("volume_profile", {})
    for k in ("poc", "vah", "val"):
        if vp.get(k):
            pts.append(vp[k])
    liq = ctx.get("liquidity", {})
    pts += list(liq.get("equal_highs", [])) + list(liq.get("equal_lows", []))
    for ob_key in ("bullish_ob", "bearish_ob"):
        z = (ctx.get("order_blocks", {}) or {}).get(ob_key)
        if z:
            pts += [z["low"], z["high"]]
    # Key EMAs (dynamic levels) from 1D and 4H.
    inds = ctx.get("inds_by_tf", {})
    for tf in ("1d", "4h"):
        for span in ("ema50", "ema200"):
            v = inds.get(tf, {}).get(span)
            if v:
                pts.append(v)

    above = [p for p in pts if p > price * 1.0005]
    below = [p for p in pts if p < price * 0.9995]
    resistance = min(above) if above else None
    support = max(below) if below else None
    return support, resistance


def format_reversal(ctx: dict[str, Any]) -> str:
    mtf = ctx.get("reversal_mtf", {})
    per_tf = mtf.get("per_tf", {})
    price = ctx.get("price")
    lines = [
        f"🔄 Анализ разворота {config.SYMBOL_DISPLAY} | 1H/4H/12H/1D",
        f"Цена: {_fmt_price(price)}",
        "",
    ]

    if not per_tf:
        lines.append("Нет данных по таймфреймам.")
        return "\n".join(lines)

    # Per-timeframe status line.
    lines.append("По таймфреймам:")
    for tf in ("1h", "4h", "12h", "1d"):
        r = per_tf.get(tf)
        if not r:
            continue
        if r.get("bullish_reversal"):
            mark = f"🟢 дно ({r.get('bull_score')})" + (" сильное" if r.get("bull_strong") else "")
        elif r.get("bearish_reversal"):
            mark = f"🔴 пик ({r.get('bear_score')})" + (" сильный" if r.get("bear_strong") else "")
        elif r.get("bull_score"):
            mark = f"🟢· слабо ({r.get('bull_score')}/2)"
        elif r.get("bear_score"):
            mark = f"🔴· слабо ({r.get('bear_score')}/2)"
        else:
            mark = "—"
        lines.append(f"  {_TF_LABEL[tf]}: {mark}")

    bull_tfs = mtf.get("bull_tfs") or []
    bear_tfs = mtf.get("bear_tfs") or []
    lines.append("")

    if mtf.get("combined_bullish"):
        lines.append(f"🟢 ДНО подтверждено на {len(bull_tfs)} ТФ: "
                     f"{', '.join(_TF_LABEL[t] for t in bull_tfs)}")
        lines += _tf_factors(per_tf, bull_tfs, "factors_bull")
    elif mtf.get("combined_bearish"):
        lines.append(f"🔴 ПИК подтверждён на {len(bear_tfs)} ТФ: "
                     f"{', '.join(_TF_LABEL[t] for t in bear_tfs)}")
        lines += _tf_factors(per_tf, bear_tfs, "factors_bear")
    elif bull_tfs:
        lines.append(f"🟢 Признаки дна на {', '.join(_TF_LABEL[t] for t in bull_tfs)} "
                     f"(нужно ≥2 ТФ для подтверждения)")
    elif bear_tfs:
        lines.append(f"🔴 Признаки пика на {', '.join(_TF_LABEL[t] for t in bear_tfs)} "
                     f"(нужно ≥2 ТФ для подтверждения)")
    else:
        any_weak = any((r.get("bull_score") or r.get("bear_score"))
                       for r in per_tf.values())
        if any_weak:
            lines.append("Слабые звоночки есть, но ни на одном ТФ не набралось "
                         "≥2 факторов — разворот не подтверждён.")
        else:
            lines.append("Признаков разворота сейчас нет ни на одном ТФ.")
        lines += _reversal_watch(ctx, price)

    lines.append("")
    lines.append("⚠️ Развороты — это фейд движения: ниже винрейт, выше R:R. "
                 "Лучше брать в сторону старшего тренда.")
    lines.append("")
    lines += _reversal_conclusion(ctx, price)
    return "\n".join(lines)


def _reversal_conclusion(ctx: dict[str, Any], price: float | None) -> list[str]:
    if not price:
        return []
    mtf = ctx.get("reversal_mtf", {})
    inds = ctx.get("inds_by_tf", {})
    ema200_1d = inds.get("1d", {}).get("ema200")
    trend = ("бычий" if price > ema200_1d else "медвежий") if ema200_1d else None
    support, resistance = _nearest_sr(ctx, price)

    out: list[str] = []
    level_ctx = _level_context(ctx, price)
    if level_ctx:
        out += level_ctx + [""]
    out.append("📌 Вывод:")
    if mtf.get("combined_bullish"):
        n = len(mtf.get("bull_tfs", []))
        if trend == "бычий":
            out.append(f"🟢 Дно подтверждено на {n} ТФ по тренду вверх.")
        elif trend == "медвежий":
            out.append(f"🟡 Дно на {n} ТФ, но ПРОТИВ тренда вниз — отскок/ловля ножа. "
                       "Рискованно, малым объёмом и быстрой фиксацией.")
        else:
            out.append(f"🟢 Дно подтверждено на {n} ТФ.")
        if support and resistance:
            out.append(f"План лонга: вход у {_fmt_price(support)}, стоп ниже, "
                       f"цель {_fmt_price(resistance)} (+{(resistance - price) / price * 100:.1f}%).")
    elif mtf.get("combined_bearish"):
        n = len(mtf.get("bear_tfs", []))
        if trend == "медвежий":
            out.append(f"🔴 Пик подтверждён на {n} ТФ по тренду вниз.")
        elif trend == "бычий":
            out.append(f"🟡 Пик на {n} ТФ, но ПРОТИВ тренда вверх — лишь коррекция. "
                       "Шорт рискован, малым объёмом.")
        else:
            out.append(f"🔴 Пик подтверждён на {n} ТФ.")
        if support and resistance:
            out.append(f"План шорта: вход у {_fmt_price(resistance)}, стоп выше, "
                       f"цель {_fmt_price(support)} (−{(price - support) / price * 100:.1f}%).")
    else:
        trend_str = f" Глобальный тренд {trend}." if trend else ""
        out.append("⚪ Подтверждённого разворота нет — сделки нет." + trend_str)
        if support and resistance:
            out.append(f"Слежу за реакцией у {_fmt_price(support)} / {_fmt_price(resistance)}. "
                       "Алерт придёт при подтверждении на ≥2 ТФ.")
    return out


def _tf_factors(per_tf: dict, tfs: list, key: str) -> list[str]:
    """Unique reversal factors across the confirming timeframes."""
    seen: list[str] = []
    for tf in tfs:
        for f in per_tf.get(tf, {}).get(key, []):
            if f not in seen:
                seen.append(f)
    return [f"  • {f}" for f in seen]


def _reversal_watch(ctx: dict[str, Any], price: float | None) -> list[str]:
    """Context for the 'no setup yet' case: RSI zone + nearest levels to watch."""
    ind = ctx.get("ind_signal", {})
    liq = ctx.get("liquidity", {})
    out: list[str] = [""]

    rsi = ind.get("rsi")
    if rsi is not None:
        if rsi < 30:
            zone = "перепродан — близко к развороту вверх 🟢"
        elif rsi < 40:
            zone = "приближается к перепроданности"
        elif rsi > 70:
            zone = "перекуплен — близко к развороту вниз 🔴"
        elif rsi > 60:
            zone = "приближается к перекупленности"
        else:
            zone = "нейтрально"
        out.append(f"• RSI: {rsi:.0f} ({zone})")

    # Nearest support below / resistance above where a reversal may form.
    lows = [l for l in (liq.get("equal_lows") or []) if price and l <= price]
    highs = [h for h in (liq.get("equal_highs") or []) if price and h >= price]
    support = max(lows) if lows else ind.get("bb_lower")
    resistance = min(highs) if highs else ind.get("bb_upper")
    if support and price:
        out.append(f"• Зона дна (поддержка): {_fmt_price(support)} "
                   f"(−{(price - support) / price * 100:.1f}%)")
    if resistance and price:
        out.append(f"• Зона пика (сопротивление): {_fmt_price(resistance)} "
                   f"(+{(resistance - price) / price * 100:.1f}%)")

    out.append("🔔 Пришлю алерт, когда дно/пик подтвердится на ≥2 ТФ.")
    return out


def _collect_level_items(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Every level source as (lo, hi, label) — zones keep their range."""
    items: list[dict[str, Any]] = []

    def add(lo, hi, label):
        if lo is None:
            return
        hi = hi if hi is not None else lo
        items.append({"lo": float(lo), "hi": float(hi), "label": label})

    ob = ctx.get("order_blocks", {}) or {}
    for key, name in (("bullish_ob", "бычий OB"), ("bearish_ob", "медвежий OB")):
        z = ob.get(key)
        if z:
            tf = f" {z['tf'].upper()}" if z.get("tf") else ""
            add(z["low"], z["high"], f"{name}{tf}")

    fvg = ctx.get("fvg", {}) or {}
    for key, name in (("bullish_fvg", "FVG (имбаланс)"), ("bearish_fvg", "FVG (имбаланс)")):
        z = fvg.get(key)
        if z:
            add(z["low"], z["high"], name)

    vp = ctx.get("volume_profile", {}) or {}
    add(vp.get("poc"), None, "POC — магнит объёма")
    add(vp.get("vah"), None, "VAH — верх объёмной зоны")
    add(vp.get("val"), None, "VAL — низ объёмной зоны")

    liq = ctx.get("liquidity", {}) or {}
    for x in (liq.get("equal_highs") or [])[:3]:
        add(x, None, "равные хаи (скопление стопов)")
    for x in (liq.get("equal_lows") or [])[:3]:
        add(x, None, "равные лои (скопление стопов)")

    levels = ctx.get("htf_levels", {}) or {}
    for x in (levels.get("highs") or []):
        add(x, None, "уровень HTF")
    for x in (levels.get("lows") or []):
        add(x, None, "уровень HTF")

    inds = ctx.get("inds_by_tf", {}) or {}
    for tf in ("1d", "4h"):
        for span in ("ema50", "ema200"):
            v = (inds.get(tf) or {}).get(span)
            if v:
                add(v, None, f"{span.upper()} {tf.upper()}")

    eq = ctx.get("equilibrium", {}) or {}
    add(eq.get("high"), None, "хай диапазона")
    add(eq.get("low"), None, "лой диапазона")

    lm = ctx.get("liquidation_map", {}) or {}
    if lm.get("source") == "oi_estimate":
        price = ctx.get("price") or 0
        shorts = [c["price"] for c in lm.get("short_liquidations", []) if c["price"] > price]
        longs = [c["price"] for c in lm.get("long_liquidations", []) if c["price"] < price]
        if shorts:
            add(min(shorts), None, "магнит ликвидаций (шорты)")
        if longs:
            add(max(longs), None, "магнит ликвидаций (лонги)")
    return items


def _cluster_levels(items: list[dict[str, Any]], price: float,
                    atr: float | None) -> tuple[list[dict], list[dict]]:
    """Merge nearby levels into clusters; >=2 sources = strong (⭐).

    Returns (resistances above, supports below), each sorted by distance."""
    tol = max((atr or price * 0.005) * 0.35, price * 0.0015)
    above = sorted((i for i in items if (i["lo"] + i["hi"]) / 2 > price * 1.0005),
                   key=lambda i: (i["lo"] + i["hi"]) / 2)
    below = sorted((i for i in items if (i["lo"] + i["hi"]) / 2 < price * 0.9995),
                   key=lambda i: -((i["lo"] + i["hi"]) / 2))

    def merge(seq):
        clusters: list[dict[str, Any]] = []
        for it in seq:
            mid = (it["lo"] + it["hi"]) / 2
            # Merge when the level falls inside the cluster's range (+tol),
            # not merely near its midpoint — zones absorb their edge levels.
            if clusters and (clusters[-1]["lo"] - tol) <= mid <= (clusters[-1]["hi"] + tol):
                c = clusters[-1]
                c["lo"], c["hi"] = min(c["lo"], it["lo"]), max(c["hi"], it["hi"])
                if it["label"] not in c["labels"]:
                    c["labels"].append(it["label"])
                c["mid"] = (c["lo"] + c["hi"]) / 2
            else:
                clusters.append({"lo": it["lo"], "hi": it["hi"], "mid": mid,
                                 "labels": [it["label"]]})
        return clusters

    return merge(above), merge(below)


def _fmt_cluster(c: dict[str, Any], price: float) -> str:
    zone = abs(c["hi"] - c["lo"]) > price * 0.0005
    where = (f"{_fmt_price(c['lo'])}–{_fmt_price(c['hi'])}" if zone
             else _fmt_price(c["mid"]))
    dist = (c["mid"] - price) / price * 100
    star = " ⭐" if len(c["labels"]) >= 2 else ""
    labels = " + ".join(c["labels"][:3])
    return f"  {where} ({dist:+.1f}%) — {labels}{star}"


def format_levels(ctx: dict[str, Any]) -> str:
    """One ladder: resistances above, supports below, nearest first.

    Levels from every source (OB/FVG/VP/liquidity/EMA/range/liquidations) are
    clustered; a cluster backed by several sources is marked ⭐ as strong."""
    price = ctx.get("price")
    lines = [f"📐 Уровни {config.SYMBOL_DISPLAY} | цена {_fmt_price(price)}"]
    if not price:
        return "\n".join(lines)

    zone_tfs = ctx.get("zone_tfs") or []
    if zone_tfs:
        lines.append(f"Зоны и уровни: {'/'.join(t.upper() for t in zone_tfs)} "
                     "+ объём/EMA/диапазон")
    lines.append("")

    items = _collect_level_items(ctx)
    res, sup = _cluster_levels(items, price, ctx.get("atr"))

    lines.append("🔺 Сопротивления (ближайшие сверху):")
    lines += [_fmt_cluster(c, price) for c in res[:4]] or ["  —"]
    lines.append("")
    lines.append("🔻 Поддержки (ближайшие снизу):")
    lines += [_fmt_cluster(c, price) for c in sup[:4]] or ["  —"]

    eq = ctx.get("equilibrium", {}) or {}
    ind = ctx.get("ind_signal", {}) or {}
    ctx_bits = []
    if eq.get("zone"):
        zone_ru = {"discount": "🟢 дисконт", "premium": "🔴 премиум",
                   "equilibrium": "⚪ равновесие"}.get(eq["zone"], eq["zone"])
        ctx_bits.append(f"{zone_ru} ({int(eq.get('pos', 0.5) * 100)}% диапазона)")
    if ind.get("bb_squeeze"):
        ctx_bits.append("BBW: сжатие — готовится импульс")
    elif ind.get("bb_expansion"):
        ctx_bits.append("BBW: расширение")
    if ctx_bits:
        lines += ["", "⚖️ " + " · ".join(ctx_bits)]

    lines.append("")
    lines += _levels_conclusion(ctx, price, res, sup)
    return "\n".join(lines)


def _strong_cluster(clusters: list[dict], price: float) -> dict | None:
    """Nearest multi-source (⭐) cluster; falls back to the nearest one, but
    skips clusters closer than 0.5% — a working range of ±0.2% is noise."""
    if not clusters:
        return None
    meaningful = [c for c in clusters if abs(c["mid"] - price) / price >= 0.005]
    pool = meaningful or clusters
    for c in pool:
        if len(c["labels"]) >= 2:
            return c
    return pool[0]


def _levels_conclusion(ctx: dict[str, Any], price: float | None,
                       res_clusters: list[dict] | None = None,
                       sup_clusters: list[dict] | None = None) -> list[str]:
    if not price:
        return []
    inds = ctx.get("inds_by_tf", {})
    eq = ctx.get("equilibrium", {})
    ema200_1d = inds.get("1d", {}).get("ema200")

    trend = None
    if ema200_1d:
        trend = "бычий" if price > ema200_1d else "медвежий"
    zone = eq.get("zone")
    zone_ru = {"discount": "дисконте", "premium": "премиуме",
               "equilibrium": "равновесии"}.get(zone)

    out = ["📌 Вывод:"]
    parts = []
    if trend:
        parts.append(f"цена {'над' if trend == 'бычий' else 'под'} 1D EMA200 "
                     f"(глобальный тренд {trend})")
    if zone_ru:
        parts.append(f"в {zone_ru} диапазона")
    if parts:
        out.append("• " + ", ".join(parts) + ".")

    # Working range between the nearest STRONG clusters (not raw points —
    # with dozens of sources something always sits 0.2% away, which made the
    # range meaninglessly narrow).
    res_c = _strong_cluster(res_clusters or [], price)
    sup_c = _strong_cluster(sup_clusters or [], price)
    if res_c and sup_c:
        s_star = " ⭐" if len(sup_c["labels"]) >= 2 else ""
        r_star = " ⭐" if len(res_c["labels"]) >= 2 else ""
        out.append(
            f"• Рабочий диапазон: {_fmt_price(sup_c['mid'])}{s_star} "
            f"(−{(price - sup_c['mid']) / price * 100:.1f}%) ↔ "
            f"{_fmt_price(res_c['mid'])}{r_star} "
            f"(+{(res_c['mid'] - price) / price * 100:.1f}%).")

    # Directional lean from trend + premium/discount.
    if trend == "бычий" and zone == "discount":
        lean = "🟢 Преимущество у покупателей: откат в дисконте по тренду вверх — зона интереса для лонгов."
    elif trend == "бычий" and zone == "premium":
        lean = "🟡 Тренд вверх, но цена в премиуме — лонги дороже, ждать отката/подтверждения."
    elif trend == "медвежий" and zone == "premium":
        lean = "🔴 Преимущество у продавцов: отскок в премиуме по тренду вниз — зона интереса для шортов."
    elif trend == "медвежий" and zone == "discount":
        lean = "🟡 Тренд вниз, но цена в дисконте — возможен отскок, шорты рискованнее."
    else:
        lean = "⚪ Чёткого перевеса нет — ждать реакции от ближайшего уровня."
    out.append(lean)
    return out


def format_funding(funding: dict[str, Any], ls_ratio: dict[str, Any]) -> str:
    cur = funding.get("current")
    cur_pct = f"{cur * 100:.4f}%" if cur is not None else "н/д"
    lines = [
        f"💸 Funding Rate {config.SYMBOL_DISPLAY}",
        f"Текущий: {cur_pct}",
        f"Среднее (30): {funding.get('avg', 0) * 100:.4f}%",
        f"Z-score: {funding.get('zscore', 0):.2f}",
        f"Аномалия: {'⚠️ да' if funding.get('anomalous') else 'нет'}",
    ]
    ratio = ls_ratio.get("ratio")
    if ratio is not None:
        lines.append(f"Long/Short ratio: {ratio:.2f}")
    return "\n".join(lines)


def format_market(price: float | None, funding: dict[str, Any],
                  ls_ratio: dict[str, Any], oi: float | None,
                  fng: dict[str, Any]) -> str:
    """One-screen market overview: price, funding, positioning, sentiment."""
    lines = [f"💹 Рынок {config.SYMBOL_DISPLAY}", ""]
    if price:
        lines.append(f"💰 Цена: {_fmt_price(price)}")

    cur = funding.get("current")
    if cur is not None:
        mood = ("⚠️ перегрев лонгов" if funding.get("anomalous") and cur > 0 else
                "⚠️ перегрев шортов" if funding.get("anomalous") and cur < 0 else
                "норма")
        lines.append(f"💸 Funding: {cur * 100:.4f}% ({mood}, z={funding.get('zscore', 0):.1f})")

    ratio = ls_ratio.get("ratio")
    if ratio is not None:
        crowd = "лонги переполнены" if ratio > 2 else \
                "шорты переполнены" if ratio < 0.5 else "баланс"
        lines.append(f"⚖️ Long/Short: {ratio:.2f} ({crowd})")

    if oi:
        lines.append(f"📊 Open Interest: {oi:,.0f}".replace(",", " "))

    val = fng.get("value")
    if val is not None:
        emoji = "😱" if val < 25 else "😟" if val < 45 else "😐" if val < 55 else "🙂" if val < 75 else "🤑"
        lines.append(f"{emoji} Fear & Greed: {val}/100 ({fng.get('classification')})")

    return "\n".join(lines)


def format_journal(stats: dict[str, Any], open_count: int = 0) -> str:
    total = stats.get("total", 0)
    lines = [
        "📒 Статистика журнала сделок",
        f"Закрыто сделок: {total}",
    ]
    if total:
        lines += [
            f"Винрейт: {stats.get('winrate', 0)}%  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)",
            f"Средний результат: {stats.get('avg_r', 0):+}R",
            f"Суммарно: {stats.get('total_r', 0):+}R",
            f"Лучшая: {stats.get('best_r', 0):+}R | Худшая: {stats.get('worst_r', 0):+}R",
        ]
    else:
        lines.append("Пока нет закрытых сделок — статистика появится после первых исходов.")
    lines.append(f"Открыто сейчас: {open_count}")
    return "\n".join(lines)


HTF_LINE = {
    "bullish": "Бычий 🔼 — шорты заблокированы",
    "bearish": "Медвежий 🔽 — лонги заблокированы",
    "neutral": "Нейтральный ⚪ — нет чёткого направления",
}

BLOCK_REASON = {
    "htf_filter": "сигнал против старшего тренда (HTF-фильтр)",
    "diversity": "мало категориального разнообразия (нужно ≥3 категории)",
    "below_threshold": "очков недостаточно для журнала (нужно ≥5)",
    "wait_for_sweep": "впереди вероятное снятие ликвидности — ждём свип",
    "dead_zone": "середина диапазона без структуры — мёртвая зона",
    "crowded_funding": "funding экстремальный, толпа на нашей стороне — риск сквиза",
    "direction_conflict": "доказательства за лонг и шорт равны — рынок в конфликте",
    "stale_data": "данные биржи устарели — анализ по ним ненадёжен",
    "abnormal_volatility": "аномальная волатильность (ATR в экстремуме) — вне торговых условий",
}

BLOCK_HINT = {
    "bullish": "Жду совпадения импульса с трендом вверх.",
    "bearish": "Жду совпадения импульса с трендом вниз.",
    "neutral": "Жду формирования чёткого дневного тренда.",
}


def _ago(dt) -> str:
    if dt is None:
        return "ещё не было"
    from datetime import datetime, timezone
    if isinstance(dt, str):
        return dt
    delta = datetime.now(timezone.utc) - dt
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "только что"
    if mins < 60:
        return f"{mins} мин назад"
    return f"{mins // 60} ч {mins % 60} мин назад"


def _ai_status_line(s: dict[str, Any]) -> str:
    ok = s.get("ai_ok")
    if ok is True:
        return f"🧠 AI: ✅ {s.get('ai_model')}"
    if ok is False:
        return (f"🧠 AI: ⚠️ недоступен — детерминированный fallback "
                f"({s.get('ai_error')})")
    return "🧠 AI: — (проверка ещё не выполнялась)"


def format_status(s: dict[str, Any]) -> str:
    dry = s.get("dry_run")
    dry_alerts = s.get("send_dry_run_alerts")
    db_ok = s.get("db_connected")
    if dry:
        mode = ("DRY-RUN 🧪 (уведомления шлются, manual only)"
                if dry_alerts else
                "DRY-RUN 🧪 (уведомления НЕ шлются)")
    else:
        mode = "LIVE ✅ (уведомления включены)"
    lines = [
        "🩺 Статус бота",
        "",
        f"💱 Источник данных: {s.get('exchange_active') or s.get('exchange_pref')}",
        f"💰 Цена {config.SYMBOL_DISPLAY}: {_fmt_price(s.get('price'))}",
        f"🗄 База данных: {'PostgreSQL ✅' if db_ok else 'in-memory ⚠️ (без персистентности)'}",
        _ai_status_line(s),
        f"📡 Режим: {mode}",
        f"🔔 Получатели алертов: {s.get('alert_chats', 0)}",
        "",
        f"⏱ Последний анализ: {_ago(s.get('last_analysis_at'))}"
        + (f" ({s.get('last_analysis_tf')})" if s.get('last_analysis_tf') else ""),
    ]
    st = s.get("last_analysis_status")
    if st == "blocked":
        lines.append(f"   └ результат: заблокирован ({s.get('last_analysis_blocked_at')})")
    elif st:
        lines.append(f"   └ результат: {st}")
    lines += [
        "",
        f"📈 Таймфреймы: {s.get('signal_tf')} осн. + {s.get('fast_tf')} быстрый",
        f"📨 Сигналов за сутки: {s.get('signals_today', 0)} (лимит {s.get('max_per_day')})",
        f"📒 Открытых сделок: {s.get('open_trades', 0)} | закрыто: {s.get('closed_trades', 0)}"
        + (f", винрейт {s.get('winrate')}%" if s.get('closed_trades') else ""),
    ]
    return "\n".join(lines)


def format_trade_event(trade: dict[str, Any], event: dict[str, Any]) -> str:
    direction = "🟢 ЛОНГ" if trade.get("direction") == "long" else "🔴 ШОРТ"
    entry = _fmt_price(trade.get("entry_price"))
    head = f"📍 Сделка {direction} {config.SYMBOL_DISPLAY} | вход {entry}"

    if event["type"] == "tp1":
        return "\n".join([
            head,
            "🎯 TP1 достигнут!",
            f"🛡 Стоп переведён в безубыток ({_fmt_price(event.get('stop'))}).",
            "Позиция теперь без риска — дальше ведём к TP2.",
        ])

    outcome = event.get("outcome")
    pnl_r = event.get("pnl_r", 0.0)
    if outcome == "win":
        emoji, label = "✅", "TP2 достигнут — закрыто в плюс"
    elif outcome == "loss":
        emoji, label = "🛑", "Стоп — закрыто в минус"
    elif outcome == "breakeven":
        emoji, label = "➖", "Откат к безубытку после TP1 — в ноль"
    else:
        emoji, label = "ℹ️", outcome or "закрыто"
    if event.get("expired"):
        label = "Закрыто по времени (21 день)"
    return "\n".join([
        head,
        f"{emoji} {label}",
        f"Результат: {pnl_r:+.2f}R | выход {_fmt_price(event.get('exit_price'))}",
    ])


def format_last_signal_note(last: dict[str, Any] | None,
                            max_age_hours: float = 24) -> str | None:
    """A reminder of the still-relevant previous signal, or None if stale."""
    if not last:
        return None
    from datetime import datetime, timezone
    created = last.get("created_at") or last.get("timestamp")
    if created is None:
        return None
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created)
        except ValueError:
            return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600
    if age_h > max_age_hours:
        return None
    d = "🟢 ЛОНГ" if last.get("direction") == "long" else "🔴 ШОРТ"
    return (
        f"📌 Действующий сетап ({_ago(created)}): {d} от "
        f"{_fmt_price(last.get('entry_price'))}\n"
        f"🛑 SL {_fmt_price(last.get('stop_loss'))} | "
        f"🎯 TP1 {_fmt_price(last.get('target_1'))} | "
        f"TP2 {_fmt_price(last.get('target_2'))}\n"
        "Новых сетапов сверх него сейчас нет."
    )


def format_blocked(result: dict[str, Any],
                   display_price: float | None = None) -> str:
    stage = result.get("blocked_at")
    bias = result.get("htf_bias", "neutral")

    lines = ["ℹ️ Нет активного сигнала"]
    lines.append(f"Причина: {BLOCK_REASON.get(stage, stage or 'нет данных')}")

    price = result.get("price")
    if display_price is not None:
        lines.append(f"💰 Текущая цена: {_fmt_price(display_price)}")
    if price is not None:
        lines.append(f"🕯 Цена свечи / контекста: {_fmt_price(price)}")

    lines.append(f"HTF ({result.get('htf_tf', '1d').upper()}): {HTF_LINE.get(bias, bias)}")

    lines.append("")
    lines.append(BLOCK_HINT.get(bias, "Жду более сильного сетапа."))
    return "\n".join(lines)
