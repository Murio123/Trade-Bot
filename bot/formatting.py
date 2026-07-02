"""Render signals and analysis into the Telegram message format from the ТЗ."""
from __future__ import annotations

from datetime import datetime
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


def format_signal(signal: dict[str, Any]) -> str:
    ts = signal.get("timestamp")
    if isinstance(ts, datetime):
        ts_str = ts.strftime("%Y-%m-%d %H:%M UTC")
    else:
        ts_str = str(ts or "")

    direction = signal.get("direction", "")
    style = signal.get("style_label", "СИГНАЛ")
    emoji = signal.get("style_emoji", "🤖")
    lines = [
        f"{emoji} {config.SYMBOL_DISPLAY} {style} | {signal.get('timeframe', '').upper()} | {ts_str}",
        f"📊 Направление: {DIRECTION_LABEL.get(direction, direction)}",
        f"💰 Вход: {_fmt_price(signal.get('entry_price'))}",
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
        f"📈 Confluence Score: {min(signal.get('score', 0), 10)}/10",
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

    conf = signal.get("confidence")
    if conf is not None:
        lines.append(f"⚡ Уверенность ИИ: {int(round(conf * 100))}%")

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
            f"  🎯 TP1: {_fmt_price(plan['tp1'])} (R:R {plan['rr']})",
            f"  🎯 TP2: {_fmt_price(plan['tp2'])}",
        ]

    if alignment == "aligned":
        lines += ["", "✅ Разворот В СТОРОНУ тренда 1D — покупка отката, "
                      "самый надёжный тип входа."]
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
            out.append(f"🟢 Дно подтверждено на {n} ТФ ПО тренду вверх — покупка отката, "
                       "надёжный сетап.")
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
            out.append(f"🔴 Пик подтверждён на {n} ТФ ПО тренду вниз — шорт отскока, "
                       "надёжный сетап.")
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


def format_levels(ctx: dict[str, Any]) -> str:
    ob = ctx.get("order_blocks", {})
    liq = ctx.get("liquidity", {})
    vp = ctx.get("volume_profile", {})
    price = ctx.get("price")
    lines = [f"📐 Ключевые уровни {config.SYMBOL_DISPLAY} | {ctx.get('timeframe', '').upper()}",
             f"Цена: {_fmt_price(price)}", ""]

    # Nearest support / resistance across all level sources.
    support, resistance = _nearest_sr(ctx, price)
    lines.append("🎯 Ближайшие уровни:")
    if resistance is not None:
        lines.append(f"  🔺 Сопротивление: {_fmt_price(resistance)} "
                     f"(+{(resistance - price) / price * 100:.1f}%)")
    if support is not None:
        lines.append(f"  🔻 Поддержка: {_fmt_price(support)} "
                     f"(−{(price - support) / price * 100:.1f}%)")
    if support is None and resistance is None:
        lines.append("  • уровни рядом не найдены")
    lines.append("")

    bull_ob = ob.get("bullish_ob")
    bear_ob = ob.get("bearish_ob")
    zone_tfs = ctx.get("zone_tfs") or []
    suffix = f" (зоны с {'/'.join(t.upper() for t in zone_tfs)})" if zone_tfs else ""
    lines.append(f"🧱 Order Blocks{suffix}:")
    if bull_ob:
        tf = f" [{bull_ob.get('tf', '').upper()}]" if bull_ob.get("tf") else ""
        lines.append(f"  • Бычий OB{tf}: {_fmt_price(bull_ob['low'])}–{_fmt_price(bull_ob['high'])}")
    if bear_ob:
        tf = f" [{bear_ob.get('tf', '').upper()}]" if bear_ob.get("tf") else ""
        lines.append(f"  • Медвежий OB{tf}: {_fmt_price(bear_ob['low'])}–{_fmt_price(bear_ob['high'])}")
    if not bull_ob and not bear_ob:
        lines.append("  • активных OB не найдено")

    lines.append("")
    lines.append("💧 Ликвидность:")
    eq_h = liq.get("equal_highs") or []
    eq_l = liq.get("equal_lows") or []
    if eq_h:
        lines.append("  • Equal highs: " + ", ".join(_fmt_price(x) for x in eq_h))
    if eq_l:
        lines.append("  • Equal lows: " + ", ".join(_fmt_price(x) for x in eq_l))
    if not eq_h and not eq_l:
        lines.append("  • кластеры стопов не обнаружены")

    lines.append("")
    lines.append("📊 Volume Profile:")
    lines.append(f"  • POC: {_fmt_price(vp.get('poc'))}")
    lines.append(f"  • VAH: {_fmt_price(vp.get('vah'))}")
    lines.append(f"  • VAL: {_fmt_price(vp.get('val'))}")

    # Liquidation magnets (OI-based estimate).
    liq_map = ctx.get("liquidation_map", {})
    if liq_map.get("source") == "oi_estimate" and price:
        shorts = [c["price"] for c in liq_map.get("short_liquidations", []) if c["price"] > price]
        longs = [c["price"] for c in liq_map.get("long_liquidations", []) if c["price"] < price]
        nearest_short = min(shorts) if shorts else None
        nearest_long = max(longs) if longs else None
        if nearest_short or nearest_long:
            lines.append("")
            lines.append("💥 Зоны ликвидаций (магниты):")
            if nearest_short:
                lines.append(f"  • Шорты сверху: {_fmt_price(nearest_short)} "
                             f"(+{(nearest_short - price) / price * 100:.1f}%)")
            if nearest_long:
                lines.append(f"  • Лонги снизу: {_fmt_price(nearest_long)} "
                             f"(−{(price - nearest_long) / price * 100:.1f}%)")

    fvg = ctx.get("fvg", {})
    bull_fvg = fvg.get("bullish_fvg")
    bear_fvg = fvg.get("bearish_fvg")
    if bull_fvg or bear_fvg:
        lines.append("")
        lines.append("🧩 FVG (имбаланс):")
        if bull_fvg:
            lines.append(f"  • Бычий: {_fmt_price(bull_fvg['low'])}–{_fmt_price(bull_fvg['high'])}")
        if bear_fvg:
            lines.append(f"  • Медвежий: {_fmt_price(bear_fvg['low'])}–{_fmt_price(bear_fvg['high'])}")

    eq = ctx.get("equilibrium", {})
    if eq.get("eq"):
        zone_ru = {"discount": "🟢 дисконт (зона лонгов)",
                   "premium": "🔴 премиум (зона шортов)",
                   "equilibrium": "⚪ равновесие"}.get(eq.get("zone"), eq.get("zone"))
        lines.append("")
        lines.append(f"⚖️ Диапазон: {_fmt_price(eq.get('low'))}–{_fmt_price(eq.get('high'))}")
        lines.append(f"  Равновесие: {_fmt_price(eq.get('eq'))} | сейчас {zone_ru} "
                     f"({int(eq.get('pos', 0.5) * 100)}%)")

    # EMA dynamic levels (1D / 4H, 50 & 200).
    inds = ctx.get("inds_by_tf", {})
    ema_rows = []
    for tf in ("1d", "4h"):
        for span in ("ema200", "ema50"):
            v = inds.get(tf, {}).get(span)
            if v and price:
                arrow = "🔺" if v > price else "🔻"
                sign = "+" if v > price else "−"
                ema_rows.append(f"  • {tf.upper()} {span.upper()}: {_fmt_price(v)} "
                                f"({sign}{abs(v - price) / price * 100:.1f}%) {arrow}")
    if ema_rows:
        lines.append("")
        lines.append("📈 EMA (динамические уровни):")
        lines += ema_rows

    ind = ctx.get("ind_signal", {})
    bbw = ind.get("bbw")
    if bbw is not None:
        regime = "сжатие 🔸" if ind.get("bb_squeeze") else (
            "расширение 🔶" if ind.get("bb_expansion") else "норма")
        lines.append("")
        lines.append(f"📏 BBW: {bbw:.4f} ({regime})")

    # Synthesis.
    support, resistance = _nearest_sr(ctx, price)
    lines.append("")
    lines += _levels_conclusion(ctx, price, support, resistance)
    return "\n".join(lines)


def _levels_conclusion(ctx: dict[str, Any], price: float | None,
                       support: float | None, resistance: float | None) -> list[str]:
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

    if resistance and support:
        out.append(f"• Диапазон работы: поддержка {_fmt_price(support)} "
                   f"(−{(price - support) / price * 100:.1f}%) ↔ сопротивление "
                   f"{_fmt_price(resistance)} (+{(resistance - price) / price * 100:.1f}%).")

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


def format_fear(fng: dict[str, Any]) -> str:
    val = fng.get("value")
    cls = fng.get("classification")
    if val is None:
        return "😶 Индекс страха/жадности недоступен."
    emoji = "😱" if val < 25 else "😟" if val < 45 else "😐" if val < 55 else "🙂" if val < 75 else "🤑"
    return f"{emoji} Индекс страха и жадности: {val}/100 ({cls})"


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
}

BLOCK_HINT = {
    "bullish": "Жду совпадения импульса с трендом вверх.",
    "bearish": "Жду совпадения импульса с трендом вниз.",
    "neutral": "Жду формирования чёткого дневного тренда.",
}


DECISION_EMOJI = {
    "STRONG BUY": "🟢🟢", "BUY": "🟢", "WEAK BUY": "🟢",
    "NO TRADE": "⚪", "WEAK SELL": "🔴", "SELL": "🔴", "STRONG SELL": "🔴🔴",
}

SECTION_LABEL = {
    "trend_alignment": "Тренд",
    "market_structure": "Структура",
    "liquidity": "Ликвидность",
    "volume": "Объём",
    "momentum": "Импульс",
    "derivatives": "Деривативы",
    "macro": "Макро",
    "historical": "История",
    "risk_profile": "R:R",
    "execution": "Исполнение",
}


def format_deep(quality: dict[str, Any], ctx: dict[str, Any], ai_text: str) -> str:
    decision = quality.get("decision", "NO TRADE")
    overall = quality.get("overall", 0)
    direction = quality.get("direction")
    scores = quality.get("scores", {})
    plan = quality.get("plan", {})
    hist = ctx.get("historical", {})
    vol = ctx.get("volatility", {})

    lines = [
        f"🏛 ГЛУБОКИЙ АНАЛИЗ {config.SYMBOL_DISPLAY} | 1D/12H/4H",
        f"Цена: {_fmt_price(ctx.get('price'))}",
        "",
        f"{DECISION_EMOJI.get(decision, '')} Решение: {decision}",
        f"📊 Trade Quality Score: {overall}/100",
        "",
        "Разбивка (взвешенная):",
    ]
    for item in quality.get("breakdown", []):
        lines.append(f"• {item['label']}: {item['earned']}/{item['max']}")

    if direction and decision != "NO TRADE":
        lines += [
            "",
            f"🎯 План ({'ЛОНГ' if direction == 'long' else 'ШОРТ'}):",
            f"Вход: {_fmt_price(plan.get('entry'))}",
            f"🛑 Стоп: {_fmt_price(plan.get('stop'))} (за структурой)",
            f"🎯 TP1: {_fmt_price(plan.get('tp1'))}",
            f"🎯 TP2: {_fmt_price(plan.get('tp2'))}",
            f"🎯 TP3: {_fmt_price(plan.get('tp3'))}",
            f"R:R ≈ {plan.get('rr')} | макс. просадка {plan.get('max_drawdown_pct')}%",
        ]

    # For NO TRADE the full plan is hidden — still show the hypothetical R:R.
    if decision == "NO TRADE" and direction and plan.get("rr") is not None:
        lines += [
            "",
            f"ℹ️ Если бы вход ({'ЛОНГ' if direction == 'long' else 'ШОРТ'}): "
            f"стоп {_fmt_price(plan.get('stop'))}, R:R ≈ {plan.get('rr')}",
        ]

    rev = ctx.get("reversal", {})
    if rev.get("bullish_reversal") or rev.get("bearish_reversal"):
        is_bull = rev.get("bullish_reversal")
        factors = rev.get("factors_bull") if is_bull else rev.get("factors_bear")
        strong = rev.get("bull_strong") if is_bull else rev.get("bear_strong")
        lines += [
            "",
            f"🔄 Истощение тренда: {'возможное ДНО 🟢' if is_bull else 'возможный ПИК 🔴'}"
            f"{' (сильное)' if strong else ''}",
            "  • " + "\n  • ".join(factors),
        ]

    em = (vol.get("expected_move") or {}).get("7d", {})
    lines += [
        "",
        f"📈 Волатильность: {vol.get('regime')}, ATR-перцентиль {vol.get('atr_percentile')}%",
        f"Ожидаемое движение 7д: ±{em.get('pct')}%" if em else "",
        f"🕰 Аналоги: {hist.get('matches', 0)} | "
        f"в сторону сделки {_hist_dir_pct(hist, direction)}% | "
        f"ср. {hist.get('avg_return')}% | conf {hist.get('confidence')}",
    ]

    if ai_text:
        lines += ["", "— — —", ai_text]

    if decision.startswith("WEAK"):
        lines += ["", "⚠️ Слабый сетап — рассматривать осторожно, уменьшенным объёмом."]
    elif decision == "NO TRADE":
        lines += ["", "💡 Кэш — тоже позиция. Сделка не форсируется."]

    return "\n".join(lines)


def _hist_dir_pct(hist: dict[str, Any], direction: str | None) -> Any:
    bp = hist.get("bullish_pct")
    if bp is None:
        return "н/д"
    return round(bp if direction == "long" else 100 - bp, 1)


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


def format_status(s: dict[str, Any]) -> str:
    dry = s.get("dry_run")
    db_ok = s.get("db_connected")
    lines = [
        "🩺 Статус бота",
        "",
        f"💱 Источник данных: {s.get('exchange_active') or s.get('exchange_pref')}",
        f"💰 Цена {config.SYMBOL_DISPLAY}: {_fmt_price(s.get('price'))}",
        f"🗄 База данных: {'PostgreSQL ✅' if db_ok else 'in-memory ⚠️ (без персистентности)'}",
        f"📡 Режим: {'DRY-RUN 🧪 (уведомления НЕ шлются)' if dry else 'LIVE ✅ (уведомления включены)'}",
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


def format_blocked(result: dict[str, Any]) -> str:
    stage = result.get("blocked_at")
    bias = result.get("htf_bias", "neutral")
    long_s = result.get("long_score")
    short_s = result.get("short_score")

    lines = ["ℹ️ Нет активного сигнала"]
    lines.append(f"Причина: {BLOCK_REASON.get(stage, stage or 'нет данных')}")

    price = result.get("price")
    if price is not None:
        lines.append(f"Цена: {_fmt_price(price)}")

    lines.append(f"HTF ({result.get('htf_tf', '1d').upper()}): {HTF_LINE.get(bias, bias)}")

    if long_s is not None and short_s is not None:
        lines.append(f"Score: лонг {long_s} / шорт {short_s}")

    # Show which categories are active so it's clear what's missing.
    cats = result.get("category_scores") or {}
    active = {k: v for k, v in cats.items() if v > 0}
    if active:
        lines.append("Категории: " + ", ".join(
            f"{CATEGORY_LABEL.get(k, k)} {v}" for k, v in active.items()))
        if stage == "diversity":
            from config import MIN_DIVERSE_CATEGORIES
            lines.append(f"Сейчас {len(active)} категория(и), нужно ≥{MIN_DIVERSE_CATEGORIES} разных.")

    lines.append("")
    lines.append(BLOCK_HINT.get(bias, "Жду более сильного сетапа."))
    return "\n".join(lines)


CATEGORY_LABEL = {
    "trend": "Тренд", "momentum": "Импульс", "volume": "Объём",
    "structure": "Структура", "macro": "Макро",
}
