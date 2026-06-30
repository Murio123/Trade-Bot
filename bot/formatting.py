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
        "📐 Risk Management (ATR-based):",
        f"🛑 Стоп: {_fmt_price(signal.get('stop_loss'))} "
        f"({signal.get('atr_multiplier_used', config.ATR_MULTIPLIER)}×ATR)",
        f"🎯 Цель 1: {_fmt_price(signal.get('target_1'))}",
        f"🎯 Цель 2: {_fmt_price(signal.get('target_2'))}",
        f"💼 Размер позиции: {signal.get('position_size')} {config.SYMBOL_DISPLAY} "
        f"({signal.get('risk_percent', config.RISK_PERCENT)}% риска)",
        "",
        f"📈 Confluence Score: {min(signal.get('score', 0), 10)}/10",
        f"• HTF Bias (1D): {BIAS_LABEL.get(signal.get('htf_bias', 'neutral'))} (фильтр пройден)",
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


def format_reversal_alert(ctx: dict[str, Any], direction: str,
                          factors: list[str], strong: bool) -> str:
    head = "🟢 Возможное ДНО" if direction == "bull" else "🔴 Возможный ПИК"
    tf = ctx.get("timeframe", "").upper()
    lines = [
        f"🔔 {head}{' (сильное)' if strong else ''} | {config.SYMBOL_DISPLAY} {tf}",
        f"Цена: {_fmt_price(ctx.get('price'))}",
        f"Совпало факторов: {len(factors)}",
    ]
    lines += [f"  • {f}" for f in factors]
    lines.append("")
    lines.append("⚠️ Сигнал на истощение тренда — это фейд. Жди подтверждения "
                 "входной свечой и учитывай старший тренд.")
    return "\n".join(lines)


def format_reversal(ctx: dict[str, Any]) -> str:
    rev = ctx.get("reversal", {})
    price = ctx.get("price")
    lines = [
        f"🔄 Анализ разворота {config.SYMBOL_DISPLAY} | {ctx.get('timeframe', '').upper()}",
        f"Цена: {_fmt_price(price)}",
        "",
    ]
    bull = rev.get("factors_bull") or []
    bear = rev.get("factors_bear") or []
    if not bull and not bear:
        lines.append("Признаков истощения/разворота сейчас нет.")
        lines.append("Цена не на свинговом экстремуме — жду формирования дна/пика.")
        return "\n".join(lines)

    if rev.get("bullish_reversal"):
        lines.append(f"🟢 Возможное ДНО{' (сильное)' if rev.get('bull_strong') else ''} — "
                     f"{rev.get('bull_score')} подтверждения:")
        lines += [f"  • {f}" for f in bull]
    elif bull:
        lines.append(f"🟢 Слабые признаки дна ({rev.get('bull_score')}/2):")
        lines += [f"  • {f}" for f in bull]

    if rev.get("bearish_reversal"):
        lines.append(f"🔴 Возможный ПИК{' (сильный)' if rev.get('bear_strong') else ''} — "
                     f"{rev.get('bear_score')} подтверждения:")
        lines += [f"  • {f}" for f in bear]
    elif bear:
        lines.append(f"🔴 Слабые признаки пика ({rev.get('bear_score')}/2):")
        lines += [f"  • {f}" for f in bear]

    lines.append("")
    lines.append("⚠️ Развороты — это фейд движения: ниже винрейт, выше R:R. "
                 "Лучше брать в сторону старшего тренда.")
    return "\n".join(lines)


def format_levels(ctx: dict[str, Any]) -> str:
    ob = ctx.get("order_blocks", {})
    liq = ctx.get("liquidity", {})
    vp = ctx.get("volume_profile", {})
    lines = [f"📐 Ключевые уровни {config.SYMBOL_DISPLAY} | {ctx.get('timeframe', '').upper()}",
             f"Цена: {_fmt_price(ctx.get('price'))}", ""]

    bull_ob = ob.get("bullish_ob")
    bear_ob = ob.get("bearish_ob")
    lines.append("🧱 Order Blocks:")
    if bull_ob:
        lines.append(f"  • Бычий OB: {_fmt_price(bull_ob['low'])}–{_fmt_price(bull_ob['high'])}")
    if bear_ob:
        lines.append(f"  • Медвежий OB: {_fmt_price(bear_ob['low'])}–{_fmt_price(bear_ob['high'])}")
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

    ind = ctx.get("ind_signal", {})
    bbw = ind.get("bbw")
    if bbw is not None:
        regime = "сжатие 🔸" if ind.get("bb_squeeze") else (
            "расширение 🔶" if ind.get("bb_expansion") else "норма")
        lines.append("")
        lines.append(f"📏 BBW: {bbw:.4f} ({regime})")
    return "\n".join(lines)


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
    "htf_filter": "сигнал против дневного тренда (HTF-фильтр)",
    "diversity": "мало категориального разнообразия (нужно ≥3 категории)",
    "below_threshold": "очков недостаточно для журнала (нужно ≥5)",
    "wait_for_sweep": "впереди вероятное снятие ликвидности — ждём свип",
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

    lines.append(f"HTF (1D): {HTF_LINE.get(bias, bias)}")

    if long_s is not None and short_s is not None:
        lines.append(f"Score: лонг {long_s} / шорт {short_s}")

    lines.append("")
    lines.append(BLOCK_HINT.get(bias, "Жду более сильного сетапа."))
    return "\n".join(lines)
