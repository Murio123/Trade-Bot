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
    lines = [
        f"🤖 {config.SYMBOL_DISPLAY} СИГНАЛ | {signal.get('timeframe', '').upper()} | {ts_str}",
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


def format_journal(stats: dict[str, Any]) -> str:
    return "\n".join([
        "📒 Статистика журнала сделок",
        f"Всего сделок: {stats.get('total', 0)}",
        f"Винрейт: {stats.get('winrate', 0)}%  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)",
        f"Средний R/R: {stats.get('avg_r', 0)}R",
    ])


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
