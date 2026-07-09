"""Telegram command handlers.

Commands: /signal /levels /funding /fear /backtest /ask /journal (+ /start /help).
Shared objects (Binance client, latest context cache) live in ``application.bot_data``.
"""
from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

import config
from analyzer.news import get_fear_greed
from ai import claude
from bot import formatting, guide, journal, keyboards
from database import db
from pipeline import gather_market_context, run_cascade

log = logging.getLogger(__name__)

HELP_TEXT = (
    "🤖 BTC Signal Bot — все команды\n\n"
    "🎯 Сигналы:\n"
    "/signal — свинг (вход 4H, структура 12H, тренд 1D)\n"
    "/intraday — интрадей (вход 15m, тренд 1H)\n"
    "/position — 🌊 позиционный (движения 2000-5000 пт, холд дни)\n\n"
    "🔍 Анализ:\n"
    "/reversal — дно/пик по 4 ТФ + план входа\n"
    "/levels — ключевые уровни + график\n"
    "/deep — глубокий разбор со score /100\n"
    "/market — цена, funding, L/S, Fear&Greed\n\n"
    "📒 Учёт:\n"
    "/journal — винрейт и R по сделкам\n"
    "/setalert <цена> — алерт по уровню (/alerts — список, /delalert — удалить)\n"
    "/backtest [swing|intraday] — подбор порога по истории\n\n"
    "⚙️ Сервис:\n"
    "/status — здоровье бота | /testalert — проверка уведомлений\n"
    "/guide — 📖 подробный гид по всем функциям\n\n"
    "💬 Любой текст без команды — вопрос к ИИ с рыночным контекстом."
)

WELCOME_TEXT = (
    "🤖 Привет! Я BTC Signal Bot.\n\n"
    "Анализирую BTC/USDT на нескольких таймфреймах и присылаю сигналы.\n"
    "Выбери действие на кнопках ниже 👇"
)


async def _gatekeeper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reject unknown users before any handler runs.

    Every message costs money (/ask calls the Anthropic API, /backtest burns
    CPU), so only whitelisted chat/user ids may interact. Empty whitelist =
    open mode (warned at startup).
    """
    from telegram.ext import ApplicationHandlerStop

    allowed = set(config.TELEGRAM_ALLOWED_CHAT_IDS)
    if not allowed and config.ALLOW_PUBLIC_ACCESS:
        return  # deliberately open bot
    ids = set()
    if update.effective_chat:
        ids.add(str(update.effective_chat.id))
    if update.effective_user:
        ids.add(str(update.effective_user.id))
    if ids & allowed:
        return
    log.warning("Rejected unauthorized access from %s", ids or "unknown")
    try:
        if update.callback_query:
            await update.callback_query.answer("⛔ Доступ ограничен", show_alert=False)
        elif update.effective_message:
            if not allowed:
                # First-run onboarding: closed by default, but tell the owner
                # their chat_id so they can whitelist themselves.
                chat_id = (update.effective_chat.id
                           if update.effective_chat else "?")
                await update.effective_message.reply_text(
                    "⛔ Доступ закрыт: whitelist не настроен.\n"
                    f"Ваш chat_id: {chat_id}\n"
                    "Добавьте его в TELEGRAM_ALLOWED_CHAT_IDS (или "
                    "TELEGRAM_ALERT_CHAT_IDS) в Railway и перезапустите бота.")
            else:
                await update.effective_message.reply_text(
                    "⛔ Это приватный бот, доступ ограничен.")
    except Exception:  # noqa: BLE001
        pass
    raise ApplicationHandlerStop


def _binance(context: ContextTypes.DEFAULT_TYPE):
    return context.application.bot_data["binance"]


async def _fresh_context(context: ContextTypes.DEFAULT_TYPE,
                         profile_name: str = "swing") -> dict[str, Any]:
    binance = _binance(context)
    ctx = await gather_market_context(binance, profile_name=profile_name)
    if profile_name == "swing":
        context.application.bot_data["last_context"] = ctx
    return ctx


async def _signal_display_price(context: ContextTypes.DEFAULT_TYPE) -> float | None:
    try:
        return float(await _binance(context).current_price())
    except Exception as exc:  # noqa: BLE001
        log.warning("signal display price fetch failed: %s", exc)
        return None


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Show the persistent reply keyboard and an inline quick-menu.
    await update.effective_message.reply_text(
        WELCOME_TEXT, reply_markup=keyboards.main_reply_keyboard()
    )
    await update.effective_message.reply_text(
        keyboards.MAIN_MENU_TITLE, reply_markup=keyboards.main_inline_keyboard()
    )


def _submenu_handler(section: str):
    """Reply-keyboard section press: send the submenu as a new message
    (reply buttons cannot edit anything — there is no source message to edit)."""
    async def _handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            keyboards.submenu_title(section),
            reply_markup=keyboards.submenu_keyboard(section))
    return _handler


menu_analyze_cmd = _submenu_handler("analyze")
menu_market_cmd = _submenu_handler("market")
menu_history_cmd = _submenu_handler("history")


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline navigation (callback_data 'menu:<section>'): edit in place."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    section = (query.data or "").split(":", 1)[-1]
    markup = keyboards.submenu_keyboard(section)
    if markup is None:  # "menu:main" and anything unknown -> main menu
        text, markup = keyboards.MAIN_MENU_TITLE, keyboards.main_inline_keyboard()
    else:
        text = keyboards.submenu_title(section)
    try:
        await query.edit_message_text(text, reply_markup=markup)
    except Exception as exc:  # noqa: BLE001 — e.g. "Message is not modified"
        log.debug("menu edit skipped: %s", exc)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        HELP_TEXT, reply_markup=keyboards.main_reply_keyboard()
    )


async def guide_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        guide.INTRO, reply_markup=guide.menu_keyboard()
    )


async def guide_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle guide navigation (callback_data 'help:<topic>')."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    key = (query.data or "").split(":", 1)[1] if ":" in (query.data or "") else "menu"
    if key == "menu":
        await query.edit_message_text(guide.INTRO, reply_markup=guide.menu_keyboard())
    else:
        await query.edit_message_text(guide.get_topic(key),
                                      reply_markup=guide.back_keyboard())


async def _run_signal(update: Update, context: ContextTypes.DEFAULT_TYPE,
                      profile_name: str,
                      display_price: float | None = None) -> None:
    from signal_engine.profiles import get_profile
    profile = get_profile(profile_name)
    await update.effective_message.reply_text(
        f"{profile['emoji']} Анализирую ({profile['label']})…")
    try:
        ctx = await _fresh_context(context, profile_name=profile_name)
        tf = profile["entry"]
        delivered_today = await db.signals_today(config.SYMBOL, timeframe=tf)
        last_signal = await db.last_signal(config.SYMBOL, timeframe=tf)
        open_now = await db.open_trades()
        result = await run_cascade(ctx, delivered_today, last_signal,
                                   interpret=True, profile_name=profile_name,
                                   open_trades=open_now)
    except Exception as exc:  # noqa: BLE001
        log.exception("signal (%s) failed", profile_name)
        await update.effective_message.reply_text(f"⚠️ Ошибка анализа: {exc}")
        return

    if result.get("status") == "blocked":
        text = formatting.format_blocked(result, display_price=display_price)
        # Continuity: no NEW setup does not mean the previous one vanished.
        note = formatting.format_last_signal_note(
            last_signal, max_age_hours=profile["cooldown_hours"] * 3)
        if note:
            text += "\n\n" + note
        await update.effective_message.reply_text(text)
        return

    # Persist what the user saw, so /signal has memory: cooldown works against
    # it and the setup can be shown later instead of "нет позиций".
    # Same guard as the scheduled stream: while a trade of this profile is
    # unresolved, the row is stored UNDELIVERED (it must not feed the cooldown
    # or daily limit), no journal trade is opened, and the reply is rendered
    # as an analysis update — never as a new actionable alert.
    risk_capped = False
    duplicate_suppressed = False
    if result.get("status") in ("alert", "journal"):
        if result["status"] == "alert":
            duplicate_suppressed = await journal.has_active_trade(
                config.SYMBOL, result.get("analysis_type"),
                result.get("timeframe"))
        record = {**result,
                  "delivered": result["status"] == "alert" and not duplicate_suppressed}
        signal_id = await db.insert_signal(record)
        if result["status"] == "alert":
            if duplicate_suppressed:
                log.info("signal alert suppressed: active signal already open "
                         "(manual /signal — informational reply only)")
            # Same aggregate risk cap as the scheduled stream.
            elif await journal.can_open_new_trade(config.SYMBOL):
                await journal.record_signal_as_trade(signal_id, result)
            else:
                risk_capped = True

    display_result = dict(result)
    display_result["display_price"] = (
        display_price if display_price is not None
        else result.get("executable_price_at_decision")
    )
    text = formatting.format_signal(display_result)
    if duplicate_suppressed:
        text = ("⚠️ Активный сигнал уже открыт. Новый сигнал не отправлен "
                "и не добавлен в журнал.\n"
                "Только обновление анализа, не новая торговая идея.\n\n" + text)
    if risk_capped:
        text += "\n\n" + formatting.format_risk_cap_note()
    if result.get("status") == "cooldown":
        text += "\n\n⏳ Это действующий сетап (в пределах cooldown) — не новый вход."
    await update.effective_message.reply_text(text)
    await _send_chart(update, ctx, result)


async def _send_chart(update: Update, ctx: dict[str, Any],
                      signal: dict[str, Any] | None = None) -> None:
    """Render and attach a chart; chart failures never break the reply."""
    import asyncio
    from bot import charts
    try:
        if signal is not None:
            path = await asyncio.to_thread(charts.render_signal_chart, ctx, signal)
        else:
            path = await asyncio.to_thread(charts.render_levels_chart, ctx)
        with open(path, "rb") as fh:
            await update.effective_message.reply_photo(photo=fh)
    except Exception as exc:  # noqa: BLE001
        log.warning("chart rendering failed: %s", exc)


async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    display_price = await _signal_display_price(context)
    await _run_signal(update, context, "swing", display_price=display_price)


async def intraday_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    display_price = await _signal_display_price(context)
    await _run_signal(update, context, "intraday", display_price=display_price)


async def position_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    display_price = await _signal_display_price(context)
    await _run_signal(update, context, "position", display_price=display_price)


async def reversal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("🔄 Ищу признаки дна/пика…")
    try:
        ctx = await _fresh_context(context, profile_name="swing")
    except Exception as exc:  # noqa: BLE001
        await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
        return
    await update.effective_message.reply_text(formatting.format_reversal(ctx))


async def levels_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        ctx = await _fresh_context(context)
    except Exception as exc:  # noqa: BLE001
        await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
        return
    await update.effective_message.reply_text(formatting.format_levels(ctx))
    await _send_chart(update, ctx)


async def funding_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    binance = _binance(context)
    try:
        funding = await binance.funding_rate()
        ls = await binance.long_short_ratio()
    except Exception as exc:  # noqa: BLE001
        await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
        return
    await update.effective_message.reply_text(formatting.format_funding(funding, ls))


async def market_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """One-screen market overview (price, funding, L/S, OI, Fear & Greed)."""
    binance = _binance(context)
    import asyncio
    price, funding, ls, oi, fng = await asyncio.gather(
        binance.current_price(), binance.funding_rate(),
        binance.long_short_ratio(), binance.open_interest(),
        get_fear_greed(), return_exceptions=True)
    def _ok(v, d):
        return d if isinstance(v, Exception) else v
    await update.effective_message.reply_text(formatting.format_market(
        _ok(price, None), _ok(funding, {}), _ok(ls, {}), _ok(oi, None), _ok(fng, {})))


async def fear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    fng = await get_fear_greed()
    await update.effective_message.reply_text(formatting.format_fear(fng))


async def journal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = await journal.get_stats(config.SYMBOL)
    open_trades = await db.open_trades()
    await update.effective_message.reply_text(
        formatting.format_journal(stats, open_count=len(open_trades))
    )


# Mode order + labels for the history screens.
_MODES = (("intraday", "⚡ Интрадей"), ("swing", "📊 Свинг"), ("position", "🌊 Позиционный"))


def _forecast_line(sig: dict[str, Any]) -> str:
    d = "🟢 ЛОНГ" if sig.get("direction") == "long" else "🔴 ШОРТ"
    created = sig.get("created_at")
    when = ""
    if created is not None:
        try:
            when = f" · {formatting.fmt_display_time(created, '%d.%m %H:%M')}"
        except (TypeError, ValueError, AttributeError):
            pass
    def p(v: Any) -> str:
        return f"{v:,.0f}".replace(",", " ") if v else "—"
    return (f"{d}{when}\n"
            f"Вход {p(sig.get('entry_price'))} | 🛑 {p(sig.get('stop_loss'))} | "
            f"🎯 {p(sig.get('target_1'))} / {p(sig.get('target_2'))}")


async def last_forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The most recent stored signal for the symbol, regardless of mode."""
    sig = await db.last_signal(config.SYMBOL)
    if not sig:
        await update.effective_message.reply_text(
            "📌 Прогнозов пока нет — запусти анализ через «📊 Новый анализ».")
        return
    mode = dict(_MODES).get(sig.get("analysis_type") or "", sig.get("timeframe") or "")
    await update.effective_message.reply_text(
        f"📌 Последний прогноз ({mode}):\n\n" + _forecast_line(sig))


async def recent_forecasts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Latest stored signal per mode (intraday / swing / position)."""
    parts = ["📌 Последние прогнозы по режимам:"]
    for mode, label in _MODES:
        sig = await db.last_signal(config.SYMBOL, analysis_type=mode)
        parts.append(f"\n{label}:")
        parts.append(_forecast_line(sig) if sig else "— пока нет")
    await update.effective_message.reply_text("\n".join(parts))


async def forecast_results_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hit-rates from the forecast outcome tracker, per mode and overall."""
    parts = ["✅ Результаты прогнозов (по данным outcome-трекера):"]
    any_data = False
    for mode, label in list(_MODES) + [(None, "Σ Все режимы")]:
        stats = await db.forecast_stats(config.SYMBOL, analysis_type=mode)
        if not stats.get("total"):
            parts.append(f"\n{label}: данных пока нет")
            continue
        any_data = True
        parts.append(
            f"\n{label}: {stats['total']} прогнозов "
            f"(в работе: {stats['unresolved']})\n"
            f"TP1 {stats['tp1_rate']}% | TP2 {stats['tp2_rate']}% | "
            f"стопов: {stats['stop_hits']}")
    if not any_data:
        parts.append("\nТаблица наполняется автоматически по мере "
                     "отслеживания прогнозов — загляни позже.")
    await update.effective_message.reply_text("\n".join(parts))


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from backtest import run_backtest
    arg = (context.args[0].lower() if context.args else "swing")
    profile = arg if arg in ("swing", "intraday", "position") else "swing"
    await update.effective_message.reply_text(
        f"⏳ Запускаю бэктест ({profile})… это ~10-20с.")
    binance = _binance(context)
    try:
        report = await run_backtest(binance, profile_name=profile)
    except Exception as exc:  # noqa: BLE001
        log.exception("backtest failed")
        await update.effective_message.reply_text(f"⚠️ Ошибка бэктеста: {exc}")
        return
    await update.effective_message.reply_text(report)


async def setalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/setalert <цена> [above|below] — notify when price crosses the level."""
    binance = _binance(context)
    args = context.args or []
    if not args:
        await update.effective_message.reply_text(
            "Использование: /setalert <цена> [above|below]\n"
            "Например: /setalert 60000 — направление определю сам по текущей цене.")
        return
    try:
        level = float(args[0].replace(",", "").replace(" ", ""))
    except ValueError:
        await update.effective_message.reply_text("⚠️ Не понял цену. Пример: /setalert 60000")
        return

    direction = args[1].lower() if len(args) > 1 and args[1].lower() in ("above", "below") else None
    if direction is None:
        try:
            price = await binance.current_price()
            direction = "above" if level > price else "below"
        except Exception:  # noqa: BLE001
            direction = "above"

    chat_id = str(update.effective_chat.id)
    alert_id = await db.add_price_alert(chat_id, config.SYMBOL, level, direction)
    arrow = "выше ↑" if direction == "above" else "ниже ↓"
    level_str = f"{level:,.0f}".replace(",", " ")
    await update.effective_message.reply_text(
        f"🔔 Алерт #{alert_id}: сообщу, когда {config.SYMBOL_DISPLAY} будет "
        f"{arrow} {level_str}\n"
        f"(проверка каждые {config.ALERT_CHECK_INTERVAL_MINUTES} мин; "
        "список — /alerts)")


async def alerts_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    items = [a for a in await db.active_price_alerts(config.SYMBOL)
             if str(a.get("chat_id")) == chat_id]
    if not items:
        await update.effective_message.reply_text(
            "Активных алертов нет. Создать: /setalert <цена>")
        return
    lines = ["🔔 Твои ценовые алерты:"]
    for a in items:
        arrow = "↑ выше" if a["direction"] == "above" else "↓ ниже"
        lines.append(f"  #{a['id']}: {arrow} {a['level']:,.0f}".replace(",", " "))
    lines.append("\nУдалить: /delalert <id>")
    await update.effective_message.reply_text("\n".join(lines))


async def delalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args or not args[0].isdigit():
        await update.effective_message.reply_text("Использование: /delalert <id> (см. /alerts)")
        return
    ok = await db.delete_price_alert(int(args[0]), str(update.effective_chat.id))
    await update.effective_message.reply_text(
        "🗑 Алерт удалён." if ok else "⚠️ Алерт не найден (см. /alerts).")


async def testalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Diagnose why auto-alerts may not arrive, and send a test if configured."""
    from bot import alerts
    problems = []
    if config.DRY_RUN and not config.SEND_DRY_RUN_ALERTS:
        problems.append("• DRY_RUN=true и SEND_DRY_RUN_ALERTS=false — "
                        "dry-run отправка алертов выключена.")
    if not config.TELEGRAM_ALERT_CHAT_IDS:
        problems.append("• TELEGRAM_ALERT_CHAT_IDS пуст — нет получателей. "
                        "Впиши свой chat_id (узнать: @userinfobot).")
    if problems:
        await update.effective_message.reply_text(
            "❌ Автосигналы сейчас НЕ придут:\n" + "\n".join(problems)
            + "\n\nПосле исправления снова нажми /testalert."
        )
        return
    sent = await alerts.send_signal_alert(
        context.application.bot,
        "✅ Тестовый алерт: канал автосигналов настроен и работает.\n"
        "Реальные сигналы придут сюда же, когда сетап наберёт порог.")
    if not sent:
        await update.effective_message.reply_text(
            "❌ Тестовый алерт не был отправлен. Проверь Telegram получателей "
            "и логи отправки.")
        return
    mode = "DRY-RUN / MANUAL ONLY" if config.DRY_RUN else "LIVE"
    await update.effective_message.reply_text(
        f"✅ Конфигурация в порядке ({mode}, получателей: "
        f"{len(config.TELEGRAM_ALERT_CHAT_IDS)}).\n"
        "Отправил тестовый алерт получателям — проверь, что он пришёл.\n\n"
        "Если тест пришёл, но сигналов нет — значит сетап пока не набирает "
        "порог (score ≥8). Это нормально: используй /signal, чтобы видеть "
        "текущий расклад, или снизь SCORE_ALERT_MIN.")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    binance = _binance(context)
    bot_data = context.application.bot_data

    price = None
    try:
        price = await binance.current_price()
    except Exception as exc:  # noqa: BLE001
        log.warning("status_cmd: price fetch failed: %s", exc)

    try:
        signals_today = len(await db.signals_today(config.SYMBOL))
        open_trades = len(await db.open_trades())
        stats = await db.journal_stats(config.SYMBOL)
    except Exception:  # noqa: BLE001
        signals_today, open_trades, stats = 0, 0, {}

    s = {
        "exchange_pref": config.EXCHANGE,
        "exchange_active": getattr(binance, "active_name", None),
        "price": price,
        "db_connected": db.pool is not None,
        "ai_ok": (bot_data.get("ai_health") or {}).get("ok"),
        "ai_error": (bot_data.get("ai_health") or {}).get("error"),
        "ai_model": config.ANTHROPIC_MODEL,
        "dry_run": config.DRY_RUN,
        "send_dry_run_alerts": config.SEND_DRY_RUN_ALERTS,
        "alert_chats": len(config.TELEGRAM_ALERT_CHAT_IDS),
        "last_analysis_at": bot_data.get("last_analysis_at"),
        "last_analysis_tf": bot_data.get("last_analysis_tf"),
        "last_analysis_status": bot_data.get("last_analysis_status"),
        "last_analysis_blocked_at": bot_data.get("last_analysis_blocked_at"),
        "signal_tf": "4H (свинг)",
        "fast_tf": "15m (интрадей)" if config.ENABLE_FAST_ANALYSIS else "—",
        "signals_today": signals_today,
        "max_per_day": config.MAX_SIGNALS_PER_DAY,
        "open_trades": open_trades,
        "closed_trades": stats.get("total", 0),
        "winrate": stats.get("winrate", 0),
    }
    await update.effective_message.reply_text(formatting.format_status(s))


# Поля решения, которые /deep только ОБЪЯСНЯЕТ — движок уже их посчитал и
# сохранил. forecast_to_deep_input их лишь переносит, ничего не пересчитывая.
_DEEP_PASSTHROUGH = (
    "analysis_status", "candidate_direction", "final_bias",
    "long_score", "short_score", "raw_confidence",
    "expected_move_min_pct", "expected_move_max_pct",
    "expected_move_points", "expected_move_percent", "expected_move_atr",
    "stop_loss", "take_profit_levels", "risk_reward",
    "no_trade_reasons", "blocked_gate",
    "market_regime", "volatility_regime",
    "strategy_version", "context_version", "entry_zone",
)


def forecast_to_deep_input(forecast: dict[str, Any]) -> dict[str, Any]:
    """Сохранённая forecasts-строка -> вход для render_deep. Чистая функция:
    ничего не решает и не выдумывает направление — только переносит уже
    посчитанные поля. Синтезирует ТОЛЬКО совместимый ``status``, чтобы
    render_deep не приписал ложный blocked_gate="unknown" (в строке нет ключа
    ``status``, а его дефолт там — "blocked"). Вход не мутируется."""
    deep: dict[str, Any] = {k: forecast.get(k) for k in _DEEP_PASSTHROUGH}

    analysis_status = forecast.get("analysis_status")
    blocked_gate = forecast.get("blocked_gate")
    if analysis_status == "ENTER":
        status = "alert"
    elif analysis_status == "WAIT":
        status = "journal"
    elif analysis_status == "NO_TRADE":
        status = "blocked" if blocked_gate else "ignored"
    else:
        status = "ignored"
    deep["status"] = status
    return deep


async def deep_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Объяснить ПОСЛЕДНИЙ сохранённый прогноз движка. /deep ничего не решает
    и не пересчитывает: saved forecast — источник истины."""
    from signal_engine.deep_renderer import render_deep
    from signal_engine.profiles import get_profile

    analysis_type = get_profile("swing")["analysis_type"]
    forecast = await db.latest_forecast(symbol=config.SYMBOL,
                                        analysis_type=analysis_type)
    if forecast is None:
        await update.effective_message.reply_text(
            "📭 Пока нет сохранённого прогноза. Дождитесь ближайшего анализа."
        )
        return

    sections = render_deep(forecast_to_deep_input(forecast))
    await update.effective_message.reply_text(
        formatting.format_deep_sections(sections)
    )


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args) if context.args else ""
    if not question:
        await update.effective_message.reply_text("Использование: /ask <ваш вопрос>")
        return
    # The cached swing context refreshes hourly; for a live question re-gather
    # when it is older than 30 minutes (a stale price misleads the answer).
    from datetime import datetime, timedelta, timezone
    ctx = context.application.bot_data.get("last_context")
    ts = (ctx or {}).get("timestamp")
    stale = ts is None or (datetime.now(timezone.utc) - ts) > timedelta(minutes=30)
    if ctx is None or stale:
        try:
            ctx = await _fresh_context(context)
        except Exception as exc:  # noqa: BLE001
            if ctx is None:
                await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
                return
            log.warning("ask: context refresh failed, using stale cache: %s", exc)
    market_context = _ask_context(ctx)
    answer = await claude.ask(question, market_context)
    await update.effective_message.reply_text(answer)


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route reply-keyboard button presses; otherwise treat text as /ask."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    command = keyboards.LABEL_TO_COMMAND.get(text)
    if command == "ask_prompt":
        await update.effective_message.reply_text(
            "🧠 Напиши свой вопрос обычным сообщением — я отвечу с учётом "
            "текущего рынка. Например: «Стоит ли ждать откат к 80k?»"
        )
        return
    if command:
        handler = COMMAND_DISPATCH.get(command)
        if handler:
            await handler(update, context)
            return

    # Not a button -> free-form question to Claude.
    context.args = text.split()
    await ask_cmd(update, context)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline-menu taps (callback_data 'cmd:<name>')."""
    query = update.callback_query
    if not query:
        return
    await query.answer()  # stop the loading spinner
    data = query.data or ""
    if not data.startswith("cmd:"):
        return
    command = data.split(":", 1)[1]
    handler = COMMAND_DISPATCH.get(command)
    if handler:
        await handler(update, context)


def _ask_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-serialisable slice of the market context for Claude."""
    ind = ctx.get("ind_signal", {})
    return {
        "symbol": ctx.get("symbol"),
        "timeframe": ctx.get("timeframe"),
        "price": ctx.get("price"),
        "rsi": ind.get("rsi"),
        "macd_hist": ind.get("macd_hist"),
        "ema20": ind.get("ema20"),
        "ema50": ind.get("ema50"),
        "ema200": ind.get("ema200"),
        "atr": ctx.get("atr"),
        "funding": (ctx.get("funding") or {}).get("current"),
        "open_interest": ctx.get("open_interest"),
        "long_short_ratio": (ctx.get("long_short_ratio") or {}).get("ratio"),
        "cvd_bullish": (ctx.get("cvd") or {}).get("cvd_bullish"),
        "order_blocks": {
            "bullish": (ctx.get("order_blocks") or {}).get("bullish_ob"),
            "bearish": (ctx.get("order_blocks") or {}).get("bearish_ob"),
        },
        "volume_profile": {
            "poc": (ctx.get("volume_profile") or {}).get("poc"),
            "vah": (ctx.get("volume_profile") or {}).get("vah"),
            "val": (ctx.get("volume_profile") or {}).get("val"),
        },
        "macro": ctx.get("macro"),
        "best_session": (ctx.get("sessions") or {}).get("best_session"),
    }


# Maps an internal command name to its handler (used by reply-keyboard
# buttons and inline-menu callbacks alike).
COMMAND_DISPATCH = {
    "menu_analyze": menu_analyze_cmd,
    "menu_market": menu_market_cmd,
    "menu_history": menu_history_cmd,
    "last_forecast": last_forecast_cmd,
    "recent_forecasts": recent_forecasts_cmd,
    "forecast_results": forecast_results_cmd,
    "signal": signal_cmd,
    "intraday": intraday_cmd,
    "position": position_cmd,
    "deep": deep_cmd,
    "reversal": reversal_cmd,
    "levels": levels_cmd,
    "market": market_cmd,
    "funding": funding_cmd,
    "fear": fear_cmd,
    "journal": journal_cmd,
    "alerts": alerts_cmd,
    "backtest": backtest_cmd,
    "status": status_cmd,
    "testalert": testalert_cmd,
    "guide": guide_cmd,
    "help": help_cmd,
}

# Shown in the Telegram "/" command menu.
BOT_COMMANDS = [
    ("signal", "📊 Свинг-сигнал (вход 4H)"),
    ("intraday", "⚡ Интрадей-сигнал (вход 15m)"),
    ("position", "🌊 Позиционный (движения 2000-5000 пт)"),
    ("reversal", "🔄 Дно/пик по 4 ТФ"),
    ("levels", "📐 Ключевые уровни"),
    ("deep", "🏛 Глубокий анализ /100"),
    ("market", "💹 Рынок: цена, funding, L/S, F&G"),
    ("journal", "📒 Статистика сделок"),
    ("setalert", "🔔 Поставить алерт по цене"),
    ("alerts", "📋 Мои ценовые алерты"),
    ("backtest", "📈 Подбор порога по истории"),
    ("status", "🩺 Статус бота"),
    ("guide", "📖 Гид по функциям"),
    ("ask", "🧠 Вопрос к ИИ"),
    ("help", "❓ Все команды"),
]


async def _post_init(application) -> None:
    """Register the '/' command menu once the bot is initialised."""
    from telegram import BotCommand

    try:
        await application.bot.set_my_commands(
            [BotCommand(c, d) for c, d in BOT_COMMANDS]
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("set_my_commands failed: %s", exc)


def register_handlers(application) -> None:
    from telegram.ext import (CallbackQueryHandler, CommandHandler,
                              MessageHandler, TypeHandler, filters)

    # Access control runs before everything else (group -1).
    if config.TELEGRAM_ALLOWED_CHAT_IDS:
        application.add_handler(TypeHandler(Update, _gatekeeper), group=-1)
    else:
        log.warning("TELEGRAM_ALLOWED_CHAT_IDS is empty -> bot is OPEN to anyone; "
                    "any stranger can spend your Anthropic tokens via /ask")

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("signal", signal_cmd))
    application.add_handler(CommandHandler("intraday", intraday_cmd))
    application.add_handler(CommandHandler("position", position_cmd))
    application.add_handler(CommandHandler("deep", deep_cmd))
    application.add_handler(CommandHandler("reversal", reversal_cmd))
    application.add_handler(CommandHandler("levels", levels_cmd))
    application.add_handler(CommandHandler("market", market_cmd))
    application.add_handler(CommandHandler("funding", funding_cmd))
    application.add_handler(CommandHandler("fear", fear_cmd))
    application.add_handler(CommandHandler("journal", journal_cmd))
    application.add_handler(CommandHandler("backtest", backtest_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("testalert", testalert_cmd))
    application.add_handler(CommandHandler("setalert", setalert_cmd))
    application.add_handler(CommandHandler("alerts", alerts_cmd))
    application.add_handler(CommandHandler("delalert", delalert_cmd))
    application.add_handler(CommandHandler("guide", guide_cmd))
    application.add_handler(CommandHandler("ask", ask_cmd))
    application.add_handler(CallbackQueryHandler(button_callback, pattern=r"^cmd:"))
    application.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^menu:"))
    application.add_handler(CallbackQueryHandler(guide_callback, pattern=r"^help:"))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_message)
    )
