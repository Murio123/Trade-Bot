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
from bot import formatting, journal, keyboards
from database import db
from pipeline import gather_market_context, run_cascade

log = logging.getLogger(__name__)

HELP_TEXT = (
    "🤖 BTC Signal Bot\n\n"
    "Нажимай кнопки ниже или используй команды:\n"
    "/signal — текущий сигнал (включая слабые 5-7)\n"
    "/deep — глубокий институциональный анализ (1D/12H/4H, score /100)\n"
    "/levels — ключевые уровни (OB, ликвидность, volume profile)\n"
    "/funding — funding rate + аномальность\n"
    "/fear — индекс страха/жадности\n"
    "/backtest — результаты стратегии за период\n"
    "/journal — статистика журнала (винрейт, R/R)\n"
    "/ask <вопрос> — свободный вопрос к Claude с рыночным контекстом\n\n"
    "💬 Любой текст без команды я восприму как вопрос к ИИ."
)

WELCOME_TEXT = (
    "🤖 Привет! Я BTC Signal Bot.\n\n"
    "Анализирую BTC/USDT на нескольких таймфреймах и присылаю сигналы.\n"
    "Выбери действие на кнопках ниже 👇"
)


def _binance(context: ContextTypes.DEFAULT_TYPE):
    return context.application.bot_data["binance"]


async def _fresh_context(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    binance = _binance(context)
    ctx = await gather_market_context(binance, signal_timeframe=config.SIGNAL_TIMEFRAME)
    context.application.bot_data["last_context"] = ctx
    return ctx


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Show the persistent reply keyboard and an inline quick-menu.
    await update.effective_message.reply_text(
        WELCOME_TEXT, reply_markup=keyboards.main_reply_keyboard()
    )
    await update.effective_message.reply_text(
        "Быстрое меню:", reply_markup=keyboards.main_inline_keyboard()
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        HELP_TEXT, reply_markup=keyboards.main_reply_keyboard()
    )


async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("⏳ Анализирую рынок…")
    try:
        ctx = await _fresh_context(context)
        delivered_today = await db.signals_today(config.SYMBOL)
        last_signal = await db.last_signal(config.SYMBOL)
        result = await run_cascade(ctx, delivered_today, last_signal, interpret=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("signal_cmd failed")
        await update.effective_message.reply_text(f"⚠️ Ошибка анализа: {exc}")
        return

    if result.get("status") == "blocked":
        await update.effective_message.reply_text(formatting.format_blocked(result))
        return
    await update.effective_message.reply_text(formatting.format_signal(result))


async def levels_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        ctx = await _fresh_context(context)
    except Exception as exc:  # noqa: BLE001
        await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
        return
    await update.effective_message.reply_text(formatting.format_levels(ctx))


async def funding_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    binance = _binance(context)
    try:
        funding = await binance.funding_rate()
        ls = await binance.long_short_ratio()
    except Exception as exc:  # noqa: BLE001
        await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
        return
    await update.effective_message.reply_text(formatting.format_funding(funding, ls))


async def fear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    fng = await get_fear_greed()
    await update.effective_message.reply_text(formatting.format_fear(fng))


async def journal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = await journal.get_stats(config.SYMBOL)
    open_trades = await db.open_trades()
    await update.effective_message.reply_text(
        formatting.format_journal(stats, open_count=len(open_trades))
    )


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from backtest import run_backtest
    await update.effective_message.reply_text("⏳ Запускаю бэктест…")
    binance = _binance(context)
    try:
        report = await run_backtest(binance)
    except Exception as exc:  # noqa: BLE001
        log.exception("backtest failed")
        await update.effective_message.reply_text(f"⚠️ Ошибка бэктеста: {exc}")
        return
    await update.effective_message.reply_text(report)


async def deep_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from pipeline import gather_swing_context
    from signal_engine.quality_score import compute_quality_score
    from ai import swing_analysis

    await update.effective_message.reply_text(
        "🏛 Запускаю глубокий институциональный анализ (1D/12H/4H)… это займёт ~10-20с."
    )
    binance = _binance(context)
    try:
        ctx = await gather_swing_context(binance)
        quality = compute_quality_score(ctx)
        ai_text = await swing_analysis.generate_report(ctx, quality)
    except Exception as exc:  # noqa: BLE001
        log.exception("deep_cmd failed")
        await update.effective_message.reply_text(f"⚠️ Ошибка глубокого анализа: {exc}")
        return
    await update.effective_message.reply_text(formatting.format_deep(quality, ctx, ai_text))


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args) if context.args else ""
    if not question:
        await update.effective_message.reply_text("Использование: /ask <ваш вопрос>")
        return
    ctx = context.application.bot_data.get("last_context")
    if ctx is None:
        try:
            ctx = await _fresh_context(context)
        except Exception as exc:  # noqa: BLE001
            await update.effective_message.reply_text(f"⚠️ Ошибка: {exc}")
            return
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
    "signal": signal_cmd,
    "deep": deep_cmd,
    "levels": levels_cmd,
    "funding": funding_cmd,
    "fear": fear_cmd,
    "journal": journal_cmd,
    "backtest": backtest_cmd,
    "help": help_cmd,
}

# Shown in the Telegram "/" command menu.
BOT_COMMANDS = [
    ("signal", "Текущий сигнал"),
    ("deep", "Глубокий институциональный анализ"),
    ("levels", "Ключевые уровни"),
    ("funding", "Funding rate"),
    ("fear", "Индекс страха/жадности"),
    ("journal", "Статистика журнала"),
    ("backtest", "Бэктест стратегии"),
    ("ask", "Вопрос к ИИ"),
    ("help", "Помощь"),
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
                              MessageHandler, filters)

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("signal", signal_cmd))
    application.add_handler(CommandHandler("deep", deep_cmd))
    application.add_handler(CommandHandler("levels", levels_cmd))
    application.add_handler(CommandHandler("funding", funding_cmd))
    application.add_handler(CommandHandler("fear", fear_cmd))
    application.add_handler(CommandHandler("journal", journal_cmd))
    application.add_handler(CommandHandler("backtest", backtest_cmd))
    application.add_handler(CommandHandler("ask", ask_cmd))
    application.add_handler(CallbackQueryHandler(button_callback, pattern=r"^cmd:"))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_message)
    )
