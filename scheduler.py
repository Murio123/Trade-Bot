"""APScheduler wiring: periodic analysis + price-alert checks.

- Full analysis every ANALYSIS_INTERVAL_HOURS (default 4h) on the 4H/1H frame.
- A daily job re-evaluates the 1D bias.
- Price-alert checks every ALERT_CHECK_INTERVAL_MINUTES.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from bot import alerts, formatting, journal
from database import db
from pipeline import gather_market_context, run_cascade

log = logging.getLogger(__name__)


async def analysis_job(application, timeframe: str | None = None) -> None:
    binance = application.bot_data["binance"]
    timeframe = timeframe or config.SIGNAL_TIMEFRAME
    log.info("Running scheduled analysis (%s)…", timeframe)
    try:
        ctx = await gather_market_context(binance, signal_timeframe=timeframe)
        # Cache the primary timeframe context for /ask and /levels.
        if timeframe == config.SIGNAL_TIMEFRAME:
            application.bot_data["last_context"] = ctx
        delivered_today = await db.signals_today(config.SYMBOL)
        # Per-timeframe cooldown so 15m and 4h don't suppress each other.
        last_signal = await db.last_signal(config.SYMBOL, timeframe=timeframe)
        result = await run_cascade(ctx, delivered_today, last_signal, interpret=True)
    except Exception:  # noqa: BLE001
        log.exception("analysis_job (%s) failed", timeframe)
        return

    # Record run telemetry for /status.
    application.bot_data["last_analysis_at"] = datetime.now(timezone.utc)
    application.bot_data["last_analysis_tf"] = timeframe
    application.bot_data["last_analysis_status"] = result.get("status")
    application.bot_data["last_analysis_blocked_at"] = result.get("blocked_at")

    status = result.get("status")
    if status == "blocked":
        log.info("[%s] Signal blocked at: %s", timeframe, result.get("blocked_at"))
        return

    # Persist anything at journal threshold or above.
    if result.get("score", 0) >= config.SCORE_JOURNAL_MIN:
        record = {**result, "delivered": status == "alert"}
        signal_id = await db.insert_signal(record)

        if status == "alert":
            text = formatting.format_signal(result)
            await alerts.send_signal_alert(application.bot, text)
            await db.mark_delivered(signal_id)
            await journal.record_signal_as_trade(signal_id, result)
            log.info("Delivered alert signal #%s (score %s)", signal_id, result["score"])
        else:
            log.info("Stored journal signal #%s (score %s)", signal_id, result["score"])


async def price_alert_job(application) -> None:
    binance = application.bot_data["binance"]
    try:
        price = await binance.current_price()
    except Exception:  # noqa: BLE001
        log.exception("price_alert_job: failed to fetch price")
        return
    alerts_list = await db.active_price_alerts(config.SYMBOL)
    for a in alerts_list:
        hit = (a["direction"] == "above" and price >= a["level"]) or \
              (a["direction"] == "below" and price <= a["level"])
        if hit:
            note = f" ({a['note']})" if a.get("note") else ""
            text = (f"🔔 Алерт по уровню {config.SYMBOL_DISPLAY}: цена {price:,.0f} "
                    f"{'≥' if a['direction'] == 'above' else '≤'} {a['level']:,.0f}{note}")
            await alerts.send_message(application.bot, str(a["chat_id"]), text)
            await db.trigger_price_alert(a["id"])


async def resolve_trades_job(application) -> None:
    """Check open journal trades for stop/target hits and record outcomes."""
    binance = application.bot_data["binance"]
    open_trades = await db.open_trades()
    if not open_trades:
        return
    try:
        df = await binance.klines("1h", limit=1000)  # ~41 days of coverage
        price = float(df["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        log.exception("resolve_trades_job: failed to fetch klines")
        return

    for trade in open_trades:
        result = journal.resolve_trade(trade, df, price)
        if not result:
            continue
        await db.close_trade(trade["id"], result["exit_price"],
                             result["outcome"], result["pnl_r"])
        log.info("Trade #%s closed: %s (%.2fR)",
                 trade["id"], result["outcome"], result["pnl_r"])
        await _notify_trade_closed(application, trade, result)


async def _notify_trade_closed(application, trade: dict, result: dict) -> None:
    emoji = {"win": "✅", "loss": "🛑", "breakeven": "➖"}.get(result["outcome"], "ℹ️")
    direction = "ЛОНГ" if trade.get("direction") == "long" else "ШОРТ"
    outcome_ru = {"win": "цель достигнута", "loss": "стоп",
                  "breakeven": "закрыта по времени"}.get(result["outcome"], result["outcome"])
    text = (f"{emoji} Сделка закрыта: {direction} {config.SYMBOL_DISPLAY}\n"
            f"Итог: {outcome_ru} | {result['pnl_r']:+.2f}R\n"
            f"Выход: {result['exit_price']:,.0f}")
    await alerts.broadcast(application.bot, text)


def build_scheduler(application) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="UTC")
    # Primary analysis on the main timeframe (e.g. 4h) every N hours.
    scheduler.add_job(
        analysis_job, "interval", hours=config.ANALYSIS_INTERVAL_HOURS,
        args=[application, config.SIGNAL_TIMEFRAME], id="analysis", next_run_time=None,
    )
    # Fast intraday analysis on a shorter timeframe (e.g. 15m) every M minutes.
    if config.ENABLE_FAST_ANALYSIS and config.FAST_TIMEFRAME != config.SIGNAL_TIMEFRAME:
        scheduler.add_job(
            analysis_job, "interval", minutes=config.FAST_INTERVAL_MINUTES,
            args=[application, config.FAST_TIMEFRAME], id="fast_analysis",
            next_run_time=None,
        )
    scheduler.add_job(
        price_alert_job, "interval", minutes=config.ALERT_CHECK_INTERVAL_MINUTES,
        args=[application], id="price_alerts",
    )
    scheduler.add_job(
        resolve_trades_job, "interval", minutes=config.TRADE_CHECK_INTERVAL_MINUTES,
        args=[application], id="resolve_trades",
    )
    return scheduler
