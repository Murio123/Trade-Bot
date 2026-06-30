"""APScheduler wiring: periodic analysis + price-alert checks.

- Full analysis every ANALYSIS_INTERVAL_HOURS (default 4h) on the 4H/1H frame.
- A daily job re-evaluates the 1D bias.
- Price-alert checks every ALERT_CHECK_INTERVAL_MINUTES.
"""
from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from bot import alerts, formatting, journal
from database import db
from pipeline import gather_market_context, run_cascade

log = logging.getLogger(__name__)


async def analysis_job(application) -> None:
    binance = application.bot_data["binance"]
    log.info("Running scheduled analysis…")
    try:
        ctx = await gather_market_context(binance, signal_timeframe="4h")
        application.bot_data["last_context"] = ctx
        delivered_today = await db.signals_today(config.SYMBOL)
        last_signal = await db.last_signal(config.SYMBOL)
        result = await run_cascade(ctx, delivered_today, last_signal, interpret=True)
    except Exception:  # noqa: BLE001
        log.exception("analysis_job failed")
        return

    status = result.get("status")
    if status == "blocked":
        log.info("Signal blocked at: %s", result.get("blocked_at"))
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


def build_scheduler(application) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        analysis_job, "interval", hours=config.ANALYSIS_INTERVAL_HOURS,
        args=[application], id="analysis", next_run_time=None,
    )
    scheduler.add_job(
        price_alert_job, "interval", minutes=config.ALERT_CHECK_INTERVAL_MINUTES,
        args=[application], id="price_alerts",
    )
    return scheduler
