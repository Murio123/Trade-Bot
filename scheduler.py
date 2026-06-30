"""APScheduler wiring: periodic analysis + price-alert checks.

- Full analysis every ANALYSIS_INTERVAL_HOURS (default 4h) on the 4H/1H frame.
- A daily job re-evaluates the 1D bias.
- Price-alert checks every ALERT_CHECK_INTERVAL_MINUTES.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from bot import alerts, formatting, journal
from database import db
from pipeline import gather_market_context, run_cascade

log = logging.getLogger(__name__)


async def analysis_job(application, profile_name: str = "swing") -> None:
    from signal_engine.profiles import get_profile
    binance = application.bot_data["binance"]
    profile = get_profile(profile_name)
    timeframe = profile["entry"]
    log.info("Running scheduled analysis [%s, %s]…", profile_name, timeframe)
    try:
        ctx = await gather_market_context(binance, profile_name=profile_name)
        # Cache the swing context for /ask, /levels.
        if profile_name == "swing":
            application.bot_data["last_context"] = ctx
        # Per-style daily budget and cooldown (keyed by the entry timeframe).
        delivered_today = await db.signals_today(config.SYMBOL, timeframe=timeframe)
        last_signal = await db.last_signal(config.SYMBOL, timeframe=timeframe)
        result = await run_cascade(ctx, delivered_today, last_signal,
                                   interpret=True, profile_name=profile_name)
    except Exception:  # noqa: BLE001
        log.exception("analysis_job [%s] failed", profile_name)
        return

    # Record run telemetry for /status.
    application.bot_data["last_analysis_at"] = datetime.now(timezone.utc)
    application.bot_data["last_analysis_tf"] = f"{profile_name}/{timeframe}"
    application.bot_data["last_analysis_status"] = result.get("status")
    application.bot_data["last_analysis_blocked_at"] = result.get("blocked_at")

    # Proactive reversal (bottom/top) alert from the already-gathered context.
    await _maybe_reversal_alert(application, ctx, profile_name, timeframe)

    status = result.get("status")
    if status == "blocked":
        log.info("[%s] Signal blocked at: %s", profile_name, result.get("blocked_at"))
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
        evaluation = journal.evaluate_trade(trade, df, price)
        if not evaluation:
            continue
        # Advance stored stage / stop, or close the trade.
        if evaluation["closed"]:
            c = evaluation["closed"]
            await db.close_trade(trade["id"], c["exit_price"], c["outcome"], c["pnl_r"])
            log.info("Trade #%s closed: %s (%.2fR)", trade["id"], c["outcome"], c["pnl_r"])
        else:
            await db.advance_trade_stage(
                trade["id"], evaluation["new_stage"], evaluation["current_stop"])
            log.info("Trade #%s -> stage %s", trade["id"], evaluation["new_stage"])

        for event in evaluation["events"]:
            text = formatting.format_trade_event(trade, event)
            await alerts.broadcast(application.bot, text)


async def _maybe_reversal_alert(application, ctx: dict, profile_name: str,
                                timeframe: str) -> None:
    """Emit a proactive bottom/top alert when enough exhaustion factors align."""
    if not config.ENABLE_REVERSAL_ALERTS:
        return
    rev = ctx.get("reversal") or {}
    min_factors = config.REVERSAL_ALERT_MIN_FACTORS
    if rev.get("bull_score", 0) >= min_factors and rev.get("bull_score", 0) >= rev.get("bear_score", 0):
        direction, factors, strong = "bull", rev.get("factors_bull", []), rev.get("bull_strong")
    elif rev.get("bear_score", 0) >= min_factors:
        direction, factors, strong = "bear", rev.get("factors_bear", []), rev.get("bear_strong")
    else:
        return

    # De-duplicate: skip if the same direction fired recently near this price.
    price = ctx.get("price")
    atr = ctx.get("atr") or (price * 0.01 if price else 0)
    key = f"last_reversal_alert_{timeframe}"
    last = application.bot_data.get(key)
    now = datetime.now(timezone.utc)
    if last and last["direction"] == direction:
        within_cooldown = (now - last["time"]) < timedelta(hours=config.REVERSAL_ALERT_COOLDOWN_HOURS)
        near_price = price is not None and abs(price - last["price"]) < atr * 0.5
        if within_cooldown and near_price:
            return

    text = formatting.format_reversal_alert(ctx, direction, factors, bool(strong))
    await alerts.broadcast(application.bot, text)
    application.bot_data[key] = {"direction": direction, "price": price, "time": now}
    log.info("[%s] Reversal alert: %s (%d factors)", profile_name, direction, len(factors))


def build_scheduler(application) -> AsyncIOScheduler:
    from signal_engine.profiles import PROFILES
    scheduler = AsyncIOScheduler(timezone="UTC")
    # Swing analysis (4h entry, 1D/12H/4H) every N hours.
    scheduler.add_job(
        analysis_job, "interval", hours=config.ANALYSIS_INTERVAL_HOURS,
        args=[application, "swing"], id="analysis_swing", next_run_time=None,
    )
    # Intraday analysis (15m entry, 4H/1H/15m) every M minutes.
    if config.ENABLE_FAST_ANALYSIS:
        scheduler.add_job(
            analysis_job, "interval",
            minutes=PROFILES["intraday"]["interval_minutes"],
            args=[application, "intraday"], id="analysis_intraday", next_run_time=None,
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
