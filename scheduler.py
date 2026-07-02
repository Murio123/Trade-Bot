"""APScheduler wiring: analysis streams, trade resolution, price alerts.

- Swing analysis hourly (1H entry) and intraday every 15m (15m entry) — the
  cadence comes from each profile's interval_minutes.
- resolve_trades_job replays open trades on their own timeframe every
  TRADE_CHECK_INTERVAL_MINUTES.
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
            await _send_signal_chart(application, ctx, result)
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
    """Check open journal trades for stop/target hits and record outcomes.

    Each trade is replayed on its OWN timeframe (a 15m intraday trade judged
    on 1H candles would often see both stop and target inside one bar and be
    scored as a loss by the conservative rule). Falls back to 1H when the
    trade's timeframe history no longer reaches back to its open time.
    """
    binance = application.bot_data["binance"]
    open_trades = await db.open_trades()
    if not open_trades:
        return

    # Fetch each needed timeframe once.
    needed = {trade.get("timeframe") or "1h" for trade in open_trades} | {"1h"}
    frames: dict[str, object] = {}
    for tf in needed:
        try:
            frames[tf] = await binance.klines(tf, limit=1000)
        except Exception:  # noqa: BLE001
            log.exception("resolve_trades_job: failed to fetch %s klines", tf)
    if "1h" not in frames:
        return
    price = float(frames["1h"]["close"].iloc[-1])

    for trade in open_trades:
        df = journal.pick_frame(trade, frames)
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


async def _send_signal_chart(application, ctx: dict, signal: dict) -> None:
    """Attach a chart to a delivered auto-signal; failures never block it."""
    import asyncio
    from bot import charts
    try:
        path = await asyncio.to_thread(charts.render_signal_chart, ctx, signal)
        await alerts.broadcast_photo(application.bot, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("signal chart failed: %s", exc)


async def _maybe_reversal_alert(application, ctx: dict, profile_name: str,
                                timeframe: str) -> None:
    """Proactive bottom/top alert.

    Fires only when ALL hold:
      - reversal confirmed on >= REVERSAL_ALERT_MIN_TFS timeframes,
      - the last closed 1H candle already turned in the reversal direction
        (no knife-catching alerts while price is still falling),
      - the throttle allows it (cooldown; re-fire inside the cooldown only
        when MORE timeframes confirm than at the previous alert).
    """
    if not config.ENABLE_REVERSAL_ALERTS:
        return
    from signal_engine.vetoes import reversal_alert_allowed

    mtf = ctx.get("reversal_mtf") or {}
    min_tfs = config.REVERSAL_ALERT_MIN_TFS
    if mtf.get("combined_bullish") and mtf.get("bull_tf_count", 0) >= min_tfs:
        direction, tfs, confirmed = "bull", mtf["bull_tfs"], mtf.get("bull_candle_confirm")
    elif mtf.get("combined_bearish") and mtf.get("bear_tf_count", 0) >= min_tfs:
        direction, tfs, confirmed = "bear", mtf["bear_tfs"], mtf.get("bear_candle_confirm")
    else:
        return
    if not confirmed:
        log.info("Reversal %s on %d TFs but no 1H confirmation candle yet — waiting",
                 direction, len(tfs))
        return

    now = datetime.now(timezone.utc)
    last = application.bot_data.get("last_reversal_alert")
    if not reversal_alert_allowed(last, direction, len(tfs), now,
                                  config.REVERSAL_ALERT_COOLDOWN_HOURS):
        return

    per_tf = mtf.get("per_tf", {})
    key_f = "factors_bull" if direction == "bull" else "factors_bear"
    factors: list[str] = []
    for tf in tfs:
        for f in per_tf.get(tf, {}).get(key_f, []):
            if f not in factors:
                factors.append(f)

    text = formatting.format_reversal_alert(ctx, direction, factors,
                                            len(tfs) >= 3, tfs)
    await alerts.broadcast(application.bot, text)
    application.bot_data["last_reversal_alert"] = {
        "direction": direction, "price": ctx.get("price"),
        "time": now, "tf_count": len(tfs),
    }
    log.info("Reversal alert: %s on %d TFs (%s)", direction, len(tfs), ",".join(tfs))


def build_scheduler(application) -> AsyncIOScheduler:
    from signal_engine.profiles import PROFILES
    scheduler = AsyncIOScheduler(timezone="UTC")
    # Swing analysis (1H entry, 1D/12H/4H context) on the profile's cadence.
    scheduler.add_job(
        analysis_job, "interval", minutes=PROFILES["swing"]["interval_minutes"],
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
