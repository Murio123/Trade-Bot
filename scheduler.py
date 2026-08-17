"""APScheduler wiring: analysis streams, trade resolution, price alerts.

- Analysis jobs are CRON-aligned to the close of each profile's entry candle
  (4H streams fire minutes after a 4H close, intraday after each 15m close),
  so the analysed close price is fresh by construction. Missed slots are NOT
  replayed after a restart (misfire_grace_time) — a stale signal must never
  be published late.
- Every analysis run is recorded in the forecasts ledger (blocked and WAIT
  included); one row per closed entry candle per mode (dedup).
- resolve_trades_job replays open trades on their own timeframe every
  TRADE_CHECK_INTERVAL_MINUTES; outcome_tracking_job measures what price did
  after every recorded forecast.
- Price-alert checks every ALERT_CHECK_INTERVAL_MINUTES.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from bot import alerts, formatting, journal
from database import db
from forecast_lifecycle import enrich_forecast_with_lifecycle
from pipeline import _drop_unclosed, gather_market_context, run_cascade

log = logging.getLogger(__name__)


async def _last_closed_candle_time(binance, timeframe: str):
    """Close time of the last CLOSED entry candle (cheap 2-bar fetch)."""
    df = _drop_unclosed(await binance.klines(timeframe, limit=2))
    if df is None or len(df) == 0:
        return None
    return df["close_time"].iloc[-1].to_pydatetime()


async def analysis_job(application, profile_name: str = "swing") -> None:
    from signal_engine.forecast_record import build_forecast_record
    from signal_engine.profiles import get_profile
    binance = application.bot_data["binance"]
    profile = get_profile(profile_name)
    timeframe = profile["entry"]
    analysis_type = profile.get("analysis_type", "SWING")
    job_fired_at = datetime.now(timezone.utc)
    log.info("Running scheduled analysis [%s, %s]…", profile_name, timeframe)

    # Candle-level dedup BEFORE the heavy work: the same closed entry candle
    # is never analysed twice for a mode (restart, manual kick, overlapping
    # slots). The forecasts UNIQUE constraint is the backstop.
    if config.ENABLE_FORECAST_LEDGER:
        try:
            candle_close = await _last_closed_candle_time(binance, timeframe)
            if candle_close is not None and await db.forecast_exists(
                    config.SYMBOL, analysis_type, candle_close):
                log.info("[%s] candle %s already analysed — skipping",
                         profile_name, candle_close)
                return
        except Exception:  # noqa: BLE001
            log.exception("dedup pre-check failed [%s] — continuing", profile_name)

    try:
        ctx = await gather_market_context(binance, profile_name=profile_name)
        ctx["job_fired_at"] = job_fired_at
        # Cache the swing context for /ask, /levels.
        if profile_name == "swing":
            application.bot_data["last_context"] = ctx
        # Per-mode daily budget and cooldown: SWING and POSITIONAL share the
        # 4H timeframe, so the analysis_type filter is what keeps their
        # cooldowns/limits independent. Cooldown counts from the last
        # DELIVERED alert: journal-only records and repeated "cooldown"
        # evaluations of a persisting setup must not keep re-arming the window.
        delivered_today = await db.signals_today(
            config.SYMBOL, timeframe=timeframe, analysis_type=analysis_type)
        last_signal = await db.last_delivered_signal(
            config.SYMBOL, timeframe=timeframe, analysis_type=analysis_type)
        open_now = await db.open_trades()
        result = await run_cascade(ctx, delivered_today, last_signal,
                                   interpret=True, profile_name=profile_name,
                                   open_trades=open_now)
    except Exception:  # noqa: BLE001
        log.exception("analysis_job [%s] failed", profile_name)
        return

    # Record run telemetry for /status.
    application.bot_data["last_analysis_at"] = datetime.now(timezone.utc)
    application.bot_data["last_analysis_tf"] = f"{profile_name}/{timeframe}"
    application.bot_data["last_analysis_status"] = result.get("status")
    application.bot_data["last_analysis_blocked_at"] = result.get("blocked_at")

    # Forecast ledger: EVERY run is recorded — blocked, WAIT and NO_TRADE
    # included. Failures here must never block alert delivery.
    forecast_id = None
    if config.ENABLE_FORECAST_LEDGER:
        try:
            fc = build_forecast_record(result, ctx, profile)
            if fc:
                # Analytics-only lifecycle metadata, computed AFTER the decision
                # and record, BEFORE persistence. Failure-safe: {} on any error,
                # so the forecast still inserts (columns stay NULL). Never feeds
                # back into the decision, Telegram or risk path.
                fc = {**fc, **await enrich_forecast_with_lifecycle(db, fc)}
                forecast_id = await db.insert_forecast(fc)
        except Exception:  # noqa: BLE001
            log.exception("forecast ledger insert failed [%s]", profile_name)

    # Observation-only profiles (the counter-trend bounce stream) stop here:
    # the run is recorded in the forecast ledger and nothing else happens — no
    # signal row, no mark_delivered, no Telegram, no journal trade. Returning
    # before _maybe_reversal_alert is deliberate: that alert is already
    # broadcast by the trend streams, and an observation stream must not be
    # able to put a message in front of the user.
    if profile.get("observation_only"):
        log.info("[%s] observation-only: forecast #%s recorded, nothing delivered",
                 profile_name, forecast_id)
        return

    # Proactive reversal (bottom/top) alert from the already-gathered context.
    await _maybe_reversal_alert(application, ctx, profile_name, timeframe)

    status = result.get("status")
    if status == "blocked":
        log.info("[%s] Signal blocked at: %s", profile_name, result.get("blocked_at"))
        return
    if status == "cooldown":
        # The same setup was already recorded when it first fired; storing it
        # again every cycle would only bloat the table.
        log.info("[%s] Signal in cooldown (score %s) — not re-recorded",
                 profile_name, result.get("score"))
        return

    # Persist anything at journal threshold or above.
    if result.get("score", 0) >= config.SCORE_JOURNAL_MIN:
        record = {**result, "delivered": False}
        signal_id = await db.insert_signal(record)
        if forecast_id:
            try:
                await db.link_forecast_signal(forecast_id, signal_id)
            except Exception:  # noqa: BLE001
                log.exception("forecast->signal link failed")

        if status == "alert":
            # Delivery guard, not a decision change: while a journal trade for
            # this symbol+profile is still open, the forecast and signal rows
            # above are kept, but no Telegram alert, no mark_delivered and no
            # second journal trade.
            if await journal.has_active_trade(config.SYMBOL, analysis_type,
                                              timeframe):
                log.info("signal alert suppressed: active signal already open "
                         "(signal #%s, %s)", signal_id, profile_name)
                return
            # Aggregate risk cap across profiles: over the cap the alert is
            # still sent (with a warning) but no journal trade is opened.
            can_open = await journal.can_open_new_trade(config.SYMBOL)
            text = formatting.format_signal(result)
            if not can_open:
                text += "\n\n" + formatting.format_risk_cap_note()
            sent = await alerts.send_signal_alert(application.bot, text)
            if sent:
                await db.mark_delivered(signal_id)
                await _send_signal_chart(application, ctx, result)
                if can_open:
                    await journal.record_signal_as_trade(signal_id, result)
                else:
                    log.warning("Risk cap: %s open trades >= MAX_OPEN_TRADES=%s — "
                                "alert #%s delivered without a journal trade",
                                config.SYMBOL, config.MAX_OPEN_TRADES, signal_id)
                log.info("Delivered alert signal #%s (score %s)",
                         signal_id, result["score"])
            else:
                log.info("Alert signal #%s was not delivered; journal trade not opened",
                         signal_id)
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

    The still-forming last candle is deliberately INCLUDED here (unlike the
    analysis path): stops and targets trigger on touch in live trading, so
    the forming bar's high/low so far is the honest signal. Dropping it would
    delay outcome detection by up to one bar of the trade's timeframe.
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
            # D1.2 item D: lifecycle events are subject to the same dry-run
            # gating as the signal alert that opened the trade, so they go
            # through the same path and carry the same banner. Plain
            # broadcast() silently dropped them in DRY_RUN and, when it did
            # send, sent them unlabelled.
            await alerts.send_signal_alert(application.bot, text)


async def outcome_tracking_job(application) -> None:
    """Measure what price did after every recorded forecast.

    Idempotent recomputation from 1H klines: horizon returns fill in as their
    horizons elapse (censored = NULL until then), MFE/MAE and TP/stop touches
    are monotone, unresolved rows are kept — never deleted.
    """
    if not config.ENABLE_FORECAST_LEDGER:
        return
    from analyzer.outcomes import measure_outcome
    from validation.funding import series_from_binance_records
    binance = application.bot_data["binance"]
    try:
        pending = await db.forecasts_pending_outcomes(config.SYMBOL)
    except Exception:  # noqa: BLE001
        log.exception("outcome_tracking_job: pending query failed")
        return
    if not pending:
        return
    try:
        df = await binance.klines("1h", limit=1000)
    except Exception:  # noqa: BLE001
        log.exception("outcome_tracking_job: failed to fetch 1h klines")
        return
    # P1: realized funding over each forecast's measured horizon. One fetch
    # covers every pending row (72h horizons, 1000 settlements back). On
    # failure `funding` stays None and measure_outcome fails closed — it writes
    # NULL for net_after_costs rather than a funding-free number, and the
    # remaining columns (returns, MFE/MAE, TP/stop touches) still update.
    funding = None
    try:
        records = await binance.funding_history(config.SYMBOL, limit=1000)
        funding = series_from_binance_records(records, symbol=config.SYMBOL)
    except Exception:  # noqa: BLE001
        log.warning("outcome_tracking_job: funding history unavailable; "
                    "net_after_costs will be left NULL this cycle",
                    exc_info=True)
    now = datetime.now(timezone.utc)
    updated = 0
    for fc in pending:
        try:
            out = measure_outcome(fc, df, now,
                                  config.TAKER_FEE_PCT, config.SLIPPAGE_PCT,
                                  funding)
            if out:
                await db.upsert_outcome(out)
                updated += 1
        except Exception:  # noqa: BLE001
            log.exception("outcome update failed for forecast #%s", fc.get("id"))
    if updated:
        log.info("outcome_tracking_job: updated %d forecast outcomes", updated)


async def _send_signal_chart(application, ctx: dict, signal: dict) -> None:
    """Attach a chart to a delivered auto-signal; failures never block it."""
    import asyncio
    from bot import charts
    try:
        path = await asyncio.to_thread(charts.render_signal_chart, ctx, signal)
        await alerts.broadcast_photo(application.bot, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("signal chart failed: %s", exc)


REVERSAL_STATE_KEY = "last_reversal_alert"

# The swing/position/intraday jobs share one event loop and compute the same
# 1H/4H/12H/1D reversal verdict. Without a lock, two jobs landing on the same
# minute both pass the cooldown check before either records state -> duplicate
# "bottom/top" alerts. The lock makes check -> record -> broadcast atomic.
_REVERSAL_LOCK = asyncio.Lock()


async def _load_reversal_state(application) -> dict[str, Any] | None:
    """Last reversal alert: bot_data cache first, then the DB (post-restart).

    The DB stores ``time`` as an ISO string; convert it back to an aware
    datetime so the cooldown arithmetic keeps working.
    """
    cached = application.bot_data.get(REVERSAL_STATE_KEY)
    if cached is not None:
        return cached
    try:
        stored = await db.get_state(REVERSAL_STATE_KEY)
    except Exception:  # noqa: BLE001
        log.exception("failed to load reversal alert state")
        return None
    if not stored:
        return None
    ts = stored.get("time")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        stored["time"] = ts
    application.bot_data[REVERSAL_STATE_KEY] = stored
    return stored


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
    from signal_engine.htf_filter import get_htf_bias
    from signal_engine.vetoes import (reversal_alert_allowed,
                                      reversal_alert_min_tfs,
                                      reversal_trend_alignment)

    mtf = ctx.get("reversal_mtf") or {}
    if mtf.get("combined_bullish"):
        direction, tfs, confirmed = "bull", mtf["bull_tfs"], mtf.get("bull_candle_confirm")
    elif mtf.get("combined_bearish"):
        direction, tfs, confirmed = "bear", mtf["bear_tfs"], mtf.get("bear_candle_confirm")
    else:
        return

    # Link to the HTF (1D) trend: with-trend reversals are prime entries,
    # counter-trend ones must clear a higher multi-TF bar.
    htf_bias = get_htf_bias(ctx.get("inds_by_tf", {}).get("1d") or ctx.get("ind_1d", {}))
    alignment = reversal_trend_alignment(direction, htf_bias)
    min_tfs = reversal_alert_min_tfs(config.REVERSAL_ALERT_MIN_TFS, alignment)
    if len(tfs) < min_tfs:
        return
    if not confirmed:
        log.info("Reversal %s on %d TFs but no 1H confirmation candle yet — waiting",
                 direction, len(tfs))
        return

    async with _REVERSAL_LOCK:
        now = datetime.now(timezone.utc)
        last = await _load_reversal_state(application)
        if not reversal_alert_allowed(last, direction, len(tfs), now,
                                      config.REVERSAL_ALERT_COOLDOWN_HOURS):
            return
        # Record the state BEFORE broadcasting: a concurrent stream then sees
        # the fresh cooldown, and a partial send failure means at worst one
        # missed alert — never a duplicate.
        state = {
            "direction": direction, "price": ctx.get("price"),
            "time": now, "tf_count": len(tfs),
        }
        application.bot_data[REVERSAL_STATE_KEY] = state
        # Persist so the cooldown survives restarts/redeploys.
        try:
            await db.set_state(REVERSAL_STATE_KEY, state)
        except Exception:  # noqa: BLE001
            log.exception("failed to persist reversal alert state")

    per_tf = mtf.get("per_tf", {})
    key_f = "factors_bull" if direction == "bull" else "factors_bear"
    factors: list[str] = []
    for tf in tfs:
        for f in per_tf.get(tf, {}).get(key_f, []):
            if f not in factors:
                factors.append(f)

    text = formatting.format_reversal_alert(ctx, direction, factors,
                                            len(tfs) >= 3, tfs,
                                            alignment=alignment)
    # Same DRY-RUN consistency fix as the lifecycle events above.
    await alerts.send_signal_alert(application.bot, text)
    # Prime (with-trend) entries also get the chart with the zones.
    if alignment == "aligned" and ctx.get("df_signal") is not None:
        try:
            import asyncio
            from bot import charts
            path = await asyncio.to_thread(charts.render_levels_chart, ctx)
            await alerts.broadcast_photo(application.bot, path)
        except Exception as exc:  # noqa: BLE001
            log.warning("reversal chart failed: %s", exc)
    log.info("Reversal alert: %s on %d TFs (%s, %s)", direction, len(tfs),
             ",".join(tfs), alignment)


# Cron alignment per entry timeframe: fire ~1 minute after the candle closes
# (the exchange needs a moment to finalise the bar). misfire_grace_time keeps
# a slot missed during downtime from firing late with a stale close.
_ANALYSIS_CRON: dict[str, tuple[dict[str, Any], int]] = {
    "15m": ({"minute": "1,16,31,46"}, 60),
    "1h": ({"minute": "1"}, 120),
    "4h": ({"hour": "0,4,8,12,16,20", "minute": "1"}, 300),
    "1d": ({"hour": "0", "minute": "1"}, 600),
}


def _analysis_trigger(profile: dict[str, Any]) -> tuple[CronTrigger, int]:
    fields, grace = _ANALYSIS_CRON.get(profile["entry"], ({"minute": "1"}, 120))
    return CronTrigger(timezone="UTC", **fields), grace


def build_scheduler(application) -> AsyncIOScheduler:
    from signal_engine.profiles import PROFILES
    scheduler = AsyncIOScheduler(timezone="UTC")
    # Analysis streams, each aligned to its entry-candle close. NOTE: the old
    # interval jobs were added with next_run_time=None, which APScheduler
    # treats as PAUSED — the recurring analyses never actually fired.
    streams = [("swing", True),
               ("position", config.ENABLE_POSITION_ANALYSIS),
               ("intraday", config.ENABLE_FAST_ANALYSIS),
               ("bounce", config.ENABLE_BOUNCE_PROFILE)]
    for name, enabled in streams:
        if not enabled:
            continue
        trigger, grace = _analysis_trigger(PROFILES[name])
        scheduler.add_job(
            analysis_job, trigger, args=[application, name],
            id=f"analysis_{name}", coalesce=True, misfire_grace_time=grace,
        )
    scheduler.add_job(
        price_alert_job, "interval", minutes=config.ALERT_CHECK_INTERVAL_MINUTES,
        args=[application], id="price_alerts",
    )
    scheduler.add_job(
        resolve_trades_job, "interval", minutes=config.TRADE_CHECK_INTERVAL_MINUTES,
        args=[application], id="resolve_trades",
    )
    scheduler.add_job(
        outcome_tracking_job, "interval",
        minutes=config.OUTCOME_TRACK_INTERVAL_MINUTES,
        args=[application], id="outcome_tracking",
    )
    return scheduler
