"""Entry point: wire up the Telegram bot, database, scheduler and run forever.

Designed for Railway `worker: python main.py`. Uses python-telegram-bot's
async Application with long-polling and an in-process APScheduler.
"""
from __future__ import annotations

import asyncio
import logging
import signal as os_signal

import config
from analyzer.exchange import create_market_client
from bot.handlers import _post_init, register_handlers
from database import db
from scheduler import build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("btc_signal_bot")


async def _amain() -> None:
    missing = config.missing_required()
    if missing:
        log.error("Missing required environment variables: %s — cannot start. "
                  "Set them in Railway / your environment and restart.",
                  ", ".join(missing))
        return
    recommended = config.missing_recommended()
    if recommended:
        log.warning("Missing recommended environment variables: %s — the bot "
                    "runs degraded (no AI / no persistence respectively).",
                    ", ".join(recommended))

    from telegram.ext import Application

    await db.connect()

    binance = create_market_client()

    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .build()
    )
    application.bot_data["binance"] = binance
    register_handlers(application)

    scheduler = build_scheduler(application)

    log.info("Starting bot (DRY_RUN=%s, symbol=%s, exchange=%s)…",
             config.DRY_RUN, config.SYMBOL, config.EXCHANGE)

    # Fail-fast AI check: a bad ANTHROPIC_MODEL / key must not degrade
    # silently. The bot still runs without AI (deterministic fallback).
    from ai import claude as ai_claude
    ai_health = await ai_claude.healthcheck()
    application.bot_data["ai_health"] = ai_health
    if ai_health["ok"]:
        log.info("AI healthcheck OK (model=%s)", config.ANTHROPIC_MODEL)
    else:
        log.error(
            "AI DEGRADED (model=%s): %s — сигналы идут с детерминированной "
            "confidence, /ask недоступен. Проверь ANTHROPIC_API_KEY / "
            "ANTHROPIC_MODEL в Railway.",
            config.ANTHROPIC_MODEL, ai_health["error"],
        )

    await application.initialize()
    await application.start()

    # Loud degradation: without Postgres the cooldown / daily-limit / journal
    # state lives in memory and resets on every redeploy (duplicate alerts
    # become possible). The owner must know, not discover it from the logs.
    if db.pool is None:
        from bot import alerts as _alerts
        log.error("PostgreSQL unavailable — running WITHOUT persistence "
                  "(cooldowns and limits reset on every restart)")
        await _alerts.broadcast(
            application.bot,
            "⚠️ БД недоступна: бот работает без персистентности.\n"
            "Cooldown, дневной лимит и журнал сбросятся при рестарте — "
            "возможны повторные алерты. Проверь DATABASE_URL в Railway "
            "(/status покажет, когда БД вернётся).",
        )

    scheduler.start()
    await application.updater.start_polling(drop_pending_updates=True)

    # Run an initial analysis shortly after startup.
    from scheduler import analysis_job
    scheduler.add_job(analysis_job, args=[application], id="initial_analysis")

    # S4A.1: Railway's container filesystem is ephemeral, so a redeploy starts
    # with no snapshot at all and the spot scanner would stay dark until the
    # next two-hourly slot. One controlled refresh at startup closes that gap.
    #
    # Scheduled rather than awaited: the build takes ~33s and startup must not
    # block on Binance's spot API. The job skips immediately when a fresh
    # snapshot survived the restart, and it cannot raise — a spot refresh that
    # fails leaves every other part of the bot running.
    if config.ENABLE_SPOT_SNAPSHOT_REFRESH:
        from scheduler import spot_snapshot_job
        scheduler.add_job(spot_snapshot_job, args=[application],
                          id="initial_spot_snapshot")

    stop_event = asyncio.Event()

    def _stop(*_):
        log.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:  # pragma: no cover (e.g. Windows)
            pass

    try:
        await stop_event.wait()
    finally:
        log.info("Shutting down…")
        scheduler.shutdown(wait=False)
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await binance.close()
        await db.close()


def main() -> None:
    try:
        asyncio.run(_amain())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
