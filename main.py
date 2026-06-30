"""Entry point: wire up the Telegram bot, database, scheduler and run forever.

Designed for Railway `worker: python main.py`. Uses python-telegram-bot's
async Application with long-polling and an in-process APScheduler.
"""
from __future__ import annotations

import asyncio
import logging
import signal as os_signal

import config
from analyzer.binance import BinanceClient
from bot.handlers import register_handlers
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
        log.error("Missing required environment variables: %s", ", ".join(missing))
        log.error("Set them in Railway / your environment and restart.")
        # Telegram token is mandatory to even start polling.
        if "TELEGRAM_BOT_TOKEN" in missing:
            return

    from telegram.ext import Application

    await db.connect()

    binance = BinanceClient()

    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )
    application.bot_data["binance"] = binance
    register_handlers(application)

    scheduler = build_scheduler(application)

    log.info("Starting bot (DRY_RUN=%s, symbol=%s)…", config.DRY_RUN, config.SYMBOL)

    await application.initialize()
    await application.start()
    scheduler.start()
    await application.updater.start_polling(drop_pending_updates=True)

    # Run an initial analysis shortly after startup.
    from scheduler import analysis_job
    scheduler.add_job(analysis_job, args=[application], id="initial_analysis")

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
