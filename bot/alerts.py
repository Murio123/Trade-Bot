"""Outbound notifications.

Respects DRY_RUN: in dry-run mode generic messages are logged but not sent.
Signal alerts can opt into SEND_DRY_RUN_ALERTS for manual dry-run delivery.
"""
from __future__ import annotations

import logging
from typing import Iterable

import config

log = logging.getLogger(__name__)


def should_send_alert(allow_dry_run_alerts: bool = False) -> bool:
    return not config.DRY_RUN or allow_dry_run_alerts


def _dry_run_manual_prefix(text: str) -> str:
    if not config.DRY_RUN:
        return text
    return (
        "🧪 DRY-RUN / MANUAL ONLY\n"
        "Сделки не открываются автоматически.\n\n"
        f"{text}"
    )


async def send_message(bot, chat_id: str, text: str,
                       allow_dry_run_alerts: bool = False) -> bool:
    if not should_send_alert(allow_dry_run_alerts):
        log.info("[DRY_RUN] would send to %s:\n%s", chat_id, text)
        return False
    try:
        await bot.send_message(chat_id=chat_id, text=text)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to send message to %s: %s", chat_id, exc)
        return False


async def broadcast(bot, text: str, chat_ids: Iterable[str] | None = None,
                    allow_dry_run_alerts: bool = False) -> bool:
    chat_ids = list(chat_ids) if chat_ids is not None else config.TELEGRAM_ALERT_CHAT_IDS
    if not chat_ids:
        log.warning("No alert chat ids configured; skipping broadcast")
        if not should_send_alert(allow_dry_run_alerts):
            log.info("[DRY_RUN] alert text:\n%s", text)
        return False
    delivered = True
    for cid in chat_ids:
        sent = await send_message(
            bot, cid, text, allow_dry_run_alerts=allow_dry_run_alerts)
        delivered = delivered and sent
    return delivered


async def send_signal_alert(bot, text: str) -> bool:
    allow_dry_run = config.DRY_RUN and config.SEND_DRY_RUN_ALERTS
    return await broadcast(
        bot, _dry_run_manual_prefix(text),
        allow_dry_run_alerts=allow_dry_run)


async def broadcast_photo(bot, photo_path: str, caption: str = "",
                          chat_ids: Iterable[str] | None = None,
                          allow_dry_run_alerts: bool = False) -> bool:
    chat_ids = list(chat_ids) if chat_ids is not None else config.TELEGRAM_ALERT_CHAT_IDS
    if not should_send_alert(allow_dry_run_alerts):
        log.info("[DRY_RUN] would send photo %s to %s", photo_path, chat_ids)
        return False
    if not chat_ids:
        log.warning("No alert chat ids configured; skipping photo broadcast")
        return False
    delivered = True
    for cid in chat_ids:
        try:
            with open(photo_path, "rb") as fh:
                await bot.send_photo(chat_id=cid, photo=fh, caption=caption[:1024])
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to send photo to %s: %s", cid, exc)
            delivered = False
    return delivered
