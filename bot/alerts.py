"""Outbound notifications.

Respects DRY_RUN: in dry-run mode messages are logged but not sent unless
SEND_DRY_RUN_ALERTS is explicitly enabled for manual dry-run notifications.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

import config

log = logging.getLogger(__name__)


def should_send_alert() -> bool:
    return not config.DRY_RUN or config.SEND_DRY_RUN_ALERTS


def _dry_run_manual_prefix(text: str) -> str:
    if not config.DRY_RUN:
        return text
    return (
        "🧪 DRY-RUN / MANUAL ONLY\n"
        "Сделки не открываются автоматически.\n\n"
        f"{text}"
    )


async def send_message(bot, chat_id: str, text: str) -> None:
    if not should_send_alert():
        log.info("[DRY_RUN] would send to %s:\n%s", chat_id, text)
        return
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to send message to %s: %s", chat_id, exc)


async def broadcast(bot, text: str, chat_ids: Iterable[str] | None = None) -> None:
    chat_ids = list(chat_ids) if chat_ids is not None else config.TELEGRAM_ALERT_CHAT_IDS
    if not chat_ids:
        log.warning("No alert chat ids configured; skipping broadcast")
        if not should_send_alert():
            log.info("[DRY_RUN] alert text:\n%s", text)
        return
    for cid in chat_ids:
        await send_message(bot, cid, text)


async def send_signal_alert(bot, text: str) -> None:
    await broadcast(bot, _dry_run_manual_prefix(text))


async def broadcast_photo(bot, photo_path: str, caption: str = "",
                          chat_ids: Iterable[str] | None = None) -> None:
    chat_ids = list(chat_ids) if chat_ids is not None else config.TELEGRAM_ALERT_CHAT_IDS
    if not should_send_alert():
        log.info("[DRY_RUN] would send photo %s to %s", photo_path, chat_ids)
        return
    for cid in chat_ids:
        try:
            with open(photo_path, "rb") as fh:
                await bot.send_photo(chat_id=cid, photo=fh, caption=caption[:1024])
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to send photo to %s: %s", cid, exc)
