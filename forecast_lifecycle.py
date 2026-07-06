"""Stage 15B2: failure-safe DB wrapper that enriches a forecast row with
setup-lifecycle analytics right before it is persisted.

Runtime glue only: it reads the previous comparable forecast and delegates to
the pure classifier/field-builder (analyzer.setup_lifecycle). It never mutates
the caller's record and never raises — any failure degrades to an empty dict so
``insert_forecast`` still runs and the lifecycle columns simply stay NULL.

Analytics metadata only: no decision logic, no scoring, no final_gate, no risk,
no Telegram, no AI, no exchange execution / order / API-key code lives here.
"""
from __future__ import annotations

import logging
from typing import Any

from analyzer.setup_lifecycle import build_lifecycle_fields

log = logging.getLogger(__name__)


async def enrich_forecast_with_lifecycle(db, fc: dict[str, Any]) -> dict[str, Any]:
    """Return persist-ready lifecycle fields for ``fc``; ``{}`` on any failure.

    Reads the strictly-previous comparable forecast (same symbol + analysis_type,
    signal_candle_close_time earlier than ``fc``'s) and classifies the transition.
    ``fc`` is treated as read-only. Never propagates lifecycle errors to the
    caller — a broken enrichment must not block the forecast insert.
    """
    try:
        previous = await db.previous_forecast(
            symbol=fc["symbol"],
            analysis_type=fc["analysis_type"],
            before=fc["signal_candle_close_time"],
        )
        return build_lifecycle_fields(previous, fc)
    except Exception:  # noqa: BLE001 — lifecycle is analytics; never block insert
        log.exception("lifecycle enrichment failed [%s/%s] — inserting without it",
                      fc.get("symbol"), fc.get("analysis_type"))
        return {}
