"""Persistence of the reversal-alert cooldown state across restarts."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from database import Database
from scheduler import REVERSAL_STATE_KEY, _load_reversal_state


class _FakeApp:
    def __init__(self):
        self.bot_data: dict = {}


def test_reversal_state_survives_restart(monkeypatch):
    """State written before a 'restart' must be readable after it, with the
    ISO time string converted back to an aware datetime."""
    db = Database(dsn=None)  # memory store mirrors the Postgres JSON round-trip
    monkeypatch.setattr("scheduler.db", db)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    asyncio.run(db.set_state(REVERSAL_STATE_KEY, {
        "direction": "bull", "price": 60000.0, "time": now, "tf_count": 2,
    }))

    fresh_app = _FakeApp()  # bot_data is empty, as after a redeploy
    state = asyncio.run(_load_reversal_state(fresh_app))

    assert state is not None
    assert state["direction"] == "bull"
    assert state["tf_count"] == 2
    assert isinstance(state["time"], datetime)
    assert state["time"].tzinfo is not None
    assert abs(state["time"] - now) < timedelta(seconds=1)
    # Cached for subsequent calls.
    assert fresh_app.bot_data[REVERSAL_STATE_KEY] is state


def test_reversal_state_empty_on_first_run(monkeypatch):
    db = Database(dsn=None)
    monkeypatch.setattr("scheduler.db", db)
    assert asyncio.run(_load_reversal_state(_FakeApp())) is None


def test_concurrent_streams_send_one_reversal_alert(monkeypatch):
    """H2 regression: swing + intraday jobs firing on the same minute must
    produce exactly ONE reversal alert, not two."""
    import config
    import scheduler as sched

    db = Database(dsn=None)
    monkeypatch.setattr("scheduler.db", db)
    monkeypatch.setattr(config, "ENABLE_REVERSAL_ALERTS", True)
    monkeypatch.setattr(config, "REVERSAL_ALERT_MIN_TFS", 2)

    sent = []

    async def fake_broadcast(bot, text):
        await asyncio.sleep(0.01)  # widen the race window
        sent.append(text)

    # Reversal alerts go through send_signal_alert since the D1.2 item D
    # dry-run consistency fix; the one-alert-under-race invariant is
    # unchanged.
    monkeypatch.setattr("scheduler.alerts.send_signal_alert", fake_broadcast)
    monkeypatch.setattr("scheduler.formatting.format_reversal_alert",
                        lambda *a, **k: "reversal!")

    ctx = {
        "price": 60000.0,
        "df_signal": None,
        "inds_by_tf": {},
        "ind_1d": {},
        "reversal_mtf": {
            "combined_bullish": True,
            "bull_tfs": ["1h", "4h"],
            "bull_candle_confirm": True,
            "per_tf": {},
        },
    }
    app = _FakeApp()
    app.bot = None

    async def race():
        await asyncio.gather(
            sched._maybe_reversal_alert(app, ctx, "swing", "1h"),
            sched._maybe_reversal_alert(app, ctx, "intraday", "15m"),
        )

    asyncio.run(race())
    assert len(sent) == 1
    # State recorded both in the cache and durably.
    assert app.bot_data[sched.REVERSAL_STATE_KEY]["direction"] == "bull"
    assert asyncio.run(db.get_state(sched.REVERSAL_STATE_KEY)) is not None


def test_reversal_state_cooldown_blocks_after_restart(monkeypatch):
    """The whole point: a redeploy inside the cooldown must not re-alert."""
    from signal_engine.vetoes import reversal_alert_allowed

    db = Database(dsn=None)
    monkeypatch.setattr("scheduler.db", db)

    now = datetime.now(timezone.utc)
    asyncio.run(db.set_state(REVERSAL_STATE_KEY, {
        "direction": "bull", "price": 60000.0,
        "time": now - timedelta(hours=1), "tf_count": 2,
    }))
    state = asyncio.run(_load_reversal_state(_FakeApp()))

    # Same direction, same tf count, inside the 4h cooldown -> blocked.
    assert not reversal_alert_allowed(state, "bull", 2, now, cooldown_hours=4)
    # Escalation to more timeframes is still allowed.
    assert reversal_alert_allowed(state, "bull", 3, now, cooldown_hours=4)
