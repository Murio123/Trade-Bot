"""Telegram messages show times in DISPLAY_TZ (UTC+3); storage stays UTC."""
from __future__ import annotations

from datetime import datetime, timezone

import config
from bot import formatting


def test_display_tz_default_is_utc_plus_3():
    assert config.DISPLAY_TZ.utcoffset(None).total_seconds() == 3 * 3600
    assert config.DISPLAY_TZ_LABEL == "UTC+3"


def test_invalid_display_tz_offset_env_falls_back(monkeypatch):
    # A malformed env value must not crash config import — it falls back to +3.
    import importlib
    monkeypatch.setenv("DISPLAY_TZ_OFFSET_HOURS", "not-a-number")
    try:
        cfg = importlib.reload(config)
        assert cfg.DISPLAY_TZ_OFFSET_HOURS == 3
        assert cfg.DISPLAY_TZ_LABEL == "UTC+3"
    finally:
        monkeypatch.delenv("DISPLAY_TZ_OFFSET_HOURS")
        importlib.reload(config)


def test_out_of_range_display_tz_offset_falls_back(monkeypatch):
    # timezone() rejects offsets beyond ±24h — an out-of-range env value must
    # fall back to +3 instead of crashing config import.
    import importlib
    monkeypatch.setenv("DISPLAY_TZ_OFFSET_HOURS", "1000")
    try:
        cfg = importlib.reload(config)
        assert cfg.DISPLAY_TZ_OFFSET_HOURS == 3
        assert cfg.DISPLAY_TZ.utcoffset(None).total_seconds() == 3 * 3600
    finally:
        monkeypatch.delenv("DISPLAY_TZ_OFFSET_HOURS")
        importlib.reload(config)


def test_fmt_display_time_shifts_aware_utc():
    dt = datetime(2026, 7, 9, 0, 31, tzinfo=timezone.utc)
    assert formatting.fmt_display_time(dt) == "2026-07-09 03:31 UTC+3"


def test_fmt_display_time_treats_naive_as_utc():
    dt = datetime(2026, 7, 9, 23, 30)
    assert formatting.fmt_display_time(dt) == "2026-07-10 02:30 UTC+3"


def test_format_signal_header_uses_display_tz():
    text = formatting.format_signal({
        "timestamp": datetime(2026, 7, 9, 6, 16, tzinfo=timezone.utc),
        "direction": "short",
        "timeframe": "15m",
    })
    header = text.splitlines()[0]
    assert "09:16 UTC+3" in header
    assert "06:16" not in header


def test_format_signal_string_timestamp_passes_through():
    text = formatting.format_signal({
        "timestamp": "2026-07-07 12:00 UTC",
        "direction": "long",
        "timeframe": "4h",
    })
    assert "2026-07-07 12:00 UTC" in text.splitlines()[0]
