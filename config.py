"""Central configuration. All secrets come from Railway environment variables.

Nothing here is hard-coded except sane defaults that are safe to expose.
"""
from __future__ import annotations

import os


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    if val is not None:
        val = val.strip()
    return val or default


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- Telegram -------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
# Comma-separated list of chat ids that receive automatic alerts.
TELEGRAM_ALERT_CHAT_IDS = [
    c.strip() for c in (_get("TELEGRAM_ALERT_CHAT_IDS", "") or "").split(",") if c.strip()
]

# --- Anthropic ------------------------------------------------------------
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
# Latest capable Sonnet for signal interpretation + chat.
ANTHROPIC_MODEL = _get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- Market data sources --------------------------------------------------
BINANCE_API_KEY = _get("BINANCE_API_KEY")          # read-only, optional for public endpoints
BINANCE_API_SECRET = _get("BINANCE_API_SECRET")
BINANCE_FAPI_BASE = _get("BINANCE_FAPI_BASE", "https://fapi.binance.com")

COINGLASS_API_KEY = _get("COINGLASS_API_KEY")
GLASSNODE_API_KEY = _get("GLASSNODE_API_KEY")
CRYPTOQUANT_API_KEY = _get("CRYPTOQUANT_API_KEY")
NEWSAPI_KEY = _get("NEWSAPI_KEY")
FRED_API_KEY = _get("FRED_API_KEY")

# --- Database -------------------------------------------------------------
# Railway provides DATABASE_URL automatically for the PostgreSQL plugin.
DATABASE_URL = _get("DATABASE_URL")

# --- Trading instrument ---------------------------------------------------
SYMBOL = _get("SYMBOL", "BTCUSDT")
SYMBOL_DISPLAY = _get("SYMBOL_DISPLAY", "BTC")

# --- Signal engine tuning -------------------------------------------------
MIN_DIVERSE_CATEGORIES = _get_int("MIN_DIVERSE_CATEGORIES", 3)
COOLDOWN_HOURS = _get_int("COOLDOWN_HOURS", 4)
MAX_SIGNALS_PER_DAY = _get_int("MAX_SIGNALS_PER_DAY", 3)
ATR_MULTIPLIER = _get_float("ATR_MULTIPLIER", 1.5)
RISK_PERCENT = _get_float("RISK_PERCENT", 1.0)
ACCOUNT_BALANCE = _get_float("ACCOUNT_BALANCE", 10000.0)

# Score thresholds (see ТЗ "Финальная классификация").
SCORE_ALERT_MIN = _get_int("SCORE_ALERT_MIN", 8)     # 8-10 -> push to Telegram
SCORE_JOURNAL_MIN = _get_int("SCORE_JOURNAL_MIN", 5)  # 5-7 -> store, available via /signal

# --- Scheduler ------------------------------------------------------------
ANALYSIS_INTERVAL_HOURS = _get_int("ANALYSIS_INTERVAL_HOURS", 4)
ALERT_CHECK_INTERVAL_MINUTES = _get_int("ALERT_CHECK_INTERVAL_MINUTES", 5)

# --- Operating mode -------------------------------------------------------
# Dry-run: run the whole pipeline but do NOT send Telegram alerts (ТЗ step 10).
DRY_RUN = _get_bool("DRY_RUN", True)

# HTTP behaviour
HTTP_TIMEOUT = _get_float("HTTP_TIMEOUT", 15.0)


def missing_required() -> list[str]:
    """Return the list of required env vars that are not set."""
    required = {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "DATABASE_URL": DATABASE_URL,
    }
    return [k for k, v in required.items() if not v]
