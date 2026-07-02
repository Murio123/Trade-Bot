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
# Who may talk to the bot at all. Every message costs money (/ask hits the
# Anthropic API), so unknown users are rejected. Falls back to the alert
# recipients. When BOTH are empty the bot is closed by default: strangers get
# their chat_id so the owner can whitelist themselves during first setup.
# Set ALLOW_PUBLIC_ACCESS=true to deliberately run an open bot.
TELEGRAM_ALLOWED_CHAT_IDS = [
    c.strip() for c in (_get("TELEGRAM_ALLOWED_CHAT_IDS", "") or "").split(",") if c.strip()
] or list(TELEGRAM_ALERT_CHAT_IDS)
ALLOW_PUBLIC_ACCESS = _get_bool("ALLOW_PUBLIC_ACCESS", False)

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

# --- Exchange selection ---------------------------------------------------
# Which market-data backend to prefer: "binance", "bybit", or "auto".
# "auto" tries Bybit first (broadly reachable from cloud regions) and falls
# back to Binance — and vice-versa — automatically on geo-block (HTTP 451).
EXCHANGE = (_get("EXCHANGE", "auto") or "auto").lower()

# --- Signal engine tuning -------------------------------------------------
MIN_DIVERSE_CATEGORIES = _get_int("MIN_DIVERSE_CATEGORIES", 3)
# Aggregate risk cap: max simultaneously open (virtual) positions per symbol
# across ALL profiles. Beyond it alerts are still sent (with a warning) but
# no new journal trade is opened — the extra entry is over risk budget.
MAX_OPEN_TRADES = _get_int("MAX_OPEN_TRADES", 3)
COOLDOWN_HOURS = _get_int("COOLDOWN_HOURS", 4)
MAX_SIGNALS_PER_DAY = _get_int("MAX_SIGNALS_PER_DAY", 3)
ATR_MULTIPLIER = _get_float("ATR_MULTIPLIER", 1.5)
RISK_PERCENT = _get_float("RISK_PERCENT", 1.0)
ACCOUNT_BALANCE = _get_float("ACCOUNT_BALANCE", 10000.0)

# --- Data-quality gates (Part A quality pass) -------------------------------
# Last closed candle older than this many bars of the entry TF -> NO_TRADE.
MAX_DATA_AGE_BARS = _get_float("MAX_DATA_AGE_BARS", 3.0)
# ATR percentile at/above which volatility is abnormal -> NO_TRADE.
ABNORMAL_VOL_PERCENTILE = _get_float("ABNORMAL_VOL_PERCENTILE", 95.0)
# Funding |z-score| that counts as "stretched" (contrarian macro points).
# Must stay BELOW the crowded-funding veto extreme (FUNDING_Z_EXTREME=2.0 in
# signal_engine/vetoes.py): score at the signal level, veto at the extreme.
FUNDING_Z_SIGNAL = _get_float("FUNDING_Z_SIGNAL", 1.5)
# Mandatory NO_TRADE thresholds (Part B).
MIN_RISK_REWARD = _get_float("MIN_RISK_REWARD", 1.5)   # to TP2
MIN_CONFIDENCE = _get_float("MIN_CONFIDENCE", 0.3)

# Score thresholds (see ТЗ "Финальная классификация").
SCORE_ALERT_MIN = _get_int("SCORE_ALERT_MIN", 8)     # 8-10 -> push to Telegram
SCORE_JOURNAL_MIN = _get_int("SCORE_JOURNAL_MIN", 5)  # 5-7 -> store, available via /signal

# --- Trading costs (used by the backtest to report NET results) ------------
TAKER_FEE_PCT = _get_float("TAKER_FEE_PCT", 0.05)   # % per side (Binance futures taker)
SLIPPAGE_PCT = _get_float("SLIPPAGE_PCT", 0.03)     # % per round trip, conservative

# --- Scheduler ------------------------------------------------------------
# Analysis cadence lives in signal_engine/profiles.py (per-profile
# interval_minutes: swing hourly, intraday every 15m).
ALERT_CHECK_INTERVAL_MINUTES = _get_int("ALERT_CHECK_INTERVAL_MINUTES", 5)
# How often open trades are checked for stop/target hits.
TRADE_CHECK_INTERVAL_MINUTES = _get_int("TRADE_CHECK_INTERVAL_MINUTES", 15)

# Proactive reversal (bottom/top) alerts.
ENABLE_REVERSAL_ALERTS = _get_bool("ENABLE_REVERSAL_ALERTS", True)
# How many timeframes (of 1H/4H/12H/1D) must confirm before alerting.
REVERSAL_ALERT_MIN_TFS = _get_int("REVERSAL_ALERT_MIN_TFS", 2)
REVERSAL_ALERT_COOLDOWN_HOURS = _get_int("REVERSAL_ALERT_COOLDOWN_HOURS", 4)
# Toggle for the intraday (15m) analysis stream; its cadence/timeframes are
# defined by the "intraday" profile in signal_engine/profiles.py.
ENABLE_FAST_ANALYSIS = _get_bool("ENABLE_FAST_ANALYSIS", True)
# Toggle for the position (4H entry, 1D structure) stream targeting
# multi-day 2000-5000pt moves.
ENABLE_POSITION_ANALYSIS = _get_bool("ENABLE_POSITION_ANALYSIS", True)

# --- Operating mode -------------------------------------------------------
# Dry-run: run the whole pipeline but do NOT send Telegram alerts (ТЗ step 10).
DRY_RUN = _get_bool("DRY_RUN", True)

# HTTP behaviour
HTTP_TIMEOUT = _get_float("HTTP_TIMEOUT", 15.0)


def missing_required() -> list[str]:
    """Env vars without which the bot cannot start at all."""
    required = {"TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN}
    return [k for k, v in required.items() if not v]


def missing_recommended() -> list[str]:
    """Env vars the bot degrades gracefully without (but shouldn't in prod):
    no AI interpretation / no persistence respectively."""
    recommended = {
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "DATABASE_URL": DATABASE_URL,
    }
    return [k for k, v in recommended.items() if not v]
