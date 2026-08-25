"""Telegram keyboards: a persistent reply keyboard + an inline menu.

The reply keyboard sits under the input field and is always available; its
buttons send the label text, which text_message routes to the right handler.
The inline menu (shown on /start) uses callback_data for a tappable grid.

Navigation is two-level. D1.3 reduces the main menu to the three sections the
product actually has — market analysis, forecast, journal — instead of the
previous six mixed entries. Inline transitions edit the message in place via
callback_data "menu:<section>"; actions keep the "cmd:<name>" namespace.

Superseded labels and sections stay in LABEL_TO_COMMAND / SUBMENUS: a reply
keyboard already sitting in a user's chat keeps sending the OLD label text,
and an inline menu in message history keeps its OLD callback_data, so dropping
either mapping would strand every client that has not pressed /start again.
"""
from __future__ import annotations

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup,
                      KeyboardButton, ReplyKeyboardMarkup)

# --- current main-menu labels (D1.3) ----------------------------------------
BTN_MARKET_ANALYSIS = "📊 Анализ рынка"
BTN_FORECAST = "🔮 Прогноз"
BTN_JOURNAL = "📓 Торговый журнал"
# S4A: the spot facts screener. A section, not an action — it opens its own
# inline tree under the "spot:" callback namespace.
BTN_SPOT_COINS = "🪙 Спот / Монеты"

BTN_BACK = "⬅️ Назад"

# --- legacy labels (removed from the keyboard, still routed) ---------------
BTN_ANALYZE = "📊 Новый анализ"
BTN_LAST_FORECAST = "📌 Последний прогноз"
BTN_MARKET = "📈 Рынок"
BTN_HISTORY = "📒 История"
BTN_ALERTS = "🔔 Алерты"
BTN_STATUS = "🟢 Статус"
BTN_OLD_SIGNAL = "📊 Свинг"
BTN_OLD_INTRADAY = "⚡ Интрадей"
BTN_OLD_POSITION = "🌊 Позиция"
BTN_OLD_REVERSAL = "🔄 Дно/Пик"
BTN_OLD_LEVELS = "📐 Уровни"
BTN_OLD_MARKET = "💹 Рынок"
BTN_OLD_JOURNAL = "📒 Журнал"
BTN_OLD_STATUS = "🩺 Статус"
BTN_FUNDING = "💸 Funding"
BTN_BACKTEST = "📈 Бэктест"
BTN_ASK = "🧠 Спросить ИИ"
BTN_HELP = "❓ Помощь"

# Main menu rows of (label, command). Sections dispatch to menu_* commands
# which open the corresponding submenu; the rest are direct actions.
_LAYOUT = [
    [(BTN_MARKET_ANALYSIS, "menu_market")],
    [(BTN_FORECAST, "menu_forecast")],
    [(BTN_JOURNAL, "menu_journal")],
    [(BTN_SPOT_COINS, "menu_spot")],
]

# Forecast submenu labels — the one place the spot/futures split is named.
BTN_SPOT = "🟢 Спот"
BTN_FUT_SWING = "📈 Фьючерсы — свинг"
BTN_FUT_INTRADAY = "⚡ Фьючерсы — интрадей"

# section -> (title, rows of (label, callback_data)).
SUBMENUS: dict[str, tuple[str, list[list[tuple[str, str]]]]] = {
    "forecast": ("🔮 Прогноз — выбери рынок:", [
        [(BTN_SPOT, "cmd:position")],
        [(BTN_FUT_SWING, "cmd:signal")],
        [(BTN_FUT_INTRADAY, "cmd:intraday")],
        [(BTN_BACK, "menu:main")],
    ]),
    "journal": ("📓 Торговый журнал:", [
        [("📌 Открытые прогнозы", "cmd:recent_forecasts")],
        [("📜 История", "cmd:forecast_results")],
        [("📊 Статистика", "cmd:journal")],
        [(BTN_BACK, "menu:main")],
    ]),
    # Reached from the "📊 Анализ рынка" main-menu button AND from the old
    # "menu:market" callback_data still sitting in message history — one
    # section serves both, so nothing loses functionality either way.
    "market": ("📊 Анализ рынка — что показать?", [
        [("📊 Общий анализ", "cmd:market")],
        [("📐 Уровни", "cmd:levels")],
        [("💰 Funding", "cmd:funding")],
        [("🔄 Reversal", "cmd:reversal")],
        [(BTN_BACK, "menu:main")],
    ]),
    # --- superseded sections, still reachable from old inline messages ------
    "analyze": ("📊 Новый анализ — выбери режим:", [
        [("⚡ Интрадей", "cmd:intraday"), ("📊 Свинг", "cmd:signal")],
        [("🌊 Позиционный", "cmd:position"), ("🔄 Разворот", "cmd:reversal")],
        [(BTN_BACK, "menu:main")],
    ]),
    "history": ("📒 История — что показать?", [
        [("📌 Последние прогнозы", "cmd:recent_forecasts")],
        [("✅ Результаты прогнозов", "cmd:forecast_results")],
        [("📒 Журнал сделок", "cmd:journal")],
        [(BTN_BACK, "menu:main")],
    ]),
}

MAIN_MENU_TITLE = "Быстрое меню:"


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    rows = [[KeyboardButton(label) for label, _ in row] for row in _LAYOUT]
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        input_field_placeholder="Кнопка — раздел, текст — вопрос к Claude…",
    )


def main_inline_keyboard() -> InlineKeyboardMarkup:
    def cb(cmd: str) -> str:
        # Section buttons navigate ("menu:analyze"), actions dispatch ("cmd:...").
        #
        # Only sections that ARE static submenus navigate that way. "menu_spot"
        # is a section too, but its screen is built from live snapshot state —
        # a title and a fixed button grid cannot express "32 монеты, данные от
        # 24.08" — so it dispatches to a handler like an action does.
        section = cmd.removeprefix("menu_")
        if cmd.startswith("menu_") and section in SUBMENUS:
            return "menu:" + section
        return f"cmd:{cmd}"

    rows = [[InlineKeyboardButton(label, callback_data=cb(cmd))
             for label, cmd in row] for row in _LAYOUT]
    return InlineKeyboardMarkup(rows)


def submenu_keyboard(section: str) -> InlineKeyboardMarkup | None:
    item = SUBMENUS.get(section)
    if item is None:
        return None
    _, rows = item
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=data)
          for label, data in row] for row in rows])


def submenu_title(section: str) -> str:
    item = SUBMENUS.get(section)
    return item[0] if item else MAIN_MENU_TITLE


# Map reply-keyboard label -> internal command name (incl. legacy labels).
LABEL_TO_COMMAND = {label: cmd for row in _LAYOUT for label, cmd in row}
# Forecast/journal submenu labels are inline-only, but a user can also type
# them verbatim, so route them too.
LABEL_TO_COMMAND.update({
    BTN_SPOT: "position",
    BTN_SPOT_COINS: "menu_spot",
    BTN_FUT_SWING: "signal",
    BTN_FUT_INTRADAY: "intraday",
})
LABEL_TO_COMMAND.update({
    BTN_ANALYZE: "menu_analyze",
    BTN_LAST_FORECAST: "last_forecast",
    BTN_MARKET: "menu_market",
    BTN_HISTORY: "menu_history",
    BTN_ALERTS: "alerts",
    BTN_STATUS: "status",
    BTN_OLD_SIGNAL: "signal",
    BTN_OLD_INTRADAY: "intraday",
    BTN_OLD_POSITION: "position",
    BTN_OLD_REVERSAL: "reversal",
    BTN_OLD_LEVELS: "levels",
    BTN_OLD_MARKET: "market",
    BTN_OLD_JOURNAL: "journal",
    BTN_OLD_STATUS: "status",
    BTN_FUNDING: "funding",
    BTN_BACKTEST: "backtest",
    BTN_HELP: "help",
    BTN_ASK: "ask_prompt",
})
