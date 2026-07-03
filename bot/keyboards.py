"""Telegram keyboards: a persistent reply keyboard + an inline menu.

The reply keyboard sits under the input field and is always available; its
buttons send the label text, which text_message routes to the right handler.
The inline menu (shown on /start) uses callback_data for a tappable grid.

Navigation is two-level: the main menu holds six entries, three of which open
submenus (analyze / market / history). Inline transitions edit the message in
place via callback_data "menu:<section>"; actions keep the "cmd:<name>"
namespace. Legacy labels stay in LABEL_TO_COMMAND so stale keyboards keep
working.
"""
from __future__ import annotations

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup,
                      KeyboardButton, ReplyKeyboardMarkup)

# --- current main-menu labels -----------------------------------------------
BTN_ANALYZE = "📊 Новый анализ"
BTN_LAST_FORECAST = "📌 Последний прогноз"
BTN_MARKET = "📈 Рынок"
BTN_HISTORY = "📒 История"
BTN_ALERTS = "🔔 Алерты"
BTN_STATUS = "🟢 Статус"

BTN_BACK = "⬅️ Назад"

# --- legacy labels (removed from the keyboard, still routed) ---------------
BTN_OLD_SIGNAL = "📊 Свинг"
BTN_OLD_INTRADAY = "⚡ Интрадей"
BTN_OLD_POSITION = "🌊 Позиция"
BTN_OLD_REVERSAL = "🔄 Дно/Пик"
BTN_OLD_LEVELS = "📐 Уровни"
BTN_OLD_DEEP = "🏛 Глубокий анализ"
BTN_OLD_MARKET = "💹 Рынок"
BTN_OLD_JOURNAL = "📒 Журнал"
BTN_OLD_STATUS = "🩺 Статус"
BTN_GUIDE = "📖 Гид"
BTN_FUNDING = "💸 Funding"
BTN_FEAR = "😱 Fear & Greed"
BTN_BACKTEST = "📈 Бэктест"
BTN_ASK = "🧠 Спросить ИИ"
BTN_HELP = "❓ Помощь"

# Main menu rows of (label, command). Sections dispatch to menu_* commands
# which open the corresponding submenu; the rest are direct actions.
_LAYOUT = [
    [(BTN_ANALYZE, "menu_analyze"), (BTN_LAST_FORECAST, "last_forecast")],
    [(BTN_MARKET, "menu_market"), (BTN_HISTORY, "menu_history")],
    [(BTN_ALERTS, "alerts"), (BTN_STATUS, "status")],
]

# section -> (title, rows of (label, callback_data)).
SUBMENUS: dict[str, tuple[str, list[list[tuple[str, str]]]]] = {
    "analyze": ("📊 Новый анализ — выбери режим:", [
        [("⚡ Интрадей", "cmd:intraday"), ("📊 Свинг", "cmd:signal")],
        [("🌊 Позиционный", "cmd:position"), ("🔄 Разворот", "cmd:reversal")],
        [("🏛 Глубокий анализ", "cmd:deep")],
        [(BTN_BACK, "menu:main")],
    ]),
    "market": ("📈 Рынок — что показать?", [
        [("📐 Ключевые уровни", "cmd:levels"), ("📊 Обзор рынка", "cmd:market")],
        [("💸 Funding", "cmd:funding"), ("😱 Fear & Greed", "cmd:fear")],
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
        input_field_placeholder="Кнопка — команда, текст — вопрос к ИИ…",
    )


def main_inline_keyboard() -> InlineKeyboardMarkup:
    def cb(cmd: str) -> str:
        # Section buttons navigate ("menu:analyze"), actions dispatch ("cmd:...").
        if cmd.startswith("menu_"):
            return "menu:" + cmd.removeprefix("menu_")
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
LABEL_TO_COMMAND.update({
    BTN_OLD_SIGNAL: "signal",
    BTN_OLD_INTRADAY: "intraday",
    BTN_OLD_POSITION: "position",
    BTN_OLD_REVERSAL: "reversal",
    BTN_OLD_LEVELS: "levels",
    BTN_OLD_DEEP: "deep",
    BTN_OLD_MARKET: "market",
    BTN_OLD_JOURNAL: "journal",
    BTN_OLD_STATUS: "status",
    BTN_GUIDE: "guide",
    BTN_FUNDING: "funding",
    BTN_FEAR: "fear",
    BTN_BACKTEST: "backtest",
    BTN_HELP: "help",
    BTN_ASK: "ask_prompt",
})
