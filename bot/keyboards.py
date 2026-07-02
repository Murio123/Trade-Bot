"""Telegram keyboards: a persistent reply keyboard + an inline menu.

The reply keyboard sits under the input field and is always available; its
buttons send the label text, which text_message routes to the right handler.
The inline menu (shown on /start) uses callback_data for a tappable grid.

Layout groups by task (top = most used):
    signals -> entry analysis -> big picture -> tracking/meta.
Legacy labels stay in LABEL_TO_COMMAND so stale keyboards keep working.
"""
from __future__ import annotations

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup,
                      KeyboardButton, ReplyKeyboardMarkup)

# --- current reply keyboard labels -----------------------------------------
BTN_SIGNAL = "📊 Свинг"
BTN_INTRADAY = "⚡ Интрадей"
BTN_POSITION = "🌊 Позиция"
BTN_REVERSAL = "🔄 Дно/Пик"
BTN_LEVELS = "📐 Уровни"
BTN_DEEP = "🏛 Глубокий анализ"
BTN_MARKET = "💹 Рынок"
BTN_JOURNAL = "📒 Журнал"
BTN_ALERTS = "🔔 Алерты"
BTN_STATUS = "🩺 Статус"
BTN_GUIDE = "📖 Гид"

# --- legacy labels (removed from the keyboard, still routed) ---------------
BTN_FUNDING = "💸 Funding"
BTN_FEAR = "😱 Fear & Greed"
BTN_BACKTEST = "📈 Бэктест"
BTN_ASK = "🧠 Спросить ИИ"
BTN_HELP = "❓ Помощь"

# Rows of (label, command); the first row holds the three signal streams.
_LAYOUT = [
    [(BTN_SIGNAL, "signal"), (BTN_INTRADAY, "intraday"), (BTN_POSITION, "position")],
    [(BTN_REVERSAL, "reversal"), (BTN_LEVELS, "levels")],
    [(BTN_DEEP, "deep"), (BTN_MARKET, "market")],
    [(BTN_JOURNAL, "journal"), (BTN_ALERTS, "alerts")],
    [(BTN_STATUS, "status"), (BTN_GUIDE, "guide")],
]


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    rows = [[KeyboardButton(label) for label, _ in row] for row in _LAYOUT]
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        input_field_placeholder="Кнопка — команда, текст — вопрос к ИИ…",
    )


def main_inline_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(label, callback_data=f"cmd:{cmd}")
             for label, cmd in row] for row in _LAYOUT]
    return InlineKeyboardMarkup(rows)


# Map reply-keyboard label -> internal command name (incl. legacy labels).
LABEL_TO_COMMAND = {label: cmd for row in _LAYOUT for label, cmd in row}
LABEL_TO_COMMAND.update({
    BTN_FUNDING: "funding",
    BTN_FEAR: "fear",
    BTN_BACKTEST: "backtest",
    BTN_HELP: "help",
    BTN_ASK: "ask_prompt",
})
