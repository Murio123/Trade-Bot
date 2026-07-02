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

_ROWS = [
    (BTN_SIGNAL, "signal"), (BTN_INTRADAY, "intraday"),
    (BTN_REVERSAL, "reversal"), (BTN_LEVELS, "levels"),
    (BTN_DEEP, "deep"), (BTN_MARKET, "market"),
    (BTN_JOURNAL, "journal"), (BTN_ALERTS, "alerts"),
    (BTN_STATUS, "status"), (BTN_GUIDE, "guide"),
]


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    labels = [label for label, _ in _ROWS]
    rows = [[KeyboardButton(a), KeyboardButton(b)]
            for a, b in zip(labels[::2], labels[1::2])]
    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        input_field_placeholder="Кнопка — команда, текст — вопрос к ИИ…",
    )


def main_inline_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for (l1, c1), (l2, c2) in zip(_ROWS[::2], _ROWS[1::2]):
        rows.append([InlineKeyboardButton(l1, callback_data=f"cmd:{c1}"),
                     InlineKeyboardButton(l2, callback_data=f"cmd:{c2}")])
    return InlineKeyboardMarkup(rows)


# Map reply-keyboard label -> internal command name (incl. legacy labels).
LABEL_TO_COMMAND = {label: cmd for label, cmd in _ROWS}
LABEL_TO_COMMAND.update({
    BTN_FUNDING: "funding",
    BTN_FEAR: "fear",
    BTN_BACKTEST: "backtest",
    BTN_HELP: "help",
    BTN_ASK: "ask_prompt",
})
