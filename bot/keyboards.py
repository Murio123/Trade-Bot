"""Telegram keyboards: a persistent reply keyboard + an inline menu.

The reply keyboard sits under the input field and is always available; its
buttons send the label text, which text_message routes to the right handler.
The inline menu (shown on /start) uses callback_data for a tappable grid.
"""
from __future__ import annotations

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup,
                      KeyboardButton, ReplyKeyboardMarkup)

# --- reply keyboard labels (also used for routing button presses) ---------
BTN_SIGNAL = "📊 Сигнал"
BTN_DEEP = "🏛 Глубокий анализ"
BTN_LEVELS = "📐 Уровни"
BTN_FUNDING = "💸 Funding"
BTN_FEAR = "😱 Fear & Greed"
BTN_JOURNAL = "📒 Журнал"
BTN_BACKTEST = "📈 Бэктест"
BTN_ASK = "🧠 Спросить ИИ"
BTN_HELP = "❓ Помощь"


def main_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(BTN_SIGNAL), KeyboardButton(BTN_DEEP)],
            [KeyboardButton(BTN_LEVELS), KeyboardButton(BTN_FUNDING)],
            [KeyboardButton(BTN_FEAR), KeyboardButton(BTN_JOURNAL)],
            [KeyboardButton(BTN_BACKTEST), KeyboardButton(BTN_ASK)],
            [KeyboardButton(BTN_HELP)],
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите команду или задайте вопрос…",
    )


def main_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(BTN_SIGNAL, callback_data="cmd:signal"),
                InlineKeyboardButton(BTN_DEEP, callback_data="cmd:deep"),
            ],
            [
                InlineKeyboardButton(BTN_LEVELS, callback_data="cmd:levels"),
                InlineKeyboardButton(BTN_FUNDING, callback_data="cmd:funding"),
            ],
            [
                InlineKeyboardButton(BTN_FEAR, callback_data="cmd:fear"),
                InlineKeyboardButton(BTN_JOURNAL, callback_data="cmd:journal"),
            ],
            [
                InlineKeyboardButton(BTN_BACKTEST, callback_data="cmd:backtest"),
            ],
        ]
    )


# Map reply-keyboard label -> internal command name.
LABEL_TO_COMMAND = {
    BTN_SIGNAL: "signal",
    BTN_DEEP: "deep",
    BTN_LEVELS: "levels",
    BTN_FUNDING: "funding",
    BTN_FEAR: "fear",
    BTN_JOURNAL: "journal",
    BTN_BACKTEST: "backtest",
    BTN_HELP: "help",
    BTN_ASK: "ask_prompt",
}
