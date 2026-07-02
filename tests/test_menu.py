"""Menu consistency: every button must route somewhere real."""
from unittest.mock import AsyncMock, MagicMock

import bot.handlers as h
import bot.keyboards as k


def test_every_label_routes_to_a_handler():
    for label, cmd in k.LABEL_TO_COMMAND.items():
        assert cmd == "ask_prompt" or cmd in h.COMMAND_DISPATCH, \
            f"{label!r} -> {cmd!r} has no handler"


def test_reply_keyboard_buttons_are_all_mapped():
    rk = k.main_reply_keyboard()
    for row in rk.keyboard:
        for btn in row:
            assert btn.text in k.LABEL_TO_COMMAND, f"unmapped button {btn.text!r}"


def test_inline_menu_callbacks_have_handlers():
    ik = k.main_inline_keyboard()
    for row in ik.inline_keyboard:
        for btn in row:
            cmd = btn.callback_data.split(":", 1)[1]
            assert cmd in h.COMMAND_DISPATCH, f"inline {cmd!r} has no handler"


def test_bot_command_menu_matches_registered_commands():
    # every command shown in the "/" menu must be dispatchable or a known
    # argument-taking command handled outside the dispatch map
    outside_dispatch = {"setalert", "delalert", "ask", "alerts"}
    for cmd, _desc in h.BOT_COMMANDS:
        assert cmd in h.COMMAND_DISPATCH or cmd in outside_dispatch, \
            f"/{cmd} in menu but unhandled"


def test_market_format_renders():
    from bot.formatting import format_market
    text = format_market(
        58000.0,
        {"current": 0.0008, "zscore": 2.4, "anomalous": True},
        {"ratio": 2.3}, 91234.0,
        {"value": 28, "classification": "Fear"})
    assert "Цена: 58 000" in text
    assert "перегрев лонгов" in text
    assert "лонги переполнены" in text
    assert "Fear & Greed: 28/100" in text
    # degrades with everything missing
    assert format_market(None, {}, {}, None, {}).startswith("💹 Рынок")
