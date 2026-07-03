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
            ns, _, name = btn.callback_data.partition(":")
            if ns == "menu":  # section navigation, handled by menu_callback
                assert name in k.SUBMENUS, f"inline section {name!r} unknown"
            else:
                assert name in h.COMMAND_DISPATCH, f"inline {name!r} has no handler"


def test_bot_command_menu_matches_registered_commands():
    # every command shown in the "/" menu must be dispatchable or a known
    # argument-taking command handled outside the dispatch map
    outside_dispatch = {"setalert", "delalert", "ask", "alerts"}
    for cmd, _desc in h.BOT_COMMANDS:
        assert cmd in h.COMMAND_DISPATCH or cmd in outside_dispatch, \
            f"/{cmd} in menu but unhandled"


# --- new two-level menu structure -------------------------------------------

MAIN_LABELS = ["📊 Новый анализ", "📌 Последний прогноз", "📈 Рынок",
               "📒 История", "🔔 Алерты", "🟢 Статус"]


def _labels(markup) -> list[str]:
    rows = getattr(markup, "inline_keyboard", None) or markup.keyboard
    return [btn.text for row in rows for btn in row]


def test_main_menu_has_exactly_the_six_target_buttons():
    assert _labels(k.main_reply_keyboard()) == MAIN_LABELS
    assert _labels(k.main_inline_keyboard()) == MAIN_LABELS


def test_main_inline_sections_navigate_and_actions_dispatch():
    for row in k.main_inline_keyboard().inline_keyboard:
        for btn in row:
            ns, _, name = btn.callback_data.partition(":")
            if ns == "menu":
                assert name in k.SUBMENUS, f"{btn.text!r} -> unknown section {name!r}"
            else:
                assert ns == "cmd" and name in h.COMMAND_DISPATCH, \
                    f"{btn.text!r} -> {btn.callback_data!r} has no handler"


def test_every_submenu_button_routes_and_has_back():
    for section in k.SUBMENUS:
        markup = k.submenu_keyboard(section)
        datas = [btn.callback_data for row in markup.inline_keyboard for btn in row]
        assert "menu:main" in datas, f"{section} submenu lacks a Back button"
        for data in datas:
            ns, _, name = data.partition(":")
            if ns == "cmd":
                assert name in h.COMMAND_DISPATCH, f"{data!r} has no handler"
            else:
                assert ns == "menu" and (name == "main" or name in k.SUBMENUS)


def test_legacy_labels_still_route():
    for label in ["📊 Свинг", "⚡ Интрадей", "🌊 Позиция", "🔄 Дно/Пик",
                  "📐 Уровни", "🏛 Глубокий анализ", "💹 Рынок", "📒 Журнал",
                  "🩺 Статус", "📖 Гид", "💸 Funding", "😱 Fear & Greed"]:
        cmd = k.LABEL_TO_COMMAND.get(label)
        assert cmd in h.COMMAND_DISPATCH, f"legacy {label!r} no longer routes"


def test_legacy_callback_data_still_dispatches():
    for cmd in ["signal", "intraday", "position", "reversal", "levels",
                "deep", "market", "journal", "alerts", "status", "guide"]:
        assert cmd in h.COMMAND_DISPATCH, f"old cmd:{cmd} would be dropped"


def _menu_query(data: str):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return update, query


def test_menu_callback_edits_message_to_submenu():
    import asyncio
    update, query = _menu_query("menu:analyze")
    asyncio.run(h.menu_callback(update, MagicMock()))
    query.edit_message_text.assert_called_once()
    _, kwargs = query.edit_message_text.call_args
    assert _labels(kwargs["reply_markup"]) == _labels(k.submenu_keyboard("analyze"))


def test_menu_callback_unknown_section_falls_back_to_main():
    import asyncio
    update, query = _menu_query("menu:no_such_section")
    asyncio.run(h.menu_callback(update, MagicMock()))
    _, kwargs = query.edit_message_text.call_args
    assert _labels(kwargs["reply_markup"]) == MAIN_LABELS


def test_menu_callback_swallows_not_modified_error():
    import asyncio
    update, query = _menu_query("menu:main")
    query.edit_message_text = AsyncMock(side_effect=RuntimeError("Message is not modified"))
    asyncio.run(h.menu_callback(update, MagicMock()))


def _message_update():
    update = MagicMock()
    msg = MagicMock()
    msg.reply_text = AsyncMock()
    update.effective_message = msg
    return update, msg


def test_history_screens_render_on_empty_store():
    import asyncio
    for handler in (h.last_forecast_cmd, h.recent_forecasts_cmd,
                    h.forecast_results_cmd):
        update, msg = _message_update()
        asyncio.run(handler(update, MagicMock()))
        msg.reply_text.assert_called_once()
        text = msg.reply_text.call_args[0][0]
        assert text  # renders something sensible instead of crashing


def test_submenu_handlers_send_submenu_message():
    import asyncio
    update, msg = _message_update()
    asyncio.run(h.menu_analyze_cmd(update, MagicMock()))
    _, kwargs = msg.reply_text.call_args
    assert _labels(kwargs["reply_markup"]) == _labels(k.submenu_keyboard("analyze"))


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


def test_levels_ladder_clusters_and_stars():
    from bot.formatting import format_levels
    ctx = {"timeframe": "1h", "price": 58700, "atr": 300,
           "order_blocks": {"bullish_ob": {"low": 57000, "high": 57500, "tf": "12h"}},
           "fvg": {}, "volume_profile": {"poc": 58900, "val": 57300},
           "liquidity": {}, "htf_levels": {"highs": [58910], "lows": [57000]},
           "inds_by_tf": {}, "equilibrium": {"zone": "discount", "pos": 0.4},
           "liquidation_map": {}, "ind_signal": {}, "zone_tfs": ["12h", "4h"]}
    text = format_levels(ctx)
    assert "Сопротивления" in text and "Поддержки" in text
    # POC (58 900) and the HTF level (58 910) cluster into ONE strong level
    resistances = text.split("Поддержки")[0]
    assert resistances.count("58 9") == 1 and "⭐" in resistances
    # the OB zone absorbed VAL and the 57 000 level -> one support cluster
    assert "57 000–57 500" in text
    # conclusion present
    assert "Вывод" in text
