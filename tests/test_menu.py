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
                  "📐 Уровни", "💹 Рынок", "📒 Журнал",
                  "🩺 Статус", "💸 Funding"]:
        cmd = k.LABEL_TO_COMMAND.get(label)
        assert cmd in h.COMMAND_DISPATCH, f"legacy {label!r} no longer routes"


def test_legacy_callback_data_still_dispatches():
    for cmd in ["signal", "intraday", "position", "reversal", "levels",
                "market", "journal", "alerts", "status"]:
        assert cmd in h.COMMAND_DISPATCH, f"old cmd:{cmd} would be dropped"


def test_removed_ux_no_longer_routes():
    """D1.3 cleanup: /guide, /deep, /fear are gone — dead UX must not
    silently resurrect via a stale label/callback mapping."""
    for label in ["🏛 Глубокий анализ", "📖 Гид", "😱 Fear & Greed"]:
        assert label not in k.LABEL_TO_COMMAND, f"removed label {label!r} still mapped"
    for cmd in ["deep", "guide", "fear"]:
        assert cmd not in h.COMMAND_DISPATCH, f"removed command {cmd!r} still dispatches"


def test_removed_ux_is_not_advertised_anywhere():
    """The mapping check above only guards the keyboard. A removed button
    named in any other user-facing string tells the user to press something
    that no longer exists — that is how "жми 🏛 Глубокий анализ" survived in
    an /ask error message long after /deep was deleted."""
    import pathlib
    import re

    # Button labels carry their emoji, so this does not collide with the
    # Fear & Greed *metric*, which legitimately survives inside /market.
    # The commands are matched with a boundary so that module paths like
    # tools/deep_discovery.py are not mistaken for an invitation to /deep.
    dead = [re.escape(lbl) for lbl in
            ("🏛 Глубокий анализ", "📖 Гид", "😱 Fear & Greed")]
    dead += [r"/(?:deep|guide|fear)(?![\w/])"]
    pattern = re.compile("|".join(dead))

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or ".venv" in path.parts:
            continue
        for hit in set(pattern.findall(path.read_text(encoding="utf-8"))):
            offenders.append(f"{path.relative_to(root)}: {hit!r}")
    assert not offenders, "removed UX still referenced: " + "; ".join(offenders)


def _d12_signal() -> dict:
    return {
        "timeframe": "4h", "timestamp": "2026-07-21 14:00 UTC",
        "direction": "long", "style_label": "Свинг", "style_emoji": "📊",
        "display_price": 118_420.0, "entry_price": 118_260.0,
        "stop_loss": 116_900.0, "target_1": 119_800.0, "target_2": 121_300.0,
        "position_size": 0.42, "htf_bias": "bullish", "htf_tf": "1d",
        "reasons": ["EMA aligned bullish", "Bullish order block reaction"],
    }


def test_signal_presents_structure_not_a_trade_plan():
    """D1.2 §2/§4: the numbers stay (real structural facts) but the
    prescriptive trade-plan framing does not, and every message says
    plainly that no validated model backs it."""
    from bot import formatting

    text = formatting.format_signal(_d12_signal())

    # the facts survive
    assert "116 900" in text and "119 800" in text and "121 300" in text
    assert "EMA aligned bullish" in text
    # the prescriptive framing does not
    for banned in ["Risk Management", "Стоп-лосс", "Цель 1", "Размер позиции",
                   "Удержание", "План"]:
        assert banned not in text, f"prescriptive framing {banned!r} came back"
    assert formatting.EVIDENTIARY_NOTE in text


def test_signal_and_blocked_never_show_score_or_confidence():
    """D1.2 §3: the score and the fake AI-confidence number must not
    return to any user-facing screen."""
    from bot import formatting

    texts = [
        formatting.format_signal({**_d12_signal(), "score": 8, "status": "journal"}),
        formatting.format_blocked(
            {"blocked_at": "no_trade", "price": 118_260.0,
             "htf_bias": "bullish", "htf_tf": "1d"},
            display_price=118_420.0),
    ]
    for text in texts:
        for banned in ["Score", "score", "Уверенность", "Confluence", "/10"]:
            assert banned not in text, f"{banned!r} leaked into a user screen"


def test_blocked_drops_directional_waiting_hint():
    """D1.2 §5: BLOCK_HINT's directional framing oversold a system with no
    validated edge; the diagnostic fact stays, the lean goes."""
    from bot import formatting

    assert not hasattr(formatting, "BLOCK_HINT")
    text = formatting.format_blocked(
        {"blocked_at": "no_trade", "htf_bias": "neutral", "htf_tf": "1d"})
    assert "Жду" not in text
    assert "следующей свече" in text


def test_error_template_is_shared_and_single_line_reason():
    """D1.2 §8/§22: one template for every command, and a raw multi-line
    exception never reaches the user verbatim."""
    import inspect

    from bot import formatting, handlers

    text = formatting.format_error(ValueError("binance timeout\nTraceback..."))
    assert text.startswith("⚠️ Не получилось выполнить запрос.")
    assert "binance timeout" in text
    assert "Traceback" not in text
    assert "/status" in text
    # degrades without an exception
    assert "None" not in formatting.format_error()
    # no handler keeps its own bespoke variant
    assert "⚠️ Ошибка" not in inspect.getsource(handlers)


def test_status_ai_line_reports_reachability_not_capability():
    """D1.2 §16: /status says whether the Claude API answers — it must not
    read as 'AI is validating the trades'."""
    from bot import formatting

    up = formatting.format_status({"ai_ok": True, "ai_model": "claude-x"})
    down = formatting.format_status({"ai_ok": False, "ai_error": "timeout"})
    unknown = formatting.format_status({})

    assert "Claude API: доступен (claude-x)" in up
    assert "Claude API: недоступен (timeout)" in down
    assert "проверка ещё не выполнялась" in unknown
    for text in (up, down, unknown):
        assert "🧠 AI:" not in text


def test_entry_screens_state_what_the_bot_does_not_know():
    """D1.2 §1/§2: /start and /help must not imply a validated edge, and
    must say plainly where the honest numbers are."""
    from bot import handlers

    for text in (handlers.WELCOME_TEXT, handlers.HELP_TEXT):
        assert "подтверждённой торговой модели" in text
    assert "присылаю сигналы" not in handlers.WELCOME_TEXT
    assert "/journal" in handlers.HELP_TEXT


def test_every_scheduled_message_shares_one_dry_run_path():
    """D1.2 item D: lifecycle events and reversal alerts used plain
    broadcast(), so in DRY_RUN they were dropped entirely and never carried
    the '🧪 DRY-RUN / MANUAL ONLY' banner the signal alert gets — despite
    being subject to the same gating."""
    import pathlib

    # Read the source instead of importing: importing scheduler needs a live
    # event loop, so an earlier test that ran asyncio.run() (and closed the
    # loop it created) made this guard fail purely on file ordering.
    src = (pathlib.Path(__file__).resolve().parent.parent / "scheduler.py"
           ).read_text(encoding="utf-8")
    # broadcast_photo is a separate, non-textual path and stays as-is
    assert "alerts.broadcast(" not in src
    assert src.count("alerts.send_signal_alert(") == 3


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
    assert "Fear & Greed: 28/100" in text
    # D1.2 §11: raw numbers, no interpretation layered on top
    assert "0.0800%" in text and "z=2.4" in text and "2.30" in text
    for banned in ("перегрев", "переполнены", "баланс", "норма"):
        assert banned not in text, f"alarmist reading {banned!r} came back"
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
