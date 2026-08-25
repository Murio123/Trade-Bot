"""S4A — the spot facts screener as a user meets it.

Three things are under test and only one of them is ordinary software.

The ordinary one: the screens render, the pager pages, the buttons route.

The second: **the screener fails closed.** A stale price shown as a current
one is indistinguishable from a fresh one at the moment a person acts on it,
so every path that cannot prove freshness must refuse rather than degrade.

The third, and the reason this stage was allowed to exist at all: **nothing
here may claim a predictive edge.** S3 measured six ranking rules against
`CLEAN_2X(180d)` and every one of them selected a top quintile that doubled
*less* often than the eligible universe. A product built on that result may
order coins by an observed fact; it may not combine facts into a score, and it
may not use language that implies a forecast. Those are asserted below on the
rendered strings and on the executable code, not left to review.
"""
from __future__ import annotations

import ast
import json
import re
import tokenize
from pathlib import Path

import pytest

from spot_market import snapshot as snap_mod
from spot_market import text as spot_text
from spot_market import views as spot_views
from spot_market.snapshot import (COIN_FIELDS, MAX_BAR_AGE_HOURS,
                                  MAX_SNAPSHOT_AGE_HOURS, SnapshotUnavailable,
                                  parse_snapshot)

REPO = Path(__file__).resolve().parents[1]
HOUR = 3_600_000
NOW = 1_704_067_200_000


# --- fixtures ---------------------------------------------------------------

def coin(symbol: str, **over) -> dict:
    base = {
        "symbol": symbol,
        "base_asset": symbol.removesuffix("USDT"),
        "price": 10.0,
        "ret_30d": 0.10,
        "ret_90d": 0.20,
        "rel_btc_90d": 0.05,
        "drawdown_180d": -0.25,
        "median_quote_volume_30d": 25_000_000.0,
        "realized_vol_30d": 0.8,
        "relative_volume": 0.1,
        "bars": 500,
        "listing_age_days": 500,
        "last_close_ms": NOW - HOUR,
    }
    base.update(over)
    base["percentiles"] = {f: 50 for f in COIN_FIELDS}
    return base


def payload(coins=None, **over) -> dict:
    coins = coins or [coin(f"C{i}USDT", price=float(i + 1),
                           rel_btc_90d=0.01 * i,
                           median_quote_volume_30d=1e7 * (i + 1),
                           drawdown_180d=-0.05 * (i + 1),
                           relative_volume=0.01 * i)
                      for i in range(25)]
    out = {
        "schema": "spot.snapshot/1",
        "venue": "binance_spot",
        "source": "test",
        "generated_at_ms": NOW - HOUR,
        "data_asof_ms": NOW - HOUR,
        "thresholds": {},
        "counts": {},
        "btc": {
            "symbol": "BTCUSDT", "price": 60000.0, "ret_30d": 0.05,
            "ret_90d": -0.02, "distance_above_ma200": 0.11,
            "realized_vol_30d": 0.45, "regime": "range",
            "last_close_ms": NOW - HOUR,
        },
        "coins": coins,
    }
    out.update(over)
    return out


@pytest.fixture
def snap():
    return parse_snapshot(payload(), now_ms=NOW)


@pytest.fixture
def screener(tmp_path, monkeypatch):
    """The real bot adapter, pointed at a fixture snapshot on disk."""
    from bot import spot_screener as ss

    path = tmp_path / "current.json"

    def write(data: dict) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    write(payload())
    monkeypatch.setattr(ss.snap_mod, "DEFAULT_SNAPSHOT_PATH", str(path))
    monkeypatch.setattr(ss.snap_mod, "_now_ms", lambda: NOW)
    ss._write_fixture = write  # type: ignore[attr-defined]
    return ss


# --- failing closed ---------------------------------------------------------

def test_a_missing_snapshot_is_refused_rather_than_rendered_empty(tmp_path):
    with pytest.raises(SnapshotUnavailable) as exc:
        snap_mod.load_snapshot(str(tmp_path / "nope.json"), now_ms=NOW)
    assert exc.value.reason == "missing"


def test_stale_data_fails_closed():
    # The declared clock alone. The stronger case — producer and bars stale
    # together — is `test_stale_data_fails_closed_when_every_clock_moves_together`
    # below, and the case this one cannot see is
    # `test_a_stale_coin_cannot_hide_behind_a_fresh_top_level_clock`.
    old = payload(data_asof_ms=NOW - (MAX_BAR_AGE_HOURS + 1) * HOUR)
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(old, now_ms=NOW)
    assert exc.value.reason == "stale_data"


def test_a_snapshot_that_stopped_being_rebuilt_fails_closed():
    """Fresh bars are not enough if nothing has rebuilt the file since.

    The two clocks are independent: a builder that died yesterday leaves a
    file whose bars still look recent for another day.
    """
    old = payload(generated_at_ms=NOW - (MAX_SNAPSHOT_AGE_HOURS + 1) * HOUR)
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(old, now_ms=NOW)
    assert exc.value.reason == "stale_snapshot"


def test_a_snapshot_just_inside_both_limits_is_served():
    ok = payload(generated_at_ms=NOW - (MAX_SNAPSHOT_AGE_HOURS - 1) * HOUR,
                 data_asof_ms=NOW - (MAX_BAR_AGE_HOURS - 1) * HOUR)
    assert parse_snapshot(ok, now_ms=NOW).universe_size == 25


@pytest.mark.parametrize("mutate,reason", [
    (lambda p: p.update(schema="spot.snapshot/9"), "schema"),
    (lambda p: p.update(coins=[]), "empty"),
    (lambda p: p["coins"][0].pop("drawdown_180d"), "malformed"),
    (lambda p: p["coins"][0].update(price=float("nan")), "malformed"),
    (lambda p: p["coins"][0].pop("percentiles"), "malformed"),
    (lambda p: p.update(btc={}), "malformed"),
    (lambda p: p["btc"].update(regime="altseason"), "malformed"),
    (lambda p: p.update(generated_at_ms=NOW + 10 * HOUR), "clock"),
])
def test_missing_or_broken_data_fails_closed(mutate, reason):
    p = payload()
    mutate(p)
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(p, now_ms=NOW)
    assert exc.value.reason == reason


def test_a_duplicated_symbol_is_refused():
    p = payload(coins=[coin("AAAUSDT"), coin("AAAUSDT")])
    with pytest.raises(SnapshotUnavailable, match="duplicate"):
        parse_snapshot(p, now_ms=NOW)


def test_the_unavailable_screen_names_the_reason_and_shows_no_numbers(screener):
    screener._write_fixture(payload(
        data_asof_ms=NOW - (MAX_BAR_AGE_HOURS + 5) * HOUR))
    text, kb = screener.render("view:strength")
    assert "устарел" in text
    assert "stale_data" in text
    assert "C0USDT" not in text and "$" not in text
    assert [b.callback_data for r in kb.inline_keyboard for b in r] == \
        ["menu:main"]


def test_every_route_fails_closed_when_the_snapshot_is_gone(screener,
                                                            tmp_path):
    (tmp_path / "current.json").unlink()
    for route in ("menu", "find", "view:strength", "all:0", "coin:C0USDT",
                  "btc"):
        text, _ = screener.render(route)
        assert "недоступен" in text, route
    # The static methodology page is the one screen that needs no data, and
    # it must keep working: it is where a confused user is sent.
    assert "КАК РАБОТАЕТ СКАНЕР" in screener.render("how")[0]


# --- the screens ------------------------------------------------------------

def test_the_spot_menu_opens_and_offers_the_four_entries(screener):
    text, kb = screener.render("menu")
    assert "СПОТ / МОНЕТЫ" in text
    assert [b.callback_data for r in kb.inline_keyboard for b in r] == [
        "spot:find", "spot:all:0", "spot:btc", "spot:how", "menu:main"]


def test_find_coins_offers_views_and_never_a_single_top_list(screener):
    text, kb = screener.render("find")
    data = [b.callback_data for r in kb.inline_keyboard for b in r]
    assert data == ["spot:view:strength", "spot:view:liquidity",
                    "spot:view:drawdown", "spot:view:activity", "spot:menu"]
    assert "Единого «топа» здесь нет" in text


def test_a_view_returns_a_short_readable_shortlist(screener):
    text, kb = screener.render("view:liquidity")
    coins = [b.callback_data for r in kb.inline_keyboard for b in r
             if b.callback_data.startswith("spot:coin:")]
    assert len(coins) == spot_views.SHORTLIST_SIZE
    assert "C24USDT" in coins[0]  # the largest median volume
    assert view_caption_is_present(text)


def view_caption_is_present(text: str) -> bool:
    return any(v.caption[:30] in text for v in spot_views.VIEWS)


def test_an_unknown_view_falls_back_to_the_chooser(screener):
    """Old inline keyboards outlive the views they were built from."""
    text, _ = screener.render("view:alpha_score")
    assert "НАЙТИ МОНЕТЫ" in text


def test_all_coins_paginates_instead_of_dumping_the_universe(screener):
    first, kb = screener.render("all:0")
    assert first.count("USDT") == spot_views.PAGE_SIZE
    assert "страница 1/3" in first
    nav = [b.callback_data for r in kb.inline_keyboard for b in r]
    assert "spot:all:1" in nav and "spot:nop" in nav

    second, _ = screener.render("all:1")
    assert "страница 2/3" in second
    assert set(re.findall(r"C\d+USDT", first)).isdisjoint(
        re.findall(r"C\d+USDT", second))


def test_the_pager_clamps_rather_than_erroring_on_a_stale_page_number(screener):
    for route in ("all:99", "all:-4", "all:not-a-number"):
        text, _ = screener.render(route)
        assert "ВСЕ МОНЕТЫ" in text, route
    assert "страница 3/3" in screener.render("all:99")[0]


def test_the_universe_numbering_runs_across_pages(screener):
    assert "\n11. " in screener.render("all:1")[0]


def test_a_coin_card_shows_the_measured_fields_and_a_context_paragraph(
        screener):
    text, kb = screener.render("coin:C3USDT")
    assert "🪙 C3USDT" in text
    for label in ("Цена:", "30 дней:", "90 дней:", "Сила к BTC (90д):",
                  "Просадка от максимума (180д):", "Оборот (30д, медиана):",
                  "Волатильность (30д, годовая):", "📊 Контекст"):
        assert label in text, label
    assert [b.callback_data for r in kb.inline_keyboard for b in r] == \
        ["spot:all:0", "spot:menu"]


def test_a_coin_that_left_the_universe_says_so_instead_of_showing_a_blank_card(
        screener):
    text, _ = screener.render("coin:GONEUSDT")
    assert "не проходит текущие условия" in text
    assert "Цена:" not in text


def test_the_btc_screen_is_context_and_says_so(screener):
    text, _ = screener.render("btc")
    assert "РЕЖИМ BTC" in text
    assert "range" in text
    assert "альтсезона" in text and "не следует" in text


def test_the_methodology_screen_states_what_the_scanner_cannot_do(screener):
    text, _ = screener.render("how")
    assert "не предсказывает" in text
    assert "ни один не оказался лучше" in text


# --- no predictive claim ----------------------------------------------------

def all_screens(screener) -> dict[str, str]:
    routes = ["menu", "find", "all:0", "btc", "how", "coin:C3USDT",
              "coin:GONEUSDT"]
    routes += [f"view:{v.vid}" for v in spot_views.VIEWS]
    return {r: screener.render(r)[0] for r in routes}


# Words that must never appear at all, in any form. Every one of them is
# either a trade instruction or a futures concept, and neither belongs in a
# spot facts screen even inside a denial.
NEVER = ("BUY", "SELL", "LONG", "SHORT", "лонг", "шорт", "плеч", "ликвидац",
         "маржа", "маржин", "стоп-лосс", "тейк", "R:R", "фандинг", "funding",
         "уверенност", "confidence", "точк вход", "целевая цена", "тейк-профит",
         "покупай", "продавай", "набирай")


def test_no_screen_contains_a_trade_instruction_or_a_futures_concept(screener):
    for route, text in all_screens(screener).items():
        low = text.lower()
        for word in NEVER:
            assert word.lower() not in low, f"{route}: {word}"


# Words that are legitimate ONLY inside a denial. The scanner has to be able
# to say "this is not a forecast"; it must never say "this is the forecast".
HEDGED = ("прогноз", "вероятн", "рейтинг", "удвоен", "сигнал", "предсказ",
          "2x", "рекоменд")
NEGATIONS = ("не ", "нет ", "ни ", "без ", "вместо ")


def test_every_predictive_word_that_appears_appears_inside_a_denial(screener):
    """The disclaimer must be the only place these words can live.

    Banning them outright would ban the sentence the whole stage exists to
    print. Requiring a negation in the same sentence is the weaker rule that
    still catches the failure being guarded against — a screen that quietly
    starts calling its shortlist a forecast.
    """
    for route, text in all_screens(screener).items():
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            low = sentence.lower()
            if not any(w in low for w in HEDGED):
                continue
            assert any(n in low for n in NEGATIONS), f"{route}: {sentence!r}"


def test_no_screen_shows_a_probability_of_anything(screener):
    """No percentage may sit next to a claim about the future."""
    for route, text in all_screens(screener).items():
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            if re.search(r"\d+(?:[.,]\d+)?\s*%", sentence):
                low = sentence.lower()
                assert not any(w in low for w in
                               ("шанс", "вероятн", "прогноз")), \
                    f"{route}: {sentence!r}"


def test_the_no_claim_line_is_on_every_screen_that_shows_numbers(screener):
    for route in ("menu", "view:strength", "view:drawdown", "coin:C3USDT"):
        assert spot_text.NO_CLAIM in screener.render(route)[0], route


# --- no composite, no combiner ----------------------------------------------

def code_only(path: Path) -> str:
    """Executable source with comments and string literals stripped.

    This project necessarily *writes about* composite scores at length —
    explaining their absence is most of the S3 record. Scanning raw text would
    punish the explanation and reward deleting it.
    """
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


S4A_MODULES = [REPO / "spot_market" / "snapshot.py",
               REPO / "spot_market" / "views.py",
               REPO / "spot_market" / "text.py",
               REPO / "bot" / "spot_screener.py"]


def test_no_module_contains_a_score_or_a_combiner():
    for path in S4A_MODULES:
        code = code_only(path)
        for banned in ("composite", "total_score", "weighted_score", "weight",
                       "confluence", "combine_features", "alpha_score",
                       "probability", "confidence", "predict"):
            assert banned not in code, f"{path.name}: {banned}"


def test_each_view_orders_by_exactly_one_measured_field():
    for view in spot_views.VIEWS:
        assert view.field in COIN_FIELDS
    assert len({v.field for v in spot_views.VIEWS}) == len(spot_views.VIEWS)


def test_a_view_cannot_be_declared_on_anything_but_a_measured_field():
    with pytest.raises(ValueError, match="measured field"):
        spot_views.View(vid="x", label="x", title="x", caption="x",
                        field="alpha_score", descending=True)


def test_a_views_order_depends_on_its_own_field_and_on_nothing_else(snap):
    """The behavioural form of "no hidden combination".

    Every other field is perturbed hard enough to dominate any weighting
    scheme. If the order moves, something is reading a field it does not
    declare.
    """
    view = spot_views.VIEWS_BY_ID["liquidity"]
    before = [c.symbol for c in spot_views.order_by(snap.coins, view)]

    p = payload()
    for i, c in enumerate(p["coins"]):
        for f in COIN_FIELDS:
            if f != view.field:
                c[f] = float(len(p["coins"]) - i) * 3.0
            c["percentiles"][f] = 100 - i
    after = [c.symbol for c in
             spot_views.order_by(parse_snapshot(p, now_ms=NOW).coins, view)]
    assert after == before


def test_ordering_is_deterministic_and_breaks_ties_on_the_symbol(snap):
    view = spot_views.VIEWS_BY_ID["strength"]
    assert ([c.symbol for c in spot_views.order_by(snap.coins, view)]
            == [c.symbol for c in spot_views.order_by(snap.coins, view)])

    tied = parse_snapshot(payload(coins=[coin("BBBUSDT"), coin("AAAUSDT"),
                                         coin("CCCUSDT")]), now_ms=NOW)
    for v in spot_views.VIEWS:
        assert [c.symbol for c in spot_views.order_by(tied.coins, v)] == \
            ["AAAUSDT", "BBBUSDT", "CCCUSDT"], v.vid


def test_percentiles_are_never_summed_or_averaged_anywhere():
    """No arithmetic over the percentile mapping outside a single lookup."""
    for path in S4A_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                assert name not in ("sum", "mean", "fsum"), \
                    f"{path.name}: {name}() over facts"


# --- explicit symbols, no futures state -------------------------------------

def test_a_coin_is_only_ever_looked_up_by_an_explicit_symbol(snap):
    assert snap.by_symbol("C3USDT").symbol == "C3USDT"
    assert snap.by_symbol("NOPEUSDT") is None
    with pytest.raises(TypeError):
        snap.by_symbol()  # type: ignore[call-arg]


def test_the_only_hard_coded_symbol_is_the_benchmark():
    """D1.3's lesson: a process-global BTCUSDT is how multi-asset dies."""
    for path in S4A_MODULES:
        src = path.read_text(encoding="utf-8")
        for match in re.findall(r'"([A-Z0-9]{2,15}USDT)"', src):
            assert match == "BTCUSDT", f"{path.name}: {match}"
    assert snap_mod.BTC_SYMBOL == "BTCUSDT"


def test_the_screener_does_not_touch_futures_open_trades_or_the_database():
    for path in S4A_MODULES:
        code = code_only(path)
        for banned in ("database", "open_trades", "signal_engine", "pipeline",
                       "analyzer", "scheduler", "risk"):
            assert banned not in code, f"{path.name}: {banned}"


# --- the architectural boundary ---------------------------------------------

def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_spot_market_imports_neither_the_runtime_nor_the_research_package():
    """It sits between the two and may depend on neither."""
    forbidden = {"spot", "bot", "config", "database", "scheduler", "analyzer",
                 "signal_engine", "pipeline", "risk", "tools", "ai",
                 "labeling", "validation"}
    for path in (REPO / "spot_market").rglob("*.py"):
        for name in imports_of(path):
            assert name.split(".")[0] not in forbidden, \
                f"{path.name} imports {name}"


# The two production modules allowed to import `spot_market`, and what each
# is for. S4A had one; S4A.1 added the second so the scheduler can keep the
# snapshot fresh. The list is short and explicit because its whole value is
# that adding to it takes a deliberate edit — `handlers.py` appearing here
# would mean the screener had reached the database and the futures pipeline.
SPOT_MARKET_CONSUMERS = {
    "bot/spot_screener.py",   # renders the Telegram screens
    "scheduler.py",           # runs the two-hourly refresh job (S4A.1)
}


def test_only_the_declared_consumers_import_spot_market():
    skip = {".venv", ".git", "__pycache__", "scratchpad", "tests",
            "spot_market"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        if str(py.relative_to(REPO)) in SPOT_MARKET_CONSUMERS:
            continue
        if any(n.split(".")[0] == "spot_market" for n in imports_of(py)):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, offenders


def test_the_research_package_never_enters_the_bot_process():
    """S4A.1's load-bearing property, and the reason the builder is a
    subprocess rather than an import.

    `tools/spot_snapshot.py` imports `spot.features` and `spot.universe`. If
    anything reachable from `main.py` imported IT, offline research code would
    be loaded into the live runtime — the direction the S-track's import rule
    exists to forbid. A module name in a string is not an import, and this is
    what makes that claim checkable rather than rhetorical.
    """
    skip = {".venv", ".git", "__pycache__", "scratchpad", "tests", "tools",
            "spot", "labeling", "validation"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        for name in imports_of(py):
            root = name.split(".")[0]
            if root == "spot" or (root == "tools"
                                  and "spot_snapshot" in name):
                offenders.append(f"{py.relative_to(REPO)} imports {name}")
    assert not offenders, offenders


def test_the_adapter_does_not_import_the_research_package():
    """`spot/` stays research. S4A reads a file it produced, not its code."""
    names = imports_of(REPO / "bot" / "spot_screener.py")
    assert not any(n == "spot" or n.startswith("spot.") for n in names)


def test_the_snapshot_builder_is_the_only_place_research_and_product_meet():
    """`tools/` is where the two sides are allowed to touch, by design."""
    names = imports_of(REPO / "tools" / "spot_snapshot.py")
    assert any(n.startswith("spot.") for n in names)
    assert not any(n.split(".")[0] in {"bot", "spot_market", "database",
                                       "signal_engine"} for n in names)


# --- menu wiring ------------------------------------------------------------

def test_the_spot_section_is_reachable_from_the_main_menu():
    import bot.handlers as h
    import bot.keyboards as k

    assert k.LABEL_TO_COMMAND[k.BTN_SPOT_COINS] == "menu_spot"
    assert h.COMMAND_DISPATCH["menu_spot"] is not None
    labels = [b.text for row in k.main_reply_keyboard().keyboard for b in row]
    assert k.BTN_SPOT_COINS in labels


def test_the_inline_main_menu_dispatches_spot_rather_than_navigating():
    """`menu:spot` would render a static title; the section needs live state."""
    import bot.keyboards as k

    data = {b.text: b.callback_data
            for row in k.main_inline_keyboard().inline_keyboard for b in row}
    assert data[k.BTN_SPOT_COINS] == "cmd:menu_spot"
    assert data[k.BTN_FORECAST] == "menu:forecast"


def test_the_slash_command_and_the_callback_pattern_are_registered():
    import bot.handlers as h

    assert ("spot", "🪙 Спот: сканер монет по фактам") in h.BOT_COMMANDS
    assert h.COMMAND_DISPATCH["spot"] is h.spot_screener.spot_menu_cmd
    src = (REPO / "bot" / "handlers.py").read_text(encoding="utf-8")
    assert 'pattern=r"^spot:"' in src


def test_the_spot_callback_namespace_does_not_collide_with_the_old_ones(
        screener):
    """`cmd:` and `menu:` keyboards predate this feature and still exist."""
    seen = set()
    for route in ("menu", "find", "all:0", "view:strength", "coin:C3USDT",
                  "btc", "how"):
        _, kb = screener.render(route)
        seen.update(b.callback_data
                    for row in kb.inline_keyboard for b in row)
    for data in seen:
        assert data.startswith("spot:") or data == "menu:main", data
        assert len(data.encode()) <= 64, f"{data} exceeds Telegram's limit"


@pytest.mark.asyncio
async def test_the_callback_handler_edits_the_message_in_place(screener):
    from unittest.mock import AsyncMock, MagicMock

    update = MagicMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = "spot:btc"
    await screener.spot_callback(update, MagicMock())
    update.callback_query.answer.assert_awaited()
    text = update.callback_query.edit_message_text.await_args.args[0]
    assert "РЕЖИМ BTC" in text


@pytest.mark.asyncio
async def test_the_page_counter_button_does_nothing(screener):
    from unittest.mock import AsyncMock, MagicMock

    update = MagicMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = "spot:nop"
    await screener.spot_callback(update, MagicMock())
    update.callback_query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_reply_keyboard_press_sends_the_section(screener):
    from unittest.mock import AsyncMock, MagicMock

    update = MagicMock()
    update.effective_message = AsyncMock()
    await screener.spot_menu_cmd(update, MagicMock())
    text = update.effective_message.reply_text.await_args.args[0]
    assert "СПОТ / МОНЕТЫ" in text


# --- rendering helpers ------------------------------------------------------

def test_prices_keep_enough_digits_for_a_cheap_coin():
    assert spot_text.price(0.00003214).count("0") >= 4
    assert spot_text.price(77458.0) == "$77 458"


def test_a_drawdown_that_rounds_to_zero_does_not_read_as_a_gain():
    assert spot_text.pct(0.0) == "0.0%"
    assert spot_text.pct(-0.00001) == "0.0%"
    assert spot_text.pct(0.184) == "+18.4%"


def test_the_universe_size_is_declined_correctly():
    assert spot_text.coins_word(1) == "монета"
    assert spot_text.coins_word(32) == "монеты"
    assert spot_text.coins_word(11) == "монет"
    assert spot_text.coins_word(25) == "монет"


def test_the_context_paragraph_is_a_pure_function_of_the_numbers(snap):
    c = snap.coins[0]
    assert spot_text.context_paragraph(c) == spot_text.context_paragraph(c)
    assert spot_text.context_paragraph(c).endswith(".")


# --- the freshness claim is re-derived, not trusted --------------------------

def test_a_stale_coin_cannot_hide_behind_a_fresh_top_level_clock():
    """The hole a two-clock check leaves open on its own.

    `data_asof_ms` is written by the producer. If the reader takes it on
    trust, one coin whose last bar closed a year ago renders as an ordinary
    card while every screen still looks current — the exact failure the
    fail-closed rule exists to prevent, and invisible from the outside.
    """
    p = payload()
    p["coins"][3]["last_close_ms"] = NOW - 365 * 24 * HOUR
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(p, now_ms=NOW)
    assert exc.value.reason == "inconsistent"


def test_a_stale_benchmark_cannot_hide_either():
    p = payload()
    p["btc"]["last_close_ms"] = NOW - 30 * 24 * HOUR
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(p, now_ms=NOW)
    assert exc.value.reason == "inconsistent"


def test_a_bar_dated_in_the_future_is_refused_as_a_clock_fault():
    p = payload()
    p["coins"][0]["last_close_ms"] = NOW + 48 * HOUR
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(p, now_ms=NOW)
    assert exc.value.reason == "clock"


def test_stale_data_fails_closed_when_every_clock_moves_together():
    """The honest version of a stale snapshot: producer and bars agree.

    The earlier test moved only the top-level field, which a reader that
    re-derives its own anchor would still catch. This one is what a real
    stalled pipeline looks like, and it must fail as `stale_data`.
    """
    old_ms = NOW - (MAX_BAR_AGE_HOURS + 5) * HOUR
    p = payload(data_asof_ms=old_ms)
    for c in p["coins"]:
        c["last_close_ms"] = old_ms
    p["btc"]["last_close_ms"] = old_ms
    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot(p, now_ms=NOW)
    assert exc.value.reason == "stale_data"


def test_a_snapshot_that_understates_its_own_freshness_is_still_served():
    """Conservative in the safe direction is not an error."""
    p = payload(data_asof_ms=NOW - 3 * HOUR)
    assert parse_snapshot(p, now_ms=NOW).universe_size == 25


def test_the_builder_declares_the_oldest_bar_and_the_reader_agrees():
    """The producer's own output must survive the reader's re-derivation.

    Asserted because the two rules live in different files and could drift
    apart silently: the builder would keep publishing and the reader would
    keep refusing.
    """
    from tests.test_spot_snapshot import NOW as BNOW
    from tests.test_spot_snapshot import frame, panel
    from tools.spot_snapshot import build_snapshot

    frames, meta = panel()
    frames["C0USDT"] = frame(400, ends_ms=BNOW - 20 * HOUR)
    built = build_snapshot(frames, meta, BNOW, source="test")
    snap = parse_snapshot(built, now_ms=BNOW)
    assert snap.data_asof_ms == min(
        [c.last_close_ms for c in snap.coins] + [snap.btc.last_close_ms])
