"""S4A: the Telegram adapter for the spot facts screener.

**This is the only module in `bot/` allowed to import `spot_market`.** The
boundary is one file wide on purpose and `tests/test_spot_screener.py` asserts
it: a screener that leaks into `handlers.py` would leak into everything
`handlers.py` already touches — the database, the futures pipeline, the risk
engine — and the spot/runtime separation would stop being checkable.

The direction of the whole feature, top to bottom:

    tools/spot_snapshot.py  builds a snapshot file  (offline, may be slow)
    spot_market/            reads it, or fails closed  (fast, pure)
    bot/spot_screener.py    turns that into screens    (this file)

Nothing here computes a market fact, and nothing here can: every number it
prints came out of the snapshot. That is also what makes the screens instant —
a button press opens one JSON file, never a research job.

Callback namespace `spot:` — separate from the existing `cmd:` and `menu:`
namespaces so an old keyboard sitting in someone's chat history cannot collide
with it:

    spot:menu             the section root
    spot:find             pick a factual view
    spot:view:<vid>       one view's shortlist
    spot:all:<page>       the browsable universe
    spot:coin:<SYMBOL>    one asset's card
    spot:btc              the benchmark's context
    spot:how              what the scanner does and does not do
"""
from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from spot_market import snapshot as snap_mod
from spot_market import text as spot_text
from spot_market import views as spot_views

log = logging.getLogger(__name__)

BACK_TO_MAIN = ("⬅️ Назад", "menu:main")


# --- keyboards --------------------------------------------------------------

def _markup(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=data) for label, data in r]
         for r in rows])


def spot_menu_keyboard() -> InlineKeyboardMarkup:
    return _markup([
        [("🔎 Найти монеты", "spot:find")],
        [("📋 Все монеты", "spot:all:0")],
        [("₿ Режим BTC", "spot:btc")],
        [("ℹ️ Как работает сканер", "spot:how")],
        [BACK_TO_MAIN],
    ])


def views_keyboard() -> InlineKeyboardMarkup:
    return _markup([[(v.label, f"spot:view:{v.vid}")]
                    for v in spot_views.VIEWS] + [[("⬅️ Назад", "spot:menu")]])


def _coin_rows(symbols: list[str], per_row: int = 2
               ) -> list[list[tuple[str, str]]]:
    rows: list[list[tuple[str, str]]] = []
    for i in range(0, len(symbols), per_row):
        rows.append([(s, f"spot:coin:{s}") for s in symbols[i:i + per_row]])
    return rows


def shortlist_keyboard(symbols: list[str]) -> InlineKeyboardMarkup:
    return _markup(_coin_rows(symbols) + [[("⬅️ К срезам", "spot:find")],
                                          [("🪙 Спот", "spot:menu")]])


def universe_keyboard(symbols: list[str], index: int, total: int
                      ) -> InlineKeyboardMarkup:
    """Coin buttons plus a pager. The counter is a button that does nothing.

    Telegram has no inert label, and leaving the middle slot empty would put
    the two arrows next to each other where a mis-tap costs a page. `spot:nop`
    is answered and ignored.
    """
    prev_i = index - 1 if index > 0 else index
    next_i = index + 1 if index < total - 1 else index
    nav = [("◀️", f"spot:all:{prev_i}"),
           (f"{index + 1}/{total}", "spot:nop"),
           ("▶️", f"spot:all:{next_i}")]
    return _markup(_coin_rows(symbols) + [nav, [("🪙 Спот", "spot:menu")]])


def coin_keyboard(back: str) -> InlineKeyboardMarkup:
    return _markup([[("⬅️ Назад", back)], [("🪙 Спот", "spot:menu")]])


# --- screens ----------------------------------------------------------------

def _load():
    """Read the snapshot. Raises `SnapshotUnavailable`; never returns stale.

    The path is looked up on the module rather than bound as a default, so a
    test can point the whole screener at a fixture snapshot and exercise the
    real loader instead of a stub of it.
    """
    return snap_mod.load_snapshot(snap_mod.DEFAULT_SNAPSHOT_PATH)


def _unavailable_screen(exc: snap_mod.SnapshotUnavailable
                        ) -> tuple[str, InlineKeyboardMarkup]:
    log.info("spot screener unavailable: %s", exc)
    return spot_text.unavailable(exc), _markup([[BACK_TO_MAIN]])


def render(route: str) -> tuple[str, InlineKeyboardMarkup]:
    """Route -> (text, keyboard). Pure apart from reading the snapshot file.

    Kept separate from the Telegram plumbing so every screen can be inspected
    in a test as the string a user would actually see.
    """
    # The methodology page describes the scanner, not the market, so it needs
    # no data and must survive the outage — it is where a user looking at the
    # unavailable screen is sent to find out what the scanner even is.
    if route == "how":
        return spot_text.HOW_IT_WORKS, _markup([[("⬅️ Назад", "spot:menu")]])

    try:
        snap = _load()
    except snap_mod.SnapshotUnavailable as exc:
        return _unavailable_screen(exc)

    if route in ("menu", ""):
        return spot_text.spot_menu(snap), spot_menu_keyboard()

    if route == "find":
        return spot_text.find_intro(snap), views_keyboard()

    if route.startswith("view:"):
        vid = route.split(":", 1)[1]
        view = spot_views.VIEWS_BY_ID.get(vid)
        if view is None:
            return spot_text.find_intro(snap), views_keyboard()
        coins = spot_views.shortlist(snap.coins, view)
        return (spot_text.shortlist_screen(view, coins, snap),
                shortlist_keyboard([c.symbol for c in coins]))

    if route.startswith("all:"):
        raw = route.split(":", 1)[1]
        index = int(raw) if raw.lstrip("-").isdigit() else 0
        page, idx, total = spot_views.page(snap.symbols(), index)
        return (spot_text.universe_page(page, idx, total, snap),
                universe_keyboard(page, idx, total))

    if route.startswith("coin:"):
        symbol = route.split(":", 1)[1]
        coin = snap.by_symbol(symbol)
        if coin is None:
            # A stale keyboard can name a coin that has since left the
            # universe. Say so rather than rendering an empty card.
            return (f"🪙 {symbol}\n\nЭта монета сейчас не проходит текущие "
                    f"условия сканера — она либо перестала торговаться, либо "
                    f"её оборот опустился ниже порога.\n\n"
                    f"{spot_text.freshness_line(snap)}",
                    coin_keyboard("spot:all:0"))
        return spot_text.coin_card(coin, snap), coin_keyboard("spot:all:0")

    if route == "btc":
        return spot_text.btc_screen(snap.btc, snap), _markup(
            [[("⬅️ Назад", "spot:menu")]])

    return spot_text.spot_menu(snap), spot_menu_keyboard()


# --- handlers ---------------------------------------------------------------

async def spot_menu_cmd(update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply-keyboard press or /spot: send the section as a new message."""
    text, markup = render("menu")
    await update.effective_message.reply_text(text, reply_markup=markup)


async def spot_callback(update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline navigation inside the `spot:` namespace, edited in place."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if not data.startswith("spot:"):
        return
    route = data.split(":", 1)[1]
    if route == "nop":  # the page counter
        return
    text, markup = render(route)
    try:
        await query.edit_message_text(text, reply_markup=markup)
    except Exception as exc:  # noqa: BLE001 — e.g. "Message is not modified"
        log.debug("spot screen edit skipped: %s", exc)
