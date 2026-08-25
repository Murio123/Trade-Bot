"""S4A: several factual views instead of one ranking.

This file is where the S3 result becomes a product decision rather than a
disclaimer. S3 measured six ranking rules against `CLEAN_2X(180d)` and every
one of them selected a top quintile that doubled *less* often than the
eligible universe (lifts 0.847-0.939, `S3_NO_SIGNAL`). So the project knows two
things: these quantities are real and measurable, and it does not know how to
combine them into a selection.

A single "Top coins" list would have to combine them. It would need weights,
and every weight would be an unmeasured claim about relative importance —
exactly the composite that S3's absence of a combiner exists to prevent. So
there is no single list. There are four views, **each ordered by exactly one
observed field**, and the user picks which fact they are currently interested
in. That is honest about what is known: the ordering inside a view is a
property of the field, not a forecast about the coin.

The rules this file must keep:

  * one view = one field, read straight off `CoinFacts`;
  * no arithmetic between fields, ever — no sums, no products, no weights;
  * every order is total and deterministic, ties broken on symbol name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from spot_market.snapshot import COIN_FIELDS, CoinFacts

# How many assets a view shows. Small on purpose: the point is a shortlist a
# person will actually read, not a leaderboard.
SHORTLIST_SIZE = 8

# How many symbols one page of the full universe holds.
PAGE_SIZE = 10


@dataclass(frozen=True)
class View:
    """One factual ordering of the current universe.

    `field` names a single measured quantity. There is no second field and no
    weight, and `shortlist` refuses any field that is not one of the measured
    ones, so a composite cannot be introduced by passing a clever string.
    """
    vid: str
    label: str
    title: str
    # What the ordering does and does not mean, in the user's language.
    caption: str
    field: str
    descending: bool

    def __post_init__(self) -> None:
        if self.field not in COIN_FIELDS:
            raise ValueError(f"{self.vid}: {self.field!r} is not a measured "
                             f"field")


VIEWS: tuple[View, ...] = (
    View(
        vid="strength",
        label="🔥 Сильнее рынка",
        title="🔥 Сильнее рынка — 90 дней",
        caption="Отсортировано по одному факту: доходность за 90 дней "
                "относительно BTC. Это описание прошлого, а не прогноз.",
        field="rel_btc_90d",
        descending=True,
    ),
    View(
        vid="liquidity",
        label="💧 Самые ликвидные",
        title="💧 Самые ликвидные",
        caption="Отсортировано по одному факту: медианный дневной оборот за "
                "30 дней. Ликвидность говорит о том, насколько легко войти и "
                "выйти, и ничего не говорит о будущей цене.",
        field="median_quote_volume_30d",
        descending=True,
    ),
    View(
        vid="drawdown",
        label="📉 В глубокой просадке",
        title="📉 В глубокой просадке",
        caption="Отсортировано по одному факту: насколько цена ниже максимума "
                "за 180 дней. Глубокая просадка — это факт, а не повод "
                "покупать: она одинаково часто бывает у монет, которые потом "
                "восстановились, и у тех, которые не восстановились.",
        field="drawdown_180d",
        descending=False,
    ),
    View(
        vid="activity",
        label="⚡ Высокая активность",
        title="⚡ Высокая активность",
        caption="Отсортировано по одному факту: оборот за последние 30 дней "
                "против собственного оборота за 180 дней. Показывает, где "
                "стало заметно больше торговли, чем обычно бывает у этой же "
                "монеты.",
        field="relative_volume",
        descending=True,
    ),
)

VIEWS_BY_ID = {v.vid: v for v in VIEWS}


def order_by(coins: Iterable[CoinFacts], view: View) -> list[CoinFacts]:
    """A total, deterministic order over one field.

    The symbol tiebreak is not cosmetic: without it the order of two coins
    with identical values would depend on snapshot insertion order, and the
    same screen would reshuffle between builds for no observable reason.
    """
    sign = -1.0 if view.descending else 1.0
    return sorted(coins, key=lambda c: (sign * c.field(view.field), c.symbol))


def shortlist(coins: Iterable[CoinFacts], view: View, *,
              size: int = SHORTLIST_SIZE) -> list[CoinFacts]:
    """The first `size` assets of a view's order."""
    if size <= 0:
        raise ValueError("shortlist size must be positive")
    return order_by(coins, view)[:size]


def page_count(total: int, page_size: int = PAGE_SIZE) -> int:
    if total <= 0:
        return 1
    return (total + page_size - 1) // page_size


def page(symbols: list[str], index: int, page_size: int = PAGE_SIZE
         ) -> tuple[list[str], int, int]:
    """One page of an alphabetical symbol list, clamped into range.

    Returns (symbols_on_page, page_index, page_total). The index is clamped
    rather than rejected: a stale inline keyboard in a user's chat history can
    point at page 12 of a universe that has since shrunk to 8 pages, and the
    right answer there is the last page, not an error.
    """
    total = page_count(len(symbols), page_size)
    idx = max(0, min(int(index), total - 1))
    start = idx * page_size
    return symbols[start:start + page_size], idx, total
