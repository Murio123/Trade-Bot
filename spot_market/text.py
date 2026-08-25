"""S4A: deterministic Russian rendering of spot facts.

Every string below is a pure function of numbers that are already in the
snapshot. There is no model here and no language model here: the same facts
always produce the same sentence, which is what makes the wording auditable at
all. An LLM gloss could only sit downstream of this text, and could never
change what is selected or in what order — but S4A does not add one, because a
fluent sentence is exactly how an unvalidated claim gets in.

The wording rules, which the tests enforce rather than trust:

  * no probability, no forecast, no target price, no "2x";
  * no Buy / Sell / Long / Short, no entry, no stop, no leverage;
  * no confidence, no score, no rating;
  * anything that sounds like advice is phrased as an observation with its
    window attached ("за 90 дней"), because a number without its window is
    where a fact turns into a claim.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from spot_market.snapshot import (MAX_BAR_AGE_HOURS, MAX_SNAPSHOT_AGE_HOURS,
                                  BtcContext, CoinFacts, MarketSnapshot,
                                  SnapshotUnavailable)
from spot_market.views import PAGE_SIZE, View

# The one sentence the whole stage exists to keep visible.
NO_CLAIM = ("Это факты о текущем состоянии рынка, а не прогноз и не сигнал "
            "на покупку.")

REGIME_WORDS = {
    "bull": "цена выше 200-дневной средней, средняя растёт",
    "bear": "цена ниже 200-дневной средней, средняя снижается",
    "range": "цена и 200-дневная средняя не сходятся в одном направлении",
    "unknown": "истории недостаточно для оценки",
}


def price(value: float) -> str:
    """Enough digits to be exact, never more. A $0.00003 coin needs eight."""
    v = abs(value)
    if v >= 1000:
        return f"${value:,.0f}".replace(",", " ")
    if v >= 1:
        return f"${value:,.2f}".replace(",", " ")
    if v >= 0.01:
        return f"${value:.4f}"
    if v >= 0.0001:
        return f"${value:.6f}"
    return f"${value:.8f}"


def pct(value: float, digits: int = 1) -> str:
    """A fraction as a signed percentage.

    A value that rounds to zero loses its sign: "просадка +0.0%" reads as a
    gain that is not there, and "-0.0%" reads as a rounding artifact, which
    it is.
    """
    scaled = value * 100
    if abs(round(scaled, digits)) == 0.0:
        return f"{0.0:.{digits}f}%"
    return f"{scaled:+.{digits}f}%"


def coins_word(n: int) -> str:
    """Russian plural for «монета». 32 монеты, 11 монет, 21 монета."""
    if 11 <= n % 100 <= 14:
        return "монет"
    last = n % 10
    if last == 1:
        return "монета"
    if 2 <= last <= 4:
        return "монеты"
    return "монет"


def volume(value: float) -> str:
    if value >= 1e9:
        return f"${value / 1e9:.1f} млрд"
    if value >= 1e6:
        return f"${value / 1e6:.1f} млн"
    if value >= 1e3:
        return f"${value / 1e3:.0f} тыс"
    return f"${value:.0f}"


def percentile(value: int | None) -> str:
    return "—" if value is None else f"{value}-й перцентиль"


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%d.%m.%Y %H:%M UTC")


def freshness_line(snap: MarketSnapshot) -> str:
    n = snap.universe_size
    return f"Данные: {_ts(snap.data_asof_ms)} · вселенная {n} {coins_word(n)}"


# --- the fail-closed screen -------------------------------------------------

_UNAVAILABLE = {
    "missing": "Снимок рынка ещё не построен.",
    "unreadable": "Файл снимка рынка повреждён.",
    "malformed": "Снимок рынка неполный.",
    "schema": "Снимок рынка в неизвестном формате.",
    "empty": "В снимке рынка нет ни одной монеты.",
    "stale_snapshot": "Снимок рынка устарел — обновление не завершено.",
    "stale_data": "Рыночные данные устарели — обновление не завершено.",
    "clock": "Время в снимке рынка не сходится с текущим.",
}


def unavailable(exc: SnapshotUnavailable) -> str:
    """What the user sees instead of numbers. Never a partial screen.

    Failing closed only helps if the failure is legible: the user must be able
    to tell "not built yet" from "stopped updating", because only the second
    one means something broke.
    """
    head = _UNAVAILABLE.get(exc.reason, "Снимок рынка недоступен.")
    return (f"⚠️ {head}\n\n"
            f"Сканер временно недоступен — показывать устаревшие цены как "
            f"текущие нельзя.\n\n"
            f"Требуется свежесть: данные не старше {MAX_BAR_AGE_HOURS} ч, "
            f"снимок не старше {MAX_SNAPSHOT_AGE_HOURS} ч.\n"
            f"Причина: {exc.reason} — {exc.detail}")


# --- screens ----------------------------------------------------------------

def spot_menu(snap: MarketSnapshot) -> str:
    return "\n".join([
        "🪙 СПОТ / МОНЕТЫ",
        "",
        f"Сейчас в наблюдении {snap.universe_size} "
        f"{coins_word(snap.universe_size)}, которые торгуются на бирже прямо "
        f"сейчас и проходят порог ликвидности.",
        "",
        NO_CLAIM,
        "",
        freshness_line(snap),
    ])


def find_intro(snap: MarketSnapshot) -> str:
    return "\n".join([
        "🔎 НАЙТИ МОНЕТЫ",
        "",
        "Единого «топа» здесь нет — и это не упущение. Исследование S3 "
        "проверило шесть способов ранжировать монеты и ни один не отбирал "
        "будущих лидеров лучше, чем вся вселенная целиком. Поэтому вместо "
        "одного рейтинга — четыре независимых среза, каждый отсортирован "
        "ровно по одному наблюдаемому факту.",
        "",
        "Выбери, какой факт тебя сейчас интересует:",
        "",
        freshness_line(snap),
    ])


def shortlist_screen(view: View, coins: list[CoinFacts],
                     snap: MarketSnapshot) -> str:
    lines = [view.title, "", view.caption, ""]
    for i, c in enumerate(coins, 1):
        lines.append(f"{i}. {c.base_asset} — {price(c.price)}")
        lines.append(f"    {_view_value_line(view, c)}")
        lines.append(f"    30д {pct(c.ret_30d)} · к BTC (90д) "
                     f"{pct(c.excess_over_btc_90d)} · просадка "
                     f"{pct(c.drawdown_180d)}")
    lines += ["", NO_CLAIM, "", freshness_line(snap)]
    return "\n".join(lines)


def _view_value_line(view: View, c: CoinFacts) -> str:
    """The one field this view orders by, spelled out."""
    p = percentile(c.percentile(view.field))
    if view.field == "median_quote_volume_30d":
        return f"Оборот (30д, медиана): {volume(c.median_quote_volume_30d)} · {p}"
    if view.field == "rel_btc_90d":
        return f"Сила к BTC (90д): {pct(c.excess_over_btc_90d)} · {p}"
    if view.field == "drawdown_180d":
        return f"Просадка от максимума (180д): {pct(c.drawdown_180d)} · {p}"
    if view.field == "relative_volume":
        return (f"Оборот против своей нормы: "
                f"×{math.exp(c.relative_volume):.2f} · {p}")
    return f"{view.field}: {c.field(view.field):.4f} · {p}"


def universe_page(symbols: list[str], index: int, total: int,
                  snap: MarketSnapshot, page_size: int = PAGE_SIZE) -> str:
    # Numbering runs across the whole universe, not per page: "1." on page
    # four would make three different coins the first coin.
    lines = [f"📋 ВСЕ МОНЕТЫ — страница {index + 1}/{total}", ""]
    for i, sym in enumerate(symbols, index * page_size + 1):
        lines.append(f"{i}. {sym}")
    lines += ["", "Нажми на монету, чтобы открыть карточку.", "",
              freshness_line(snap)]
    return "\n".join(lines)


def coin_card(c: CoinFacts, snap: MarketSnapshot) -> str:
    lines = [
        f"🪙 {c.symbol}",
        "",
        f"Цена: {price(c.price)}",
        f"30 дней: {pct(c.ret_30d)}",
        f"90 дней: {pct(c.ret_90d)}",
        f"Сила к BTC (90д): {pct(c.excess_over_btc_90d)} · "
        f"{percentile(c.percentile('rel_btc_90d'))}",
        f"Просадка от максимума (180д): {pct(c.drawdown_180d)}",
        f"Оборот (30д, медиана): {volume(c.median_quote_volume_30d)} · "
        f"{percentile(c.percentile('median_quote_volume_30d'))}",
        f"Волатильность (30д, годовая): {c.realized_vol_30d * 100:.0f}% · "
        f"{percentile(c.percentile('realized_vol_30d'))}",
        f"История: {c.bars} дневных баров ({c.listing_age_days} дней на бирже)",
        "",
        "📊 Контекст",
        context_paragraph(c),
        "",
        NO_CLAIM,
        "",
        freshness_line(snap),
    ]
    return "\n".join(lines)


def context_paragraph(c: CoinFacts) -> str:
    """Deterministic prose, assembled from thresholds and nothing else.

    Each clause restates a number that is already on the card. Nothing is
    inferred, ranked or predicted — the paragraph exists so the card reads as
    a description rather than a table, not so it reads as an opinion.

    Three sentences, each built from one group of facts: where the price
    stands, how the asset sits in the universe, and whether trading is unusual
    for it. No clause depends on another clause's value, so no combination can
    creep in through the wording.
    """
    sentences: list[str] = []

    excess = c.excess_over_btc_90d
    if excess > 0.05:
        price_clause = (f"За последние 90 дней монета сильнее BTC на "
                        f"{abs(excess) * 100:.0f}%")
    elif excess < -0.05:
        price_clause = (f"За последние 90 дней монета слабее BTC на "
                        f"{abs(excess) * 100:.0f}%")
    else:
        price_clause = "За последние 90 дней монета идёт примерно вровень с BTC"

    dd = c.drawdown_180d
    if dd > -0.03:
        price_clause += " и держится у максимума за 180 дней"
    else:
        price_clause += (f" и остаётся на {abs(dd) * 100:.0f}% ниже максимума "
                         f"за 180 дней")
    sentences.append(price_clause + ".")

    liq = c.percentile("median_quote_volume_30d")
    vol = c.percentile("realized_vol_30d")
    place: list[str] = []
    if liq is not None:
        if liq >= 80:
            place.append("ликвидность — в верхней части доступной вселенной")
        elif liq >= 40:
            place.append("ликвидность — в средней части доступной вселенной")
        else:
            place.append("ликвидность — в нижней части доступной вселенной, "
                         "вход и выход дороже")
    if vol is not None:
        if vol >= 80:
            place.append("волатильность выше, чем у большинства монет "
                         "вселенной")
        elif vol < 20:
            place.append("волатильность ниже, чем у большинства монет "
                         "вселенной")
    if place:
        joined = ", ".join(place)
        sentences.append(joined[0].upper() + joined[1:] + ".")

    if c.relative_volume > 0.20:
        sentences.append(
            f"Торговый оборот за последний месяц примерно в "
            f"{math.exp(c.relative_volume):.1f} раза выше собственной "
            f"полугодовой нормы.")
    elif c.relative_volume < -0.20:
        sentences.append("Торговый оборот за последний месяц ниже собственной "
                         "полугодовой нормы.")

    return " ".join(sentences)


def btc_screen(btc: BtcContext, snap: MarketSnapshot) -> str:
    return "\n".join([
        "₿ РЕЖИМ BTC",
        "",
        f"Цена: {price(btc.price)}",
        f"30 дней: {pct(btc.ret_30d)}",
        f"90 дней: {pct(btc.ret_90d)}",
        f"К 200-дневной средней: {pct(btc.distance_above_ma200)}",
        f"Волатильность (30д, годовая): {btc.realized_vol_30d * 100:.0f}%",
        "",
        f"Состояние: {btc.regime} — {REGIME_WORDS[btc.regime]}.",
        "",
        "Это контекст для чтения альткоинов: они обычно двигаются вместе с "
        "BTC. Из состояния BTC не следует ни начало альтсезона, ни момент "
        "входа — такой метод в проекте не проверен.",
        "",
        freshness_line(snap),
    ])


HOW_IT_WORKS = "\n".join([
    "ℹ️ КАК РАБОТАЕТ СКАНЕР",
    "",
    "Сканер показывает монеты, которые торгуются прямо сейчас, имеют не "
    "меньше полугода истории и проходят порог ликвидности. Для каждой он "
    "считает несколько наблюдаемых величин: динамику за 30 и 90 дней, силу "
    "относительно BTC, просадку от максимума, оборот и волатильность — и "
    "показывает, где монета находится по этим величинам относительно "
    "остальных.",
    "",
    "Единого рейтинга нет специально. Вместо него — четыре среза, каждый "
    "отсортирован ровно по одному факту. Ни один срез не смешивает признаки "
    "между собой.",
    "",
    "Чего сканер не делает. Он не предсказывает рост, не считает "
    "вероятность удвоения и не даёт сигналов на покупку или продажу. "
    "Исследование проекта проверило шесть способов отбирать монеты по этим "
    "же признакам и ни один не оказался лучше, чем вся вселенная целиком. "
    "Поэтому здесь показываются факты, а решение остаётся за тобой.",
    "",
    "Если данные устарели, сканер отключается и говорит об этом вместо того, "
    "чтобы показать вчерашние цены как сегодняшние.",
])
