# S4A — the facts-only spot screener, as shipped

Recorded 2026-08-25. **This is a product stage, not a research stage.** It
registers no trial, measures no hypothesis and changes no verdict. The spot
family's `n_trials` is still 6, `CLEAN_2X` is still +100% / −40% / 180 days,
and S3 is still `S3_NO_SIGNAL`.

It implements option **A** from `reports/s3/SPOT_TRACK_STATUS.md` §2: ship a
facts screener with no selection claim. Options B and C remain untaken.

---

## 1. What was built

A Telegram section that answers *"what is currently true about the spot
market"* and refuses to answer *"what should I buy"*. The user gets a
browsable universe of currently tradable, currently liquid assets; four
single-fact orderings over it; a card per asset; and BTC's own state as
context.

The one design decision worth arguing about is §7. Everything else follows
from it.

## 2. Menu structure

Main menu (reply keyboard and inline, both):

```
📊 Анализ рынка
🔮 Прогноз
📓 Торговый журнал
🪙 Спот / Монеты        ← new
```

Inside `🪙 Спот / Монеты`:

```
🔎 Найти монеты      → 🔥 Сильнее рынка / 💧 Самые ликвидные
                       📉 В глубокой просадке / ⚡ Высокая активность
📋 Все монеты        → paginated, 10 per page, a button per coin
₿ Режим BTC
ℹ️ Как работает сканер
⬅️ Назад
```

Also reachable as `/spot`.

Callback namespace `spot:` — new, and separate from the existing `cmd:` and
`menu:` namespaces so that inline keyboards already sitting in a user's chat
history cannot collide with it. Routes: `spot:menu`, `spot:find`,
`spot:view:<vid>`, `spot:all:<page>`, `spot:coin:<SYMBOL>`, `spot:btc`,
`spot:how`, `spot:nop` (the inert page counter).

Nothing existing was redesigned. Every superseded label and callback still
routes where it used to; `tests/test_menu.py` still asserts that. The single
change to old behaviour is that `main_inline_keyboard()` now emits
`cmd:menu_spot` rather than `menu:spot` for the new section, because the spot
root screen is built from live snapshot state and a static submenu title
cannot carry "34 монеты, данные от 24.08".

## 3. Data source

| layer | file | what it does |
|---|---|---|
| builder | `tools/spot_snapshot.py` | fetches / reads, screens, measures, writes one JSON |
| reader | `spot_market/` | reads that JSON, or refuses |
| adapter | `bot/spot_screener.py` | turns it into Telegram screens |

The builder has two sources, one schema (`spot.snapshot/1`):

* **`--source live`** — Binance spot `/api/v3/exchangeInfo` for what actually
  trades right now, plus the last 400 daily bars per candidate via
  `/api/v3/klines`. Needs no local panel. Measured: **467 symbols screened in
  33 s**. This is the deployable path.
* **`--source panel`** — the verified S1 panel on disk through
  `tools/spot_dataset.load_panel`. No network. Used by anyone who wants to
  rebuild from data they already trust.

Both produce byte-identical schemas and were run: live gives 34 eligible,
panel gives 32, and the difference is two symbols listed after the panel's
last ingestion.

The snapshot is written with a temp-file rename, because a Telegram handler
may be reading it at the moment it is replaced and a half-written JSON would
fail closed for the wrong reason.

## 4. Freshness rule — frozen

Two independent clocks, **both** must pass, checked on every read:

```
now - data_asof_ms      <= 48 h     the oldest last-closed bar behind any coin shown
now - generated_at_ms   <= 24 h     when the snapshot file itself was built
```

Neither implies the other. The first catches a build that ran against a stale
panel; the second catches a snapshot that was fresh when written and has since
been left to rot, which is what happens when the producer stops running. A
snapshot whose timestamps are ahead of now is refused as `clock` rather than
served.

`data_asof_ms` is the **minimum** last-bar close over every coin shown plus
BTC, not the maximum. The freshness promise has to hold for every coin on the
screen, so the weakest link is what gets published.

**And the reader does not take that value on trust.** It re-derives the oldest
bar behind everything it is about to show and refuses a file that claims to be
fresher than its own contents (`inconsistent`). A Codex audit found this gap:
with the two clocks checked in isolation, a snapshot with a current top-level
timestamp and one coin whose last bar closed a year ago rendered that coin as
an ordinary card, and every screen still looked normal. A bar dated ahead of
now is refused as `clock`. Four regression tests cover it, including one that
asserts the builder's own output survives the reader's re-derivation — the two
rules live in different files and could otherwise drift apart silently.

On failure the user sees a named reason and **no numbers at all** — never a
partial screen:

```
⚠️ Рыночные данные устарели — обновление не завершено.

Сканер временно недоступен — показывать устаревшие цены как текущие нельзя.

Требуется свежесть: данные не старше 48 ч, снимок не старше 24 ч.
Причина: stale_data — last closed bar is 53.0h old, limit 48h
```

`ℹ️ Как работает сканер` is the one screen that survives the outage: it
describes the scanner, not the market, and it is where a confused user is
sent.

## 5. Current eligibility — and why it is not the S2 rule

S2 asks *could a person have bought this on 2019-04-01*. S4A asks *can a
person buy this today*. Confusing the two is exactly the S2 E8 defect, which
admitted 938 rows of coins that had already stopped trading, and in a live
scanner the same mistake would put delisted assets in front of a user as
current opportunities.

An asset appears today only if all of the following hold:

| rule | threshold | origin |
|---|---|---|
| venue status is `TRADING` right now | — | new, from live `exchangeInfo` |
| not pegged / wrapped / leveraged | `spot.universe` lists | S2, unchanged |
| visible daily bars | ≥ 200 | S2 |
| listing age | ≥ 180 days | S2 |
| median quote volume, last 30 bars | ≥ $5 M | S2 |
| **last closed bar age** | **≤ 48 h from now** | **new — replaces E8** |
| last 30 bars are consecutive | span == 29 days | S2, and runs *after* the line above |

The order of the last two lines **is** the correction, and the code says so.
`stale` is anchored to `now`; a symbol whose last bar closed weeks ago fails
it no matter how tidy its own history is. `inactive` is then allowed to count
backwards from the symbol's last bar, because staleness has already
established that the last bar is recent. On its own the second check is the
defect — a coin delisted in 2022 has thirty perfectly consecutive final bars
forever — which is why it never runs on its own.
`tests/test_spot_snapshot.py` asserts both halves, including that the dead
frame's own `tail(30)` really is regular.

The historical point-in-time universe in `spot/universe.py` was not touched.

## 6. Fields shown

Per asset, all measured on bars that had closed before the build instant (the
day in progress is never read):

`price`, `ret_30d`, `ret_90d`, `rel_btc_90d`, `drawdown_180d`,
`median_quote_volume_30d`, `realized_vol_30d`, `relative_volume`, plus
`bars`, `listing_age_days`, `last_close_ms`.

Three of these reuse S3 definitions verbatim — B5 (relative strength), F3
(relative participation), F4 (drawdown) — because the numbers a user sees
should be the numbers the research measured. **This is not a rehabilitation of
those features.** S3's verdict stands; nothing here combines two of them or
attaches a weight to one. B4, F1 and F2 are not displayed at all, except that
F2's definition supplies BTC's own distance from its 200-day mean on the
benchmark screen.

Percentiles are attached per field, mid-rank over the current eligible
universe. They stay per field: a mean of percentiles is a composite score
wearing a different hat, and `test_percentiles_are_never_summed_or_averaged`
rejects it at the AST level.

## 7. Shortlist logic — four views, no ranking

**There is no single "Top coins" list, and that is the finding turned into a
design.** A single list would have to combine fields; every weight would be an
unmeasured claim about relative importance; S3 measured six such orderings and
every one of them selected a top quintile that doubled *less* often than the
eligible universe (lifts 0.847–0.939).

So: four views, **each ordered by exactly one observed field**, `SHORTLIST_SIZE
= 8`.

| view | ordered by | direction |
|---|---|---|
| 🔥 Сильнее рынка | `rel_btc_90d` | desc |
| 💧 Самые ликвидные | `median_quote_volume_30d` | desc |
| 📉 В глубокой просадке | `drawdown_180d` | asc |
| ⚡ Высокая активность | `relative_volume` | desc |

`View.__post_init__` refuses any field that is not one of the measured ones,
so a composite cannot be smuggled in as a clever string. Ordering is total and
deterministic, ties broken on symbol name — without that, two coins with equal
values would reshuffle between builds for no observable reason.

The strongest guard is behavioural rather than lexical:
`test_a_views_order_depends_on_its_own_field_and_on_nothing_else` perturbs
every *other* field hard enough to dominate any weighting scheme and asserts
the order does not move.

The drawdown view carries its own caption, because it is the one that most
invites a misreading: *«Глубокая просадка — это факт, а не повод покупать: она
одинаково часто бывает у монет, которые потом восстановились, и у тех, которые
не восстановились.»* F4's ρ = +0.136 observation stays
`OBSERVATION_NOT_HYPOTHESIS` and gets no extra weight here — the view exists
because drawdown is a fact a person wants to see, not because it was measured
to work.

`📋 Все монеты` is ordered alphabetically, which asserts nothing about merit.

## 8. What is explicitly NOT claimed

Not computed, not displayed, not present in the code:

* probability of 2x, of any move, or of anything else;
* expected return, target price, confidence, AI confidence;
* a predictive score, composite score, confluence score, alpha score;
* any weighted combination of features;
* Buy / Sell / Long / Short, entry, stop-loss, take-profit;
* leverage, liquidation, margin, position size, futures R:R, funding-based
  recommendation.

Negative context is phrased as *«требует осторожности»*, never as a short.

Two tests carry this. `test_no_screen_contains_a_trade_instruction_or_a_futures_concept`
bans a list of words outright. `test_every_predictive_word_that_appears_appears_inside_a_denial`
handles the harder case: the scanner *must* be able to print "это не прогноз",
so words like «прогноз», «вероятность», «рейтинг», «2x» are permitted only in
a sentence that also contains a negation. A screen that quietly starts calling
its shortlist a forecast fails.

No LLM is involved anywhere in S4A. The `📊 Контекст` paragraph on a coin card
is assembled deterministically from thresholds on numbers already printed
above it, in three sentence groups, with no clause depending on another
clause's value.

## 9. Architectural boundary

```
tools/spot_snapshot.py   research side · may import spot/ · network + panel
        ↓ writes data/spot/snapshot/current.json
spot_market/             product side  · imports nothing but stdlib
        ↓
bot/spot_screener.py     the ONE bot module that may import spot_market
```

Asserted, not promised:

* `spot_market/` imports none of `spot`, `bot`, `config`, `database`,
  `scheduler`, `analyzer`, `signal_engine`, `pipeline`, `risk`, `tools`, `ai`,
  `labeling`, `validation`;
* `bot/spot_screener.py` does not import `spot` — it reads a file that package
  produced, not its code;
* no file outside `spot_market/` and `bot/spot_screener.py` imports
  `spot_market`;
* `tools/spot_snapshot.py` imports `spot.*` but not `bot`, `spot_market`,
  `database` or `signal_engine`.

The existing import guards in `tests/test_g1_integration.py` were **not
weakened**. Production still does not import `spot`, and `spot` still does not
import production; the new package sits between them and depends on neither.

Multi-asset by construction: every lookup takes an explicit symbol,
`by_symbol()` has no default, and `test_the_only_hard_coded_symbol_is_the_benchmark`
scans all four modules for symbol literals and permits exactly one — `BTCUSDT`,
used as the benchmark. Nothing reads `database.open_trades()` or any futures
state; the D1.3 blocker cannot recur here because no futures state is touched
at all.

Performance: a button press opens one JSON file. The multi-year research
computation lives in the builder and cannot be reached from a handler.

## 10. Tests

**2151 passed** (2054 before S4A; +97).

New: `tests/test_spot_snapshot.py` (22) and `tests/test_spot_screener.py` (60).
Amended: `tests/test_menu.py` (main menu is four sections),
`tests/test_spot_governance.py` (+3, §11).

Coverage against the stage's requirements:

| requirement | test |
|---|---|
| new menu routes | `test_the_spot_section_is_reachable_from_the_main_menu` |
| old buttons still routable | `test_every_label_routes_to_a_handler` (existing) |
| spot menu / find / all / coin / BTC / help | one test each |
| pagination, clamping, cross-page numbering | 3 tests |
| no probability of 2x | `test_no_screen_shows_a_probability_of_anything` |
| no BUY/SELL, no futures fields | `test_no_screen_contains_a_trade_instruction_or_a_futures_concept` |
| no predictive confidence | `test_no_module_contains_a_score_or_a_combiner` |
| no composite score / weight combiner | same, plus the AST percentile test |
| **delisted asset cannot appear today** | `test_a_delisted_asset_cannot_appear_in_todays_scanner`, `test_the_recency_test_is_anchored_to_now_not_to_the_symbols_own_history` |
| stale data fails closed | `test_stale_data_fails_closed`, `test_a_snapshot_that_stopped_being_rebuilt_fails_closed` |
| missing / broken data fails closed | 8 parametrised cases + `test_every_route_fails_closed_when_the_snapshot_is_gone` |
| a stale coin cannot hide behind a fresh clock | `test_a_stale_coin_cannot_hide_behind_a_fresh_top_level_clock` + 3 (audit round 1) |
| deterministic ordering | `test_ordering_is_deterministic_and_breaks_ties_on_the_symbol` |
| explicit-symbol behaviour | `test_a_coin_is_only_ever_looked_up_by_an_explicit_symbol`, `test_the_only_hard_coded_symbol_is_the_benchmark` |
| no dependence on futures open trades | `test_the_screener_does_not_touch_futures_open_trades_or_the_database` |
| boundary stays explicit | 4 import-direction tests |
| S2/S3 verdicts unchanged | `tests/test_spot_governance.py` |
| push blocker recorded resolved | `test_the_push_blocker_is_recorded_as_resolved` |

## 11. Governance correction (Phase 0)

`reports/s3/spot_governance.json` and `SPOT_TRACK_STATUS.md` §4 recorded the
branch as 12 commits ahead with `pushed: false` and an HTTP 403 blocker. That
is no longer true: origin is at `79a1899`, 0 ahead / 0 behind.

The record now says `pushed: true`, `commits_ahead_of_origin: 0`,
`push_blocker: null`, and keeps the 403 in a `push_blocker_history` block
marked `RESOLVED` with the date. It is marked rather than deleted, because a
record that erases what it corrected cannot be checked later.

**This is an operational fact only.** No verdict, number or artifact moved
with it, and `test_resolving_the_push_blocker_changed_no_research_fact`
asserts that.

## 12. Screens, as they actually render

Built live, 2026-08-25, 34 assets:

```
🪙 СПОТ / МОНЕТЫ

Сейчас в наблюдении 34 монеты, которые торгуются на бирже прямо сейчас
и проходят порог ликвидности.

Это факты о текущем состоянии рынка, а не прогноз и не сигнал на покупку.

Данные: 24.08.2026 23:59 UTC · вселенная 34 монеты
```

```
🔥 Сильнее рынка — 90 дней

Отсортировано по одному факту: доходность за 90 дней относительно BTC.
Это описание прошлого, а не прогноз.

1. TUT — $0.0771
    Сила к BTC (90д): +642.2% · 98-й перцентиль
    30д +402.3% · к BTC (90д) +642.2% · просадка -61.7%
2. ALLO — $0.2685
    Сила к BTC (90д): +213.3% · 95-й перцентиль
    30д -30.1% · к BTC (90д) +213.3% · просадка -48.1%
...
```

```
🪙 SUIUSDT

Цена: $0.8153
30 дней: +14.7%
90 дней: -18.5%
Сила к BTC (90д): -20.1% · 14-й перцентиль
Просадка от максимума (180д): -38.8%
Оборот (30д, медиана): $12.7 млн · 55-й перцентиль
Волатильность (30д, годовая): 72% · 39-й перцентиль
История: 1210 дневных баров (1210 дней на бирже)

📊 Контекст
За последние 90 дней монета слабее BTC на 20% и остаётся на 39% ниже
максимума за 180 дней. Ликвидность — в средней части доступной вселенной.
Торговый оборот за последний месяц ниже собственной полугодовой нормы.

Это факты о текущем состоянии рынка, а не прогноз и не сигнал на покупку.

Данные: 24.08.2026 23:59 UTC · вселенная 34 монеты
```

```
₿ РЕЖИМ BTC

Цена: $77 458
30 дней: +20.3%
90 дней: +2.0%
К 200-дневной средней: +12.1%
Волатильность (30д, годовая): 43%

Состояние: range — цена и 200-дневная средняя не сходятся в одном направлении.

Это контекст для чтения альткоинов: они обычно двигаются вместе с BTC.
Из состояния BTC не следует ни начало альтсезона, ни момент входа —
такой метод в проекте не проверен.
```

## 13. Known limitations

1. **Nothing rebuilds the snapshot automatically.** The builder is a CLI. In
   production the file will age past 24 h and the screener will correctly go
   dark. Wiring it to a schedule is §14, and it was deliberately left out of
   this stage rather than touching `scheduler.py`.
2. **The universe is small — 34 assets.** That is the $5 M median-volume floor
   meeting a thin market: 350 of 467 screened symbols are excluded as
   illiquid, 83 for too little history. The threshold is S2's and was not
   re-tuned here, because changing a frozen screening threshold to make a
   product screen look fuller is how a threshold stops meaning anything. If it
   should be lower for the product, that is a decision to take explicitly.
3. **Daily bars only.** Prices are yesterday's close, not a live tick. The
   screener is for deciding what to look at, not for timing anything, and the
   freshness line says which day the numbers are from.
4. **No intraday or weekly views**, no filtering or search within `Все
   монеты`, no sorting inside the coin card, no watchlist.
5. **Live mode makes ~470 requests per build.** Fine at hourly cadence,
   rate-limited by the client's own backoff, but not something to run per user
   request — and it structurally cannot be, since no handler can reach it.
6. **The four views are a judgement about what a person wants to see**, not a
   measured claim that these four fields are the useful ones. Nothing here
   establishes that.
7. **Tradability is enforced by the builder, not by the reader.** The snapshot
   carries no per-coin tradability flag, so a hand-written file naming a
   delisted symbol with fresh timestamps would render. The audit raised this
   and did not treat it as a defect: no builder path produces such a file —
   live mode fetches only `TRADING` candidates and `screen_now` requires
   `trading_now` — and the reader already refuses anything whose bars are
   stale. It is recorded because the guarantee lives on one side of the
   boundary and the guarantee's consumer lives on the other.

## 14. Exact next product step

**Schedule the builder.** Add a job that runs
`python -m tools.spot_snapshot --source live` every 1–2 hours and writes to
`data/spot/snapshot/current.json`, so the screener stops going dark after a
day. Live mode needs no panel, so it works on Railway as-is. This is a
deployment/scheduling change; it touches no research code and registers no
trial.

Everything else stays where `SPOT_TRACK_STATUS.md` §3 left it: S4 predictive
scanner not started, S3b not started, `CLEAN_2X` unchanged, Phase A′ paused,
G1 `G1_INDETERMINATE`.
