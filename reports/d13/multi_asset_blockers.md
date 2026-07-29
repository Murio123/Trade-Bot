# D1.3 — BTC hardcoding that will block ETH/SOL

Recorded while building the three-section button menu. **Nothing here was
changed in D1.3** — no multi-asset refactor was performed. This is the list a
future multi-asset task has to work through, in the order the dependencies
actually run.

## 1. The symbol is process-global, not per-request

`config.SYMBOL` / `config.SYMBOL_DISPLAY` (`config.py:76-77`) are read once from
the environment at import time. Every caller reaches for the global rather than
receiving a symbol:

- `bot/handlers.py` — 10 call sites pass `config.SYMBOL` into db/journal calls.
- `scheduler.py` — 9 call sites.
- `pipeline.py` — 3 call sites.
- `bot/formatting.py`, `bot/charts.py` — interpolate `config.SYMBOL_DISPLAY`
  into user-facing text and chart titles.

**Consequence:** one running process can only ever be one asset. Supporting
ETH/SOL means threading a symbol through the request path, not adding a menu.

**Mitigating factor:** the storage layer is already symbol-aware, so the
refactor is plumbing rather than redesign. `db.last_signal`, `db.signals_today`,
`db.forecast_stats`, `journal.get_stats`, `journal.can_open_new_trade` and
`journal.has_active_trade` all take an explicit `symbol` argument today.

## 2. `db.open_trades()` is symbol-blind

`database.py:391` selects every row with `outcome = 'open'` with no symbol
filter. Its callers (`bot/handlers.py`, `scheduler.py`) currently get away with
it because only one symbol exists.

**Consequence:** the moment a second asset is added, this is a correctness bug,
not just a limitation — the aggregate risk cap in `journal.can_open_new_trade`
and the open-position count on the journal screen would silently mix assets,
and an open ETH trade would block a BTC signal.

This is the single highest-risk item on the list.

## 3. The exchange client is constructed per-process

`bot_data["binance"]` is a single client instance shared by every handler
(`bot/handlers.py:_binance`). Market data calls (`current_price`,
`funding_rate`, `long_short_ratio`, `open_interest`) carry no symbol argument at
the call site. Multi-asset needs either a client per symbol or a symbol
parameter on each method.

## 4. Product copy names BTC directly

`HELP_TEXT` and `WELCOME_TEXT` (`bot/handlers.py:23,44`) say "BTC Signal Bot".
The menu labels added in D1.3 are deliberately asset-neutral — "📊 Анализ
рынка", "🔮 Прогноз", "📓 Торговый журнал", "🟢 Спот", "📈 Фьючерсы — свинг",
"⚡ Фьючерсы — интрадей" — so the navigation itself needs no rework when a
second asset arrives; only these two strings and a symbol picker do.

## 5. Profile tuning constants are BTC-scaled

`signal_engine/profiles.py` carries absolute point thresholds — e.g.
`POSITION_MIN_MOVE_PTS` defaults to 3000, `minimum_expected_move_points` on
each profile. These are BTC price-scale numbers; on SOL they are nonsense.
They would have to become percentage- or ATR-relative, which is a change to
decision logic and therefore out of scope for any UX task.

## Suggested order for a future multi-asset task

1. Fix `db.open_trades()` to take a symbol (correctness, independent of any
   multi-asset work — worth doing on its own).
2. Thread a symbol through the handler → pipeline → db path, defaulting to
   `config.SYMBOL`.
3. Make the exchange client symbol-aware.
4. Convert the profile point thresholds to relative units (decision-logic
   change: needs its own validation, not a UX ticket).
5. Only then add an asset picker to the menu.
