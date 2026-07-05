# Trading Journal v2 (Stage 13)

Read-only аналитический слой, который классифицирует **уже персистируемые**
строки `forecasts` + `forecast_outcomes` в человекочитаемые метки решения —
**без изменения торгового поведения**.

## Что это НЕ делает

- не меняет схему БД, `database.py`, миграции, backfill;
- не меняет `scheduler.py`, `pipeline.py`, producer (`analyzer/outcomes.py`),
  `forecasts_pending_outcomes`, outcome-tracking для NO_TRADE;
- не трогает scoring, thresholds, `final_gate`, `run_cascade`, risk,
  Telegram-вывод, Railway/env, `DRY_RUN`;
- не ходит в Binance/сеть, не пишет в БД (только `SELECT`);
- не импортируется runtime-кодом.

## Компоненты

| Файл | Роль |
|---|---|
| `analyzer/journal_classify.py` | Чистый leaf-классификатор (stdlib/typing). Формула R не дублируется и не меняется — `realized_r` приходит на вход как уже измеренное поле outcome. |
| `tools/trading_journal.py` | Read-only CLI/агрегатор поверх `--input` JSON или `--database-url` (SELECT). |

## Классификация

`classify(forecast, outcome) -> str`, где `outcome` может быть `None`.

### ENTER

| Условие | Метка |
|---|---|
| нет outcome / не `resolved` | `unresolved` |
| `resolved`, `realized_r > 0` | `good_trade` |
| `resolved`, `realized_r < 0` | `bad_trade` |
| `resolved`, `realized_r == 0` (безубыток) | `none` |
| `resolved`, `realized_r is None` (R не посчитан) | `none` |

`realized_r` берётся как есть из `forecast_outcomes` (Stage 11). Классификатор
его **не пересчитывает**.

### WAIT / NO_TRADE / blocked (гипотетический исход)

| Условие | Метка |
|---|---|
| нет `candidate_direction` (long/short) | `none` |
| нет достаточных уровней (стоп + TP1) | `no_levels` |
| нет outcome / не `resolved` | `none` |
| гипотетически сработал бы чистый стоп (`stop_hit` и не `tp1_hit`) | `avoided_loss` |
| гипотетически отработал бы (`tp2_hit`, либо `tp1_hit` без стопа) | `missed_opportunity` |
| иначе (breakeven / неоднозначно) | `none` |

Трактовка «отработал бы / чистый стоп» совпадает с win/loss-ветками
`analyzer.realized_r` и `_favorable` в `tools/forecast_metrics.py`.

**Принцип честности:** отсутствие уровней/направления/исхода никогда не
становится `good/bad/avoided/missed` — только `none` / `no_levels` /
`unresolved`.

### Практическое покрытие

`WAIT` (journal/cooldown) уже попадают в `forecasts_pending_outcomes`, поэтому у
них обычно есть измеренный outcome → достижимы `avoided_loss` /
`missed_opportunity`. `NO_TRADE`/blocked outcome сейчас **не** измеряется
(вне scope Stage 13), поэтому такие строки, как правило, дают `none` /
`no_levels`. Это ограничение осознанное; включение outcome-tracking для
NO_TRADE — отдельный будущий stage.

## Запуск

```bash
# offline из JSON-экспорта {forecasts, outcomes}
python -m tools.trading_journal --input export.json
python -m tools.trading_journal --input export.json --json

# read-only из живой БД (только SELECT)
python -m tools.trading_journal --database-url "$DATABASE_URL"
```

Сводка: total forecasts; разбивки по `analysis_status` / `analysis_type` /
`candidate_direction` / `classification`; ENTER (good/bad/unresolved/none);
WAIT+NO_TRADE (avoided/missed/no_levels/none); средний `realized_r` по ENTER;
инвентаризация пропусков (`missing_realized_r`, `missing_levels`,
`missing_direction`, `unresolved_outcomes`, `missing_market_regime`,
`missing_volatility_regime`).
