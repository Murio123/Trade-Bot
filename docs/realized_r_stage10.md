# Stage 10 — Realized R / Per-Forecast R Definition

> **Stage 11 update (реализовано, Option A без backfill):** формула вынесена в
> leaf-модуль `analyzer/realized_r.py` (единый источник; `tools/realized_r.py` и
> producer `analyzer/outcomes.py` используют её, без дублирования). Добавлена
> additive-nullable колонка `forecast_outcomes.realized_r DOUBLE PRECISION`
> (`ADD COLUMN IF NOT EXISTS`, в `_OUTCOME_COLS`); `measure_outcome` пишет raw R
> (без округления). Backfill НЕ делается — старые resolved-строки остаются NULL,
> покрытие растёт вперёд. `forecast_metrics.py` / `MISSING_DATA` / golden не
> тронуты (follow-up). См. чек-лист §8.

**Тип этапа:** фиксация формулы + offline read-only инструмент (Option B).

**Что Stage 10 делает:** фиксирует честное определение per-forecast `realized_r`
и добавляет offline-калькулятор `tools/realized_r.py` (+ тесты), считающий R из
**уже персистируемых** полей `forecasts` + `forecast_outcomes`.

**Что Stage 10 НЕ делает:** не меняет схему БД, `database.py`, `MIGRATIONS`,
`_OUTCOME_COLS`, `upsert_outcome`, `analyzer/outcomes.measure_outcome`,
`bot/journal.evaluate_trade`, scheduler / pipeline / scoring / risk / Telegram;
не добавляет колонку `forecast_outcomes.realized_r`; не делает backfill; не
подключает Binance execution. Модуль не импортируется runtime-кодом и не пишет
в БД (только SELECT через read-only загрузчики `forecast_metrics`).

Парный к `MISSING_DATA` в `tools/forecast_metrics.py` (пункт «honest
per-forecast R») и к плану расширения схемы в `docs/data_quality_stage8.md`.

---

## 1. Источник данных (read-only)

Только уже сохранённые поля, никакого нового обращения к klines:

- `forecasts`: `analysis_status`, `candidate_direction`, `stop_loss`,
  `take_profit_levels`, `executable_price_at_decision`, `signal_close_price`.
- `forecast_outcomes`: `tp1_hit`, `tp2_hit`, `stop_hit`, `return_72h`,
  `resolved`, `reference_price` (если присутствует).

Якорь-«вход» (`ref`): `reference_price` из outcome, иначе — та же дефиниция,
что в `measure_outcome`: `executable_price_at_decision → signal_close_price`.

---

## 2. Формула

```
risk = |reference_price - stop_loss|
sign = +1 (long) / -1 (short)
ref  = reference_price
R    = sign * (exit - ref) / risk        # единое с journal/backtest определение
```

`realized_r(forecast, outcome)`:

| # | Условие | Результат |
|---|---------|-----------|
| 1 | не ENTER / нет direction / нет outcome / unresolved / нет ref / нет stop / risk==0 | `None` |
| 2 | `stop_hit and not tp1_hit` | `-1.0` |
| 3 | `tp2_hit` (и tp2 задан) | `sign*(tp2-ref)/risk` |
| 4 | `tp1_hit and stop_hit` | `0.0` (после TP1 — безубыток) |
| 5a | `tp1_hit`, TP2 в сетапе отсутствует | `sign*(tp1-ref)/risk` (single-target вин) |
| 5b | `tp1_hit`, TP2 есть, но не взят, стопа нет | mark-to-market (см. §4) |
| 6 | ни TP, ни стоп, но `resolved` | mark-to-market |
| 7 | mark-to-market | `(return_72h/100)*ref/risk`; `None`, если `return_72h` None |

**Округление:** pure `realized_r` возвращает полную точность `float` (кроме
точных `-1.0` / `0.0`). Округление до 4 знаков — только на публичном
summary/CLI-выводе. Один подход, покрыт тестами (`pytest.approx`).

**Симметрия LONG/SHORT:** обеспечивается множителем `sign`; зеркальный сетап даёт
одинаковый R (тест `test_long_short_symmetry_mirror`).

---

## 3. Edge cases (все покрыты тестами)

LONG и SHORT (зеркально): стоп-only `-1R`; TP2 `+RR`; TP1→стоп `0R`;
single-target TP1 `+R`; TP1 без TP2/стопа → MTM; ни TP ни стоп (resolved) → MTM;
unresolved → `None`; нет `stop_loss` → `None`; нет `reference_price` → `None`;
`risk==0` → `None`; `WAIT` → `None`; `NO_TRADE` → `None`.

`realized_r_summary` игнорирует `None` в average/median/распределении и отдаёт
`kind_counts` (loss / tp2_win / breakeven / tp1_single_win / mark_to_market) и
`none_reasons` (причины недоступности).

---

## 4. Почему TP1 mid-case не exact journal parity

`measure_outcome` **не моделирует безубыток**: после касания TP1 он продолжает
проверять исходный стоп, а не entry. Поэтому касание безубытка (entry) после TP1
в outcome-флагах **не хранится**. Для случая «TP1 достигнут, TP2 и исходный стоп
— нет» точный journal-R (безубыток vs плавающая позиция) невосстановим из флагов.
Честный компромисс — mark-to-market по `return_72h`. Это единственная ветка, где
`realized_r` не тождественен `journal.pnl_r`; она явно помечена (`kind =
mark_to_market`) и задокументирована.

## 5. reference_price vs journal entry_price

`reference_price` (`executable_price_at_decision`) — proxy входа на уровне
**анализа**, доступный для всей популяции прогнозов (включая недоставленные).
`journal.entry_price` — цена входа доставленной сделки. Поэтому популяция
`realized_r` шире journal-подмножества и не сравнивается с `journal.pnl_r`
«в лоб». Это by design: `realized_r` даёт clean-R взгляд на **все** прогнозы,
journal — на реально открытые сделки.

## 6. Почему это offline metric, а не production decision logic

`realized_r` — измерение постфактум (нужен закрытый 72h-outcome). Он ничего не
предсказывает и не участвует в scoring / gating / risk / Telegram. Модуль не
импортируется runtime-кодом (guard-тест), не пишет в БД, использует только
SELECT. Никакого нового lookahead: берётся то же 72h-окно, что уже измерил
`measure_outcome`; `unresolved → None` (censoring сохраняется).

## 7. Почему schema-колонка отложена (до Stage 11)

Формула сначала должна отстояться на реальной выборке. Пока `realized_r`
пересчитывается детерминированно оффлайн из существующих полей — колонка не
нужна и лишь повышает риск (правки `MIGRATIONS` / `_OUTCOME_COLS` /
`upsert_outcome`). Добавляем колонку только когда определение подтверждено.

---

## 8. Stage 11 checklist (persist `forecast_outcomes.realized_r`) — ВЫПОЛНЕНО

1. ✅ Формула зафиксирована; вынесена в leaf-модуль `analyzer/realized_r.py`
   (stdlib-only, без импортов database/tools/outcomes — единый источник).
2. ✅ `ALTER TABLE forecast_outcomes ADD COLUMN IF NOT EXISTS realized_r DOUBLE
   PRECISION;` (nullable, без default, без backfill) в `database.MIGRATIONS`
   + колонка в `SCHEMA`.
3. ✅ `realized_r` добавлен в `_OUTCOME_COLS`.
4. ✅ `analyzer/outcomes.measure_outcome` пишет `out["realized_r"]` через
   `analyzer.realized_r.realized_r` (raw float, без округления). Producer —
   только `outcome_tracking_job`, вне decision-path.
5. ✅ Толерантность к NULL: `upsert_outcome` строит колонки динамически;
   старые outcome-dict без ключа вставляются без ошибки.
6. ✅ Тесты: schema/migration/`_OUTCOME_COLS`, upsert персистит (memory),
   legacy-dict без `realized_r`, measure_outcome для ENTER/не-ENTER/unresolved,
   идемпотентность, guard'ы (tools не импортируется runtime; нет write-SQL).
7. ⏸ Отложено (follow-up): интеграция persisted `realized_r` в
   `forecast_metrics.py` и замена пункта «honest per-forecast R» в
   `MISSING_DATA` (тянет golden/ожидания) — не требуется для Stage 11.

Инварианты (держатся и в Stage 11): production behavior не меняется без
отдельного подтверждения; никаких destructive-миграций; никакого backfill;
Binance execution не подключается.
