# Stage 8 — Data Quality & Missing Context

> **Stage 9 update (реализовано, Option B):** в `forecasts` добавлены 4 additive
> nullable-поля — `market_regime`, `volatility_regime`, `strategy_version`,
> `context_version`. Заполняются в `build_forecast_record` из уже существующих
> `result` / `ctx["volatility"]["regime"]` / `config.{STRATEGY,CONTEXT}_VERSION`;
> старые строки остаются NULL; `tools/forecast_metrics.py` получил разрезы
> `by_market_regime` / `by_volatility_regime` (NULL → `"unknown"`). Отложены:
> `context_quality_score`, `data_quality_flags`, sidecar `forecast_context`
> (см. критерии ниже).
>
> **Stage 11 update (реализовано, Option A без backfill):** добавлена
> additive-nullable колонка `forecast_outcomes.realized_r DOUBLE PRECISION`;
> формула — leaf-модуль `analyzer/realized_r.py` (единый источник для producer
> `analyzer/outcomes.py` и offline `tools/realized_r.py`). Backfill не делается,
> старые resolved-строки остаются NULL. См. `docs/realized_r_stage10.md` §8.

**Тип этапа:** аудит + фиксация плана (Option B). Read-only.

**Что Stage 8 НЕ делает:** не меняет схему БД, не добавляет миграций, не меняет
production-поведение, не трогает `database.py` / `scheduler.py` / `pipeline.py` /
`backtest.py` / `config.py` / `bot/` / `ai/` / `risk/` / `signal_engine/` /
`contracts/`, не подключает Binance execution, не меняет Telegram output, AI-модель
или prompts.

**Что Stage 8 делает:** фиксирует карту пробелов данных и добавляет read-only
offline-аудитор (`tools/data_quality_audit.py`), который измеряет реальную
заполненность уже персистируемых данных и классифицирует запланированные Stage 9
поля. Схема остаётся неизменной; любое поле добавляется только в Stage 9,
additive-nullable.

Документ — единый источник правды для будущего расширения схемы. Он парный к
инвентарю `MISSING_DATA` в `tools/forecast_metrics.py` (Stage 6) и оценкам в
`tools/strategy_report.py` (Stage 7).

---

## 1. Текущее состояние сбора данных

Пишущий путь (production, **в Stage 8 не меняется**):
`scheduler → pipeline.run_cascade → signal_engine/forecast_record.build_forecast_record
→ database.insert_forecast`, а также outcome-трекинг →
`database.upsert_outcome`, и журнал сделок → `database.insert_trade` /
`database.close_trade`.

Ключевой факт архитектуры: `insert_forecast` строит колонки динамически —
`cols = [c for c in _FORECAST_COLS if c in fc]`. Отсутствующие ключи молча
пропускаются, читатели используют `.get()` и толерантны к NULL. Поэтому будущее
additive-nullable расширение (Stage 9) **не ломает** ни запись, ни чтение.

### Уже сохраняется

**`forecasts`** — append-only ledger, одна строка на прогон (включая
blocked / NO_TRADE). UNIQUE `(symbol, analysis_type, signal_candle_close_time)`:

`symbol, analysis_type, timeframe, signal_candle_close_time, decision_time,
data_freshness_seconds, decision_latency_seconds, candidate_direction,
final_bias, analysis_status, blocked_gate, long_score, short_score,
raw_confidence, calibrated_confidence (всегда NULL), expected_move_points,
expected_move_percent, expected_move_atr, entry_zone, signal_close_price,
executable_price_at_decision, stop_loss, take_profit_levels, tp2_source,
risk_reward, no_trade_reasons, prompt_version, model_version, signal_id`.

**`forecast_outcomes`** — измерения на прогноз; right-censoring явный
(нерезолвнутые не удаляются):

`anchor_time, reference_price, return_{1h,4h,12h,24h,72h}, mfe_points,
mae_points, reached_{500,1500,3000}, tp1_hit, tp2_hit, stop_hit,
net_after_costs, resolved`.

**`signals`** — только journal+ записи (не blocked):

`entry_price, stop_loss, target_1, target_2, position_size, score, confidence,
category_scores, reasons, ai_text, delivered, analysis_type`.

**`trades_journal`** — единственный источник чистого **R**:

`entry_price, exit_price, stop_loss, tp1, tp2, stage, current_stop, outcome
(win|loss|breakeven), pnl_r, notes, timeframe, symbol, analysis_type`.

## 2. Какие метрики уже считаются

Read-only offline инструменты (SELECT-only, не импортируются runtime-кодом):

- **`tools/forecast_metrics.py`** (Stage 6): core hit/direction-accuracy/
  precision, MFE/MAE, journal clean-R equity/drawdown/profit-factor/expectancy,
  бакеты по `analysis_type / timeframe / direction / status / confidence /
  score / freshness / weekday-hour / blocked_gate`, raw-Brier calibration,
  execution realism (slippage, freshness/latency impact, net_after_costs).
- **`tools/strategy_report.py`** (Stage 7): интерпретация поверх метрик —
  warnings/insights/recommendations, всё под sample-size гейтами; сильные
  выводы подавляются на малой выборке.

Единственное поле из числа персистируемых, которое пока не даёт метрик, —
`calibrated_confidence` (колонка есть, всегда NULL).

---

## 3. Missing context inventory (Stage 6/7, авторитетно)

| # | Missing | Причина | Есть в runtime? |
|---|---|---|---|
| 1 | **market_regime** | не персистится в `forecasts` | ✅ `signal_engine/regime.py::detect_regime`, `contracts.TechnicalContext.market_regime` — нужна только протяжка |
| 2 | **volatility_regime** | не персистится (только `expected_move_atr`) | ✅ `analyzer/volatility.py` → expansion/compression/normal; `contracts.base.VolatilityRegime` |
| 3 | **calibrated_confidence** | колонка есть, всегда NULL | ⚠️ нужна калибровочная стадия, не только протяжка |
| 4 | **TP3** | нигде не хранится | ❌ ни одна стратегия не эмитит третий тейк |
| 5 | **honest per-forecast R** | outcomes хранят %-за-горизонт, не clean R | ⚠️ выводимо из stop/entry/tp, но не считается/не хранится |
| 6 | **full live-execution PnL** | реальные ордера не подключены; PnL модельный | ❌ Binance execution сознательно не подключён |

---

## 4. Critical fields для Stage 9 (высший приоритет)

Данные уже считаются в runtime — не хватает только протяжки/версионирования:

- **`market_regime`** (`forecasts`) — самый ценный недостающий разрез:
  позволяет условную по режиму оценку эджа (работает ли сигнал только в
  экспансии / в тренде). Нулевые новые вычисления.
- **`volatility_regime`** (`forecasts`) — то же для волатильности.
- **`strategy_version`** (`forecasts`) — без версии стратегии нельзя честно
  атрибутировать изменения исходов изменениям стратегии во времени.
- **`context_version`** (`forecasts`) — парная к `strategy_version`; версия
  набора контекста, для честного A/B по стадиям.
- **`realized_r`** (`forecast_outcomes`) — честный per-forecast R, чтобы
  clean-R анализ покрывал всю популяцию прогнозов, а не только journal-подмножество.

## 5. Nice-to-have (реально, но вторично)

- **`context_quality_score`**, **`data_quality_flags`** (`forecasts`) — агрегат
  и per-block `BlockMeta.degraded` (источник уже есть); фильтрация degraded-прогонов.
- **`funding_regime`, `oi_regime`, `correlation_regime`, `liquidity_regime`,
  `macro_regime`** (sidecar `forecast_context`) — сырьё частично собирается в
  `unified_context` (OI hist, correlations, macro), но режим-метки не выводятся.
- **`setup_id` / `scenario_id`** — `contracts.Scenario.setup_type` — эвристика;
  стабильный id требует дизайна.
- **`gate_trace` summary** — `blocked_gate` (один гейт) уже пишется; полный
  упорядоченный trace — JSONB в sidecar.

## 6. Dangerous / redundant (НЕ добавлять)

- **`confidence_bucket`** — чистая производная `raw_confidence`; уже считается
  offline (`bucket_by(... CONFIDENCE_EDGES)`). Хранение раздувает строку и несёт
  риск drift определения границ. Считать offline, не персистить.
- **`TP3` до того, как стратегия его эмитит** — схема под несуществующий выход.
- **stored `expected_r`, если выводимо** — выводимо из
  `stop_loss`/`take_profit_levels`/entry; отдельная колонка избыточна.
- **отдельная model/prompt version table** — `prompt_version`/`model_version`
  уже per-row в `forecasts`; отдельная таблица преждевременна.

---

## 7. Рекомендованный Stage 9 schema plan

Правила (все обязательны):

- **additive nullable columns only** — `ADD COLUMN IF NOT EXISTS`, все NULLable,
  без non-null defaults (нет rewrite таблицы). Sidecar — `IF NOT EXISTS`.
- **no destructive migration** — никаких rename / drop / type-change.
- **no backfill required** — исторические строки остаются NULL (честно, не
  выдумывается); в аналитике падают в бакет `unknown`.
- **code must tolerate NULL** — читатели уже используют `.get()` + None-guards.
- **tools must show missing/unknown for NULL** — уже установленный паттерн
  (`MISSING_DATA` секция, `unknown`-бакеты, этот аудитор).

Размещение (item 9–12 из аудита): скаляры режима/версии/качества → «горячая»
`forecasts`; `realized_r` → `forecast_outcomes`; высококардинальный кластер
режимов + `gate_trace` + JSONB-блобы → sidecar-таблица `forecast_context`
(ключ = `forecast_id`), чтобы не раздувать горячую строку.

Предлагаемое (Stage 9, не сейчас):

```
-- forecasts (скаляры)
market_regime          TEXT
volatility_regime      TEXT
strategy_version       TEXT
context_version        TEXT
context_quality_score  DOUBLE PRECISION
data_quality_flags     JSONB

-- forecast_outcomes
realized_r             DOUBLE PRECISION

-- forecast_context (sidecar, forecast_id PK -> forecasts.id)
funding_regime, oi_regime, correlation_regime, liquidity_regime, macro_regime  TEXT
gate_trace, regime_snapshot                                                    JSONB
```

---

## 8. Stage 9 checklist

Перед Stage 9:

1. Зафиксировать семантику `strategy_version` / `context_version` (константы в
   `config`, правило инкремента).
2. Зафиксировать точное определение `realized_r`.
3. Утвердить, какие поля идут в `forecasts`, какие в `forecast_context`.

В Stage 9 (по порядку, каждое проверяемо):

4. Добавить `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (nullable) в
   `database.MIGRATIONS`; sidecar через `... IF NOT EXISTS`.
5. Расширить `_FORECAST_COLS` / `_OUTCOME_COLS` новыми колонками.
6. Наполнить значения в `signal_engine/forecast_record.build_forecast_record`
   из уже присутствующих `result` / `ctx` (regime, quality, versions).
7. Убедиться, что запись остаётся толерантной (ключ отсутствует → колонка NULL).
8. Расширить `tools/forecast_metrics.py` бакетами по режиму (активны только
   когда колонка не-NULL; иначе `unknown`).
9. Тесты: идемпотентность миграции, insert с/без новых полей, читатель
   толерантен к NULL, `realized_r` агрегация, `forecast_context` join.
10. Прогнать этот аудитор до/после — coverage новых полей растёт с NULL к >0,
    статусы уходят из `available_runtime_not_persisted` в `present`.

Инварианты (должны держаться и в Stage 9): production behavior не меняется без
отдельного подтверждения; никаких destructive-миграций; никакого backfill;
Binance execution не подключается.

---

## 9. Stage 8 deliverables (этот этап)

- `docs/data_quality_stage8.md` — этот документ.
- `tools/data_quality_audit.py` — offline read-only аудитор: per-table field
  coverage + классификация планируемых полей (`present` /
  `available_runtime_not_persisted` / `derived_offline` / `not_available_yet` /
  `missing`). Работает из `--input JSON` или `--database-url` (SELECT-only).
  Не импортируется runtime-кодом, не пишет в БД, не выполняет миграций.
- `tests/test_data_quality_audit.py` — покрытие: coverage-математика, NULL,
  detection present/missing полей, JSON-путь, пустой датасет, read-only guard
  (нет пишущих SQL-глаголов), не импортируется production, golden не тронуты,
  CLI exit 0.

Запуск аудита:

```
.venv/bin/python -m tools.data_quality_audit --input tests/fixtures/forecast_metrics_sample.json
```
