# Stage 12 — Bollinger / Volatility Context

**Тип этапа:** context-only, Option A (pure leaf-модуль). Read-only, без БД, без
decision-path, без миграций.

**Что Stage 12 делает:** добавляет `analyzer/bollinger_context.py` —
детерминированную чистую функцию, считающую Bollinger как **контекст
волатильности/режима** для будущей аналитики (journal, deep analysis, strategy
report, возможный persist в отдельной стадии).

**Что Stage 12 НЕ делает / не трогает:** `analyzer/indicators.py`,
`signal_engine/`, `pipeline.py`, `scheduler.py`, `database.py`, `contracts/`,
`bot/formatting.py`, `tools/forecast_metrics.py`, `tools/strategy_report.py`,
`tests/golden/`; не меняет scoring, thresholds, final_gate, run_cascade,
Telegram, Railway/env; не добавляет миграций и не персистит. CoinGlass — вне
Stage 12 (будущий 12B).

---

## 1. Почему отдельный модуль, а не расширение существующего

Bollinger в репозитории уже разделён на два слоя, и оба трогать нельзя:

- **Decision-path** (`analyzer/indicators.py`): `bb_upper/lower/mid/bbw/bbw_avg`,
  `bb_squeeze/bb_expansion`, `bb_breakout_up/down` — потребляются
  `signal_engine/confluence.py`, `analyzer/reversal.py`, `bot/formatting.py`
  (scoring + Telegram).
- **Shadow-контекст** (`unified_context.py::stage2_volatility`):
  `compression_score` (= BBW-перцентиль) в `volatility_stage2`; попадает в
  `contracts.VolatilityContext` и в golden `unified_swing.json`.

Чтобы не задеть ни decision-path, ни golden, Stage 12 — **самостоятельный
leaf-модуль**: он ничего из вышеперечисленного не импортирует и не изменяет.

## 2. Источник данных

Klines DataFrame с колонкой `close` (обычно 4H entry-фрейм). Модуль использует
только `close`; ширина/перцентиль/%b считаются из скользящих статистик закрытий.

## 3. Формулы и поля

Параметры: `LENGTH=20`, `STD=2.0` (классические полосы).

```
mid   = SMA(close, 20)
sd    = STD(close, 20, ddof=0)
upper = mid + 2*sd
lower = mid - 2*sd
width = (upper - lower) / mid
%b    = (close - lower) / (upper - lower)
```

| Поле | Смысл |
|---|---|
| `bb_middle`, `bb_upper`, `bb_lower` | полосы последней свечи |
| `bb_width` | нормированная ширина `(upper-lower)/mid` |
| `bb_width_percentile` | перцентиль текущей ширины в её собственной истории (0..100) |
| `bb_percent_b` | положение цены относительно полос (внутри ~[0,1], вне — за пределами) |
| `bb_squeeze` | `width_percentile <= 20` — сжатие/coiling |
| `bb_expansion` | `width_percentile >= 80` — расширение |
| `close_outside_upper` / `close_outside_lower` | закрытие за полосой (`%b>1` / `%b<0`) |
| `band_walk_direction` | `up`/`down`/`None` — последние 3 `%b` подряд у одной границы (`>=0.8` / `<=0.2`): тренд «идёт по полосе» |
| `mean_reversion_risk` | `high`/`medium`/`low`/`None` (см. §4) |
| `regime_clue` | строка-подсказка режима волатильности (context-only) |

`regime_clue` ∈ `{squeeze_coiling, expansion_trend_walk,
expansion_breakout_risk, trend_band_walk, range_mean_reversion,
neutral_range}` — **подсказка**, не сигнал.

## 4. band-walk vs mean-reversion (почему не наивно)

Пробой полосы сам по себе НЕ означает разворот:

- **band-walk** (тренд у полосы): `%b` держится у границы несколько баров →
  сильный тренд, растяжение может продолжаться → `mean_reversion_risk = medium`.
- **растяжение без band-walk** (одиночный вынос за полосу): `%b` вне полосы, но
  тренда вдоль границы нет → повышенный риск возврата → `mean_reversion_risk =
  high`.
- `%b` близко к границе (`>=0.9`/`<=0.1`) без band-walk → `medium`.
- иначе → `low`.

Так Bollinger трактуется как контекст (squeeze/expansion/band-walk/reversion
risk), а НЕ как «верхняя полоса = SHORT, нижняя = LONG».

## 5. Инварианты (context-only)

- НЕ эмитит `LONG/SHORT/NEUTRAL`, `ENTER/WAIT/NO_TRADE`.
- НЕ влияет на `confidence`, `realized_r`, scoring, gating, Telegram.
- Детерминированный; graceful `None` при `len(close) < 25` или схлопнутых полосах.
- Не импортирует `database / tools / signal_engine / bot / ai / risk /
  contracts / pipeline / scheduler` (guard-тест).

## 6. Тесты (`tests/test_bollinger_context.py`)

Короткий df → все None; `%b` внутри/выше/ниже полос; squeeze/expansion по
перцентилю; band-walk up/down; `mean_reversion_risk=high` при выносе за полосу
без band-walk; отсутствие торгового направления/статуса; детерминизм;
отсутствие запрещённых импортов.

## 7. Будущие стадии (не сейчас)

- **12B** — CoinGlass / деривативный контекст (отдельно).
- **persist** (следующая стадия по образцу Stage 11): additive nullable
  колонка/JSONB `bollinger_context`, продюсер в outcome/forecast, бакеты в
  `forecast_metrics` — только после стабилизации определений, с миграцией
  `ADD COLUMN IF NOT EXISTS`, без backfill.
- **strategy_report** — интерпретация bb-режимов поверх метрик (после persist).
