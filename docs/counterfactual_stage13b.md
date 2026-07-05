# Offline Counterfactual Outcomes (Stage 13B)

Read-only offline-аналитика: «а что было бы, если бы бот вошёл?» — для решений,
где входа **не было** (WAIT / NO_TRADE / blocked / journal-only).

## Зачем

Trading Journal v2 (Stage 13) классифицирует ENTER и те WAIT, у которых уже
измерен runtime-исход. Но для **NO_TRADE / blocked** исход не меряется вовсе
(`outcome_tracking_job` их исключает). Из-за этого не видно:

- какие NO_TRADE были **avoided_loss** (правильно не вошли — сработал бы стоп);
- какие NO_TRADE были **missed_opportunity** (зря не вошли — отработал бы TP);
- какие blocked-сигналы спасли от убытка, а какие гейты слишком строгие.

Counterfactual считает гипотетический исход постфактум из исторических свечей и
раскладывает не-ENTER решения по этим категориям, с разбивкой по
`blocked_gate` / `analysis_type` / режимам.

## Компоненты

| Файл | Роль |
|---|---|
| `analyzer/counterfactual.py` | Чистое ядро (Step 1). Reuse `analyzer.outcomes.measure_outcome` — та же дефиниция TP/stop/MFE/MAE и тот же 72h-горизонт. |
| `tools/counterfactual_report.py` | Read-only offline CLI (Step 2): читает forecasts (JSON/JSONL) + klines (JSON/CSV), агрегирует отчёт. |

## Почему offline-only и без записи в БД

- **Offline:** klines берутся только из локального файла (`--klines`); CLI не
  ходит в сеть, не подключается к Binance, не читает/не пишет продакшн-БД. Это
  делает прогон детерминированным, воспроизводимым в CI и безопасным.
- **Без записи в БД (пока):** Stage 13B — чистая наблюдаемость. Персистентность
  counterfactual-исходов (колонки/таблица) — отдельный будущий stage, требующий
  additive-nullable миграции и отдельного approval. Здесь ничего не пишется, не
  мигрируется, не бэкфиллится.

## Как это помогает

Для каждого не-ENTER прогноза с направлением, уровнями (стоп + TP1) и доступным
окном свечей CLI определяет:

- **avoided_loss** — гипотетически сработал бы чистый стоп → не войти было верно;
- **missed_opportunity** — гипотетически отработал бы TP → движение упущено;
- **none** — resolved, но ни чистого стопа, ни чистого профита.

Разбивка **по `blocked_gate`** прямо показывает, какие гейты чаще всего режут
`missed_opportunity` (кандидаты на пересмотр) против `avoided_loss` (работают).

## Как классифицируется гипотетический исход

Метка avoided_loss / missed_opportunity выводится из тех же хитов и того же
критерия «сетап отработал», что и Trading Journal v2
(`analyzer.journal_classify`), поэтому результат стыкуется с журналом. Формула
`realized_r` **не меняется** и для не-ENTER остаётся `None`.

## Ограничения (важно, читать честно)

- Работает **только** когда есть `candidate_direction`, `stop_loss` + `TP1` и
  свечи после `decision_time`. Иначе — честные `no_direction` / `no_levels` /
  `unresolved`; пропуск данных **не** превращается в win/loss.
- Ранние блокировки (`direction_conflict`, `htf_filter`, `dead_zone` и т.п.) не
  несут уровней → `no_levels`; для них counterfactual в принципе невозможен.
- **Не доказывает прибыльность:** это гипотетика на прошлых свечах, без
  проскальзывания входа, частичных заполнений и реального исполнения.
- **Не меняет сигналы, scoring, gating, confidence, risk, Telegram или live
  trading.** `DRY_RUN` не затрагивается. Это только измерение и отчёт.
- Censoring: пока не прошло 72h от решения — строка `unresolved`, не успех/провал.

## Форматы входа

**Forecasts** (`--input`): JSON (`{"forecasts": [...]}` или голый список) либо
JSONL (по объекту в строке). Поля: `analysis_status`, `candidate_direction`,
`analysis_type`, `decision_time`, `reference_price`/`entry_price`/
`signal_close_price`, `stop_loss`, `take_profit_levels`, `blocked_gate`,
`market_regime`, `volatility_regime`.

**Klines** (`--klines`): JSON-список или CSV с колонками `open_time`/`timestamp`,
`open`, `high`, `low`, `close`. `close_time` достраивается из шага `open_time`.
`--now` (опц.) задаёт cutoff; по умолчанию — время последней свечи.

## Пример запуска

```bash
# forecasts из JSONL + klines из CSV, текстовый отчёт
python -m tools.counterfactual_report --input forecasts.jsonl --klines klines.csv

# JSON-вход, машиночитаемый вывод
python -m tools.counterfactual_report --input forecasts.json --klines klines.json --json
```
