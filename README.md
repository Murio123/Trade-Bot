# BTC Signal Bot 🤖

AI-Telegram-бот для торговых сигналов BTC/USDT (фьючерсы). Два независимых
потока сигналов (свинг и интрадей), Smart-Money анализ по зонам старших
таймфреймов, 7-уровневый каскад шумоподавления, ведение сделок по этапам и
честный бэктест с учётом комиссий.

> ⚠️ Не является финансовой рекомендацией. Используйте на свой риск.

## Два потока сигналов

| | 📊 Свинг | ⚡ Интрадей |
|---|---|---|
| Вход | 1H | 15m |
| Тренд (HTF-фильтр) | 1D | 1H |
| Зоны (OB/FVG/цели) | 12H + 4H | 4H + 1H |
| Согласие ТФ | 1H/4H/12H/1D | 15m/1H/4H |
| Cooldown | 8 ч | 2 ч |
| Анализ | каждый час | каждые 15 мин |

Параметры профилей: `signal_engine/profiles.py` (цели/стоп настраиваются через
`SWING_TARGETS`, `INTRADAY_TARGETS`, `*_ATR_MULT`).

## Каскад шумоподавления (7 уровней)

1. **HTF-фильтр** — сигнал против старшего тренда блокируется полностью.
2. **Confluence Score** — очки в 5 категорий (тренд/импульс/объём/структура/макро);
   Smart-Money факторы весят больше: HTF Order Block +3, снятие ликвидности +3,
   FVG +2, premium/discount +2, разворот на экстремуме +2/+3.
3. **Разнообразие** — минимум 3 разные категории (`MIN_DIVERSE_CATEGORIES`).
4. **Конфликты** — при вероятном свипе ликвидности вход откладывается.
5. **Согласие ТФ** — направленный множитель уверенности (4/4 → ×1.2, против → ×0.7).
6. **Cooldown/дедуп** — по каждому потоку отдельно.
7. **Дневной лимит** — жёсткий cap `MAX_SIGNALS_PER_DAY` (по таймфрейму, скользящие 24 часа); сигналы сверх лимита сохраняются в журнал (/signal), но не отправляются.

Классификация: score ≥8 → авто-уведомление; 5-7 → журнал (`/signal`); <5 → игнор.

## Команды

| Команда | Действие |
|---|---|
| `/signal`, `/intraday` | Свинг / интрадей сигнал по запросу |
| `/deep` | Институциональный разбор 1D/12H/4H со взвешенным score /100 |
| `/reversal` | Дно/пик: истощение тренда сразу по 1H/4H/12H/1D (≥2 ТФ = подтверждение) |
| `/levels` | Ближайшие S/R, OB/FVG с HTF, Volume Profile, ликвидации, EMA, вывод |
| `/funding`, `/fear` | Деривативы и индекс страха/жадности |
| `/journal` | Реальный винрейт и R по закрытым сделкам |
| `/backtest [swing\|intraday]` | Перебор порогов на истории, NET (с комиссиями), проекция %/мес |
| `/status`, `/testalert` | Здоровье бота и проверка канала уведомлений |
| `/guide` | Встроенный гид по всем функциям |
| `/ask <вопрос>` | Вопрос к Claude с рыночным контекстом (любой текст = /ask) |

Интерфейс — кнопки (reply-клавиатура + inline-меню), появляются после `/start`.

## Ведение сделок

Каждый отправленный сигнал ведётся как позиция:
`TP1 достигнут → стоп в безубыток → TP2 (win) / возврат к БУ (0R) / стоп (−1R)`,
с уведомлением на каждом этапе. Исходы проверяются на **родном таймфрейме**
сделки каждые 15 минут; статистика — в `/journal`.

## Архитектура

```
main.py                 точка входа: Telegram polling + APScheduler
pipeline.py             сбор контекста (HTF-зоны, равновесие, развороты) + каскад
scheduler.py            свинг/интрадей анализ, resolve сделок, ценовые алерты
backtest.py             бэктест живых профилей: NET результат, порог, проекция
config.py               конфигурация из env (см. .env.example)
database.py             PostgreSQL (asyncpg) + in-memory fallback, миграции

analyzer/               binance, bybit, exchange (auto-failover 451→запасная),
                        indicators (EMA/RSI/MACD/BB/ATR/TSI/VWAP/StochRSI),
                        order_blocks, fvg, liquidity, volume_profile,
                        equilibrium (premium/discount), reversal (6 факторов),
                        divergence, cvd, structure (BOS/CHoCH), volatility,
                        historical (аналоги), correlation, macro, onchain, news

signal_engine/          htf_filter, confluence, conflict_resolver,
                        mtf_confidence, cooldown, daily_limiter,
                        profiles (свинг/интрадей), quality_score (/100 для /deep)

bot/                    handlers (+гейткипер доступа), keyboards, formatting,
                        alerts (DRY_RUN-aware), journal (лайфцикл), guide
ai/                     claude (сигналы, /ask), swing_analysis (/deep нарратив)
tests/                  pytest: каскад, лайфцикл сделок, анализаторы
```

## Деплой (Railway)

1. Подключить репозиторий + плагин **PostgreSQL** (`DATABASE_URL` через
   `${{Postgres.DATABASE_URL}}`).
2. Обязательные переменные: `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`,
   `TELEGRAM_ALERT_CHAT_IDS` (chat_id получателя).
3. Безопасность: `TELEGRAM_ALLOWED_CHAT_IDS` — кто вообще может писать боту
   (по умолчанию = получатели алертов; чужие отклоняются).
4. Старт в **`DRY_RUN=true`**: анализ идёт, уведомления не шлются. Для
   manual-only Telegram-сигналов без авто-сделок можно включить
   `SEND_DRY_RUN_ALERTS=true`. Проверить `/status` и `/testalert`, затем
   `DRY_RUN=false` для LIVE-уведомлений.
5. Источник данных: `EXCHANGE=auto` (Bybit↔Binance c автофейловером при 451).

Полный список переменных — в `.env.example`. Все внешние источники, кроме
биржи/Telegram/Anthropic, опциональны (graceful degradation).

## Тесты

```bash
pip install pytest && python -m pytest tests/ -q
```
