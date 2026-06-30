# BTC Signal Bot 🤖

AI-powered Telegram бот для торговых сигналов BTC/USDT (фьючерсы). Анализирует
рынок на нескольких таймфреймах, прогоняет каждый потенциальный сигнал через
семиуровневый каскад шумоподавления и формулирует итоговый сигнал текстом через
Claude.

> ⚠️ Не является финансовой рекомендацией. Используйте на свой риск.

## Архитектура

```
main.py            — точка входа: Telegram polling + APScheduler
pipeline.py        — сбор рыночного контекста + запуск каскада
scheduler.py       — анализ каждые 4ч, проверка ценовых алертов
config.py          — конфигурация из переменных окружения
database.py        — PostgreSQL (asyncpg) + in-memory fallback
backtest.py        — упрощённый бэктест стратегии

analyzer/          — источники данных и индикаторы
  binance.py         цена, klines, funding, OI, long/short ratio, aggTrades
  indicators.py      EMA / RSI / MACD / Bollinger+BBW / ATR
  divergence.py      авто-детект RSI/MACD дивергенций
  order_blocks.py    Order Blocks + Break-of-Structure
  liquidity.py       equal highs/lows + sweep detection
  volume_profile.py  POC / VAH / VAL
  liquidation_map.py Coinglass / OI-based зоны ликвидаций
  cvd.py             Cumulative Volume Delta из aggTrades
  macro.py           DXY / US10Y корреляция (FRED или Stooq)
  session_stats.py   волатильность по сессиям (Азия/Лондон/Нью-Йорк)
  onchain.py         Glassnode (netflow, MVRV, SOPR, whales)
  news.py            NewsAPI sentiment + Fear & Greed

signal_engine/     — каскад генерации сигнала (уровни 1-7)
  htf_filter.py      L1: HTF bias filter (блокирующий)
  confluence.py      L2-3: confluence scoring + категориальное разнообразие
  conflict_resolver.py L4: разрешение конфликтов
  mtf_confidence.py  L5: мультитаймфреймовое согласие
  cooldown.py        L6: cooldown + дедупликация
  daily_limiter.py   L7: дневной лимит сигналов

risk/position_sizing.py — ATR-based sizing
ai/claude.py            — интерпретация сигнала + /ask чат-режим
bot/                    — Telegram интерфейс
  handlers.py            команды
  alerts.py              отправка уведомлений (учитывает DRY_RUN)
  journal.py             журнал сделок + статистика
  formatting.py          рендер сообщений
```

## Каскад генерации сигнала

| Уровень | Модуль | Тип |
|---|---|---|
| 1. HTF Bias Filter | `htf_filter` | блокирующий |
| 2. Confluence Scoring | `confluence` | скоринг |
| 3. Категориальное разнообразие (≥3) | `confluence` | блокирующий |
| 4. Conflict Resolution | `conflict_resolver` | блокирующий |
| 5. MTF Confidence | `mtf_confidence` | модификатор |
| 6. Cooldown / дедуп | `cooldown` | блокирующий |
| 7. Дневной лимит | `daily_limiter` | блокирующий |

Итоговая классификация по score: **8-10** → Telegram уведомление; **5-7** →
запись в БД (доступно по `/signal`); **<5** → игнорируется.

## Команды

| Команда | Действие |
|---|---|
| `/signal` | Текущий сигнал (включая слабые 5-7) |
| `/levels` | Order Blocks, ликвидность, volume profile |
| `/funding` | Funding rate + историческая аномальность |
| `/fear` | Индекс страха/жадности |
| `/backtest` | Результаты стратегии за период |
| `/journal` | Винрейт и средний R/R |
| `/ask <вопрос>` | Свободный вопрос к Claude с рыночным контекстом |

Любое текстовое сообщение без команды обрабатывается как `/ask`.

## Деплой на Railway

1. Создайте проект и подключите этот репозиторий.
2. Добавьте плагин **PostgreSQL** — Railway проставит `DATABASE_URL`.
3. Заполните переменные окружения (см. `.env.example`). Обязательны:
   `TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`, `DATABASE_URL`.
4. `Procfile` уже задаёт процесс: `worker: python main.py`.
5. **Dry-run**: оставьте `DRY_RUN=true` для теста на реальных данных без
   отправки уведомлений. Убедитесь, что анализ идёт (логи), затем поставьте
   `DRY_RUN=false` и задайте `TELEGRAM_ALERT_CHAT_IDS`.

## Локальный запуск

```bash
pip install -r requirements.txt
cp .env.example .env   # заполните ключи
export $(grep -v '^#' .env | xargs)
python main.py
```

Без `DATABASE_URL` бот использует in-memory хранилище; без опциональных
ключей соответствующие модули просто не дают очков (graceful degradation).

## Graceful degradation

Каждый внешний источник опционален. Если ключ не задан или API недоступно —
модуль возвращает нейтральный результат, а движок продолжает работу на
доступных данных. Обязателен только Binance (публичные эндпоинты), Telegram и
Anthropic (для текста сигнала; есть детерминированный fallback).
