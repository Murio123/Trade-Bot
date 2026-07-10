"""Stage C1.2: offline, source-pinned historical kline cache.

Read-only measurement/collection tool. НЕ торгует, НЕ отправляет ордера, НЕ
пишет в БД, НЕ меняет схему, scoring, thresholds, pipeline, scheduler или
backtest. НЕ импортируется runtime-кодом — только ручной CLI-запуск.

Зачем отдельный слой кэша (см. C1.1, drift points D14/D15):

  * D14 — вспомогательные фреймы в backtest берутся одним запросом limit=500,
    тогда как история entry-таймфрейма пагинируется. Кэш пагинирует ВСЁ, без
    12-страничного safety cap, поэтому глубина набора задаётся явно (--bars),
    а не límit'ом одного запроса.

  * D15 — Bybit не отдаёт taker_buy_base, Binance отдаёт. Failover-клиент
    может незаметно переключить источник посреди набора и смешать две разные
    CVD-семантики в одном датасете. Поэтому здесь биржа ПРИБИТА ГВОЗДЯМИ:
    клиент инстанцируется напрямую, никакого failover, никакого смешивания
    источников в одном файле. Наличие taker_buy_base фиксируется в манифесте
    (has_taker_buy_base), чтобы потребитель датасета знал, доступен ли CVD.

Датасет пишется в Parquet, если parquet-движок уже доступен в окружении;
иначе — в CSV. Новых зависимостей тул не вводит.

Рядом с датасетом пишется sidecar-манифест JSON: происхождение данных, границы
выборки, и ОТЧЁТ О ПРОПУСКАХ. Пропуски не глотаются молча: если спейсинг
open_time не совпадает с шагом таймфрейма, это видно в манифесте
(has_gaps/gap_count/missing_bars/gap_examples). Наличие пропусков не считается
фатальной ошибкой — датасет всё равно пишется, но помечен честно.

Запуск:

    .venv/bin/python -m tools.kline_cache \
      --exchange binance \
      --symbol BTCUSDT \
      --timeframe 15m \
      --bars 70000 \
      --outdir data/klines
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from analyzer.binance import BinanceClient
from analyzer.bybit import BybitClient

# Шаг таймфрейма в миллисекундах — эталон для проверки спейсинга open_time.
INTERVAL_MS: dict[str, int] = {
    "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
    "1d": 86_400_000, "1w": 604_800_000,
}

# Биржа пинуется явно. Никакого failover-клиента, никакого auto-режима.
EXCHANGES = ("binance", "bybit")

PAGE_LIMIT = 1000
PAGE_SLEEP_S = 0.2
GAP_EXAMPLES_MAX = 5


def create_client(exchange: str) -> Any:
    """Прямой публичный клиент одной биржи. Без ключей, без failover."""
    if exchange == "binance":
        return BinanceClient()
    if exchange == "bybit":
        return BybitClient()
    raise ValueError(f"unknown exchange: {exchange!r}")


def _to_ms(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("open_time must not be a bool")
    if isinstance(value, (int, float)):
        return int(value)
    return int(pd.Timestamp(value).timestamp() * 1000)


def _iso(open_time_ms: int) -> str:
    return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Сборка страниц
# ---------------------------------------------------------------------------

def assemble(frames: list[pd.DataFrame], bars: int) -> tuple[pd.DataFrame, int]:
    """Склеить страницы: дедуп по open_time, хронологический порядок, срез до N.

    Возвращает (df, duplicate_count_removed). Триммится ХВОСТ — то есть в
    датасете остаются самые СВЕЖИЕ bars свечей, а не самые старые.
    """
    if not frames:
        return pd.DataFrame(), 0
    full = pd.concat(frames, ignore_index=True)
    before = len(full)
    full = full.drop_duplicates("open_time")
    duplicates_removed = before - len(full)
    full = full.sort_values("open_time").reset_index(drop=True)
    if bars > 0 and len(full) > bars:
        full = full.iloc[-bars:].reset_index(drop=True)
    return full, duplicates_removed


async def fetch_history(client: Any, symbol: str, timeframe: str, bars: int,
                        page_limit: int = PAGE_LIMIT,
                        sleep_s: float = PAGE_SLEEP_S) -> tuple[pd.DataFrame, int]:
    """Пагинация назад по end_time до набора `bars` свечей.

    Никакого safety cap на число страниц: глубину задаёт вызывающий. Запросы
    строго последовательные, между страницами пауза ~sleep_s (вежливость к
    rate-limit, и одна биржа на весь набор — ротации источника нет).

    Терминирующие условия:
      * пустая страница        — старее данных нет;
      * короткая страница      — естественный конец истории;
      * набрано >= bars строк  — цель достигнута.
    """
    frames: list[pd.DataFrame] = []
    end_time: int | None = None
    collected = 0
    while collected < bars:
        df = await client.klines(timeframe, limit=page_limit, symbol=symbol,
                                 end_time=end_time)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        collected += len(df)
        if len(df) < page_limit:
            break  # короткая страница = дальше истории нет
        if collected >= bars:
            break
        oldest = df["open_time"].iloc[0]
        end_time = _to_ms(oldest) - 1
        await asyncio.sleep(sleep_s)
    return assemble(frames, bars)


# ---------------------------------------------------------------------------
# Отчёт о пропусках
# ---------------------------------------------------------------------------

def gap_report(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Сверить спейсинг open_time с шагом таймфрейма.

    gap_count    — сколько РАЗРЫВОВ (мест склейки с дырой);
    missing_bars — сколько свечей суммарно недостаёт.
    Пропуски только регистрируются; падать по ним по умолчанию нечестно —
    у биржи бывают реальные простои, и датасет с честной пометкой полезнее
    отсутствующего датасета.
    """
    expected = INTERVAL_MS.get(timeframe, 0)
    report: dict[str, Any] = {
        "expected_interval_ms": expected,
        "has_gaps": False,
        "gap_count": 0,
        "missing_bars": 0,
        "gap_examples": [],
    }
    if expected <= 0 or df is None or len(df) < 2:
        return report

    times = [_to_ms(v) for v in df["open_time"].tolist()]
    examples: list[dict[str, Any]] = []
    for prev, curr in zip(times, times[1:]):
        delta = curr - prev
        if delta <= expected:
            continue
        missing = delta // expected - 1
        if missing <= 0:
            continue
        report["gap_count"] += 1
        report["missing_bars"] += int(missing)
        if len(examples) < GAP_EXAMPLES_MAX:
            examples.append({
                "after_open_time": prev,
                "after_open_time_iso": _iso(prev),
                "before_open_time": curr,
                "before_open_time_iso": _iso(curr),
                "missing_bars": int(missing),
            })
    report["has_gaps"] = report["gap_count"] > 0
    report["gap_examples"] = examples
    return report


# ---------------------------------------------------------------------------
# Манифест и запись
# ---------------------------------------------------------------------------

def build_manifest(df: pd.DataFrame, *, exchange: str, symbol: str,
                   timeframe: str, requested_bars: int,
                   duplicate_count_removed: int, file_format: str,
                   data_file: str) -> dict[str, Any]:
    first_ms = _to_ms(df["open_time"].iloc[0]) if len(df) else None
    last_ms = _to_ms(df["open_time"].iloc[-1]) if len(df) else None
    gaps = gap_report(df, timeframe)
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_bars": requested_bars,
        "actual_bars": int(len(df)),
        "first_open_time": first_ms,
        "last_open_time": last_ms,
        "first_open_time_iso": _iso(first_ms) if first_ms is not None else None,
        "last_open_time_iso": _iso(last_ms) if last_ms is not None else None,
        # D15: Bybit не отдаёт taker_buy_base -> CVD по этому датасету недоступен.
        "has_taker_buy_base": "taker_buy_base" in df.columns,
        "file_format": file_format,
        "data_file": data_file,
        "expected_interval_ms": gaps["expected_interval_ms"],
        "has_gaps": gaps["has_gaps"],
        "gap_count": gaps["gap_count"],
        "missing_bars": gaps["missing_bars"],
        "gap_examples": gaps["gap_examples"],
        "duplicate_count_removed": int(duplicate_count_removed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # Датасет собран одним пинованным источником, без failover-ротации.
        "source_pinned": True,
    }


def parquet_available() -> bool:
    """Parquet — только если движок УЖЕ есть. Новых зависимостей не вводим."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


def dataset_stem(exchange: str, symbol: str, timeframe: str) -> str:
    """Единый источник правды об именовании файлов датасета.

    И писатель (write_dataset), и читатель (tools.kline_dataset) обязаны
    получать имя отсюда, иначе конвенция со временем разъедется.
    """
    return f"{exchange}_{symbol}_{timeframe}"


def write_dataset(df: pd.DataFrame, outdir: str, *, exchange: str, symbol: str,
                  timeframe: str, requested_bars: int,
                  duplicate_count_removed: int) -> tuple[str, str, dict[str, Any]]:
    import os

    os.makedirs(outdir, exist_ok=True)
    stem = dataset_stem(exchange, symbol, timeframe)
    use_parquet = parquet_available()
    file_format = "parquet" if use_parquet else "csv"
    data_file = f"{stem}.{file_format}"
    data_path = os.path.join(outdir, data_file)
    manifest_path = os.path.join(outdir, f"{stem}.manifest.json")

    manifest = build_manifest(
        df, exchange=exchange, symbol=symbol, timeframe=timeframe,
        requested_bars=requested_bars,
        duplicate_count_removed=duplicate_count_removed,
        file_format=file_format, data_file=data_file,
    )

    if use_parquet:
        df.to_parquet(data_path, index=False)
    else:
        df.to_csv(data_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)
    return data_path, manifest_path, manifest


def format_report(manifest: dict[str, Any], data_path: str,
                  manifest_path: str) -> str:
    lines = [
        f"source-pinned kline cache — {manifest['exchange']} "
        f"{manifest['symbol']} {manifest['timeframe']}",
        f"  bars           : {manifest['actual_bars']} "
        f"(requested {manifest['requested_bars']})",
        f"  range          : {manifest['first_open_time_iso']} .. "
        f"{manifest['last_open_time_iso']}",
        f"  taker_buy_base : {manifest['has_taker_buy_base']}"
        + ("" if manifest["has_taker_buy_base"] else "  (CVD unavailable)"),
        f"  duplicates     : {manifest['duplicate_count_removed']} removed",
    ]
    if manifest["has_gaps"]:
        lines.append(f"  GAPS           : {manifest['gap_count']} gap(s), "
                     f"{manifest['missing_bars']} bar(s) missing")
        for ex in manifest["gap_examples"]:
            lines.append(f"     after {ex['after_open_time_iso']} -> "
                         f"{ex['before_open_time_iso']} "
                         f"({ex['missing_bars']} missing)")
    else:
        lines.append("  gaps           : none")
    lines.append(f"  data           : {data_path}")
    lines.append(f"  manifest       : {manifest_path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    client = create_client(args.exchange)
    try:
        df, duplicates = await fetch_history(
            client, args.symbol, args.timeframe, args.bars)
    finally:
        await client.close()
    data_path, manifest_path, manifest = write_dataset(
        df, args.outdir, exchange=args.exchange, symbol=args.symbol,
        timeframe=args.timeframe, requested_bars=args.bars,
        duplicate_count_removed=duplicates,
    )
    return manifest, data_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline source-pinned historical kline cache (Stage C1.2).")
    # Биржа ОБЯЗАТЕЛЬНА и без default: молчаливый выбор источника — это ровно
    # тот класс ошибок (D15), который тул должен исключать.
    parser.add_argument("--exchange", choices=EXCHANGES, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", required=True, choices=sorted(INTERVAL_MS))
    parser.add_argument("--bars", type=int, required=True)
    parser.add_argument("--outdir", default="data/klines")
    parser.add_argument("--json", action="store_true",
                        help="вывести манифест как JSON")
    args = parser.parse_args(argv)

    if args.bars <= 0:
        parser.error("--bars must be positive")

    manifest, data_path, manifest_path = asyncio.run(_run(args))
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(manifest, data_path, manifest_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
