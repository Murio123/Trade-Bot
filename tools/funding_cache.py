"""P1: offline, source-pinned historical funding-settlement cache.

Read-only collection tool. НЕ торгует, НЕ отправляет ордера, НЕ пишет в БД, НЕ
меняет схему, scoring, thresholds, pipeline, scheduler или backtest. НЕ
импортируется runtime-кодом — только ручной CLI-запуск.

Зачем отдельный датасет (P1, `reports/c50/P1_TASK.md`):

  * Каждое место, считающее издержки, платило комиссии и проскальзывание и НЕ
    платило funding, хотя перп держит позицию через сеттлменты. Чтобы считать
    РЕАЛИЗОВАННЫЙ funding, нужна история ставок, а не текущая ставка.

  * `analyzer.binance.funding_rate` для этого не подходит: он сворачивает
    историю в z-score для вето «crowded funding». Здесь нужны сами сеттлменты.

  * Приближение «последняя ставка x время удержания» запрещено ТЗ P1 и было бы
    неверным: 15% сеттлментов на BTCUSDT отрицательные, знак меняется внутри
    сделки.

Биржа ПРИБИТА ГВОЗДЯМИ (та же причина, что D15 в kline_cache): failover-клиент
мог бы незаметно сменить источник посреди набора и смешать две разные
funding-семантики в одном файле.

Шаг сеттлментов НЕ ЗАХАРДКОЖЕН в 8h: Binance переводил часть символов на 4h, и
датасет должен сообщать фактический шаг, а не подтверждать допущение. Модальный
интервал считается по данным и пишется в манифест.

`markPrice` у старых записей приходит пустой строкой — в датасет он попадает как
NaN, и потребитель не имеет права на него опираться.

Запуск:

    .venv/bin/python -m tools.funding_cache \
      --exchange binance \
      --symbol BTCUSDT \
      --start 2022-04-01 \
      --outdir data/funding
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from analyzer.binance import BinanceClient
# Имя файла датасета живёт в validation.funding: его читает production-код,
# который не имеет права импортировать tools.*. Писатель берёт имя оттуда.
from validation.funding import dataset_stem

# Только Binance: у Bybit другой контракт funding-истории, и смешивать две
# семантики в одном файле — ровно тот класс ошибок, который тул исключает.
EXCHANGES = ("binance",)

PAGE_LIMIT = 1000
PAGE_SLEEP_S = 0.2
GAP_EXAMPLES_MAX = 5

# Разрыв объявляется, когда интервал превышает модальный более чем в
# GAP_TOLERANCE раз. Порог не 1.0, потому что fundingTime имеет джиттер в
# миллисекундах (…400002 вместо …400000) — это не пропуск.
GAP_TOLERANCE = 1.5


class FundingCacheError(Exception):
    pass


def create_client(exchange: str) -> Any:
    if exchange == "binance":
        return BinanceClient()
    raise ValueError(f"unknown exchange: {exchange!r}")


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _to_ms(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("timestamp must not be a bool")
    if isinstance(value, (int, float)):
        return int(value)
    return int(pd.Timestamp(value).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Сборка страниц
# ---------------------------------------------------------------------------

def assemble(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    """Склеить страницы: дедуп по funding_time, хронологический порядок.

    Возвращает (df, duplicates_removed). Ничего не триммится: границы набора
    задаются вызывающим через --start/--end, а не молчаливым срезом.
    """
    if not records:
        return pd.DataFrame(columns=["funding_time", "funding_rate",
                                     "mark_price", "rate_type"]), 0
    df = pd.DataFrame({
        "funding_time": [_to_ms(r["fundingTime"]) for r in records],
        "funding_rate": [float(r["fundingRate"]) for r in records],
        # Пустая строка у старых записей -> NaN, а не 0.0: ноль здесь был бы
        # выдуманной ценой.
        "mark_price": [pd.to_numeric(r.get("markPrice"), errors="coerce")
                       for r in records],
        "rate_type": [r.get("rateType") for r in records],
    })
    before = len(df)
    df = df.drop_duplicates("funding_time")
    duplicates_removed = before - len(df)
    df = df.sort_values("funding_time").reset_index(drop=True)
    return df, duplicates_removed


async def fetch_history(client: Any, symbol: str, *, start_ms: int,
                        end_ms: int | None = None,
                        page_limit: int = PAGE_LIMIT,
                        sleep_s: float = PAGE_SLEEP_S,
                        ) -> tuple[pd.DataFrame, int]:
    """Пагинация ВПЕРЁД по startTime от start_ms до end_ms (или до конца).

    Никакого safety cap на число страниц: глубину задаёт вызывающий. Запросы
    строго последовательные, между страницами пауза ~sleep_s.

    Терминирующие условия:
      * пустая страница   — данных дальше нет;
      * короткая страница — естественный конец истории;
      * курсор ушёл за end_ms.
    """
    records: list[dict[str, Any]] = []
    cursor = start_ms
    while True:
        page = await client.funding_history(symbol, start_time=cursor,
                                           end_time=end_ms, limit=page_limit)
        if not page:
            break
        records.extend(page)
        if len(page) < page_limit:
            break
        last = _to_ms(page[-1]["fundingTime"])
        if end_ms is not None and last >= end_ms:
            break
        cursor = last + 1
        await asyncio.sleep(sleep_s)
    return assemble(records)


# ---------------------------------------------------------------------------
# Отчёт о пропусках
# ---------------------------------------------------------------------------

def modal_interval_ms(df: pd.DataFrame) -> int:
    """Фактический шаг сеттлментов по данным, а не по допущению.

    Джиттер в миллисекундах гасится округлением до минуты перед подсчётом
    моды: иначе 28_800_000 и 28_800_002 стали бы разными «шагами».
    """
    if df is None or len(df) < 2:
        return 0
    deltas = df["funding_time"].diff().dropna()
    if deltas.empty:
        return 0
    minutes = (deltas / 60_000).round()
    mode = minutes.mode()
    if mode.empty:
        return 0
    return int(mode.iloc[0]) * 60_000


def gap_report(df: pd.DataFrame) -> dict[str, Any]:
    """Сверить спейсинг funding_time с модальным шагом.

    gap_count           — сколько разрывов;
    missing_settlements — сколько сеттлментов суммарно недостаёт.

    Пропуски только регистрируются: датасет с честной пометкой полезнее
    отсутствующего. Решение «падать или нет» принимает потребитель —
    `validation.funding` падает, когда пропуск попал внутрь конкретной сделки.
    """
    expected = modal_interval_ms(df)
    report: dict[str, Any] = {
        "modal_interval_ms": expected,
        "has_gaps": False,
        "gap_count": 0,
        "missing_settlements": 0,
        "gap_examples": [],
    }
    if expected <= 0 or df is None or len(df) < 2:
        return report

    times = [int(t) for t in df["funding_time"].tolist()]
    examples: list[dict[str, Any]] = []
    for prev, curr in zip(times, times[1:]):
        delta = curr - prev
        if delta <= expected * GAP_TOLERANCE:
            continue
        missing = int(round(delta / expected)) - 1
        if missing <= 0:
            continue
        report["gap_count"] += 1
        report["missing_settlements"] += missing
        if len(examples) < GAP_EXAMPLES_MAX:
            examples.append({
                "after_funding_time": prev,
                "after_funding_time_iso": _iso(prev),
                "before_funding_time": curr,
                "before_funding_time_iso": _iso(curr),
                "missing_settlements": missing,
            })
    report["has_gaps"] = report["gap_count"] > 0
    report["gap_examples"] = examples
    return report


# ---------------------------------------------------------------------------
# Манифест и запись
# ---------------------------------------------------------------------------

def rate_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Описательная статистика ставок — она же ответ на вопрос «насколько
    допущение M00 (0.01 %/8h) расходится с фактом»."""
    if df is None or len(df) == 0:
        return {"mean_pct_per_settlement": None, "median_pct_per_settlement": None,
                "min_pct": None, "max_pct": None, "negative_share": None}
    pct = df["funding_rate"] * 100.0
    return {
        "mean_pct_per_settlement": float(pct.mean()),
        "median_pct_per_settlement": float(pct.median()),
        "min_pct": float(pct.min()),
        "max_pct": float(pct.max()),
        "negative_share": float((df["funding_rate"] < 0).mean()),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(df: pd.DataFrame, *, exchange: str, symbol: str,
                   requested_start_ms: int, requested_end_ms: int | None,
                   duplicates_removed: int, file_format: str,
                   data_file: str) -> dict[str, Any]:
    first = int(df["funding_time"].iloc[0]) if len(df) else None
    last = int(df["funding_time"].iloc[-1]) if len(df) else None
    gaps = gap_report(df)
    manifest = {
        "exchange": exchange,
        "symbol": symbol,
        "kind": "funding_settlements",
        "requested_start": requested_start_ms,
        "requested_start_iso": _iso(requested_start_ms),
        "requested_end": requested_end_ms,
        "requested_end_iso": (_iso(requested_end_ms)
                              if requested_end_ms is not None else None),
        "settlements": int(len(df)),
        "first_funding_time": first,
        "last_funding_time": last,
        "first_funding_time_iso": _iso(first) if first is not None else None,
        "last_funding_time_iso": _iso(last) if last is not None else None,
        # markPrice пуст у старых записей: потребитель не вправе на него
        # опираться, и манифест говорит об этом прямо.
        "mark_price_coverage": (float(df["mark_price"].notna().mean())
                                if len(df) else None),
        "file_format": file_format,
        "data_file": data_file,
        "duplicates_removed": int(duplicates_removed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pinned": True,
    }
    manifest.update(gaps)
    manifest.update(rate_stats(df))
    return manifest


def write_dataset(df: pd.DataFrame, outdir: str, *, exchange: str, symbol: str,
                  requested_start_ms: int, requested_end_ms: int | None,
                  duplicates_removed: int,
                  ) -> tuple[str, str, dict[str, Any]]:
    import os

    os.makedirs(outdir, exist_ok=True)
    stem = dataset_stem(exchange, symbol)
    data_file = f"{stem}.csv"
    data_path = os.path.join(outdir, data_file)
    manifest_path = os.path.join(outdir, f"{stem}.manifest.json")

    manifest = build_manifest(
        df, exchange=exchange, symbol=symbol,
        requested_start_ms=requested_start_ms,
        requested_end_ms=requested_end_ms,
        duplicates_removed=duplicates_removed,
        file_format="csv", data_file=data_file,
    )

    df.to_csv(data_path, index=False)
    # Хеш считается по УЖЕ записанному файлу: пин должен описывать то, что
    # прочитает потребитель, а не то, что было в памяти.
    manifest["data_sha256"] = _sha256_file(data_path)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)
    return data_path, manifest_path, manifest


def format_report(manifest: dict[str, Any], data_path: str,
                  manifest_path: str) -> str:
    interval_h = (manifest["modal_interval_ms"] / 3_600_000
                  if manifest["modal_interval_ms"] else 0)
    lines = [
        f"source-pinned funding cache — {manifest['exchange']} "
        f"{manifest['symbol']}",
        f"  settlements    : {manifest['settlements']}",
        f"  range          : {manifest['first_funding_time_iso']} .. "
        f"{manifest['last_funding_time_iso']}",
        f"  modal interval : {interval_h:g}h",
        f"  duplicates     : {manifest['duplicates_removed']} removed",
    ]
    if manifest["settlements"]:
        lines += [
            f"  mean rate      : {manifest['mean_pct_per_settlement']:.6f}% "
            f"per settlement (median {manifest['median_pct_per_settlement']:.6f}%)",
            f"  range of rates : {manifest['min_pct']:.5f}% .. "
            f"{manifest['max_pct']:.5f}%",
            f"  negative share : {manifest['negative_share'] * 100:.2f}%  "
            f"(shorts pay, longs are credited)",
            f"  mark_price     : {manifest['mark_price_coverage'] * 100:.1f}% "
            f"populated" + ("" if manifest["mark_price_coverage"] == 1.0
                            else "  (do not rely on it)"),
        ]
    if manifest["has_gaps"]:
        lines.append(f"  GAPS           : {manifest['gap_count']} gap(s), "
                     f"{manifest['missing_settlements']} settlement(s) missing")
        for ex in manifest["gap_examples"]:
            lines.append(f"     after {ex['after_funding_time_iso']} -> "
                         f"{ex['before_funding_time_iso']} "
                         f"({ex['missing_settlements']} missing)")
    else:
        lines.append("  gaps           : none")
    lines.append(f"  sha256         : {manifest.get('data_sha256', '')[:16]}")
    lines.append(f"  data           : {data_path}")
    lines.append(f"  manifest       : {manifest_path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_ts(value: str) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


async def _run(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    start_ms = _parse_ts(args.start)
    end_ms = _parse_ts(args.end) if args.end else None
    client = create_client(args.exchange)
    try:
        df, duplicates = await fetch_history(client, args.symbol,
                                             start_ms=start_ms, end_ms=end_ms)
    finally:
        await client.close()
    data_path, manifest_path, manifest = write_dataset(
        df, args.outdir, exchange=args.exchange, symbol=args.symbol,
        requested_start_ms=start_ms, requested_end_ms=end_ms,
        duplicates_removed=duplicates,
    )
    return manifest, data_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline source-pinned funding-settlement cache (P1).")
    # Источник обязателен и без default — молчаливый выбор биржи запрещён.
    parser.add_argument("--exchange", choices=EXCHANGES, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True,
                        help="ISO date/datetime, UTC when tz is absent")
    parser.add_argument("--end", default=None)
    parser.add_argument("--outdir", default="data/funding")
    parser.add_argument("--json", action="store_true",
                        help="вывести манифест как JSON")
    args = parser.parse_args(argv)

    start_ms = _parse_ts(args.start)
    if args.end and _parse_ts(args.end) <= start_ms:
        parser.error("--end must be after --start")

    manifest, data_path, manifest_path = asyncio.run(_run(args))
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(manifest, data_path, manifest_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
