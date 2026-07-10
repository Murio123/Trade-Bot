"""Stage C1.2 — offline kline cache. Никакой сети: клиенты подделываются."""
from __future__ import annotations

import asyncio
import json
import pathlib
import re

import pandas as pd
import pytest

from tools import kline_cache

INTERVAL = 900_000  # 15m
BASE_MS = 1_700_000_000_000


def _page(start_ms: int, n: int, *, interval_ms: int = INTERVAL,
          taker: bool = True, skip_after: int | None = None) -> pd.DataFrame:
    """Одна страница klines в форме, которую отдаёт реальный клиент.

    ``skip_after`` вырезает одну свечу после указанного индекса — так строится
    датасет с дырой, не ломая монотонность open_time.
    """
    times = [start_ms + i * interval_ms for i in range(n)]
    if skip_after is not None:
        times.pop(skip_after + 1)
    cols = {
        "open_time": pd.to_datetime(times, unit="ms", utc=True),
        "open": [100.0] * len(times),
        "high": [101.0] * len(times),
        "low": [99.0] * len(times),
        "close": [100.5] * len(times),
        "volume": [10.0] * len(times),
        "quote_volume": [1000.0] * len(times),
    }
    if taker:
        cols["taker_buy_base"] = [5.0] * len(times)
    return pd.DataFrame(cols)


class FakeClient:
    """Отдаёт заранее заданные страницы по порядку. Никаких HTTP-вызовов."""

    def __init__(self, pages: list[pd.DataFrame]) -> None:
        self.pages = list(pages)
        self.calls: list[dict] = []
        self.closed = False

    async def klines(self, interval, limit=300, symbol=None, end_time=None):
        self.calls.append({"interval": interval, "limit": limit,
                           "symbol": symbol, "end_time": end_time})
        if not self.pages:
            return pd.DataFrame()
        return self.pages.pop(0)

    async def close(self) -> None:
        self.closed = True


def _fetch(client, bars, page_limit=2):
    return asyncio.run(kline_cache.fetch_history(
        client, "BTCUSDT", "15m", bars, page_limit=page_limit, sleep_s=0.0))


# --- 1..5: пагинация ------------------------------------------------------

def test_pagination_assembles_multiple_pages():
    newest = _page(BASE_MS + 3 * INTERVAL, 2)   # t3, t4
    middle = _page(BASE_MS + 1 * INTERVAL, 2)   # t1, t2
    oldest = _page(BASE_MS, 1)                  # t0 — короткая, конец истории
    client = FakeClient([newest, middle, oldest])
    df, _ = _fetch(client, bars=10)

    assert len(client.calls) == 3
    assert len(df) == 5
    # первая страница без end_time, дальше курсор идёт назад по времени
    assert client.calls[0]["end_time"] is None
    assert client.calls[1]["end_time"] == BASE_MS + 3 * INTERVAL - 1
    assert client.calls[2]["end_time"] == BASE_MS + 1 * INTERVAL - 1


def test_result_is_chronologically_sorted():
    client = FakeClient([_page(BASE_MS + 2 * INTERVAL, 2), _page(BASE_MS, 1)])
    df, _ = _fetch(client, bars=10)
    times = [int(t.timestamp() * 1000) for t in df["open_time"]]
    assert times == sorted(times)
    assert times[0] == BASE_MS


def test_duplicate_open_time_rows_removed():
    first = _page(BASE_MS + 2 * INTERVAL, 2)
    overlapping = _page(BASE_MS + INTERVAL, 2)  # одна свеча общая
    client = FakeClient([first, overlapping])
    df, duplicates = _fetch(client, bars=10)

    assert duplicates == 1
    assert len(df) == 3
    assert df["open_time"].is_unique


def test_short_final_page_terminates_fetch():
    full = _page(BASE_MS + 2 * INTERVAL, 2)   # len == page_limit
    short = _page(BASE_MS, 1)                 # len < page_limit -> стоп
    never = _page(BASE_MS - 10 * INTERVAL, 2)
    client = FakeClient([full, short, never])
    df, _ = _fetch(client, bars=100)

    assert len(client.calls) == 2, "короткая страница должна остановить пагинацию"
    assert len(df) == 3
    assert len(client.pages) == 1, "третья страница не должна запрашиваться"
    assert never is not None


def test_requested_bars_trims_to_latest_n():
    client = FakeClient([_page(BASE_MS + 2 * INTERVAL, 2), _page(BASE_MS, 2)])
    df, _ = _fetch(client, bars=3)

    assert len(df) == 3
    first = int(df["open_time"].iloc[0].timestamp() * 1000)
    last = int(df["open_time"].iloc[-1].timestamp() * 1000)
    # оставлены самые СВЕЖИЕ 3 бара: самый старый (BASE_MS) отрезан
    assert first == BASE_MS + INTERVAL
    assert last == BASE_MS + 3 * INTERVAL


# --- 6: пропуски ----------------------------------------------------------

def test_gap_report_detects_missing_intervals():
    df = _page(BASE_MS, 6, skip_after=1)  # дыра после 2-го бара
    df = pd.concat([df, _page(BASE_MS + 20 * INTERVAL, 1)], ignore_index=True)
    report = kline_cache.gap_report(df, "15m")

    assert report["has_gaps"] is True
    assert report["gap_count"] == 2
    assert report["missing_bars"] == 1 + 14
    assert report["expected_interval_ms"] == INTERVAL
    assert report["gap_examples"][0]["missing_bars"] == 1
    assert "after_open_time_iso" in report["gap_examples"][0]


def test_gap_report_clean_dataset():
    report = kline_cache.gap_report(_page(BASE_MS, 5), "15m")
    assert report["has_gaps"] is False
    assert report["gap_count"] == 0
    assert report["missing_bars"] == 0
    assert report["gap_examples"] == []


# --- 7..9: манифест -------------------------------------------------------

def _manifest(df, exchange="binance"):
    return kline_cache.build_manifest(
        df, exchange=exchange, symbol="BTCUSDT", timeframe="15m",
        requested_bars=5, duplicate_count_removed=0,
        file_format="csv", data_file="x.csv")


def test_manifest_source_pinned_and_exchange():
    manifest = _manifest(_page(BASE_MS, 5), exchange="bybit")
    assert manifest["source_pinned"] is True
    assert manifest["exchange"] == "bybit"
    assert manifest["symbol"] == "BTCUSDT"
    assert manifest["timeframe"] == "15m"
    assert manifest["actual_bars"] == 5
    assert manifest["first_open_time"] == BASE_MS
    assert manifest["first_open_time_iso"].startswith("20")


def test_manifest_has_taker_buy_base_true_when_column_present():
    # Binance-подобный фрейм -> CVD доступен.
    assert _manifest(_page(BASE_MS, 3, taker=True))["has_taker_buy_base"] is True


def test_manifest_has_taker_buy_base_false_when_column_absent():
    # Bybit-подобный фрейм (D15) -> CVD по этому датасету не посчитать.
    assert _manifest(_page(BASE_MS, 3, taker=False))["has_taker_buy_base"] is False


# --- 10, 15: CLI ----------------------------------------------------------

def test_cli_requires_explicit_exchange():
    with pytest.raises(SystemExit):
        kline_cache.main(["--timeframe", "15m", "--bars", "10"])


def test_cli_json_mode_prints_manifest(tmp_path, monkeypatch, capsys):
    client = FakeClient([_page(BASE_MS, 3)])
    monkeypatch.setattr(kline_cache, "create_client", lambda exchange: client)

    rc = kline_cache.main(["--exchange", "binance", "--symbol", "BTCUSDT",
                           "--timeframe", "15m", "--bars", "3",
                           "--outdir", str(tmp_path), "--json"])
    assert rc == 0
    assert client.closed is True

    manifest = json.loads(capsys.readouterr().out)
    assert manifest["source_pinned"] is True
    assert manifest["exchange"] == "binance"
    assert manifest["actual_bars"] == 3
    assert manifest["has_taker_buy_base"] is True
    assert (tmp_path / manifest["data_file"]).exists()
    assert (tmp_path / "binance_BTCUSDT_15m.manifest.json").exists()

    sidecar = json.loads(
        (tmp_path / "binance_BTCUSDT_15m.manifest.json").read_text())
    assert sidecar["source_pinned"] is True


# --- 11..14: статические гарантии -----------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "kline_cache.py").read_text()


def test_never_uses_failover_market_client():
    # Источник должен быть пинован: клиент биржи инстанцируется напрямую.
    for forbidden in ("MarketClient", "analyzer.exchange", "create_market_client"):
        assert forbidden not in SOURCE, forbidden
    assert "from analyzer.binance import BinanceClient" in SOURCE
    assert "from analyzer.bybit import BybitClient" in SOURCE


def test_no_db_access_or_write_sql():
    lowered = SOURCE.lower()
    for verb in ("insert", "update ", "delete", "truncate", "alter",
                 "create table", "commit("):
        assert verb not in lowered, verb
    for mod in ("import database", "psycopg", "sqlalchemy", "asyncpg"):
        assert mod not in lowered, mod


def test_no_exchange_or_order_execution():
    for forbidden in ("create_order", "market_order", "limit_order",
                      "api_key", "api_secret", "exchange.create"):
        assert forbidden not in SOURCE, forbidden


def test_production_runtime_does_not_import_kline_cache():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))

    for path in runtime_files:
        if not path.exists():
            continue
        text = path.read_text()
        assert not re.search(r"\bkline_cache\b", text), f"{path} imports kline_cache"
