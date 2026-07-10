"""Stage C1.3a — строгий загрузчик датасетов. Никакой сети, только tmp-фикстуры."""
from __future__ import annotations

import json
import pathlib
import re

import pandas as pd
import pytest

from tools import kline_cache, kline_dataset
from tools.kline_dataset import DatasetError, GapRatioExceeded

INTERVAL = 900_000  # 15m
BASE_MS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    """Детерминированная ветка записи: pyarrow в окружении может появиться."""
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


def _df(n: int, *, interval_ms: int = INTERVAL, taker: bool = True,
        skip_after: int | None = None, skip_len: int = 1) -> pd.DataFrame:
    """OHLCV-фрейм; skip_after вырезает skip_len свечей -> настоящая дыра."""
    times = [BASE_MS + i * interval_ms for i in range(n + (skip_len if skip_after
                                                           is not None else 0))]
    if skip_after is not None:
        del times[skip_after + 1: skip_after + 1 + skip_len]
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
    df = pd.DataFrame(cols)
    df["close_time"] = df["open_time"] + pd.to_timedelta(interval_ms, unit="ms")
    return df


def _write(outdir, df, *, exchange="binance", symbol="BTCUSDT", timeframe="15m"):
    return kline_cache.write_dataset(
        df, str(outdir), exchange=exchange, symbol=symbol, timeframe=timeframe,
        requested_bars=len(df), duplicate_count_removed=0)


def _patch_manifest(outdir, overrides, exchange="binance", symbol="BTCUSDT",
                    timeframe="15m"):
    """Испортить манифест на диске. overrides — dict, а не **kwargs: его ключи
    (exchange/symbol/timeframe) столкнулись бы с параметрами хелпера."""
    mpath = kline_dataset.manifest_path_for(str(outdir), exchange, symbol, timeframe)
    manifest = json.loads(pathlib.Path(mpath).read_text())
    for key, value in overrides.items():
        manifest[key] = value
    pathlib.Path(mpath).write_text(json.dumps(manifest))
    return manifest


# --- 1..2: манифест первичен ----------------------------------------------

def test_manifest_must_exist(tmp_path):
    with pytest.raises(DatasetError, match="manifest not found"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_data_file_never_loaded_without_manifest(tmp_path):
    # Файл данных есть, манифеста нет -> отказ, а не «догадаемся по имени».
    _write(tmp_path, _df(5))
    (tmp_path / "binance_BTCUSDT_15m.manifest.json").unlink()
    with pytest.raises(DatasetError, match="manifest not found"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_source_pinned_false_raises(tmp_path):
    _write(tmp_path, _df(5))
    _patch_manifest(tmp_path, {"source_pinned": False})
    with pytest.raises(DatasetError, match="source_pinned"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


# --- 3: запрошенное != записанное ------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("exchange", "bybit"), ("symbol", "ETHUSDT"), ("timeframe", "1h"),
])
def test_requested_mismatch_raises(tmp_path, field, value):
    _write(tmp_path, _df(5))
    _patch_manifest(tmp_path, {field: value})
    with pytest.raises(DatasetError, match=field):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_expected_interval_mismatch_raises(tmp_path):
    _write(tmp_path, _df(5))
    _patch_manifest(tmp_path, {"expected_interval_ms": 60_000})
    with pytest.raises(DatasetError, match="expected_interval_ms"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


# --- 4..7: данные против манифеста -----------------------------------------

def test_actual_bars_mismatch_raises(tmp_path):
    _write(tmp_path, _df(5))
    _patch_manifest(tmp_path, {"actual_bars": 99})
    with pytest.raises(DatasetError, match="actual_bars"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


@pytest.mark.parametrize("field", ["first_open_time", "last_open_time"])
def test_endpoint_mismatch_raises(tmp_path, field):
    _write(tmp_path, _df(5))
    _patch_manifest(tmp_path, {field: 1})
    with pytest.raises(DatasetError, match=field):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_duplicate_open_time_raises(tmp_path):
    df = _df(5)
    dup = pd.concat([df, df.iloc[[2]]], ignore_index=True).sort_values("open_time")
    dup = dup.reset_index(drop=True)
    _write(tmp_path, dup)  # манифест согласован с данными: ловим сам дубликат
    with pytest.raises(DatasetError, match="duplicates"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_unsorted_open_time_raises(tmp_path):
    df = _df(5)
    shuffled = df.iloc[[0, 2, 1, 3, 4]].reset_index(drop=True)
    # endpoints совпадают, дублей нет -> падаем именно на монотонности
    _write(tmp_path, shuffled)
    with pytest.raises(DatasetError, match="not strictly increasing"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_duplicate_reported_as_duplicate_not_endpoint_mismatch(tmp_path):
    """Диагностика: структура open_time проверяется ДО сверки границ."""
    df = _df(5)
    dup = pd.concat([df, df.iloc[[2]]], ignore_index=True).sort_values("open_time")
    _write(tmp_path, dup.reset_index(drop=True))
    _patch_manifest(tmp_path, {"first_open_time": 1})  # границы тоже врут

    with pytest.raises(DatasetError, match="duplicates"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_unsorted_reported_as_monotonic_not_endpoint_mismatch(tmp_path):
    df = _df(5)
    _write(tmp_path, df.iloc[[0, 2, 1, 3, 4]].reset_index(drop=True))
    _patch_manifest(tmp_path, {"last_open_time": 1})

    with pytest.raises(DatasetError, match="not strictly increasing"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


# --- 8: dtype-контракт ------------------------------------------------------

def test_mixed_precision_timestamps_raise_dataset_error(tmp_path):
    """Чужой/правленый CSV падает внятно, а не непрозрачной ошибкой pandas."""
    data_path, _, _ = _write(tmp_path, _df(5))
    raw = pd.read_csv(data_path)
    raw.loc[3, "open_time"] = "2023-11-15 00:13:20.001000+00:00"
    raw.to_csv(data_path, index=False)

    with pytest.raises(DatasetError, match="open_time"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_unparseable_timestamp_raises_dataset_error(tmp_path):
    data_path, _, _ = _write(tmp_path, _df(5))
    raw = pd.read_csv(data_path)
    raw.loc[2, "close_time"] = "not-a-timestamp"
    raw.to_csv(data_path, index=False)

    with pytest.raises(DatasetError, match="close_time"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_csv_load_preserves_utc_datetimes(tmp_path):
    _write(tmp_path, _df(5))
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")

    for col in ("open_time", "close_time"):
        assert isinstance(frame.df[col].dtype, pd.DatetimeTZDtype)
        assert str(frame.df[col].dt.tz) == "UTC"
    # сравнение close_time <= t в backtest должно работать, а не молча врать
    t = frame.df["close_time"].iloc[2]
    assert (frame.df["close_time"] <= t).sum() == 3


# --- 9..10: CVD-инвариант (D15) --------------------------------------------

def test_has_taker_buy_base_true_matches_manifest(tmp_path):
    _write(tmp_path, _df(5, taker=True))
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")
    assert frame.has_taker_buy_base is True
    assert frame.cvd_method == "exact"


def test_has_taker_buy_base_false_matches_manifest(tmp_path):
    _write(tmp_path, _df(5, taker=False), exchange="bybit")
    frame = kline_dataset.load_frame(str(tmp_path), "bybit", "BTCUSDT", "15m")
    assert frame.has_taker_buy_base is False
    assert frame.cvd_method == "estimate"


def test_taker_column_lost_on_roundtrip_raises(tmp_path):
    """Манифест верит в exact-CVD, а колонки в данных нет -> отказ, не тишина."""
    df = _df(5, taker=True)
    data_path, _, _ = _write(tmp_path, df)
    df.drop(columns=["taker_buy_base"]).to_csv(data_path, index=False)

    with pytest.raises(DatasetError, match="has_taker_buy_base"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_taker_column_all_nan_raises(tmp_path):
    """Колонка есть, но пустая: compute_cvd_from_klines ушёл бы в estimate."""
    df = _df(5, taker=True)
    data_path, _, _ = _write(tmp_path, df)
    blanked = df.copy()
    blanked["taker_buy_base"] = pd.NA
    blanked.to_csv(data_path, index=False)

    with pytest.raises(DatasetError, match="has_taker_buy_base"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


# --- 11..12: пропуски -------------------------------------------------------

def test_gap_ratio_above_limit_raises(tmp_path):
    df = _df(200, skip_after=50, skip_len=5)  # 5 пропущенных из ~200 = 2.5%
    _write(tmp_path, df)
    with pytest.raises(GapRatioExceeded, match="gap ratio"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m",
                                 max_gap_ratio=0.001)


def test_small_gap_below_limit_loads_and_exposes_metadata(tmp_path):
    df = _df(2000, skip_after=100, skip_len=1)  # 1/2000 = 0.0005 < 0.001
    _write(tmp_path, df)
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m",
                                     max_gap_ratio=0.001)

    gaps = frame.gaps
    assert gaps["has_gaps"] is True
    assert gaps["gap_count"] == 1
    assert gaps["missing_bars"] == 1
    assert gaps["gap_ratio"] == pytest.approx(1 / 2000)
    assert len(gaps["gap_examples"]) == 1


def test_no_max_gap_ratio_means_no_enforcement(tmp_path):
    df = _df(200, skip_after=50, skip_len=5)
    _write(tmp_path, df)
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")
    assert frame.gaps["missing_bars"] == 5  # видно, но не фатально


def test_honest_gap_above_threshold_raises(tmp_path):
    df = _df(200, skip_after=50, skip_len=5)
    _write(tmp_path, df)
    with pytest.raises(GapRatioExceeded):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m",
                                 max_gap_ratio=0.01)


# --- манифест не источник истины о пропусках (Codex Medium, c7ea403) --------

_LIE = {"missing_bars": 0, "gap_count": 0, "has_gaps": False, "gap_examples": []}


def test_lying_manifest_gap_fields_raise(tmp_path):
    """Данные с реальной дырой + манифест, объявляющий их бездырочными."""
    _write(tmp_path, _df(200, skip_after=50, skip_len=5))
    _patch_manifest(tmp_path, _LIE)

    with pytest.raises(DatasetError, match="recomputed"):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_max_gap_ratio_uses_recomputed_not_manifest(tmp_path):
    """Враньё в манифесте не должно отключать риск-контроль max_gap_ratio."""
    _write(tmp_path, _df(200, skip_after=50, skip_len=5))
    _patch_manifest(tmp_path, _LIE)

    with pytest.raises(DatasetError):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m",
                                 max_gap_ratio=0.0)


@pytest.mark.parametrize("field,value", [
    ("gap_count", 7), ("missing_bars", 99), ("has_gaps", False),
])
def test_manifest_gap_field_mismatch_raises(tmp_path, field, value):
    _write(tmp_path, _df(2000, skip_after=100, skip_len=1))
    _patch_manifest(tmp_path, {field: value})
    with pytest.raises(DatasetError, match=field):
        kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")


def test_loaded_frame_gaps_are_recomputed_from_data(tmp_path):
    """gap_examples в манифесте затёрты, но фрейм всё равно их знает."""
    _write(tmp_path, _df(2000, skip_after=100, skip_len=1))
    # gap_examples — иллюстрация, а не инвариант: его подмена не фатальна,
    # но и не должна попасть в LoadedFrame.gaps.
    _patch_manifest(tmp_path, {"gap_examples": []})

    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m",
                                     max_gap_ratio=0.001)
    assert frame.gaps["missing_bars"] == 1
    assert frame.gaps["gap_count"] == 1
    assert frame.gaps["gap_ratio"] == pytest.approx(1 / 2000)
    assert len(frame.gaps["gap_examples"]) == 1  # пересчитано, не скопировано
    assert frame.manifest["gap_examples"] == []


def test_valid_cache_output_still_loads(tmp_path):
    """Честный, свежий вывод kline_cache грузится без единой правки."""
    _write(tmp_path, _df(50))
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m",
                                     max_gap_ratio=0.0)
    assert frame.bars == 50
    assert frame.gaps["has_gaps"] is False
    assert frame.gaps["missing_bars"] == 0
    assert frame.cvd_method == "exact"


# --- 13..14: смешение источников (D15) --------------------------------------

def _loaded(tmp_path, exchange, symbol, timeframe):
    _write(tmp_path, _df(5), exchange=exchange, symbol=symbol, timeframe=timeframe)
    return kline_dataset.load_frame(str(tmp_path), exchange, symbol, timeframe)


def test_mixed_exchange_across_timeframes_raises(tmp_path):
    frames = {"15m": _loaded(tmp_path, "binance", "BTCUSDT", "15m"),
              "1h": _loaded(tmp_path, "bybit", "BTCUSDT", "1h")}
    with pytest.raises(DatasetError, match="mixed exchanges"):
        kline_dataset.assert_single_source(frames)


def test_mixed_symbol_across_timeframes_raises(tmp_path):
    frames = {"15m": _loaded(tmp_path, "binance", "BTCUSDT", "15m"),
              "1h": _loaded(tmp_path, "binance", "ETHUSDT", "1h")}
    with pytest.raises(DatasetError, match="mixed symbols"):
        kline_dataset.assert_single_source(frames)


def test_load_frames_happy_path_single_source(tmp_path):
    _write(tmp_path, _df(5), timeframe="15m")
    _write(tmp_path, _df(5, interval_ms=3_600_000), timeframe="1h")
    frames = kline_dataset.load_frames(str(tmp_path), "binance", "BTCUSDT",
                                       ["15m", "1h"])
    assert set(frames) == {"15m", "1h"}
    assert {f.exchange for f in frames.values()} == {"binance"}


def test_load_frames_rejects_frame_from_other_exchange(tmp_path):
    _write(tmp_path, _df(5), timeframe="15m")
    _write(tmp_path, _df(5, interval_ms=3_600_000), exchange="bybit", timeframe="1h")
    # запрошена binance -> bybit-манифест для 1h просто не найдётся по имени
    with pytest.raises(DatasetError):
        kline_dataset.load_frames(str(tmp_path), "binance", "BTCUSDT",
                                  ["15m", "1h"])


# --- 15: aux-depth (D14) ----------------------------------------------------

INTRADAY = {"entry": "15m", "htf": "4h", "zone_tfs": ["4h", "2h", "1h"]}
SWING = {"entry": "4h", "htf": "1d", "zone_tfs": ["12h", "6h"]}


def test_required_timeframes_includes_1d_for_intraday():
    assert kline_dataset.required_timeframes(INTRADAY) == [
        "15m", "4h", "2h", "1h", "1d"]


def test_aux_depth_required_available_deficit():
    # 1200 баров 15m = 300 часов календаря.
    table = kline_dataset.aux_depth(INTRADAY, 1200, {
        "15m": 1200, "4h": 285, "2h": 100, "1h": 330, "1d": 50,
    })

    assert table["15m"]["required"] == 1200 + 300      # entry warmup
    assert table["4h"]["required"] == 75 + 210         # htf+zone -> warmup 210
    assert table["4h"]["roles"] == ["htf", "zone"]
    assert table["2h"]["required"] == 150 + 30
    assert table["1h"]["required"] == 300 + 30
    assert table["1d"]["required"] == 13 + 210         # ceil(300/24)=13

    assert table["4h"]["deficit"] == 0 and table["4h"]["ok"] is True
    assert table["2h"]["deficit"] == 80 and table["2h"]["ok"] is False
    assert table["1d"]["deficit"] == 223 - 50
    assert table["15m"]["deficit"] == 300

    deficits = kline_dataset.aux_depth_deficits(table)
    assert set(deficits) == {"15m", "2h", "1d"}


def test_aux_depth_swing_shares_1d_between_htf_and_regime():
    table = kline_dataset.aux_depth(SWING, 100, {"4h": 0, "1d": 0,
                                                 "12h": 0, "6h": 0})
    # 1d одновременно htf и regime -> берётся максимальный warmup, не сумма
    assert table["1d"]["roles"] == ["htf", "regime_1d"]
    assert table["1d"]["warmup"] == 210


def test_aux_depth_accepts_loaded_frames_and_manifests(tmp_path):
    _write(tmp_path, _df(5), timeframe="15m")
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")
    table = kline_dataset.aux_depth({"entry": "15m", "htf": "1d",
                                     "zone_tfs": []}, 10,
                                    {"15m": frame, "1d": frame.manifest})
    assert table["15m"]["available"] == 5
    assert table["1d"]["available"] == 5


def test_aux_depth_rejects_non_positive_walk_bars():
    with pytest.raises(ValueError):
        kline_dataset.aux_depth(INTRADAY, 0, {})


def test_aux_depth_missing_frame_counts_as_zero_available():
    table = kline_dataset.aux_depth(INTRADAY, 100, {})
    assert table["1h"]["available"] == 0
    assert table["1h"]["deficit"] == table["1h"]["required"]


# --- 16: dataset_stem -------------------------------------------------------

def test_dataset_stem_preserves_naming_convention(tmp_path):
    assert kline_cache.dataset_stem("binance", "BTCUSDT", "15m") == \
        "binance_BTCUSDT_15m"

    data_path, manifest_path, manifest = _write(tmp_path, _df(3))
    assert pathlib.Path(data_path).name == "binance_BTCUSDT_15m.csv"
    assert pathlib.Path(manifest_path).name == "binance_BTCUSDT_15m.manifest.json"
    assert manifest["data_file"] == "binance_BTCUSDT_15m.csv"

    # писатель и читатель обязаны сойтись на одном имени
    assert kline_dataset.manifest_path_for(
        str(tmp_path), "binance", "BTCUSDT", "15m") == manifest_path
    # путь к данным читатель берёт из манифеста, а не из соглашения об имени
    frame = kline_dataset.load_frame(str(tmp_path), "binance", "BTCUSDT", "15m")
    assert frame.data_path == data_path


# --- 17..20: статические гарантии -------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "kline_dataset.py").read_text()


def test_never_uses_failover_market_client():
    for forbidden in ("MarketClient", "analyzer.exchange", "create_market_client"):
        assert forbidden not in SOURCE, forbidden


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


def test_no_network_client_imports():
    # Загрузчик читает файлы; сеть ему не нужна ни в каком виде.
    for forbidden in ("httpx", "BinanceClient", "BybitClient", "requests"):
        assert forbidden not in SOURCE, forbidden


def test_production_runtime_does_not_import_kline_dataset():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))

    for path in runtime_files:
        if not path.exists():
            continue
        assert not re.search(r"\bkline_dataset\b", path.read_text()), \
            f"{path} imports kline_dataset"
