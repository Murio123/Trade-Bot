"""Stage C1.3b — offline deep-backtest core. Никакой сети, только tmp-фикстуры.

Датасеты собираются тем же писателем (tools.kline_cache.write_dataset), которым
их пишет продовый CLI, поэтому тесты проверяют реальный путь загрузки через
tools.kline_dataset, а не подсунутый в обход валидатора DataFrame.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from signal_engine.profiles import get_profile
from tools import deep_backtest, kline_cache
from tools.deep_backtest import DeepBacktestError
from tools.kline_dataset import DatasetError

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools" / "deep_backtest.py").read_text()

# Конец истории, выровненный по суточной сетке: все ТФ закрывают последнюю
# свечу ровно в END, поэтому as-of срезы у всех фреймов согласованы.
END_MS = 1_700_006_400_000  # 2023-11-15T00:00:00Z, кратно 86_400_000
assert END_MS % 86_400_000 == 0

MS = kline_cache.INTERVAL_MS

WALK_BARS = 40

# Глубины подобраны под aux_depth(intraday, WALK_BARS) с небольшим запасом.
DEPTHS = {"15m": 342, "4h": 215, "2h": 40, "1h": 45, "1d": 213}


# ---------------------------------------------------------------------------
# Синтетические фреймы
# ---------------------------------------------------------------------------

def _series(n: int, *, start: float, step: float, wobble: float) -> list[float]:
    """Детерминированный ряд: линейный дрейф + ограниченная «пила»."""
    return [start + i * step + wobble * math.sin(i * 1.7) for i in range(n)]


def _frame(timeframe: str, n: int, *, taker: bool, start: float, step: float,
           wobble: float) -> pd.DataFrame:
    interval = MS[timeframe]
    opens = [END_MS - interval * k for k in range(n, 0, -1)]
    closes = _series(n, start=start, step=step, wobble=wobble)
    df = pd.DataFrame({
        "open_time": pd.to_datetime(opens, unit="ms", utc=True),
        "open": [c - 1.0 for c in closes],
        "high": [c + 12.0 for c in closes],
        "low": [c - 12.0 for c in closes],
        "close": closes,
        "volume": [10.0 + (i % 7) for i in range(n)],
        "quote_volume": [1000.0] * n,
    })
    if taker:
        df["taker_buy_base"] = [5.0 + (i % 3) for i in range(n)]
    df["close_time"] = df["open_time"] + pd.to_timedelta(interval, unit="ms")
    return df


def _build_dataset(outdir, *, exchange: str = "binance", taker: bool = True,
                   depths: dict[str, int] | None = None) -> str:
    """Полный набор ТФ для профиля intraday: entry 15m, htf/zone 4h/2h/1h, 1d.

    1D — устойчивый аптренд (price > ema20 > ema50 > ema200), чтобы
    regime_live_parity расходился с regime_current, который у intraday всегда
    "range" (D13).
    """
    depths = depths or DEPTHS
    spec = {
        "15m": dict(start=30_000.0, step=0.4, wobble=25.0),
        "1h": dict(start=29_000.0, step=1.5, wobble=30.0),
        "2h": dict(start=28_500.0, step=3.0, wobble=40.0),
        "4h": dict(start=20_000.0, step=25.0, wobble=60.0),
        "1d": dict(start=5_000.0, step=95.0, wobble=120.0),
    }
    for tf, n in depths.items():
        df = _frame(tf, n, taker=taker, **spec[tf])
        kline_cache.write_dataset(df, str(outdir), exchange=exchange,
                                  symbol="BTCUSDT", timeframe=tf,
                                  requested_bars=n, duplicate_count_removed=0)
    return str(outdir)


TF_MINUTES = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def _random_walk_frame(timeframe: str, n: int, seed: int, *, drift: float = 0.3,
                       vol: float = 0.5, start: float = 30_000.0) -> pd.DataFrame:
    """Сидированный random-walk с настоящими тенями свечей.

    Синусоидальный ряд из ``_frame`` слишком вырожден, чтобы пройти
    has_diverse_confirmation: на нём почти не срабатывают анализаторы. Здесь
    свечи «шумят» как рынок, поэтому воронка доходит до сделок.
    """
    frac = TF_MINUTES[timeframe] / 1440.0
    rng = np.random.default_rng([seed, TF_MINUTES[timeframe]])
    closes = start * np.exp(np.cumsum(rng.normal(drift * frac,
                                                 vol * np.sqrt(frac), n)))
    opens = np.concatenate([[start], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = np.abs(rng.normal(100, 20, n))

    interval = MS[timeframe]
    open_times = [END_MS - interval * k for k in range(n, 0, -1)]
    df = pd.DataFrame({
        "open_time": pd.to_datetime(open_times, unit="ms", utc=True),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volume, "quote_volume": volume * closes,
        "taker_buy_base": volume * rng.uniform(0.3, 0.7, n),
    })
    df["close_time"] = df["open_time"] + pd.to_timedelta(interval, unit="ms")
    return df


# Сид подобран так, чтобы воронка дала и закрытые сделки, и unresolved-хвост.
OUTCOME_SEED = 6


def _build_random_walk_dataset(outdir, seed: int = OUTCOME_SEED) -> str:
    for tf, n in DEPTHS.items():
        kline_cache.write_dataset(_random_walk_frame(tf, n, seed), str(outdir),
                                  exchange="binance", symbol="BTCUSDT",
                                  timeframe=tf, requested_bars=n,
                                  duplicate_count_removed=0)
    return str(outdir)


@pytest.fixture(autouse=True)
def force_csv(monkeypatch):
    """Детерминированная ветка записи: pyarrow может присутствовать в окружении."""
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)


@pytest.fixture
def dataset(tmp_path):
    return _build_dataset(tmp_path / "binance")


@pytest.fixture(scope="module")
def _intraday_profile():
    return get_profile("intraday")


# ---------------------------------------------------------------------------
# 1..4: статические гарантии (offline-инструмент, не runtime)
# ---------------------------------------------------------------------------

def test_production_runtime_does_not_import_deep_backtest():
    runtime_dirs = ["bot", "signal_engine", "risk", "contracts", "analyzer", "ai"]
    runtime_files = [ROOT / f for f in ("main.py", "pipeline.py", "scheduler.py",
                                        "backtest.py", "database.py", "config.py",
                                        "unified_context.py", "forecast_lifecycle.py")]
    for d in runtime_dirs:
        runtime_files.extend((ROOT / d).rglob("*.py"))

    for path in runtime_files:
        if not path.exists():
            continue
        assert not re.search(r"\bdeep_backtest\b", path.read_text()), \
            f"{path} imports deep_backtest"


def test_never_uses_failover_market_client():
    for forbidden in ("MarketClient", "analyzer.exchange", "create_market_client",
                      "BinanceClient", "BybitClient", "httpx", "requests"):
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
                      "api_key", "api_secret", "exchange.create", "place_order"):
        assert forbidden not in SOURCE, forbidden


# ---------------------------------------------------------------------------
# 5..8: датасет, aux-depth, CVD
# ---------------------------------------------------------------------------

def test_requires_cached_dataset_through_kline_dataset(tmp_path):
    with pytest.raises(DatasetError, match="manifest not found"):
        deep_backtest.prepare(str(tmp_path), "binance", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)


def test_refuses_insufficient_aux_depth(dataset):
    # 15m-фрейма хватает ровно на WALK_BARS; вдвое больший обход упирается в
    # глубину старших ТФ (D14) — отказ, а не тихий пропуск баров.
    with pytest.raises(DeepBacktestError, match="insufficient dataset depth"):
        deep_backtest.prepare(dataset, "binance", "BTCUSDT", "intraday",
                              WALK_BARS * 10, 0.001, False)


def test_refuses_estimated_cvd_without_flag(tmp_path):
    outdir = _build_dataset(tmp_path / "bybit", exchange="bybit", taker=False)
    with pytest.raises(DeepBacktestError, match="taker_buy_base"):
        deep_backtest.prepare(outdir, "bybit", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)


def test_allows_estimated_cvd_only_with_explicit_flag(tmp_path):
    outdir = _build_dataset(tmp_path / "bybit", exchange="bybit", taker=False)
    _frames, _profile, _table, cvd_method = deep_backtest.prepare(
        outdir, "bybit", "BTCUSDT", "intraday", WALK_BARS, 0.001,
        allow_estimated_cvd=True)
    assert cvd_method == "estimate"


def test_exact_cvd_on_binance_dataset(dataset):
    *_, cvd_method = deep_backtest.prepare(dataset, "binance", "BTCUSDT",
                                           "intraday", WALK_BARS, 0.001, False)
    assert cvd_method == "exact"


def test_rejects_nonpositive_bars(dataset):
    with pytest.raises(DeepBacktestError, match="--bars must be positive"):
        deep_backtest.prepare(dataset, "binance", "BTCUSDT", "intraday", 0,
                              0.001, False)


# ---------------------------------------------------------------------------
# Один настоящий обход, переиспользуемый несколькими тестами (он не бесплатный)
# ---------------------------------------------------------------------------

@contextmanager
def _csv_writer():
    """module-scoped фикстуры не видят function-scoped monkeypatch."""
    original = kline_cache.parquet_available
    kline_cache.parquet_available = lambda: False
    try:
        yield
    finally:
        kline_cache.parquet_available = original


@pytest.fixture(scope="module")
def walk_report(tmp_path_factory):
    """Обход по детерминированному ряду: чистые теги режима, сделок нет."""
    outdir = tmp_path_factory.mktemp("ds")
    with _csv_writer():
        _build_dataset(outdir)
        return deep_backtest.run(str(outdir), "binance", "BTCUSDT", "intraday",
                                 WALK_BARS)


@pytest.fixture(scope="module")
def outcome_report(tmp_path_factory):
    """Обход по random-walk: воронка доходит до сделок и до unresolved-хвоста."""
    outdir = tmp_path_factory.mktemp("ds_rw")
    with _csv_writer():
        _build_random_walk_dataset(outdir)
        return deep_backtest.run(str(outdir), "binance", "BTCUSDT", "intraday",
                                 WALK_BARS)


# ---------------------------------------------------------------------------
# 9..10: skip ledger
# ---------------------------------------------------------------------------

def test_skip_ledger_invariant_holds(walk_report):
    ledger = walk_report["skip_ledger"]
    assert walk_report["bars_walked"] > 0
    assert walk_report["bars_walked"] == (
        walk_report["bars_evaluated"] + ledger["total"])
    assert ledger["total"] == sum(ledger["counts"].values())


def test_bars_walked_is_the_denominator(walk_report):
    # Знаменатель — реально пройденные бары, а не (n - start) из backtest.
    n_entry = walk_report["dataset"]["15m"]["actual_bars"]
    start = max(300, n_entry - WALK_BARS)
    assert walk_report["bars_walked"] == n_entry - 1 - start


def test_every_skip_reason_has_a_call_site():
    """Каждая причина достижима: у неё есть ровно свой вызов ledger.add в воронке."""
    for reason in deep_backtest.SKIP_REASONS:
        assert re.search(rf'\.add\(\s*"{reason}"', SOURCE), reason


def test_skip_ledger_examples_are_capped_and_shaped():
    ledger = deep_backtest.SkipLedger()
    ts = pd.Timestamp("2023-01-01T00:00:00Z")
    for i in range(deep_backtest.SKIP_EXAMPLES_MAX + 3):
        ledger.add("no_atr", i, ts, details=f"i={i}")
    assert ledger.counts["no_atr"] == deep_backtest.SKIP_EXAMPLES_MAX + 3
    examples = ledger.examples["no_atr"]
    assert len(examples) == deep_backtest.SKIP_EXAMPLES_MAX
    assert examples[0] == {"idx": 0, "open_time_iso": ts.isoformat(),
                           "details": "i=0"}
    with pytest.raises(KeyError):
        ledger.add("not_a_reason", 0, ts)


def test_skip_ledger_counts_cover_all_reasons(walk_report):
    assert set(walk_report["skip_ledger"]["counts"]) == set(deep_backtest.SKIP_REASONS)


# ---------------------------------------------------------------------------
# 11..12: timeout / unresolved
# ---------------------------------------------------------------------------

def _flat_df(n: int, price: float = 101.0) -> pd.DataFrame:
    """Плоские свечи ВЫШЕ entry (100) и ниже стопа: сами по себе они не
    закрывают сделку, поэтому каждый тест ниже задаёт ровно одно событие."""
    return pd.DataFrame({"high": [price + 0.5] * n, "low": [price - 0.5] * n,
                         "close": [price] * n})


POS = {"entry_price": 100.0, "stop_loss": 90.0, "target_1": 110.0,
       "target_2": 130.0}


def test_timeout_marks_to_market_at_horizon():
    df = _flat_df(20)
    df.loc[5, "close"] = 105.0  # close бара горизонта (entry_idx=0, hold=5)
    out = deep_backtest.resolve(df, 0, "long", POS, hold_bars=5)
    assert out["outcome"] == "timeout"
    assert out["exit_idx"] == 5
    # risk = 10, +5 пунктов -> +0.5R (до вычета издержек)
    assert out["r"] == pytest.approx(0.5)
    assert out["hit_tp1"] is False


def test_timeout_reports_hit_tp1_flag():
    df = _flat_df(20)
    df.loc[2, "high"] = 111.0    # TP1
    df.loc[5, "close"] = 102.0
    out = deep_backtest.resolve(df, 0, "long", POS, hold_bars=5)
    assert out["outcome"] == "timeout"
    assert out["hit_tp1"] is True
    assert out["r"] == pytest.approx(0.2)


def test_timeout_mark_to_market_can_be_negative_for_short():
    df = _flat_df(20)
    df.loc[3, "close"] = 104.0
    pos = {"entry_price": 100.0, "stop_loss": 110.0, "target_1": 90.0,
           "target_2": 70.0}
    out = deep_backtest.resolve(df, 0, "short", pos, hold_bars=3)
    assert out["outcome"] == "timeout"
    assert out["r"] == pytest.approx(-0.4)


def test_unresolved_when_future_data_ends_before_horizon():
    df = _flat_df(4)
    out = deep_backtest.resolve(df, 0, "long", POS, hold_bars=10)
    assert out["outcome"] == "unresolved"
    assert out["r"] is None
    assert out["exit_idx"] is None


def test_stop_tp1_tp2_semantics_match_backtest():
    # stop
    df = _flat_df(10)
    df.loc[1, "low"] = 89.0
    assert deep_backtest.resolve(df, 0, "long", POS, 5)["outcome"] == "loss"

    # TP1 затем возврат к entry -> breakeven (стоп-в-БУ действует со след. свечи)
    df = _flat_df(10)
    df.loc[1, "high"] = 111.0
    df.loc[2, "low"] = 99.0
    out = deep_backtest.resolve(df, 0, "long", POS, 5)
    assert out["outcome"] == "breakeven" and out["r"] == 0.0

    # TP1 на свече 1 не закрывает сделку в 0R сам по себе, даже если та же
    # свеча заходит под entry: брейкивен-стоп действует со СЛЕДУЮЩЕЙ.
    df = _flat_df(10)
    df.loc[1, "high"] = 111.0
    df.loc[1, "low"] = 99.0
    df.loc[3, "high"] = 131.0
    out = deep_backtest.resolve(df, 0, "long", POS, 5)
    assert out["outcome"] == "win" and out["r"] == pytest.approx(3.0)


def test_unresolved_excluded_from_aggregates_but_counted():
    setups = [
        {"idx": 0, "score": 9, "outcome": "win", "r": 2.0},
        {"idx": 100, "score": 9, "outcome": "loss", "r": -1.0},
        {"idx": 200, "score": 9, "outcome": "timeout", "r": 0.3},
        {"idx": 300, "score": 9, "outcome": "unresolved", "r": None},
    ]
    counts = deep_backtest.outcome_counts(setups)
    assert counts == {"win": 1, "loss": 1, "breakeven": 0, "timeout": 1,
                      "unresolved": 1}

    row = deep_backtest.threshold_stats(setups, 5, cooldown_bars=1)
    assert row["qualified"] == 4
    assert row["trades"] == 3                    # unresolved вне знаменателя
    assert row["unresolved"] == 1
    assert row["unresolved_share"] == pytest.approx(0.25)
    assert row["timeout_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["including_timeouts"]["trades"] == 3
    assert row["including_timeouts"]["sum_r"] == pytest.approx(1.3)
    assert row["excluding_timeouts"]["trades"] == 2
    assert row["excluding_timeouts"]["sum_r"] == pytest.approx(1.0)
    assert row["excluding_timeouts"]["win_rate"] == pytest.approx(0.5)


def test_full_walk_reaches_resolution_and_keeps_the_invariant(outcome_report):
    """Сквозной прогон, который РЕАЛЬНО доходит до сделок (иначе инвариант
    bars_walked == bars_evaluated + skips выполнялся бы вырожденно, при
    bars_evaluated == 0)."""
    counts = outcome_report["outcome_counts"]
    assert outcome_report["bars_evaluated"] > 0
    assert sum(counts[o] for o in deep_backtest.RESOLVED_OUTCOMES) == \
        outcome_report["bars_evaluated"]
    assert outcome_report["bars_walked"] == (
        outcome_report["bars_evaluated"] + outcome_report["skip_ledger"]["total"])


def test_full_walk_produces_unresolved_tail_counted_in_ledger(outcome_report):
    """Хвост истории короче max_hold_bars -> unresolved, и он ИМЕННО в ledger.

    Это не дефект фикстуры: последние max_hold_bars баров любого датасета не
    могут быть разрешены, и молча выкинуть их (как делал backtest._resolve,
    возвращая None) — значит потерять знаменатель.
    """
    counts = outcome_report["outcome_counts"]
    ledger = outcome_report["skip_ledger"]
    assert counts["unresolved"] > 0
    assert ledger["counts"]["unresolved_no_future_data"] == counts["unresolved"]
    # ...и не попал в bars_evaluated.
    assert outcome_report["bars_evaluated"] == sum(
        counts[o] for o in deep_backtest.RESOLVED_OUTCOMES)
    examples = ledger["examples"]["unresolved_no_future_data"]
    assert examples and set(examples[0]) >= {"idx", "open_time_iso"}


def test_full_walk_threshold_table_excludes_unresolved_from_trades(outcome_report):
    rows = {r["threshold"]: r for r in outcome_report["thresholds"]}
    row = rows[min(deep_backtest.THRESHOLDS)]
    assert row["qualified"] == row["trades"] + row["unresolved"]
    assert row["including_timeouts"]["trades"] == row["trades"]
    assert row["trades"] > 0
    # Издержки вычтены: R закрытых сделок не равен «сырому» -1.0 ровно.
    assert row["including_timeouts"]["sum_r"] < 0


def test_costs_are_subtracted_from_resolved_r(outcome_report):
    """На этом сиде все закрытые сделки — стопы; net-of-costs каждая ХУЖЕ -1R.

    Round-trip издержки (2 x TAKER_FEE_PCT + SLIPPAGE_PCT) вычитаются в R,
    поэтому «чистый» -1.0 недостижим. Проверка ловит потерю строки cost_r.
    """
    rows = {r["threshold"]: r for r in outcome_report["thresholds"]}
    stats = rows[min(deep_backtest.THRESHOLDS)]["including_timeouts"]
    assert stats["win_rate"] == 0.0
    assert stats["expectancy_r"] < -1.0


def test_max_hold_bars_from_forecast_horizon(_intraday_profile):
    assert deep_backtest.max_hold_bars(_intraday_profile) == math.ceil(24 / 0.25)
    assert deep_backtest.max_hold_bars(get_profile("swing")) == math.ceil(96 / 4)
    assert deep_backtest.max_hold_bars(get_profile("position")) == math.ceil(336 / 4)
    assert deep_backtest.max_hold_bars(get_profile("bounce")) == 72


def test_report_max_hold_bars_matches_profile(walk_report, _intraday_profile):
    assert walk_report["max_hold_bars"] == deep_backtest.max_hold_bars(_intraday_profile)


# ---------------------------------------------------------------------------
# 13..15: двойной тег режима
# ---------------------------------------------------------------------------

def test_regime_current_uses_backtest_semantics_and_is_range_for_intraday(
        _intraday_profile):
    """D13: у intraday htf == 4h -> в detect_regime уходит None -> всегда range.

    Даже если 4H-индикаторы описывают явный аптренд.
    """
    trending_4h = {"price": 200.0, "ema20": 190.0, "ema50": 180.0, "ema200": 170.0}
    assert deep_backtest.regime_current(_intraday_profile, trending_4h) == "range"

    swing = get_profile("swing")  # htf == 1d -> индикаторы читаются
    assert deep_backtest.regime_current(swing, trending_4h) == "trend_up"


def test_regime_live_parity_is_computed_from_1d():
    trending_1d = {"price": 200.0, "ema20": 190.0, "ema50": 180.0, "ema200": 170.0}
    assert deep_backtest.regime_live_parity(trending_1d, None) == "trend_up"
    # 1D-волатильность — часть live-семантики.
    assert deep_backtest.regime_live_parity(
        trending_1d, {"atr_percentile": 95}) == "high_volatility"
    # Нехватка 1D-истории не притворяется диапазоном.
    assert deep_backtest.regime_live_parity(None, None) == "unavailable"
    assert deep_backtest.REGIME_UNAVAILABLE == "unavailable"


def test_walk_regime_current_is_always_range_for_intraday(walk_report):
    assert list(walk_report["regime_confusion_matrix"]) == ["range"]


def test_walk_regime_live_parity_disagrees_and_is_reported(walk_report):
    matrix = walk_report["regime_confusion_matrix"]
    parity_tags = set(matrix["range"])
    assert parity_tags - {"range"}, (
        "1D-фрейм построен как аптренд: shadow-режим обязан отличаться от range")
    assert walk_report["regime_disagreement_rate"] > 0
    assert walk_report["regime_samples"] == sum(matrix["range"].values())


def test_disagreement_rate_and_matrix_helpers():
    pairs = [("range", "trend_up"), ("range", "range"), ("range", "trend_up")]
    assert deep_backtest.confusion_matrix(pairs) == {
        "range": {"trend_up": 2, "range": 1}}
    assert deep_backtest.disagreement_rate(pairs) == pytest.approx(2 / 3)
    assert deep_backtest.disagreement_rate([]) == 0.0


def test_scoring_uses_regime_current_not_live_parity(tmp_path, monkeypatch):
    """Spy: weighted_total обязан видеть ТОЛЬКО regime_current.

    Если бы D13 «починили» походя, сюда приехал бы trend_up с 1D-фрейма и
    веса категорий (а значит и пороги) поехали бы молча.
    """
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)
    outdir = _build_dataset(tmp_path / "spy")

    seen: list[str] = []
    real = deep_backtest.weighted_total

    def spy(scores, regime):
        seen.append(regime)
        return real(scores, regime)

    monkeypatch.setattr(deep_backtest, "weighted_total", spy)
    report = deep_backtest.run(outdir, "binance", "BTCUSDT", "intraday", WALK_BARS)

    assert seen, "воронка не дошла до скоринга — тест ничего не доказывает"
    assert set(seen) == {"range"}, seen

    parity_tags = set(report["regime_confusion_matrix"]["range"])
    shadow_only = parity_tags - {"range"}
    assert shadow_only, "нужен расходящийся shadow-тег, иначе spy тавтологичен"
    assert not (shadow_only & set(seen)), (
        f"regime_live_parity {shadow_only} утёк в скоринг")


# ---------------------------------------------------------------------------
# 16..18: отчёт
# ---------------------------------------------------------------------------

def test_report_has_no_recommended_threshold(walk_report):
    assert "recommended_threshold" not in walk_report
    assert "recommended_threshold" not in json.dumps(walk_report, default=str)
    # Ключа нет и в исходнике: единственное вхождение слова — в docstring и в
    # limitations, где оно объясняет, почему рекомендации НЕТ.
    assert not re.search(r'"recommended_threshold"', SOURCE)


def test_report_contains_required_fields(walk_report):
    required = {"profile", "symbol", "exchange", "bars_requested", "bars_walked",
                "cvd_method", "max_hold_bars", "dataset", "aux_depth",
                "skip_ledger", "regime_confusion_matrix",
                "regime_disagreement_rate", "outcome_counts", "thresholds",
                "limitations"}
    assert required <= set(walk_report)
    assert walk_report["profile"] == "intraday"
    assert walk_report["exchange"] == "binance"
    assert walk_report["symbol"] == "BTCUSDT"
    assert walk_report["bars_requested"] == WALK_BARS
    assert set(walk_report["outcome_counts"]) == set(deep_backtest.OUTCOMES)


def test_dataset_provenance_per_timeframe(walk_report):
    assert set(walk_report["dataset"]) == set(DEPTHS)
    for tf, row in walk_report["dataset"].items():
        assert row["actual_bars"] == DEPTHS[tf]
        assert row["gap_count"] == 0 and row["missing_bars"] == 0
        assert row["first_open_time"] < row["last_open_time"]
        assert row["has_taker_buy_base"] is True


def test_aux_depth_table_reports_no_deficit(walk_report):
    for tf, row in walk_report["aux_depth"].items():
        assert row["ok"] is True, (tf, row)
        assert row["deficit"] == 0
        assert row["available"] == DEPTHS[tf]


def test_threshold_rows_cover_all_thresholds(walk_report):
    assert [r["threshold"] for r in walk_report["thresholds"]] == \
        deep_backtest.THRESHOLDS
    for row in walk_report["thresholds"]:
        assert set(row) >= {"trades", "unresolved_share", "timeout_share",
                            "including_timeouts", "excluding_timeouts"}
        for block in ("including_timeouts", "excluding_timeouts"):
            assert set(row[block]) == {"trades", "win_rate", "expectancy_r", "sum_r"}


def test_thresholds_match_backtest_thresholds():
    import backtest
    assert deep_backtest.THRESHOLDS == backtest.THRESHOLDS


def test_limitations_block_present(walk_report):
    text = " ".join(walk_report["limitations"]).lower()
    assert len(walk_report["limitations"]) >= 5
    assert "walk-forward" in text
    assert "recommended threshold" in text
    assert "shadow" in text
    assert "funding" in text
    assert "stop" in text


def test_cli_json_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)
    outdir = _build_dataset(tmp_path / "cli")
    rc = deep_backtest.main([
        "--dataset", outdir, "--exchange", "binance", "--symbol", "BTCUSDT",
        "--profile", "intraday", "--bars", str(WALK_BARS), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cvd_method"] == "exact"
    assert payload["bars_walked"] == payload["bars_evaluated"] + \
        payload["skip_ledger"]["total"]
    assert "recommended_threshold" not in payload


def test_cli_text_output_and_failure_exit_code(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)
    outdir = _build_dataset(tmp_path / "cli2")
    assert deep_backtest.main([
        "--dataset", outdir, "--exchange", "binance", "--profile", "intraday",
        "--bars", str(WALK_BARS)]) == 0
    out = capsys.readouterr().out
    assert "deep backtest (C1.3b)" in out
    assert "skip ledger:" in out
    assert "limitations:" in out
    # Слово встречается только в отрицающей формулировке блока limitations.
    assert "No recommended threshold is produced." in out
    assert not re.search(r"recommended(?! threshold is produced)", out)

    # Отсутствующий датасет -> ненулевой код возврата, а не traceback.
    assert deep_backtest.main([
        "--dataset", str(tmp_path / "missing"), "--exchange", "binance",
        "--profile", "intraday", "--bars", "10"]) == 2
    assert "deep_backtest:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Паритет с backtest._walk
# ---------------------------------------------------------------------------

def test_deep_walk_gate_order_matches_backtest_walk():
    """Порядок гейтов — контракт C1.3b: deep_backtest ИЗМЕРЯЕТ backtest, не чинит.

    Сравниваются позиции опорных вызовов в исходниках обоих модулей.
    """
    bt = (ROOT / "backtest.py").read_text()
    gates = ["compute_indicators", "get_htf_bias", "_build_htf_zones",
             "compute_equilibrium", "detect_liquidity", "detect_reversal",
             "detect_divergence", "compute_cvd_from_klines",
             "calculate_confluence_score", "weighted_total", "apply_htf_policy",
             "has_diverse_confirmation", "dead_zone", "abnormal_volatility",
             "calculate_position", "_structural_stop", "_structure_targets",
             "effective_expected_move"]

    def order(text: str) -> list[str]:
        found = [(text.index(g + "("), g) for g in gates if g + "(" in text]
        return [g for _, g in sorted(found)]

    assert order(SOURCE) == order(bt)


def test_deep_walk_resolution_matches_backtest_resolve_when_horizon_is_long():
    """С горизонтом длиннее истории deep.resolve повторяет backtest._resolve.

    Единственное расхождение — имя исхода для «не закрылось»: backtest вернул бы
    None, deep возвращает unresolved. Всё остальное обязано совпадать.
    """
    import backtest

    cases = []
    df = _flat_df(12); df.loc[2, "low"] = 89.0; cases.append(df)
    df = _flat_df(12); df.loc[2, "high"] = 111.0; df.loc[4, "low"] = 99.0
    cases.append(df)
    df = _flat_df(12); df.loc[2, "high"] = 111.0; df.loc[5, "high"] = 131.0
    cases.append(df)
    cases.append(_flat_df(12))

    for df in cases:
        legacy = backtest._resolve(df, 0, "long", POS)
        deep = deep_backtest.resolve(df, 0, "long", POS, hold_bars=1000)
        if legacy is None:
            assert deep["outcome"] == "unresolved"
        else:
            assert deep["outcome"] == legacy["outcome"]
            assert deep["r"] == pytest.approx(legacy["r"])
