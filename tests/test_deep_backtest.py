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


def _build_random_walk_dataset(outdir, seed: int = OUTCOME_SEED,
                               depths: dict[str, int] | None = None) -> str:
    for tf, n in (depths or DEPTHS).items():
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
    # Токена нет в исходнике вообще — ни как JSON-ключа, ни в прозе комментариев
    # (Codex Low: убрано и из docstring). Прозу пишем как «совет по порогу».
    assert "recommended_threshold" not in SOURCE


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
    assert "walk_forward" not in payload      # single-run: WF key must be absent


def test_cli_text_output_and_failure_exit_code(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)
    outdir = _build_dataset(tmp_path / "cli2")
    assert deep_backtest.main([
        "--dataset", outdir, "--exchange", "binance", "--profile", "intraday",
        "--bars", str(WALK_BARS)]) == 0
    out = capsys.readouterr().out
    assert "deep backtest (C1.3d)" in out
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

    Сравниваются позиции опорных вызовов в исходниках обоих модулей. Это
    структурная растяжка, а НЕ доказательство эквивалентности: она не увидит
    изменённый аргумент, предикат, порог или новый гейт вне списка. Настоящую
    эквивалентность проверяют behavioural-тесты ниже; этот оставлен как дешёвый
    ранний сигнал о переставленном/удалённом гейте.
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


# ---------------------------------------------------------------------------
# Behavioural parity: обе воронки реально ПРОГОНЯЮТСЯ на одних и тех же свечах
# ---------------------------------------------------------------------------
#
# Почему это отдельно от source-order теста: тот сравнивает позиции вызовов в
# тексте и слеп к изменённому аргументу, предикату, порогу или новому гейту.
# Здесь backtest._walk и deep_backtest.deep_walk исполняются на ОДНИХ фреймах,
# и сравниваются их решения.
#
# Выравнивание (иначе тест сравнивал бы разные окна):
#   * n_entry = 342, PARITY_BARS = 42  -> deep.start = max(300, 342-42) = 300
#   * backtest.MAX_BARS = 1200 > 342   -> bt.start   = max(warmup, <0) = 300
#   * warmup=300 == deep_backtest.ENTRY_WARMUP
#   * общий датасет -> общие costs, thresholds и exact-CVD (есть taker_buy_base)
#
# Ограничение, зафиксированное явно: backtest._resolve не имеет горизонта и
# возвращает None для незакрытых сделок (бар молча выпадает из setups), а deep
# различает timeout/unresolved. Поэтому:
#   * гейты и сайзинг сравниваются при ЗАГЛУШЕННОМ resolve (тест A) — так
#     сравнение не зависит от разной семантики разрешения;
#   * R сравнивается на СОПОСТАВИМОМ подмножестве (тест B) — сделки, которые
#     deep закрыл штатно (win/loss/breakeven) внутри горизонта.

PARITY_BARS = 42

# Вспомогательные ТФ намеренно ГЛУБЖЕ минимума aux_depth: при 40-45 барах окна
# HTF_WINDOW=250 и ZONE_WINDOW=160 никогда не «кусаются» (срез и так короче
# окна), и паритетный тест оказался бы слеп к их изменению. Проверено
# мутацией: на мелких фреймах подмена ZONE_WINDOW=100 проходила незамеченной,
# на этих — падает. Длина обхода (PARITY_BARS) при этом не растёт, поэтому
# тест остаётся быстрым.
PARITY_DEPTHS = {"15m": 342, "4h": 300, "2h": 300, "1h": 400, "1d": 220}


def _parity_fixture(tmp_path_factory, name: str, depths: dict[str, int]):
    outdir = tmp_path_factory.mktemp(name)
    with _csv_writer():
        _build_random_walk_dataset(outdir, depths=depths)
        frames, profile, _table, cvd_method = deep_backtest.prepare(
            str(outdir), "binance", "BTCUSDT", "intraday", PARITY_BARS,
            max_gap_ratio=0.001, allow_estimated_cvd=False)
    assert cvd_method == "exact"
    dfs = {tf: f.df for tf, f in frames.items()}
    return frames, dfs, profile


@pytest.fixture(scope="module")
def parity_inputs(tmp_path_factory):
    """Глубокие aux-фреймы: окна HTF_WINDOW/ZONE_WINDOW реально усекают срез."""
    return _parity_fixture(tmp_path_factory, "parity", PARITY_DEPTHS)


@pytest.fixture(scope="module")
def parity_resolved_inputs(tmp_path_factory):
    """Мелкие aux-фреймы (DEPTHS): на них воронка даёт ЗАКРЫТЫЕ сделки.

    Нужна отдельная фикстура: на PARITY_DEPTHS структурный стоп уезжает дальше
    от входа, и внутри 41-барового хвоста истории ни одна сделка не успевает
    закрыться — сравнивать R было бы не на чем. Гейты/срезы это не проверяет
    (для них есть тесты A и C), здесь важен только штатный исход.
    """
    return _parity_fixture(tmp_path_factory, "parity_resolved", DEPTHS)


def test_parity_walk_windows_are_aligned(parity_inputs):
    """Сам тест паритета бессмыслен, если воронки идут по разным барам."""
    import backtest

    _frames, dfs, _profile = parity_inputs
    n = len(dfs["15m"])
    assert n == PARITY_DEPTHS["15m"]
    assert n - PARITY_BARS == deep_backtest.ENTRY_WARMUP        # deep.start
    assert max(deep_backtest.ENTRY_WARMUP, n - backtest.MAX_BARS) == \
        deep_backtest.ENTRY_WARMUP                              # backtest.start
    assert deep_backtest.THRESHOLDS == backtest.THRESHOLDS


def _run_backtest_walk(monkeypatch, dfs, profile, resolve_stub=None):
    """backtest._walk возвращает форматированный текст; перехватываем setups."""
    import backtest

    monkeypatch.setattr(backtest, "_report",
                        lambda setups, *args, **kwargs: setups)
    if resolve_stub is not None:
        monkeypatch.setattr(backtest, "_resolve", resolve_stub)
    return backtest._walk(dfs, profile, warmup=deep_backtest.ENTRY_WARMUP)


def test_deep_walk_gate_decisions_match_backtest_walk(parity_inputs, monkeypatch):
    """ТЕСТ A: одни свечи -> одни и те же бары проходят все гейты.

    resolve заглушён в ОБОИХ путях, поэтому разная семантика разрешения не
    маскирует расхождение гейтов. Сравниваются: индексы прошедших баров,
    направление, счёт и ПОЛНЫЙ dict позиции (entry/stop/TP1/TP2 + сайзинг).
    """
    frames, dfs, profile = parity_inputs

    bt_calls: list[tuple] = []
    dp_calls: list[tuple] = []

    def bt_resolve(df, idx, direction, pos):
        bt_calls.append((idx, direction, dict(pos)))
        return {"outcome": "win", "r": 1.0}

    def dp_resolve(df, idx, direction, pos, hold_bars):
        dp_calls.append((idx, direction, dict(pos)))
        return {"outcome": "win", "r": 1.0, "hit_tp1": False, "exit_idx": idx + 1}

    monkeypatch.setattr(deep_backtest, "resolve", dp_resolve)
    bt_setups = _run_backtest_walk(monkeypatch, dfs, profile, bt_resolve)
    deep = deep_backtest.deep_walk(frames, profile, PARITY_BARS)

    assert bt_calls, "ни один бар не прошёл гейты — тест был бы вакуумным"

    # Гейты: тот же набор баров, то же направление, тот же сайзинг/уровни.
    # Проверено мутацией: HTF_WINDOW 250->200 и ZONE_WINDOW 160->100 роняют
    # это сравнение (на мелких aux-фреймах ZONE_WINDOW проходил незамеченным —
    # отсюда PARITY_DEPTHS).
    assert dp_calls == bt_calls

    # Счёт (weighted_total) и итоговый R (значит, и cost-модель) совпадают.
    bt_rows = [(s["idx"], s["direction"], s["score"], s["r"]) for s in bt_setups]
    dp_rows = [(s["idx"], s["direction"], s["score"], s["r"])
               for s in deep["setups"]]
    assert dp_rows == bt_rows


def test_deep_walk_skips_exactly_the_bars_backtest_skips(parity_inputs, monkeypatch):
    """ТЕСТ A2: знаменатель. Пройдено баров == сетапы + пропуски, и множество
    сетапов совпадает с backtest — то есть ledger не «съел» ни одного бара,
    который backtest бы отторгoвал, и не пропустил ни одного лишнего."""
    frames, dfs, profile = parity_inputs

    def bt_resolve(df, idx, direction, pos):
        return {"outcome": "win", "r": 1.0}

    def dp_resolve(df, idx, direction, pos, hold_bars):
        return {"outcome": "win", "r": 1.0, "hit_tp1": False, "exit_idx": idx + 1}

    monkeypatch.setattr(deep_backtest, "resolve", dp_resolve)
    bt_setups = _run_backtest_walk(monkeypatch, dfs, profile, bt_resolve)
    deep = deep_backtest.deep_walk(frames, profile, PARITY_BARS)

    n = len(dfs["15m"])
    walked = n - 1 - deep_backtest.ENTRY_WARMUP
    assert deep["bars_walked"] == walked
    assert deep["bars_evaluated"] == len(bt_setups)
    assert deep["bars_walked"] == deep["bars_evaluated"] + deep["ledger"].total


def test_deep_walk_resolved_outcomes_match_backtest_on_comparable_subset(
        parity_resolved_inputs, monkeypatch):
    """ТЕСТ B: настоящий resolve в обоих путях.

    Сопоставимое подмножество — сделки, закрытые deep штатно (win/loss/
    breakeven) внутри max_hold_bars. Такая сделка закрылась раньше горизонта,
    значит backtest._resolve (без горизонта) обязан дать ТОТ ЖЕ исход и тот же R.

    Обратное неверно и не проверяется: backtest мог закрыть сделку далеко за
    горизонтом — у deep это timeout, а у баров в хвосте истории — unresolved.
    Это задокументированное расхождение C1.3b, а не дефект.
    """
    frames, dfs, profile = parity_resolved_inputs

    bt_setups = _run_backtest_walk(monkeypatch, dfs, profile)
    deep = deep_backtest.deep_walk(frames, profile, PARITY_BARS)

    bt_by_idx = {s["idx"]: s for s in bt_setups}
    comparable = [s for s in deep["setups"]
                  if s["outcome"] in ("win", "loss", "breakeven")]
    assert comparable, "нет штатно закрытых сделок — сравнивать нечего"

    for s in comparable:
        legacy = bt_by_idx.get(s["idx"])
        assert legacy is not None, (
            f"deep закрыл сетап на баре {s['idx']}, а backtest его потерял")
        assert s["direction"] == legacy["direction"]
        assert s["score"] == legacy["score"]
        assert s["outcome"] == legacy["outcome"]
        assert s["r"] == pytest.approx(legacy["r"])

    # Каждый бар, прошедший гейты у backtest, известен deep — либо как сделка,
    # либо как timeout/unresolved. Потерянных баров нет.
    deep_idx = {s["idx"] for s in deep["setups"]}
    assert set(bt_by_idx) <= deep_idx


def _indicator_input(df) -> tuple[int, int, int]:
    """Отпечаток среза, поданного в compute_indicators: (шаг ТФ, конец, длина).

    Шаг определяется по самим данным, поэтому entry/htf/zone различимы без
    знания, какой модуль их звал.
    """
    open_time = df["open_time"]
    step = int((open_time.iloc[1] - open_time.iloc[0]).total_seconds() * 1000)
    return step, int(open_time.iloc[-1].value), len(df)


def test_deep_walk_feeds_indicators_the_same_slices_as_backtest(parity_inputs,
                                                                monkeypatch):
    """ТЕСТ C: одинаковые ВХОДЫ индикаторов, а не только одинаковые решения.

    Тесты A/B сравнивают решения и могут промолчать, когда изменённая константа
    окна не меняет исход на этой фикстуре (проверено мутацией: ENTRY_WARMUP
    250 vs 300 не сдвинул ни одного сетапа). Здесь сравниваются сами срезы,
    поэтому дрейф ENTRY_WARMUP / HTF_WINDOW / ZONE_WINDOW падает немедленно.

    Сравнение множеств, а не последовательностей: deep кэширует индикаторы
    старших ТФ (_IndicatorCache), поэтому зовёт compute_indicators реже. Срезы
    entry-ТФ не кэшируются никогда, их сравниваем по порядку.
    1D-срезы deep (теневой режим) в backtest отсутствуют и отфильтрованы.

    Проверено мутацией (ENTRY_WARMUP 300->250, HTF_WINDOW 250->200,
    ZONE_WINDOW 160->100 — каждая роняет этот тест).

    Чего он НЕ ловит и почему: HTF_WARMUP (210) и ZONE_MIN_BARS (30) — это
    guard'ы «истории не хватает». На датасете, прошедшем aux_depth, срез старшего
    ТФ гарантированно >= 210 баров даже на первом баре обхода, поэтому подмена
    этих констант не меняет НИ ОДНОГО решения — она недостижима поведенчески.
    Их защищают отдельные тесты: test_refuses_insufficient_aux_depth и счётчик
    htf_insufficient_history в ledger.
    """
    import backtest

    frames, dfs, profile = parity_inputs
    entry_step = MS[profile["entry"]]

    def spy(module, sink):
        real = module.compute_indicators

        def wrapper(df, *args, **kwargs):
            sink.append(_indicator_input(df))
            return real(df, *args, **kwargs)

        monkeypatch.setattr(module, "compute_indicators", wrapper)

    bt_seen: list[tuple] = []
    dp_seen: list[tuple] = []
    spy(backtest, bt_seen)
    spy(deep_backtest, dp_seen)

    def noop_bt(df, idx, direction, pos):
        return {"outcome": "win", "r": 1.0}

    def noop_dp(df, idx, direction, pos, hold_bars):
        return {"outcome": "win", "r": 1.0, "hit_tp1": False, "exit_idx": idx + 1}

    monkeypatch.setattr(deep_backtest, "resolve", noop_dp)
    _run_backtest_walk(monkeypatch, dfs, profile, noop_bt)
    deep_backtest.deep_walk(frames, profile, PARITY_BARS)

    assert bt_seen and dp_seen

    # Entry-ТФ: точная последовательность срезов (300-баровое окно live-паритета).
    bt_entry = [rec for rec in bt_seen if rec[0] == entry_step]
    dp_entry = [rec for rec in dp_seen if rec[0] == entry_step]
    assert dp_entry == bt_entry
    assert {rec[2] for rec in bt_entry} == {deep_backtest.ENTRY_WARMUP}

    # Старшие ТФ: множества срезов (deep кэширует, backtest пересчитывает).
    bt_steps = {rec[0] for rec in bt_seen}
    bt_aux = {rec for rec in bt_seen if rec[0] != entry_step}
    dp_aux = {rec for rec in dp_seen if rec[0] != entry_step and rec[0] in bt_steps}
    assert dp_aux == bt_aux
    # 1D-срез deep существует и в backtest его нет — это теневой режим.
    assert any(rec[0] == MS["1d"] for rec in dp_seen)
    assert not any(rec[0] == MS["1d"] for rec in bt_seen)


def test_deep_walk_parity_survives_a_changed_predicate(parity_inputs, monkeypatch):
    """Мета-тест: паритетная проверка ДЕЙСТВИТЕЛЬНО ловит расхождение.

    Source-order тест слеп к смене порога. Здесь мы меняем порог, который читает
    deep_walk, и требуем, чтобы behavioural-паритет упал. Без этого «зелёный»
    паритет ничего не доказывал бы.
    """
    frames, dfs, profile = parity_inputs

    def bt_resolve(df, idx, direction, pos):
        return {"outcome": "win", "r": 1.0}

    def dp_resolve(df, idx, direction, pos, hold_bars):
        return {"outcome": "win", "r": 1.0, "hit_tp1": False, "exit_idx": idx + 1}

    monkeypatch.setattr(deep_backtest, "resolve", dp_resolve)
    # Порог min(THRESHOLDS) = 5 -> 999: ни один сетап не должен пройти.
    monkeypatch.setattr(deep_backtest, "THRESHOLDS", [999])
    bt_setups = _run_backtest_walk(monkeypatch, dfs, profile, bt_resolve)
    deep = deep_backtest.deep_walk(frames, profile, PARITY_BARS)

    assert bt_setups, "фикстура обязана давать сетапы у backtest"
    assert deep["setups"] == [], "изменённый порог не повлиял — тест слеп"
    assert deep["ledger"].counts["below_min_threshold"] > 0


# ---------------------------------------------------------------------------
# C1.3c: as-of выравнивание aux-фреймов
# ---------------------------------------------------------------------------
#
# aux_depth доказывает счёт, но не то, что бары старших ТФ лежат ДО первого
# пройденного entry-бара. Здесь строим датасет с ВЕРНЫМ счётом, но одним aux-ТФ,
# сдвинутым вперёд по времени: aux_depth его пропускает, а aux_alignment ловит.

# spec идентичен _build_dataset — цены для as-of тестов роли не играют, важны
# только временные метки.
_SHIFT_SPEC = {
    "15m": dict(start=30_000.0, step=0.4, wobble=25.0),
    "1h": dict(start=29_000.0, step=1.5, wobble=30.0),
    "2h": dict(start=28_500.0, step=3.0, wobble=40.0),
    "4h": dict(start=20_000.0, step=25.0, wobble=60.0),
    "1d": dict(start=5_000.0, step=95.0, wobble=120.0),
}


def _build_shifted_dataset(outdir, shift_tf: str, delta_ms: int,
                           depths: dict[str, int] | None = None) -> str:
    """Стандартный intraday-датасет, но shift_tf сдвинут вперёд на delta_ms.

    Сдвиг вперёд означает «этот ТФ покрывает меньше истории, чем entry» — ровно
    старый след limit=500. Счёт баров сохраняется, поэтому aux_depth проходит,
    а as-of покрытие на первом пройденном баре — нет.
    """
    depths = depths or DEPTHS
    for tf, n in depths.items():
        df = _frame(tf, n, taker=True, **_SHIFT_SPEC[tf])
        if tf == shift_tf:
            shift = pd.to_timedelta(delta_ms, unit="ms")
            df["open_time"] = df["open_time"] + shift
            df["close_time"] = df["close_time"] + shift
        kline_cache.write_dataset(df, str(outdir), exchange="binance",
                                  symbol="BTCUSDT", timeframe=tf,
                                  requested_bars=n, duplicate_count_removed=0)
    return str(outdir)


def _loaded_intraday(outdir):
    profile = get_profile("intraday")
    frames = deep_backtest.load_frames(
        outdir, "binance", "BTCUSDT",
        deep_backtest.required_timeframes(profile))
    return frames, profile


def test_walk_start_extracts_backtest_start():
    assert deep_backtest.walk_start(1000, 40) == 1000 - 40
    # хвост короче warmup -> старт упирается в ENTRY_WARMUP
    assert deep_backtest.walk_start(310, 40) == deep_backtest.ENTRY_WARMUP


def test_walk_start_shared_between_walk_and_alignment(dataset):
    """Обход и валидация обязаны считать «первый пройденный бар» одинаково."""
    frames, profile = _loaded_intraday(dataset)
    n = len(frames["15m"].df)
    i0 = deep_backtest.walk_start(n, WALK_BARS)

    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)
    expected_iso = frames["15m"].df["open_time"].iloc[i0].isoformat()
    assert align and all(
        row["first_walked_open_time_iso"] == expected_iso
        for row in align.values())

    # deep_walk стартует с того же индекса (знаменатель = n-1-i0).
    report = deep_backtest.run(dataset, "binance", "BTCUSDT", "intraday", WALK_BARS)
    assert report["bars_walked"] == n - 1 - i0


def test_aux_alignment_excludes_entry_and_passes_on_aligned_dataset(dataset):
    frames, profile = _loaded_intraday(dataset)
    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)
    assert set(align) == {"4h", "2h", "1h", "1d"}   # entry 15m исключён
    assert all(row["ok"] and row["deficit"] == 0 for row in align.values())


def test_aligned_dataset_passes_prepare(dataset):
    frames, profile, _table, cvd = deep_backtest.prepare(
        dataset, "binance", "BTCUSDT", "intraday", WALK_BARS, 0.001, False)
    assert cvd == "exact"
    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)
    assert all(row["ok"] for row in align.values())


def test_time_shifted_htf_is_count_valid_but_fails_alignment(tmp_path):
    """Счёт 4h верный (aux_depth проходит), но as-of покрытие — нет."""
    outdir = _build_shifted_dataset(tmp_path / "htf", "4h", 6 * MS["4h"])
    frames, profile = _loaded_intraday(outdir)

    depth = deep_backtest.aux_depth(profile, WALK_BARS, frames)
    assert depth["4h"]["deficit"] == 0, "счёт обязан быть валидным"

    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)
    row = align["4h"]
    assert row["ok"] is False and row["deficit"] > 0

    with pytest.raises(DeepBacktestError) as exc:
        deep_backtest.prepare(outdir, "binance", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)
    msg = str(exc.value)
    assert "4h" in msg
    assert str(row["required_asof"]) in msg
    assert str(row["available_asof"]) in msg
    assert row["first_walked_open_time_iso"] in msg
    assert "re-fetch all timeframes" in msg


def test_time_shifted_zone_fails_alignment(tmp_path):
    outdir = _build_shifted_dataset(tmp_path / "zone", "1h", 6 * MS["1h"])
    frames, profile = _loaded_intraday(outdir)

    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)
    assert align["1h"]["ok"] is False
    # сдвинут только 1h — остальные zone/htf-фреймы выровнены.
    assert align["4h"]["ok"] is True and align["2h"]["ok"] is True

    with pytest.raises(DeepBacktestError, match="1h"):
        deep_backtest.prepare(outdir, "binance", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)


def test_time_shifted_1d_fails_alignment(tmp_path):
    outdir = _build_shifted_dataset(tmp_path / "d1", "1d", 3 * MS["1d"])
    frames, profile = _loaded_intraday(outdir)

    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)
    assert align["1d"]["ok"] is False

    with pytest.raises(DeepBacktestError, match="1d"):
        deep_backtest.prepare(outdir, "binance", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)


def test_report_contains_aux_alignment(walk_report):
    align = walk_report["aux_alignment"]
    assert set(align) == {"4h", "2h", "1h", "1d"}
    for row in align.values():
        assert set(row) >= {"roles", "required_asof", "available_asof",
                            "first_walked_open_time", "first_walked_open_time_iso",
                            "deficit", "ok", "error"}
        assert row["ok"] is True and row["deficit"] == 0
        assert row["error"] is None


def test_stage_label_is_c13d(walk_report):
    assert walk_report["stage"] == "C1.3d"
    assert "recommended_threshold" not in walk_report


def test_text_report_shows_all_sections(walk_report):
    text = deep_backtest.format_report(walk_report)
    for marker in ("dataset provenance:", "aux depth (count):",
                   "aux as-of alignment", "skip ledger:",
                   "regime disagreement rate", "limitations:"):
        assert marker in text, marker
    assert f"({walk_report['stage']})" in text
    # provenance-строка выводит per-tf бары.
    assert str(DEPTHS["4h"]) in text


# ---------------------------------------------------------------------------
# C1.3c: close_time недоступен -> дефицитная строка, а не исключение (Codex Medium)
# ---------------------------------------------------------------------------
#
# kline_dataset грузит фрейм и БЕЗ close_time (колонка валидируется только если
# присутствует). aux_alignment зовёт _close_ms, который на таком фрейме поднял бы
# DatasetError. Контракт «возвращает таблицу и не бросает» обязан это пережить:
# ТФ без close_time становится строкой ok=false с полем error, и prepare()
# отказывает штатным путём выравнивания.

def _strip_close_time(outdir, tf: str) -> None:
    """Убрать close_time из уже записанного CSV одного ТФ (манифест остаётся
    валидным: actual_bars и границы считаются по open_time)."""
    stem = kline_cache.dataset_stem("binance", "BTCUSDT", tf)
    path = pathlib.Path(outdir) / f"{stem}.csv"
    raw = pd.read_csv(path)
    raw.drop(columns=["close_time"]).to_csv(path, index=False)


def test_aux_alignment_missing_aux_close_time_is_deficit_not_raise(tmp_path):
    outdir = _build_dataset(tmp_path / "noclose")
    _strip_close_time(outdir, "4h")
    frames, profile = _loaded_intraday(outdir)   # грузится: close_time опционален

    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)  # НЕ бросает
    row = align["4h"]
    assert row["ok"] is False
    assert row["error"] and "close_time" in row["error"]
    assert row["available_asof"] == 0
    assert row["deficit"] == row["required_asof"]
    # прочие aux-ТФ (с close_time) остаются выровненными
    assert align["2h"]["ok"] and align["1h"]["ok"] and align["1d"]["ok"]
    assert align["2h"]["error"] is None


def test_prepare_rejects_missing_aux_close_time(tmp_path):
    outdir = _build_dataset(tmp_path / "noclose2")
    _strip_close_time(outdir, "4h")
    frames, profile = _loaded_intraday(outdir)
    expected_iso = deep_backtest.aux_alignment(
        frames, profile, WALK_BARS)["4h"]["first_walked_open_time_iso"]

    with pytest.raises(DeepBacktestError) as exc:
        deep_backtest.prepare(outdir, "binance", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)
    msg = str(exc.value)
    assert "4h" in msg
    assert "close_time" in msg          # подлежащая ошибка вынесена в сообщение
    assert "short" in msg               # required/deficit тоже
    assert expected_iso in msg          # первый пройденный бар


def test_aux_alignment_missing_entry_close_time_flags_all_aux(tmp_path):
    outdir = _build_dataset(tmp_path / "noentryclose")
    _strip_close_time(outdir, "15m")    # entry-якорь t0 становится невычислим
    frames, profile = _loaded_intraday(outdir)

    align = deep_backtest.aux_alignment(frames, profile, WALK_BARS)  # НЕ бросает
    assert set(align) == {"4h", "2h", "1h", "1d"}
    assert all(not row["ok"] and "close_time" in (row["error"] or "")
               for row in align.values())

    with pytest.raises(DeepBacktestError, match="close_time"):
        deep_backtest.prepare(outdir, "binance", "BTCUSDT", "intraday",
                              WALK_BARS, 0.001, False)


def test_cli_exits_2_on_missing_close_time(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)
    outdir = _build_dataset(tmp_path / "cli_noclose")
    _strip_close_time(outdir, "4h")
    rc = deep_backtest.main([
        "--dataset", outdir, "--exchange", "binance", "--profile", "intraday",
        "--bars", str(WALK_BARS)])
    assert rc == 2                       # штатный код, не traceback
    err = capsys.readouterr().err
    assert "deep_backtest:" in err
    assert "close_time" in err


# ---------------------------------------------------------------------------
# C1.3d: walk-forward validation (measurement only, opt-in via --walk-forward)
# ---------------------------------------------------------------------------
#
# Геометрия/purge/embargo проверяются как чистые функции (быстро и точно); один
# настоящий WF-прогон подтверждает end-to-end форму отчёта. purge_bars привязан
# к max_hold_bars(intraday)=96, поэтому для >=3 фолдов нужен более глубокий
# entry-фрейм, чем в single-run тестах.

WF_N15 = 640
WF_BARS = WF_N15 - deep_backtest.ENTRY_WARMUP          # 340; walk_start -> 300
WF_DEPTHS = {"15m": WF_N15, "4h": 240, "2h": 90, "1h": 130, "1d": 220}


def _wf_setups(idxs, *, outcome="win", r=1.0, score=9):
    return [{"idx": i, "score": score, "outcome": outcome, "r": r,
             "regime_current": "range", "regime_live_parity": "range"}
            for i in idxs]


@pytest.fixture(scope="module")
def wf_report(tmp_path_factory):
    """Один настоящий walk-forward прогон по random-walk истории."""
    outdir = tmp_path_factory.mktemp("ds_wf")
    with _csv_writer():
        _build_random_walk_dataset(outdir, depths=WF_DEPTHS)
        return deep_backtest.run(
            str(outdir), "binance", "BTCUSDT", "intraday", WF_BARS,
            wf_overrides=dict(train_bars=80, val_bars=30, embargo_bars=10,
                              holdout_frac=0.08, min_folds=3,
                              min_trades_per_fold=1))


# --- fold geometry (pure functions) ----------------------------------------

def test_fold_windows_rolling_boundaries():
    wf = deep_backtest.WFConfig(train_bars=100, val_bars=50, purge_bars=10,
                                embargo_bars=5, holdout_bars=0, min_folds=1)
    folds = deep_backtest.fold_windows(0, 400, wf)          # gap=15, step=50
    assert [f.index for f in folds] == [0, 1, 2, 3, 4]
    assert (folds[0].train_lo, folds[0].train_hi,
            folds[0].val_lo, folds[0].val_hi) == (0, 100, 115, 165)
    assert (folds[1].train_lo, folds[1].train_hi,
            folds[1].val_lo, folds[1].val_hi) == (50, 150, 165, 215)
    for a, b in zip(folds, folds[1:]):                     # disjoint & adjacent
        assert a.val_hi <= b.val_lo


def test_fold_windows_expanding_boundaries():
    wf = deep_backtest.WFConfig(train_bars=100, val_bars=50, purge_bars=10,
                                embargo_bars=5, holdout_bars=0, min_folds=1,
                                window_mode="expanding")
    folds = deep_backtest.fold_windows(0, 400, wf)
    assert all(f.train_lo == 0 for f in folds)             # train_lo pinned
    assert [f.train_hi for f in folds] == [100, 150, 200, 250, 300]
    assert (folds[0].val_lo, folds[0].val_hi) == (115, 165)


def test_fold_windows_fails_closed_below_min_folds():
    wf = deep_backtest.WFConfig(train_bars=100, val_bars=50, purge_bars=10,
                                embargo_bars=5, holdout_bars=0, min_folds=99)
    with pytest.raises(DeepBacktestError, match="needs >= 99 folds"):
        deep_backtest.fold_windows(0, 400, wf)


def test_fold_windows_holdout_shrinks_region():
    base = deep_backtest.WFConfig(train_bars=60, val_bars=20, purge_bars=10,
                                  embargo_bars=5, holdout_bars=0, min_folds=1)
    held = deep_backtest.WFConfig(train_bars=60, val_bars=20, purge_bars=10,
                                  embargo_bars=5, holdout_bars=120, min_folds=1)
    assert len(deep_backtest.fold_windows(0, 400, held)) < \
        len(deep_backtest.fold_windows(0, 400, base))


def test_embargo_gap_is_respected():
    wf = deep_backtest.WFConfig(train_bars=100, val_bars=50, purge_bars=12,
                                embargo_bars=7, holdout_bars=0, min_folds=1)
    for f in deep_backtest.fold_windows(0, 500, wf):
        assert f.val_lo - f.train_hi == 12 + 7


def test_validation_windows_pairwise_disjoint():
    wf = deep_backtest.WFConfig(train_bars=60, val_bars=20, purge_bars=10,
                                embargo_bars=5, holdout_bars=0, min_folds=2)
    spans = [(f.val_lo, f.val_hi) for f in deep_backtest.fold_windows(0, 400, wf)]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            (lo1, hi1), (lo2, hi2) = spans[i], spans[j]
            assert hi1 <= lo2 or hi2 <= lo1


def test_wfconfig_rejects_step_below_val():
    # step < val сдвигал бы validation-окна с перекрытием и двойным счётом.
    with pytest.raises(DeepBacktestError, match="step_bars must be >= val_bars"):
        deep_backtest.WFConfig(train_bars=60, val_bars=20, purge_bars=10,
                               embargo_bars=5, holdout_bars=0, step_bars=10,
                               min_folds=1)


def test_wfconfig_for_profile_purge_floor_and_embargo():
    profile = get_profile("intraday")
    wf = deep_backtest.WFConfig.for_profile(
        profile, walked_bars=1000, train_bars=100, val_bars=50, purge_bars=1)
    # purge не может опуститься ниже max_hold_bars профиля
    assert wf.purge_bars == deep_backtest.max_hold_bars(profile)
    assert wf.embargo_bars == deep_backtest.cooldown_bars(profile)
    assert wf.step_bars == wf.val_bars                     # step по умолчанию = val


@pytest.mark.parametrize("profile_name,walked", [
    ("swing", 8760), ("position", 8760),
    ("intraday", 70080), ("bounce", 35040),
])
@pytest.mark.parametrize("holdout_frac", [0.0, 0.15])
def test_for_profile_default_sizing_fits_min_folds(profile_name, walked,
                                                   holdout_frac):
    # Регрессия C1.3d.1: дефолтный авто-сайзер (без train/val overrides) обязан
    # давать >= min_folds фолдов. Прежняя формула val = region // (min_folds+2)
    # игнорировала gap = purge + embargo и стабильно давала min_folds - 1, так
    # что дефолтный --walk-forward всегда падал fail-closed на всех профилях.
    profile = get_profile(profile_name)
    wf = deep_backtest.WFConfig.for_profile(
        profile, walked_bars=walked, holdout_frac=holdout_frac, min_folds=3)
    assert wf.val_bars > 0 and wf.train_bars > 0
    assert wf.step_bars >= wf.val_bars                     # непересекающиеся окна
    assert wf.purge_bars >= deep_backtest.max_hold_bars(profile)
    # span_lo=0, span_hi=walked -> usable region = walked - holdout, как в run().
    folds = deep_backtest.fold_windows(0, walked, wf)
    assert len(folds) >= wf.min_folds


def test_default_walk_forward_auto_sizes_without_override(tmp_path):
    # C1.3d.1: дефолтный --walk-forward (без явных train/val) авто-подбирает
    # геометрию и НЕ падает fail-closed — раньше run() поднимал DeepBacktestError.
    outdir = _build_random_walk_dataset(tmp_path / "autowf", depths=WF_DEPTHS)
    report = deep_backtest.run(
        str(outdir), "binance", "BTCUSDT", "intraday", WF_BARS,
        wf_overrides=dict(min_folds=3))
    wf = report["walk_forward"]
    assert len(wf["folds"]) >= 3
    cfg = wf["config"]
    assert cfg["train_bars"] > 0 and cfg["val_bars"] > 0
    assert cfg["step_bars"] >= cfg["val_bars"]


# --- partition / purge ------------------------------------------------------

def test_partition_purges_train_horizon_leak():
    fold = deep_backtest.Fold(0, train_lo=0, train_hi=200, val_lo=250,
                              val_hi=300, purge_bars=40, embargo_bars=10)
    hold = 60
    setups = _wf_setups([100, 190, 199, 260, 305])
    train, val = deep_backtest.partition_setups(setups, fold, hold)
    # 190+60=250>=250 и 199+60>=250 -> покидают train (горизонт залезает в val)
    assert [s["idx"] for s in train] == [100]
    assert [s["idx"] for s in val] == [260]
    assert not ({s["idx"] for s in train} & {s["idx"] for s in val})


# --- per-fold stats / aggregate --------------------------------------------

def test_fold_threshold_stats_unresolved_counted_timeout_included():
    wf = deep_backtest.WFConfig(train_bars=1, val_bars=1, purge_bars=1,
                                embargo_bars=0, holdout_bars=0, min_folds=1,
                                min_trades_per_fold=1)
    setups = [
        {"idx": 0, "score": 9, "outcome": "win", "r": 2.0},
        {"idx": 10, "score": 9, "outcome": "timeout", "r": 0.3},
        {"idx": 20, "score": 9, "outcome": "unresolved", "r": None},
    ]
    row = deep_backtest.fold_threshold_stats(setups, 5, 1, wf)
    assert row["qualified"] == 3
    assert row["trades"] == 2                              # unresolved вне знаменателя
    assert row["unresolved"] == 1
    assert row["including_timeouts"]["trades"] == 2        # timeout включён
    assert row["including_timeouts"]["sum_r"] == pytest.approx(2.3)
    assert row["excluding_timeouts"]["trades"] == 1
    assert row["median_r"] == pytest.approx(1.15)          # median([2.0, 0.3])
    assert row["profit_factor"] is None                    # убытков нет
    assert row["max_drawdown_r"] == 0.0
    assert set(row["confidence_flags"]) == {"low_trades", "high_unresolved",
                                            "high_timeout"}


def test_validation_aggregate_uses_only_confident_folds():
    wf = deep_backtest.WFConfig(train_bars=1, val_bars=1, purge_bars=1,
                                embargo_bars=0, holdout_bars=0, min_folds=1,
                                min_trades_per_fold=2)
    confident = _wf_setups([0, 10, 20], r=1.0)             # 3 трейда -> confident
    sparse = _wf_setups([0], outcome="loss", r=-1.0)       # 1 трейд -> low_trades

    def frow(setups):
        return deep_backtest.fold_threshold_stats(setups, 5, 1, wf)

    rows = [
        {"fold": 0, "train": frow(confident), "validation": frow(confident)},
        {"fold": 1, "train": frow(sparse), "validation": frow(sparse)},
    ]
    agg = deep_backtest._aggregate_validation(rows, wf)
    assert agg["confident_folds"] == 1                     # sparse отброшен
    assert agg["mean_expectancy_r"] == pytest.approx(1.0)
    assert agg["min_expectancy_r"] == pytest.approx(1.0)
    assert agg["sign_consistency"] == pytest.approx(1.0)
    assert "recommended_threshold" not in agg


def test_max_drawdown_and_profit_factor_helpers():
    assert deep_backtest._max_drawdown_r([1.0, -2.0, 0.5]) == pytest.approx(2.0)
    assert deep_backtest._profit_factor([2.0, -1.0, -1.0]) == pytest.approx(1.0)
    assert deep_backtest._profit_factor([1.0, 2.0]) is None   # убытков нет
    assert deep_backtest._max_drawdown_r([]) == 0.0


# --- end-to-end report shape -----------------------------------------------

def test_single_run_has_no_walk_forward(walk_report):
    assert "walk_forward" not in walk_report


def test_walk_forward_key_present_and_stage_is_c13d(wf_report):
    assert wf_report["stage"] == "C1.3d"
    assert "walk_forward" in wf_report
    wf = wf_report["walk_forward"]
    assert set(wf) >= {"config", "folds", "per_threshold_folds",
                       "validation_aggregate", "holdout", "warnings", "limitations"}
    assert len(wf["folds"]) >= 3


def test_walk_forward_all_thresholds_per_fold(wf_report):
    wf = wf_report["walk_forward"]
    thr_keys = {str(t) for t in deep_backtest.THRESHOLDS}
    assert set(wf["per_threshold_folds"]) == thr_keys
    assert set(wf["validation_aggregate"]) == thr_keys
    n_folds = len(wf["folds"])
    for rows in wf["per_threshold_folds"].values():
        assert len(rows) == n_folds
        for r in rows:
            assert set(r["validation"]) >= {"trades", "including_timeouts",
                                            "median_r", "max_drawdown_r",
                                            "profit_factor", "confidence_flags"}


def test_walk_forward_folds_are_purged_and_disjoint(wf_report):
    for f in wf_report["walk_forward"]["folds"]:
        assert f["val_lo"] - f["train_hi"] == f["purge_bars"] + f["embargo_bars"]
        assert f["train_hi"] <= f["val_lo"]


def test_walk_forward_folds_expose_validation_coverage(wf_report):
    # coverage/trades_per_bar/regime_disagreement считаются в fold_stats и ОБЯЗАНЫ
    # попасть в отчёт, а не теряться (Codex Medium).
    for f in wf_report["walk_forward"]["folds"]:
        assert set(f) >= {"val_coverage_bars", "val_trades_per_bar",
                          "val_regime_disagreement_rate"}
        assert f["val_coverage_bars"] == f["val_hi"] - f["val_lo"]
        assert 0.0 <= f["val_regime_disagreement_rate"] <= 1.0


def test_walk_forward_holdout_is_sealed(wf_report):
    h = wf_report["walk_forward"]["holdout"]
    assert h["sealed"] is True
    assert set(h) >= {"idx_lo", "idx_hi", "n_setups"}
    forbidden = {"win_rate", "expectancy_r", "sum_r", "thresholds",
                 "including_timeouts", "profit_factor", "median_r"}
    assert not (forbidden & set(h))


def test_walk_forward_emits_no_recommended_threshold(wf_report):
    blob = json.dumps(wf_report, default=str)
    assert "recommended_threshold" not in blob
    assert "recommended_threshold" not in deep_backtest.format_report(wf_report)
    assert "recommended_threshold" not in wf_report["walk_forward"]


def test_walk_forward_text_report_has_sections(wf_report):
    text = deep_backtest.format_report(wf_report)
    for marker in ("walk-forward (out-of-sample folds):", "folds (entry-bar index):",
                   "validation stability", "holdout (sealed):",
                   "walk-forward limitations:"):
        assert marker in text, marker


def test_cli_walk_forward_adds_section(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(kline_cache, "parquet_available", lambda: False)
    outdir = _build_random_walk_dataset(tmp_path / "cliwf", depths=WF_DEPTHS)
    rc = deep_backtest.main([
        "--dataset", outdir, "--exchange", "binance", "--profile", "intraday",
        "--bars", str(WF_BARS), "--walk-forward", "--wf-train-bars", "80",
        "--wf-val-bars", "30", "--wf-embargo-bars", "10", "--wf-min-folds", "3",
        "--wf-min-trades-per-fold", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "walk_forward" in payload
    assert payload["stage"] == "C1.3d"
    assert len(payload["walk_forward"]["folds"]) >= 3
    assert "recommended_threshold" not in json.dumps(payload, default=str)
