"""Stage 13B: pure offline counterfactual core для non-ENTER прогнозов.

Доказывает: гипотетический исход считается честно (пропуски не становятся
win/loss), результат помечен hypothetical, вход не мутируется, вывод
детерминирован, семантика совпадает с analyzer.outcomes.measure_outcome, а сам
модуль offline (нет импортов БД/scheduler/pipeline/сети).
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from analyzer.counterfactual import (
    EVALUATED, NO_DIRECTION, NO_LEVELS, UNRESOLVED, UNSUPPORTED,
    counterfactual_outcome,
)
from analyzer.outcomes import measure_outcome

UTC = timezone.utc
REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "analyzer" / "counterfactual.py"


def _klines(start: datetime, hours: int, price: float, step: float = 0.0,
            spread: float = 50.0) -> pd.DataFrame:
    rows = []
    p = price
    for i in range(hours):
        ot = start + timedelta(hours=i)
        rows.append({"open_time": pd.Timestamp(ot), "open": p,
                     "high": p + spread, "low": p - spread, "close": p + step,
                     "close_time": pd.Timestamp(ot + timedelta(hours=1))})
        p += step
    return pd.DataFrame(rows)


def _fc(**kw):
    base = {
        "id": 1,
        "analysis_status": "NO_TRADE",
        "candidate_direction": "long",
        "decision_time": datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
        "executable_price_at_decision": 100_000.0,
        "signal_close_price": 100_000.0,
        "stop_loss": 95_000.0,
        "take_profit_levels": [101_000.0, 106_000.0],
    }
    base.update(kw)
    return base


ANCHOR = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
FAR_NOW = ANCHOR + timedelta(hours=80)


# --- 1. no_direction --------------------------------------------------------

def test_missing_direction_is_no_direction():
    fc = _fc(candidate_direction=None)
    res = counterfactual_outcome(fc, _klines(ANCHOR, 80, 100_000.0), FAR_NOW)
    assert res["status"] == NO_DIRECTION
    assert res["classification_hint"] == "none"


# --- 2. no_levels -----------------------------------------------------------

def test_missing_stop_is_no_levels():
    fc = _fc(stop_loss=None)
    res = counterfactual_outcome(fc, _klines(ANCHOR, 80, 100_000.0), FAR_NOW)
    assert res["status"] == NO_LEVELS
    assert res["classification_hint"] == "no_levels"


def test_missing_tp_is_no_levels():
    fc = _fc(take_profit_levels=[])
    res = counterfactual_outcome(fc, _klines(ANCHOR, 80, 100_000.0), FAR_NOW)
    assert res["status"] == NO_LEVELS


# --- 3. avoided_loss (hypothetical clean stop) ------------------------------

def test_hypothetical_stop_is_avoided_loss():
    # Цена падает: стоп 95000 выбит, TP1 101000 недостижим -> чистый стоп.
    df = _klines(ANCHOR, 80, 100_000.0, step=-1_000.0)
    res = counterfactual_outcome(_fc(), df, FAR_NOW)
    assert res["status"] == EVALUATED
    assert res["stop_hit"] is True and res["tp1_hit"] is False
    assert res["classification_hint"] == "avoided_loss"
    assert res["realized_r"] is None  # non-ENTER: R не считается


# --- 4. missed_opportunity (hypothetical TP) --------------------------------

def test_hypothetical_tp_is_missed_opportunity():
    # Цена растёт: TP2 106000 достигнут, стоп не тронут.
    df = _klines(ANCHOR, 80, 100_000.0, step=1_000.0)
    res = counterfactual_outcome(_fc(), df, FAR_NOW)
    assert res["status"] == EVALUATED
    assert res["tp2_hit"] is True and res["stop_hit"] is False
    assert res["classification_hint"] == "missed_opportunity"


# --- 5. unresolved / censored -----------------------------------------------

def test_horizon_not_complete_is_unresolved():
    anchor = datetime.now(UTC) - timedelta(hours=2)
    fc = _fc(decision_time=anchor)
    # Плоские свечи: ни TP, ни стоп; 72h не прошли -> censored.
    df = _klines(anchor, 3, 100_000.0, step=0.0, spread=50.0)
    res = counterfactual_outcome(fc, df, anchor + timedelta(hours=2))
    assert res["status"] == UNRESOLVED
    assert res["classification_hint"] == "unresolved"


def test_no_candles_after_decision_is_unresolved():
    # Все свечи ДО decision_time -> окно пустое -> measure_outcome None.
    fc = _fc(decision_time=ANCHOR + timedelta(hours=200))
    df = _klines(ANCHOR, 10, 100_000.0)
    res = counterfactual_outcome(fc, df, ANCHOR + timedelta(hours=300))
    assert res["status"] == UNRESOLVED


# --- 6. hypothetical flag ---------------------------------------------------

def test_result_is_marked_hypothetical():
    df = _klines(ANCHOR, 80, 100_000.0, step=1_000.0)
    for fc in (_fc(), _fc(candidate_direction=None), _fc(stop_loss=None)):
        assert counterfactual_outcome(fc, df, FAR_NOW)["hypothetical"] is True


# --- 7. no mutation ---------------------------------------------------------

def test_input_forecast_and_klines_not_mutated():
    fc = _fc()  # без 'id', чтобы проверить, что инъекция id не мутирует вход
    del fc["id"]
    fc_snapshot = copy.deepcopy(fc)
    df = _klines(ANCHOR, 80, 100_000.0, step=1_000.0)
    df_snapshot = df.copy(deep=True)
    counterfactual_outcome(fc, df, FAR_NOW)
    assert fc == fc_snapshot
    assert df.equals(df_snapshot)


# --- 8. determinism ---------------------------------------------------------

def test_deterministic_output():
    df = _klines(ANCHOR, 80, 100_000.0, step=-1_000.0)
    a = counterfactual_outcome(_fc(), df, FAR_NOW)
    b = counterfactual_outcome(_fc(), df, FAR_NOW)
    assert a == b


# --- 9. reuse parity with measure_outcome -----------------------------------

def test_reuse_parity_with_measure_outcome():
    df = _klines(ANCHOR, 80, 100_000.0, step=1_000.0)
    fc = _fc()
    res = counterfactual_outcome(fc, df, FAR_NOW)
    ref = measure_outcome(fc, df, FAR_NOW, 0.05, 0.03)
    for key in ("tp1_hit", "tp2_hit", "stop_hit", "mfe_points", "mae_points"):
        assert res[key] == ref[key], key


def test_enter_is_unsupported_not_hypothetical_actual():
    # ENTER несёт фактический исход; counterfactual его не подменяет.
    res = counterfactual_outcome(_fc(analysis_status="ENTER"),
                                 _klines(ANCHOR, 80, 100_000.0, step=1_000.0), FAR_NOW)
    assert res["status"] == UNSUPPORTED


# --- 10. offline: no forbidden imports --------------------------------------

def test_module_has_no_forbidden_imports():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    tops = {n.split(".")[0] for n in imported}
    assert not (tops & {"database", "scheduler", "pipeline", "signal_engine",
                        "risk", "bot", "contracts", "tools"}), tops
    # Никакого сетевого/БД-клиента среди импортов (проза docstring не в счёт).
    assert not (tops & {"binance", "asyncpg", "aiohttp", "requests", "httpx"}), tops
    # Переиспользует producer, но не сетевой/БД-код.
    assert "analyzer.outcomes" in imported


# --- 11. realized_r formula untouched (behavioural) -------------------------

def test_realized_r_stays_none_for_non_enter():
    # Модуль не форсит R для non-ENTER — формула realized_r не тронута.
    df = _klines(ANCHOR, 80, 100_000.0, step=-1_000.0)
    res = counterfactual_outcome(_fc(analysis_status="WAIT"), df, FAR_NOW)
    assert res["realized_r"] is None
