"""Stage 12: Bollinger volatility/regime CONTEXT (Option A, pure leaf-module).

Доказывает: контекст считается детерминированно из klines, деградирует в None на
короткой истории, %b/squeeze/expansion/band-walk/mean-reversion классифицируются
корректно, модуль НЕ эмитит торговое направление/статус и НЕ импортирует
decision-path (database/tools/signal_engine/bot/ai/risk/contracts/pipeline/
scheduler).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

import analyzer.bollinger_context as bc

REPO = Path(__file__).resolve().parent.parent


def _df(closes: list[float]) -> pd.DataFrame:
    closes = [float(c) for c in closes]
    return pd.DataFrame({
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1.0] * len(closes),
    })


# 1. graceful None on short data ---------------------------------------------

def test_short_dataframe_returns_all_none():
    out = bc.bollinger_context(_df([100.0] * 10))
    assert set(out) == set(bc.OUTPUT_KEYS)
    assert all(v is None for v in out.values())


def test_none_dataframe_returns_all_none():
    assert all(v is None for v in bc.bollinger_context(None).values())


# 2-4. percent_b positioning -------------------------------------------------

def test_percent_b_inside_bands():
    closes = [100 + (1 if i % 2 else -1) for i in range(30)]
    out = bc.bollinger_context(_df(closes))
    assert 0.0 < out["bb_percent_b"] < 1.0
    assert out["close_outside_upper"] is False
    assert out["close_outside_lower"] is False


def test_percent_b_above_upper_band():
    out = bc.bollinger_context(_df([100.0] * 29 + [110.0]))
    assert out["bb_percent_b"] > 1.0
    assert out["close_outside_upper"] is True
    assert out["close_outside_lower"] is False


def test_percent_b_below_lower_band():
    out = bc.bollinger_context(_df([100.0] * 29 + [90.0]))
    assert out["bb_percent_b"] < 0.0
    assert out["close_outside_lower"] is True
    assert out["close_outside_upper"] is False


# 5-6. squeeze / expansion ---------------------------------------------------

def _alt(base: float, amp: float, n: int) -> list[float]:
    return [base + (amp if i % 2 else -amp) for i in range(n)]


def test_squeeze_classification():
    # длинная волатильная история -> недавнее сжатие: текущая ширина минимальна.
    closes = _alt(100.0, 10.0, 80) + _alt(100.0, 0.1, 21)
    out = bc.bollinger_context(_df(closes))
    assert out["bb_squeeze"] is True
    assert out["bb_expansion"] is False
    assert out["bb_width_percentile"] <= bc.SQUEEZE_PCT


def test_expansion_classification():
    # длинная спокойная история -> недавнее расширение: ширина максимальна.
    closes = _alt(100.0, 0.1, 80) + _alt(100.0, 10.0, 21)
    out = bc.bollinger_context(_df(closes))
    assert out["bb_expansion"] is True
    assert out["bb_squeeze"] is False
    assert out["bb_width_percentile"] >= bc.EXPANSION_PCT


# 7-8. band walk -------------------------------------------------------------

def test_band_walk_direction_up():
    out = bc.bollinger_context(_df([100 + i * 2 for i in range(40)]))
    assert out["band_walk_direction"] == "up"


def test_band_walk_direction_down():
    out = bc.bollinger_context(_df([100 - i * 2 for i in range(40)]))
    assert out["band_walk_direction"] == "down"


# 9. mean-reversion risk -----------------------------------------------------

def test_mean_reversion_risk_high_outside_band_without_walk():
    # Резкий выброс за верхнюю полосу без предшествующего band-walk-тренда.
    out = bc.bollinger_context(_df([100.0] * 29 + [110.0]))
    assert out["close_outside_upper"] is True
    assert out["band_walk_direction"] is None
    assert out["mean_reversion_risk"] == "high"


# 10. no trade direction / status --------------------------------------------

_FORBIDDEN_TOKENS = {"LONG", "SHORT", "BUY", "SELL", "ENTER", "WAIT",
                     "NO_TRADE", "NEUTRAL", "long", "short", "buy", "sell"}


def test_module_emits_no_trade_direction_or_status():
    out = bc.bollinger_context(_df([100 + (1 if i % 2 else -1) for i in range(30)]))
    assert set(out) == set(bc.OUTPUT_KEYS)                     # ровно контекст
    assert out["band_walk_direction"] in (None, "up", "down")  # не торговое направление
    for v in out.values():
        assert v not in _FORBIDDEN_TOKENS
    # regime_clue — подсказка о волатильности, без торговых токенов.
    assert not any(tok.lower() in str(out["regime_clue"]).lower()
                   for tok in ("long", "short", "buy", "sell", "enter"))
    for key in ("direction", "confidence", "status", "signal", "realized_r"):
        assert key not in out  # band_walk_direction — единственный *_direction


# 11. determinism ------------------------------------------------------------

def test_deterministic_same_input():
    closes = [100 + (i % 7) - 3 for i in range(60)]
    assert bc.bollinger_context(_df(closes)) == bc.bollinger_context(_df(closes))


# 12. forbidden imports absent -----------------------------------------------

def test_no_forbidden_imports():
    forbidden = {"database", "tools", "signal_engine", "bot", "ai", "risk",
                 "contracts", "pipeline", "scheduler"}
    tree = ast.parse((REPO / "analyzer" / "bollinger_context.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    leak = imported & forbidden
    assert not leak, f"leaf-модуль импортирует запрещённое: {leak}"
