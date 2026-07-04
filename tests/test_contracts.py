"""Контракты: иммутабельность, фабрики, сериализация, изоляция от production."""
from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts import (BlockMeta, Decision, GateCheck, PricePoint,
                       ReversalEffect, VolatilityRegime, price_point,
                       to_jsonable)

NOW = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def test_contracts_are_frozen():
    pp = PricePoint(price=100.0, provenance="structural")
    with pytest.raises(dataclasses.FrozenInstanceError):
        pp.price = 200.0  # type: ignore[misc]
    gate = GateCheck(name="rr", passed=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.passed = False  # type: ignore[misc]


def test_price_point_factory_normalizes_distances():
    pp = price_point(101_600.0, "structural", ref_price=100_000.0,
                     atr=800.0, tf="12h")
    assert pp.pct_distance == 1.6
    assert pp.atr_distance == 2.0
    assert pp.tf == "12h"
    # Без якоря/ATR — нормировки честно None, не нули.
    bare = price_point(101_600.0, "structural")
    assert bare.atr_distance is None and bare.pct_distance is None


def test_enums_have_expected_values():
    assert Decision.ENTER_LONG.value == "ENTER_LONG"
    assert Decision.WAIT.value == "WAIT"
    assert set(VolatilityRegime) == {
        VolatilityRegime.LOW_VOLATILITY, VolatilityRegime.COMPRESSION,
        VolatilityRegime.NORMAL_TREND, VolatilityRegime.HIGH_VOLATILITY,
        VolatilityRegime.LIQUIDATION_EVENT}
    assert ReversalEffect.BLOCK.value == "block"


def test_to_jsonable_handles_all_leaf_types():
    block = BlockMeta(source="legacy_ctx", as_of=NOW, freshness_seconds=120.0)
    payload = {
        "block": block,
        "when": pd.Timestamp("2026-07-01 12:00", tz="UTC"),
        "np_float": np.float64(1.5),
        "np_int": np.int64(7),
        "np_bool": np.bool_(True),
        "nan": float("nan"),
        "inf": float("inf"),
        "enum": Decision.WAIT,
        "nested": [PricePoint(price=1.0, provenance="atr")],
    }
    out = to_jsonable(payload)
    assert out["block"]["source"] == "legacy_ctx"
    assert out["block"]["as_of"] == "2026-07-01T12:00:00+00:00"
    assert out["when"] == "2026-07-01T12:00:00+00:00"
    assert out["np_float"] == 1.5 and out["np_int"] == 7 and out["np_bool"] is True
    assert out["nan"] is None and out["inf"] is None
    assert out["enum"] == "WAIT"
    assert out["nested"][0]["provenance"] == "atr"
    # Результат сериализуем стандартным json без default-хуков.
    json.dumps(out, sort_keys=True)


def test_production_code_does_not_import_contracts():
    """Этап 1: зависимость строго односторонняя (contracts -> production)."""
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for pkg in ("analyzer", "signal_engine", "bot", "risk", "ai"):
        for path in (root / pkg).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "import contracts" in text or "from contracts" in text:
                offenders.append(str(path))
    for name in ("pipeline.py", "scheduler.py", "backtest.py", "database.py",
                 "main.py", "config.py"):
        text = (root / name).read_text(encoding="utf-8")
        if "import contracts" in text or "from contracts" in text:
            offenders.append(name)
    assert not offenders, f"production импортирует contracts: {offenders}"
