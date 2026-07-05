"""Stage 13: pure classifier для Trading Journal v2.

Доказывает: таблица истинности классификации, отсутствие «выдумывания»
исходов при пропусках, детерминизм и независимость leaf-модуля от
database/scheduler/pipeline/tools/runtime.
"""
from __future__ import annotations

import ast
from pathlib import Path

from analyzer.journal_classify import (
    AVOIDED_LOSS, BAD_TRADE, GOOD_TRADE, MISSED_OPPORTUNITY, NO_LEVELS, NONE,
    UNRESOLVED, classify,
)

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "analyzer" / "journal_classify.py"


def _fc(**kw):
    base = {
        "analysis_status": "ENTER",
        "candidate_direction": "long",
        "final_bias": "LONG",
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
        "blocked_gate": None,
        "no_trade_reasons": None,
    }
    base.update(kw)
    return base


def _out(**kw):
    base = {
        "tp1_hit": False, "tp2_hit": False, "stop_hit": False,
        "resolved": True, "realized_r": None,
    }
    base.update(kw)
    return base


# --- ENTER ------------------------------------------------------------------

def test_enter_positive_r_is_good_trade():
    assert classify(_fc(), _out(realized_r=1.8, tp2_hit=True)) == GOOD_TRADE


def test_enter_negative_r_is_bad_trade():
    assert classify(_fc(), _out(realized_r=-1.0, stop_hit=True)) == BAD_TRADE


def test_enter_unresolved_outcome():
    assert classify(_fc(), _out(resolved=False, realized_r=None)) == UNRESOLVED


def test_enter_missing_outcome_is_unresolved():
    assert classify(_fc(), None) == UNRESOLVED


def test_enter_resolved_but_r_none_is_none_not_success():
    # resolved, но R не посчитан -> не хорошо и не плохо.
    assert classify(_fc(), _out(resolved=True, realized_r=None)) == NONE


def test_enter_breakeven_zero_r_is_none():
    assert classify(_fc(), _out(resolved=True, realized_r=0.0,
                                tp1_hit=True, stop_hit=True)) == NONE


# --- WAIT / NO_TRADE / blocked ---------------------------------------------

def test_non_enter_without_levels_is_no_levels():
    fc = _fc(analysis_status="WAIT", stop_loss=None, take_profit_levels=None)
    assert classify(fc, None) == NO_LEVELS


def test_non_enter_partial_levels_is_no_levels():
    fc = _fc(analysis_status="NO_TRADE", stop_loss=100.0, take_profit_levels=[])
    assert classify(fc, _out(resolved=True)) == NO_LEVELS


def test_non_enter_hypothetical_stop_is_avoided_loss():
    fc = _fc(analysis_status="WAIT")
    assert classify(fc, _out(resolved=True, stop_hit=True, tp1_hit=False)) == AVOIDED_LOSS


def test_non_enter_hypothetical_win_is_missed_opportunity():
    fc = _fc(analysis_status="WAIT")
    assert classify(fc, _out(resolved=True, tp2_hit=True)) == MISSED_OPPORTUNITY


def test_non_enter_tp1_without_stop_is_missed_opportunity():
    fc = _fc(analysis_status="NO_TRADE")
    assert classify(fc, _out(resolved=True, tp1_hit=True, stop_hit=False)) == MISSED_OPPORTUNITY


def test_non_enter_no_outcome_is_none():
    fc = _fc(analysis_status="NO_TRADE")
    assert classify(fc, None) == NONE


def test_non_enter_unresolved_outcome_is_none():
    fc = _fc(analysis_status="WAIT")
    assert classify(fc, _out(resolved=False)) == NONE


def test_non_enter_breakeven_outcome_is_none():
    # TP1 задет и стоп задет -> ни чистый стоп, ни чистый вин.
    fc = _fc(analysis_status="WAIT")
    assert classify(fc, _out(resolved=True, tp1_hit=True, stop_hit=True)) == NONE


def test_non_enter_missing_direction_is_none():
    fc = _fc(analysis_status="NO_TRADE", candidate_direction=None)
    assert classify(fc, _out(resolved=True, stop_hit=True)) == NONE


# --- «не выдумывать» / robustness -------------------------------------------

def test_missing_data_never_becomes_good_or_bad():
    # Пустой forecast без статуса/направления/уровней/исхода.
    empty = {"analysis_status": None, "candidate_direction": None}
    assert classify(empty, None) in (NONE, NO_LEVELS, UNRESOLVED)
    assert classify(empty, None) not in (GOOD_TRADE, BAD_TRADE,
                                         AVOIDED_LOSS, MISSED_OPPORTUNITY)


def test_deterministic():
    fc, out = _fc(), _out(realized_r=1.2, tp2_hit=True)
    results = {classify(fc, out) for _ in range(50)}
    assert results == {GOOD_TRADE}


# --- независимость leaf-модуля ----------------------------------------------

def test_classifier_does_not_import_forbidden_modules():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "database", "scheduler", "pipeline", "config",
        "analyzer.outcomes", "analyzer.realized_r",
    }
    assert not (imported & forbidden), f"запрещённые импорты: {imported & forbidden}"
    # никаких signal_engine / tools / risk / bot / contracts / ai
    for name in imported:
        top = name.split(".")[0]
        assert top not in {"signal_engine", "tools", "risk", "bot",
                           "contracts", "ai"}, f"leaf импортирует {name}"
