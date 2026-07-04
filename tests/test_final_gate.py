"""Этап 3: shared final_gate — read-only shadow / compatibility layer.

Доказывает legacy parity: final_gate строит FinalDecision/GateCheck[] из
готового legacy-`result` без потери решения, направления, точки блокировки и
причин — и делает это через делегирование в contracts.legacy (обёртка).
Golden-снапшоты этими тестами НЕ трогаются (только чтение через run_cascade).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import config
from contracts import Decision, GateCheck, decision_from_legacy
from pipeline import run_cascade
from contracts.final_gate import (GATE_BY_NAME, GATE_REGISTRY, ParityReport,
                                  compare_legacy_and_final_decision,
                                  final_decision_from_legacy_result,
                                  gate_checks_from_legacy_result,
                                  summarize_gate_checks)
from signal_engine.profiles import get_profile
from tests.synthetic_market import build_ctx
from tests.test_golden_contexts import SCENARIOS

PROFILE = get_profile("swing")
NOW = datetime(2026, 7, 1, 12, 2, tzinfo=timezone.utc)

# Минимальный ctx: decision_from_legacy читает решение из `result`, а из ctx
# берёт только meta-поля (symbol/timeframe/price/timestamp).
MIN_CTX = {"symbol": "BTCUSDT", "timeframe": "4h", "price": 100_000.0,
           "timestamp": NOW}


def _run(name: str) -> dict:
    params = SCENARIOS[name]
    ctx = build_ctx("swing", **params)
    result = asyncio.run(run_cascade(ctx, delivered_today=[], last_signal=None,
                                     interpret=False, profile_name="swing"))
    return {"ctx": ctx, "result": result}


# ---------------------------------------------------------------------------
# Реестр гейтов: границы этапа 3
# ---------------------------------------------------------------------------

def test_registry_documents_bundle_and_dedup_only():
    """no_trade sub-gates и candle_dedup — documented_only, не из result."""
    for name in ("rr", "expected_move", "confidence", "tf_conflict",
                 "position_conflict", "invalid_tp2", "missing_invalidation"):
        spec = GATE_BY_NAME[name]
        assert spec.documented_only is True
        assert spec.reachable_from_result is False
        assert spec.stage == "no_trade"     # под-гейт свёрнутого no_trade

    dedup = GATE_BY_NAME["candle_dedup"]
    assert dedup.documented_only is True
    assert dedup.reachable_from_result is False
    assert dedup.db_dependent is True

    # no_trade остаётся ОДНИМ достижимым агрегатом.
    no_trade = GATE_BY_NAME["no_trade"]
    assert no_trade.reachable_from_result is True
    assert no_trade.documented_only is False


def test_registry_marks_db_dependent_gates():
    for name in ("candle_dedup", "cooldown", "daily_limit", "position_conflict"):
        assert GATE_BY_NAME[name].db_dependent is True


def test_registry_names_unique():
    names = [g.name for g in GATE_REGISTRY]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Обёртка: вывод идентичен адаптеру contracts.legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_wrapper_matches_legacy_adapter(name):
    data = _run(name)
    result, ctx = data["result"], data["ctx"]
    final = final_decision_from_legacy_result(result, ctx, PROFILE)
    reference = decision_from_legacy(result, ctx, PROFILE)
    assert final.to_json() == reference.to_json()
    assert gate_checks_from_legacy_result(result, ctx, PROFILE) == reference.gates


# ---------------------------------------------------------------------------
# Parity на 5 golden-контекстах
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_parity_on_golden_scenarios(name):
    data = _run(name)
    result, ctx = data["result"], data["ctx"]
    final = final_decision_from_legacy_result(result, ctx, PROFILE)

    report = compare_legacy_and_final_decision(result, final)
    assert report.ok, report.diffs

    # direction совпадает.
    assert final.direction == (result.get("direction")
                               or result.get("candidate_direction"))

    # blocked/no_trade/wait/enter маппится корректно.
    status = result.get("status")
    if status == "alert":
        assert final.decision in (Decision.ENTER_LONG, Decision.ENTER_SHORT)
    elif status in ("journal", "cooldown"):
        assert final.decision is Decision.WAIT
    else:
        assert final.decision is Decision.NO_TRADE

    # candle_dedup НЕ синтезируется из result.
    assert "candle_dedup" not in {g.name for g in final.gates}
    # ни один под-гейт no_trade не развёрнут в отдельный GateCheck.
    assert not ({"rr", "expected_move", "confidence", "position_conflict"}
                & {g.name for g in final.gates})


def test_golden_scenarios_cover_multiple_outcomes():
    outcomes = {(_run(n)["result"].get("status"),
                 _run(n)["result"].get("blocked_at")) for n in SCENARIOS}
    assert len(outcomes) >= 2


# ---------------------------------------------------------------------------
# Parity по типам исходов (ручные legacy-result)
# ---------------------------------------------------------------------------

def _assert_parity(result: dict) -> None:
    final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
    report = compare_legacy_and_final_decision(result, final)
    assert report.ok, report.diffs


def test_parity_blocked_early_veto():
    for stage in ("stale_data", "abnormal_volatility", "htf_filter", "diversity"):
        result = {"status": "blocked", "blocked_at": stage, "direction": "long"}
        final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
        assert final.decision is Decision.NO_TRADE
        assert summarize_gate_checks(final.gates)["blocked_at"] == stage
        _assert_parity(result)


def test_parity_no_trade_preserves_reasons():
    result = {"status": "blocked", "blocked_at": "no_trade", "direction": "long",
              "reasons": ["conf1", "conf2"],
              "no_trade_reasons": ["risk/reward 1.10 ниже минимума 1.50",
                                   "уверенность 0.20 ниже минимума 0.30"]}
    final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
    assert final.decision is Decision.NO_TRADE
    nt_gate = next(g for g in final.gates if g.name == "no_trade")
    assert not nt_gate.passed
    for r in result["no_trade_reasons"]:
        assert r in nt_gate.detail
    assert final.reasons == ["conf1", "conf2"]
    _assert_parity(result)


def test_parity_wait_journal_and_stale():
    below = {"status": "journal", "direction": "long", "score_weighted": 6,
             "reasons": ["r"]}
    final = final_decision_from_legacy_result(below, MIN_CTX, PROFILE)
    assert final.decision is Decision.WAIT
    assert summarize_gate_checks(final.gates)["blocked_at"] == "alert_threshold"
    _assert_parity(below)

    stale = {"status": "journal", "direction": "short", "stale_decision": True,
             "score_weighted": 9}
    final = final_decision_from_legacy_result(stale, MIN_CTX, PROFILE)
    assert final.decision is Decision.WAIT
    assert summarize_gate_checks(final.gates)["blocked_at"] == "freshness"
    _assert_parity(stale)


def test_parity_daily_limit_journal():
    result = {"status": "journal", "direction": "long",
              "score_weighted": config.SCORE_ALERT_MIN + 1}
    final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
    assert final.decision is Decision.WAIT
    assert summarize_gate_checks(final.gates)["blocked_at"] == "daily_limit"
    _assert_parity(result)


def test_parity_cooldown():
    result = {"status": "cooldown", "direction": "short"}
    final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
    assert final.decision is Decision.WAIT
    assert summarize_gate_checks(final.gates)["blocked_at"] == "cooldown"
    _assert_parity(result)


def test_parity_alert_has_no_failed_gates():
    result = {"status": "alert", "direction": "long", "reasons": ["a", "b"]}
    final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
    assert final.decision is Decision.ENTER_LONG
    summary = summarize_gate_checks(final.gates)
    assert summary["failed"] == 0
    assert summary["blocked_at"] is None
    _assert_parity(result)


# ---------------------------------------------------------------------------
# summarize / compare — единичные свойства
# ---------------------------------------------------------------------------

def test_summarize_counts():
    gates = [GateCheck("a", True), GateCheck("b", True),
             GateCheck("c", False, detail="x")]
    s = summarize_gate_checks(gates)
    assert s == {"total": 3, "passed": 2, "failed": 1, "first_failed": "c",
                 "blocked_at": "c", "failed_names": ["c"],
                 "passed_names": ["a", "b"]}


def test_compare_detects_direction_mismatch():
    # Искусственно ломаем parity: направление в result не совпадёт с final.
    result = {"status": "alert", "direction": "long", "reasons": []}
    final = final_decision_from_legacy_result(result, MIN_CTX, PROFILE)
    broken = {"status": "alert", "direction": "short", "reasons": []}
    report = compare_legacy_and_final_decision(broken, final)
    assert isinstance(report, ParityReport)
    assert report.ok is False
    assert any("direction" in d for d in report.diffs)
