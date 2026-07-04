"""Stage 4: offline shadow parity runner (Option B, golden-only).

Доказывает: runner сверяет legacy-`result` и FinalDecision на всех golden
без расхождений; корректно репортит инъецированный mismatch; exit-code
семантика верна. Production-поток не задействован — runner read-only.
"""
from __future__ import annotations

import tools.shadow_parity as sp
from contracts.final_gate import ParityReport
from tests.test_golden_contexts import SCENARIOS


def test_all_golden_scenarios_parity_ok():
    results = sp.run_all()
    assert {r.name for r in results} == set(SCENARIOS)
    bad = [(r.name, r.diffs) for r in results if not r.ok]
    assert not bad, f"parity mismatch на golden: {bad}"


def test_summary_shape_clean():
    s = sp.summarize(sp.run_all())
    assert s["total"] == len(SCENARIOS)
    assert s["ok"] == s["total"]
    assert s["mismatched"] == 0
    assert s["mismatched_names"] == []


def test_main_returns_zero_and_prints(capsys):
    code = sp.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "parity:" in out
    assert "MISMATCH" not in out


def test_runner_detects_injected_mismatch(monkeypatch):
    """Подменяем parity-проверку на заведомо непройденную — runner обязан
    отметить сценарий как mismatch и вернуть ненулевой код выхода."""
    monkeypatch.setattr(
        sp, "compare_legacy_and_final_decision",
        lambda result, final: ParityReport(ok=False, diffs=["injected diff"]))

    name = sorted(SCENARIOS)[0]
    r = sp.run_scenario_parity(name, SCENARIOS[name])
    assert r.ok is False
    assert "injected diff" in r.diffs

    assert sp.main() == 1


def test_summary_and_report_on_mismatch():
    results = [
        sp.ScenarioParity("a", status="alert", blocked_at=None, ok=True),
        sp.ScenarioParity("b", status="blocked", blocked_at="htf_filter",
                          ok=False, diffs=["decision mismatch"]),
    ]
    s = sp.summarize(results)
    assert s == {"total": 2, "ok": 1, "mismatched": 1, "mismatched_names": ["b"]}

    report = sp.format_report(results)
    assert "MISMATCH" in report
    assert "decision mismatch" in report
    assert "1/2 ok" in report
