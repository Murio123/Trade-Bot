"""Stage 10: offline per-forecast realized R (Option B).

Доказывает: формула realized_r детерминирована и симметрична LONG/SHORT,
покрывает все edge cases (стоп / TP2 / breakeven / single-target / MTM /
unresolved / отсутствующие уровни / WAIT-NO_TRADE), summary игнорирует None,
источник read-only (нет пишущих SQL-глаголов), runner не импортируется
production-кодом, golden не тронуты, CLI работает на fixture и пустом входе.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

import tools.realized_r as rr

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "forecast_metrics_sample.json"
GOLDEN_DIR = REPO / "tests" / "golden"


def _enter(direction: str, ref: float, stop: float | None,
           tps: list[float] | None) -> dict:
    return {"id": 1, "analysis_status": "ENTER", "candidate_direction": direction,
            "executable_price_at_decision": ref, "stop_loss": stop,
            "take_profit_levels": tps}


def _out(**kw) -> dict:
    base = {"forecast_id": 1, "tp1_hit": False, "tp2_hit": False,
            "stop_hit": False, "return_72h": None, "resolved": True}
    base.update(kw)
    return base


# --- LONG -------------------------------------------------------------------

def test_long_stop_only_is_minus_one():
    f = _enter("long", 100050.0, 98500.0, [102000.0, 104000.0])
    assert rr.realized_r(f, _out(stop_hit=True)) == -1.0


def test_long_tp2_positive_rr():
    f = _enter("long", 100050.0, 98500.0, [102000.0, 104000.0])
    r = rr.realized_r(f, _out(tp1_hit=True, tp2_hit=True))
    assert r == pytest.approx((104000.0 - 100050.0) / 1550.0)  # ~2.548
    assert r > 0


def test_long_tp1_then_stop_is_breakeven():
    f = _enter("long", 100050.0, 98500.0, [102000.0, 104000.0])
    assert rr.realized_r(f, _out(tp1_hit=True, stop_hit=True)) == 0.0


def test_long_tp1_single_target_positive():
    f = _enter("long", 100000.0, 99000.0, [101500.0])  # только TP1
    r = rr.realized_r(f, _out(tp1_hit=True))
    assert r == pytest.approx((101500.0 - 100000.0) / 1000.0)  # 1.5


def test_long_tp1_no_tp2_no_stop_mtm():
    f = _enter("long", 100820.0, 99300.0, [102300.0, 103800.0])  # TP2 есть, но не взят
    r = rr.realized_r(f, _out(tp1_hit=True, return_72h=1.2))
    assert r == pytest.approx((1.2 / 100.0) * 100820.0 / 1520.0)


def test_long_no_tp_no_stop_resolved_mtm():
    f = _enter("long", 100000.0, 98000.0, [102000.0, 104000.0])
    r = rr.realized_r(f, _out(return_72h=0.5))
    assert r == pytest.approx((0.5 / 100.0) * 100000.0 / 2000.0)


def test_long_unresolved_is_none():
    f = _enter("long", 100000.0, 98000.0, [102000.0])
    assert rr.realized_r(f, _out(resolved=False, tp1_hit=True)) is None


def test_missing_stop_is_none():
    f = _enter("long", 100000.0, None, [102000.0])
    assert rr.realized_r(f, _out(tp2_hit=True)) is None


def test_missing_reference_is_none():
    f = {"id": 1, "analysis_status": "ENTER", "candidate_direction": "long",
         "executable_price_at_decision": None, "signal_close_price": None,
         "stop_loss": 98000.0, "take_profit_levels": [102000.0]}
    assert rr.realized_r(f, _out(tp2_hit=True)) is None


def test_zero_risk_is_none():
    f = _enter("long", 100000.0, 100000.0, [102000.0])  # ref == stop
    assert rr.realized_r(f, _out(stop_hit=True)) is None


def test_wait_status_is_none():
    f = _enter("long", 100000.0, 98000.0, [102000.0, 104000.0])
    f["analysis_status"] = "WAIT"
    assert rr.realized_r(f, _out(tp1_hit=True, tp2_hit=True)) is None


def test_no_trade_status_is_none():
    f = _enter("short", 100000.0, 101500.0, None)
    f["analysis_status"] = "NO_TRADE"
    assert rr.realized_r(f, _out(stop_hit=True)) is None


# --- SHORT (зеркально) ------------------------------------------------------

def test_short_stop_only_is_minus_one():
    f = _enter("short", 100950.0, 102500.0, [99500.0, 98000.0])
    assert rr.realized_r(f, _out(stop_hit=True)) == -1.0


def test_short_tp2_positive_rr():
    f = _enter("short", 100950.0, 102500.0, [99500.0, 98000.0])
    r = rr.realized_r(f, _out(tp1_hit=True, tp2_hit=True))
    assert r == pytest.approx((98000.0 - 100950.0) * -1.0 / 1550.0)  # ~1.903
    assert r > 0


def test_short_no_tp_no_stop_resolved_mtm():
    f = _enter("short", 100000.0, 101500.0, [98000.0, 96000.0])
    r = rr.realized_r(f, _out(return_72h=0.8))
    # return_72h уже sign-adjusted -> MTM использует его напрямую.
    assert r == pytest.approx((0.8 / 100.0) * 100000.0 / 1500.0)


def test_long_short_symmetry_mirror():
    """Зеркальный сетап (LONG вверх == SHORT вниз) даёт одинаковый R."""
    long = _enter("long", 100000.0, 99000.0, [101000.0, 102000.0])
    short = _enter("short", 100000.0, 101000.0, [99000.0, 98000.0])
    r_long = rr.realized_r(long, _out(tp1_hit=True, tp2_hit=True))
    r_short = rr.realized_r(short, _out(tp1_hit=True, tp2_hit=True))
    assert r_long == pytest.approx(r_short) == pytest.approx(2.0)


# --- summary ----------------------------------------------------------------

def test_summary_ignores_none_and_computes_stats():
    forecasts = [
        _enter("long", 100.0, 90.0, [110.0, 120.0]),   # id1 tp2 -> +2R
        {**_enter("long", 100.0, 90.0, [110.0]), "id": 2},  # id2 tp1 single -> +1R
        {**_enter("long", 100.0, 90.0, [110.0, 120.0]), "id": 3},  # id3 stop -> -1R
        {**_enter("long", 100.0, 90.0, [110.0]), "id": 4,
         "analysis_status": "WAIT"},                    # id4 WAIT -> None
        {**_enter("long", 100.0, 90.0, [110.0]), "id": 5},  # id5 unresolved -> None
    ]
    outcomes = [
        _out(forecast_id=1, tp1_hit=True, tp2_hit=True),
        _out(forecast_id=2, tp1_hit=True),
        _out(forecast_id=3, stop_hit=True),
        _out(forecast_id=4, tp1_hit=True, tp2_hit=True),
        _out(forecast_id=5, resolved=False),
    ]
    s = rr.realized_r_summary(forecasts, outcomes)
    assert s["total_forecasts"] == 5
    assert s["eligible_forecasts"] == 4          # ENTER+direction (id1,2,3,5)
    assert s["computed"] == 3                    # id1,2,3
    assert s["none_unavailable"] == 2            # id4,id5
    assert s["none_within_eligible"] == 1        # id5 (unresolved)
    assert s["average_realized_r"] == 0.6667  # summary округляет до 4 знаков
    assert s["median_realized_r"] == 1.0
    assert s["min_realized_r"] == -1.0 and s["max_realized_r"] == 2.0
    assert s["kind_counts"] == {"loss": 1, "tp1_single_win": 1, "tp2_win": 1}
    assert s["none_reasons"]["none_unresolved"] == 1
    assert s["none_reasons"]["none_not_enter"] == 1


def test_summary_empty():
    s = rr.realized_r_summary([], [])
    assert s["computed"] == 0 and s["average_realized_r"] is None
    assert s["distribution"] == {}


def test_summary_on_fixture():
    data = rr.fm.load_json(str(FIXTURE))
    s = rr.realized_r_summary(data["forecasts"], data["outcomes"])
    # ENTER long/short: ids 1,2,3,4,8. id4 unresolved -> None. computed = 4.
    assert s["eligible_forecasts"] == 5
    assert s["computed"] == 4
    assert s["kind_counts"] == {"loss": 1, "tp2_win": 3}
    assert s["none_reasons"] == {"none_not_enter": 3, "none_unresolved": 1}
    assert s["average_realized_r"] == pytest.approx(1.3434, abs=1e-3)


# --- CLI --------------------------------------------------------------------

def test_cli_exit_zero_on_fixture(capsys):
    code = rr.main(["--input", str(FIXTURE)])
    out = capsys.readouterr().out
    assert code == 0
    assert "per-forecast realized R" in out
    assert "LIMITATIONS" in out
    assert "mark-to-market" in out
    assert "reference_price" in out


def test_cli_exit_zero_on_empty(capsys):
    code = rr.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "total_forecasts: 0" in out


def test_cli_json_output(capsys):
    code = rr.main(["--input", str(FIXTURE), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    parsed = json.loads(out)
    assert parsed["computed"] == 4


# --- read-only / isolation guards -------------------------------------------

def test_source_has_no_write_sql_verbs():
    src = (REPO / "tools" / "realized_r.py").read_text(encoding="utf-8")
    forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|TRUNCATE|"
                           r"UPSERT|MERGE|CREATE)\b", src, re.IGNORECASE)
    assert not forbidden, f"найдены пишущие SQL-глаголы: {set(forbidden)}"


def test_module_does_not_import_database():
    src = (REPO / "tools" / "realized_r.py").read_text(encoding="utf-8")
    assert "import database" not in src
    assert "from database" not in src


def test_offline_tools_not_imported_by_production():
    """Stage 11: pure-формула переехала в analyzer/realized_r.py и легитимно
    используется production (analyzer/outcomes.py). Инвариант теперь: offline
    tools-слой (tools.realized_r / tools.forecast_metrics) НЕ импортируется
    runtime-кодом — аналитика не попадает в decision-path."""
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        src = py.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\b(import\s+tools|from\s+tools)\b", src):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"production импортирует offline tools: {offenders}"


def test_pure_formula_lives_in_analyzer_leaf_module():
    """Формула — единый источник в analyzer/realized_r.py; tools/realized_r.py
    её только реэкспортирует, не дублируя."""
    import analyzer.realized_r as core
    assert rr.realized_r is core.realized_r          # тот же объект, без копии
    assert rr.classify is core.classify
    # leaf-модуль: реально импортирует только stdlib/typing — никаких database /
    # tools / analyzer.outcomes (проверяем сами import-узлы, не текст docstring).
    tree = ast.parse((REPO / "analyzer" / "realized_r.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for mod in imported:
        top = mod.split(".")[0]
        assert top not in {"database", "tools"}, f"leaf-модуль импортирует {mod}"
        assert mod != "analyzer.outcomes", "leaf-модуль импортирует analyzer.outcomes"


def _golden_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_running_does_not_touch_golden():
    before = _golden_digest()
    rr.main(["--input", str(FIXTURE)])
    rr.realized_r_summary(*[rr.fm.load_json(str(FIXTURE))[k]
                            for k in ("forecasts", "outcomes")])
    assert _golden_digest() == before
