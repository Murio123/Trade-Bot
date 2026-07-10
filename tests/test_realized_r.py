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


# --- Stage B3a.2: analysis_type filter + grouping ---------------------------

def _typed(fid: int, analysis_type, **over) -> dict:
    """An ENTER long that resolves to +2R (tp2), tagged with an analysis_type."""
    f = {**_enter("long", 100.0, 90.0, [110.0, 120.0]), "id": fid,
         "analysis_type": analysis_type}
    f.update(over)
    return f


def _mixed() -> tuple[list[dict], list[dict]]:
    """One ENTER per mode, all +2R, plus one row with no analysis_type."""
    forecasts = [_typed(1, "SWING"), _typed(2, "INTRADAY"),
                 _typed(3, "POSITION"), _typed(4, "BOUNCE"), _typed(5, None)]
    outcomes = [_out(forecast_id=i, tp1_hit=True, tp2_hit=True)
                for i in range(1, 6)]
    return forecasts, outcomes


def test_no_filter_keeps_existing_global_behaviour():
    forecasts, outcomes = _mixed()
    s = rr.realized_r_summary(forecasts, outcomes)
    assert s["total_forecasts"] == 5 and s["computed"] == 5
    # unchanged: the unfiltered list is the whole population
    assert rr.filter_by_analysis_type(forecasts, None) == forecasts
    assert rr.filter_by_analysis_type(forecasts, "") == forecasts
    assert rr.filter_by_analysis_type(forecasts, "   ") == forecasts


@pytest.mark.parametrize("wanted,fid", [
    ("SWING", 1), ("INTRADAY", 2), ("POSITION", 3), ("BOUNCE", 4)])
def test_filter_selects_exactly_one_mode(wanted, fid):
    forecasts, _ = _mixed()
    rows = rr.filter_by_analysis_type(forecasts, wanted)
    assert [r["id"] for r in rows] == [fid]


@pytest.mark.parametrize("spelling", ["swing", "SWING", "Swing", "  swing  "])
def test_filter_is_case_and_whitespace_insensitive(spelling):
    forecasts, _ = _mixed()
    assert [r["id"] for r in rr.filter_by_analysis_type(forecasts, spelling)] == [1]


def test_filter_never_matches_rows_without_analysis_type():
    forecasts, _ = _mixed()
    for wanted in ("SWING", "BOUNCE", "unknown", "UNKNOWN", "None"):
        assert 5 not in [r["id"] for r in rr.filter_by_analysis_type(forecasts, wanted)]


def test_filter_on_unknown_mode_returns_nothing():
    forecasts, outcomes = _mixed()
    rows = rr.filter_by_analysis_type(forecasts, "NOPE")
    assert rows == []
    s = rr.realized_r_summary(rows, outcomes)
    assert s["computed"] == 0 and s["average_realized_r"] is None


def test_grouping_buckets_every_row_exactly_once():
    forecasts, outcomes = _mixed()
    groups = rr.summarize_by_analysis_type(forecasts, outcomes)
    assert set(groups) == {"SWING", "INTRADAY", "POSITION", "BOUNCE", "unknown"}
    assert sum(g["total_forecasts"] for g in groups.values()) == len(forecasts)
    for name, g in groups.items():
        assert g["computed"] == 1 and g["average_realized_r"] == 2.0


def test_missing_analysis_type_lands_in_the_unknown_bucket():
    forecasts, outcomes = _mixed()
    groups = rr.summarize_by_analysis_type(forecasts, outcomes)
    assert groups["unknown"]["total_forecasts"] == 1
    for name in ("SWING", "INTRADAY", "POSITION", "BOUNCE"):
        assert groups[name]["total_forecasts"] == 1


def test_bounce_does_not_blend_into_trend_modes():
    """The reason this stage exists: an observation-only BOUNCE row must never
    contribute R to a trend mode's pool."""
    forecasts = [_typed(1, "SWING"), _typed(2, "INTRADAY"), _typed(3, "POSITION"),
                 # a catastrophic bounce loss that would drag any shared pool down
                 {**_typed(4, "BOUNCE"), "stop_loss": 90.0}]
    outcomes = [_out(forecast_id=i, tp1_hit=True, tp2_hit=True) for i in (1, 2, 3)]
    outcomes.append(_out(forecast_id=4, stop_hit=True))

    groups = rr.summarize_by_analysis_type(forecasts, outcomes)
    assert groups["BOUNCE"]["average_realized_r"] == -1.0
    for name in ("SWING", "INTRADAY", "POSITION"):
        assert groups[name]["average_realized_r"] == 2.0
        assert groups[name]["computed"] == 1

    # ...while the global pool DOES mix them — that is exactly what the
    # grouped view exists to separate.
    assert rr.realized_r_summary(forecasts, outcomes)["computed"] == 4


def test_grouping_is_empty_for_no_forecasts():
    assert rr.summarize_by_analysis_type([], []) == {}


# --- Stage B3a.2: report / CLI ----------------------------------------------

def test_report_without_filter_shows_no_filtered_banner():
    forecasts, outcomes = _mixed()
    text = rr.format_report(rr.realized_r_summary(forecasts, outcomes),
                            rr.summarize_by_analysis_type(forecasts, outcomes))
    assert "[FILTERED]" not in text
    assert "by_analysis_type" in text


def test_report_with_filter_says_it_is_filtered():
    forecasts, outcomes = _mixed()
    rows = rr.filter_by_analysis_type(forecasts, "bounce")
    text = rr.format_report(rr.realized_r_summary(rows, outcomes),
                            rr.summarize_by_analysis_type(rows, outcomes),
                            "bounce")
    assert "[FILTERED]" in text
    assert "analysis_type == BOUNCE" in text      # normalised in the banner


def test_format_report_still_accepts_a_lone_summary():
    """Back-compat: the old single-argument call renders without the section."""
    text = rr.format_report(rr.realized_r_summary(*_mixed()))
    assert "per-forecast realized R" in text
    assert "  by_analysis_type:" not in text   # section omitted; the note stays


def test_cli_filter_narrows_the_population(capsys):
    """The fixture holds 7 SWING + 1 INTRADAY, so the filter must really bite."""
    rr.main(["--input", str(FIXTURE), "--json"])
    everything = json.loads(capsys.readouterr().out)
    assert set(everything["by_analysis_type"]) == {"SWING", "INTRADAY"}
    assert everything["total_forecasts"] == 8

    code = rr.main(["--input", str(FIXTURE), "--analysis-type", "SWING", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert parsed["analysis_type_filter"] == "SWING"
    assert set(parsed["by_analysis_type"]) == {"SWING"}
    assert parsed["total_forecasts"] == 7          # the INTRADAY row is gone


def test_cli_filter_on_intraday_excludes_the_swing_rows(capsys):
    rr.main(["--input", str(FIXTURE), "--analysis-type", "intraday", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["total_forecasts"] == 1
    assert set(parsed["by_analysis_type"]) == {"INTRADAY"}


def test_cli_json_keeps_the_global_summary_at_top_level(capsys):
    """Old consumers read parsed["computed"]; that contract must not break."""
    rr.main(["--input", str(FIXTURE), "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["computed"] == 4                 # same as before this stage
    assert parsed["analysis_type_filter"] is None
    assert "by_analysis_type" in parsed


def test_cli_text_output_lists_the_groups(capsys):
    rr.main(["--input", str(FIXTURE)])
    out = capsys.readouterr().out
    assert "by_analysis_type:" in out
    assert "[FILTERED]" not in out


def test_cli_filter_is_case_insensitive(capsys):
    rr.main(["--input", str(FIXTURE), "--analysis-type", "swing"])
    out = capsys.readouterr().out
    assert "[FILTERED]" in out and "analysis_type == SWING" in out


# --- Stage B3a.2: the formula is untouched ----------------------------------

def test_filtering_does_not_touch_the_formula():
    """classify/realized_r stay the analyzer leaf objects, re-exported as-is."""
    import analyzer.realized_r as core
    assert rr.realized_r is core.realized_r
    assert rr.classify is core.classify


def test_filter_and_grouping_do_not_recompute_r():
    """A grouped summary equals the summary of that group's rows alone."""
    forecasts, outcomes = _mixed()
    groups = rr.summarize_by_analysis_type(forecasts, outcomes)
    for name in ("SWING", "BOUNCE"):
        rows = rr.filter_by_analysis_type(forecasts, name)
        assert groups[name] == rr.realized_r_summary(rows, outcomes)


# --- Stage B3a.2 follow-up: the banner must not lie about the selection -----

@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  "])
def test_blank_filter_reports_unfiltered_in_text(capsys, blank):
    """A whitespace-only filter narrows nothing, so it must not claim to.

    The banner and the actual population are two views of one decision; if they
    disagree, the report misrepresents its own sample.
    """
    rr.main(["--input", str(FIXTURE), "--analysis-type", blank])
    out = capsys.readouterr().out
    assert "[FILTERED]" not in out
    assert "total_forecasts: 8" in out           # the whole fixture population


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_filter_reports_none_in_json(capsys, blank):
    rr.main(["--input", str(FIXTURE), "--analysis-type", blank, "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["analysis_type_filter"] is None
    assert parsed["total_forecasts"] == 8
    assert set(parsed["by_analysis_type"]) == {"SWING", "INTRADAY"}


def test_padded_real_filter_still_filters_and_says_so(capsys):
    """Padding around a REAL type is stripped, not treated as blank."""
    rr.main(["--input", str(FIXTURE), "--analysis-type", "  swing  ", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["analysis_type_filter"] == "swing"   # normalised, not blank
    assert parsed["total_forecasts"] == 7

    rr.main(["--input", str(FIXTURE), "--analysis-type", "  swing  "])
    out = capsys.readouterr().out
    assert "[FILTERED]" in out and "analysis_type == SWING" in out


def test_banner_presence_always_matches_the_population(capsys):
    """Invariant: [FILTERED] appears iff the population was actually narrowed."""
    for arg, narrowed in (("SWING", True), ("intraday", True),
                          ("   ", False), ("", False)):
        rr.main(["--input", str(FIXTURE), "--analysis-type", arg])
        out = capsys.readouterr().out
        assert ("[FILTERED]" in out) is narrowed, arg
        assert ("total_forecasts: 8" in out) is (not narrowed), arg
