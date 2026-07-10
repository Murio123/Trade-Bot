"""Stage B3a: the bounce observation report is read-only and honest.

Two families of guarantee:

1. PURITY — the report is analytics. It must not import the database module,
   must not contain a single writing statement, and must not reach the
   exchange. A reporting tool that can write is a reporting tool that can
   corrupt the ledger it is measuring.

2. HONESTY — a row whose outcome cannot be measured (blocked, hence no stop and
   no TP) is reported as no_levels. It is never silently folded into a win, a
   loss, or a zero. The same rule governs missing scores in the argmax census.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from tools import bounce_observation_report as bor


def _forecast(**over):
    """A BOUNCE forecast row with levels, overridable."""
    base = {
        "id": 1,
        "analysis_type": "BOUNCE",
        "analysis_status": "ENTER",
        "blocked_gate": None,
        "candidate_direction": "long",
        "long_score": 8.0,
        "short_score": 5.0,
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
    }
    base.update(over)
    return base


def _blocked(**over):
    """A blocked row: the cascade returned before position sizing."""
    return _forecast(analysis_status="NO_TRADE", stop_loss=None,
                     take_profit_levels=None, **over)


# --- 1. Purity --------------------------------------------------------------

def test_report_does_not_import_database():
    src = inspect.getsource(bor)
    assert not re.search(r"^\s*(import|from)\s+database\b", src, re.MULTILINE)
    assert "from database import" not in src


def test_report_does_not_import_runtime_decision_modules():
    src = inspect.getsource(bor)
    for module in ("pipeline", "scheduler", "signal_engine", "risk", "bot."):
        assert not re.search(rf"^\s*(import|from)\s+{re.escape(module)}",
                             src, re.MULTILINE), module


@pytest.mark.parametrize("verb", ["INSERT", "UPDATE", "DELETE", "DROP",
                                  "ALTER", "TRUNCATE", "CREATE"])
def test_report_contains_no_writing_sql(verb):
    assert verb not in inspect.getsource(bor)


def test_report_opens_no_file_for_writing():
    src = inspect.getsource(bor)
    assert not re.search(r"open\([^)]*['\"][wa]", src)


def test_sql_is_select_only():
    for sql in (bor._SQL_FORECASTS, bor._SQL_OUTCOMES):
        assert sql.strip().upper().startswith("SELECT")


# --- 2. Filtering -----------------------------------------------------------

def test_empty_input_is_handled_cleanly():
    summary = bor.summarize([], [])
    assert summary["bounce_forecasts"] == 0
    assert summary["by_analysis_status"] == {}
    assert summary["argmax_census"]["counted"] == 0
    assert summary["argmax_census"]["average_delta"] is None
    assert summary["realized_r"]["enter_rows"] == 0
    assert summary["realized_r"]["average_realized_r"] is None
    assert bor.format_report(summary)  # renders without raising


def test_non_bounce_rows_are_ignored():
    rows = [_forecast(id=1),
            _forecast(id=2, analysis_type="SWING"),
            _forecast(id=3, analysis_type="INTRADAY"),
            _forecast(id=4, analysis_type=None)]
    summary = bor.summarize(rows, [])
    assert summary["total_forecasts"] == 4
    assert summary["bounce_forecasts"] == 1


# --- 3. Counts --------------------------------------------------------------

def test_status_counts_are_correct():
    rows = [_forecast(id=1, analysis_status="ENTER"),
            _forecast(id=2, analysis_status="WAIT"),
            _forecast(id=3, analysis_status="WAIT"),
            _blocked(id=4, blocked_gate="htf_filter"),
            _forecast(id=5, analysis_status=None)]
    summary = bor.summarize(rows, [])
    assert summary["by_analysis_status"] == {
        "ENTER": 1, "NO_TRADE": 1, "WAIT": 2, "unknown": 1}


def test_blocked_gate_counts_only_blocked_rows():
    rows = [_forecast(id=1),
            _blocked(id=2, blocked_gate="htf_filter"),
            _blocked(id=3, blocked_gate="htf_filter"),
            _blocked(id=4, blocked_gate="below_threshold")]
    summary = bor.summarize(rows, [])
    assert summary["by_blocked_gate"] == {"below_threshold": 1, "htf_filter": 2}


def test_candidate_direction_distribution_counts_missing_as_none():
    rows = [_forecast(id=1, candidate_direction="long"),
            _forecast(id=2, candidate_direction="short"),
            _forecast(id=3, candidate_direction="short"),
            _blocked(id=4, candidate_direction=None)]
    summary = bor.summarize(rows, [])
    assert summary["by_candidate_direction"] == {"long": 1, "none": 1, "short": 2}


# --- 4. Argmax census -------------------------------------------------------

@pytest.mark.parametrize("delta,bucket", [
    (-9.0, "delta <= -3"),
    (-3.0, "delta <= -3"),
    (-2.9, "-3 < delta <= -1"),
    (-1.0, "-3 < delta <= -1"),
    (-0.9, "-1 < delta < 0"),
    (0.0, "delta == 0 (tie)"),
    (0.9, "0 < delta < 1"),
    (1.0, "1 <= delta < 3"),
    (2.9, "1 <= delta < 3"),
    (3.0, "delta >= 3"),
    (9.0, "delta >= 3"),
])
def test_delta_buckets_are_exhaustive_and_boundary_correct(delta, bucket):
    assert bor.delta_bucket(delta) == bucket


def test_argmax_census_buckets_and_tallies():
    rows = [
        _forecast(id=1, long_score=9.0, short_score=2.0),   # +7 -> long wins
        _forecast(id=2, long_score=2.0, short_score=9.0),   # -7 -> short wins
        _forecast(id=3, long_score=5.0, short_score=5.0),   #  0 -> tie
        _forecast(id=4, long_score=5.5, short_score=5.0),   # +0.5
    ]
    census = bor.summarize(rows, [])["argmax_census"]
    assert census["counted"] == 4
    assert census["long_wins_argmax"] == 2
    assert census["short_wins_argmax"] == 1
    assert census["ties"] == 1
    assert census["buckets"] == {"delta <= -3": 1, "delta == 0 (tie)": 1,
                                 "0 < delta < 1": 1, "delta >= 3": 1}
    assert census["max_delta"] == 7.0
    assert census["min_delta"] == -7.0


def test_missing_scores_are_not_bucketed_as_zero():
    """A pre-scoring block (stale_data) has no scores — that is not a tie."""
    rows = [_blocked(id=1, blocked_gate="stale_data",
                     long_score=None, short_score=None),
            _blocked(id=2, blocked_gate="stale_data",
                     long_score=4.0, short_score=None)]
    census = bor.summarize(rows, [])["argmax_census"]
    assert census["missing_scores"] == 2
    assert census["counted"] == 0
    assert census["ties"] == 0
    assert census["buckets"] == {}


def test_bool_scores_are_not_treated_as_numbers():
    rows = [_forecast(id=1, long_score=True, short_score=False)]
    census = bor.summarize(rows, [])["argmax_census"]
    assert census["missing_scores"] == 1
    assert census["counted"] == 0


# --- 5. no_levels honesty ---------------------------------------------------

def test_rows_without_levels_are_counted_as_no_levels():
    rows = [_forecast(id=1),
            _blocked(id=2, blocked_gate="htf_filter"),
            _blocked(id=3, blocked_gate="diversity")]
    summary = bor.summarize(rows, [])
    assert summary["no_levels"] == 2
    assert summary["no_levels_blocked"] == 2


def test_missing_tp1_alone_makes_a_row_no_levels():
    rows = [_forecast(id=1, take_profit_levels=[]),
            _forecast(id=2, stop_loss=None)]
    assert bor.summarize(rows, [])["no_levels"] == 2


def test_no_levels_rows_never_enter_the_realized_r_pool():
    """A blocked row must not become a win, a loss, or a zero."""
    rows = [_blocked(id=1, blocked_gate="htf_filter")]
    outcomes = [{"forecast_id": 1, "realized_r": -1.0, "resolved": True}]
    r = bor.summarize(rows, outcomes)["realized_r"]
    assert r["enter_rows"] == 0        # NO_TRADE is not ENTER
    assert r["computed"] == 0
    assert r["sum_realized_r"] is None


# --- 6. realized R is READ, never recomputed --------------------------------

def test_realized_r_is_read_from_outcomes():
    rows = [_forecast(id=1), _forecast(id=2)]
    outcomes = [{"forecast_id": 1, "realized_r": 2.0, "resolved": True},
                {"forecast_id": 2, "realized_r": -1.0, "resolved": True}]
    r = bor.summarize(rows, outcomes)["realized_r"]
    assert r["enter_rows"] == 2
    assert r["computed"] == 2
    assert r["kind_counts"] == {"computed": 2}
    assert r["average_realized_r"] == 0.5
    assert r["sum_realized_r"] == 1.0


def test_enter_row_without_outcome_is_unknown_not_zero():
    rows = [_forecast(id=1)]
    r = bor.summarize(rows, [])["realized_r"]
    assert r["kind_counts"] == {"no_outcome": 1}
    assert r["unknown"] == 1
    assert r["average_realized_r"] is None


def test_enter_row_with_unpersisted_realized_r_is_unknown():
    rows = [_forecast(id=1)]
    outcomes = [{"forecast_id": 1, "realized_r": None, "resolved": True}]
    r = bor.summarize(rows, outcomes)["realized_r"]
    assert r["kind_counts"] == {"not_persisted": 1}
    assert r["unknown"] == 1
    assert r["computed"] == 0


def test_report_module_defines_no_r_formula():
    """R is read from the ledger; the formula lives in analyzer/realized_r.py."""
    src = inspect.getsource(bor)
    for token in ("stop_hit", "tp1_hit", "tp2_hit", "return_72h", "reference_price"):
        assert token not in src, token


# --- 7. Rendering -----------------------------------------------------------

def test_report_has_a_limitations_section():
    text = bor.format_report(bor.summarize([_forecast(id=1)], []))
    assert "[LIMITATIONS]" in text
    assert len(bor.LIMITATIONS) >= 4


def test_limitations_name_the_four_required_caveats():
    blob = " ".join(bor.LIMITATIONS).lower()
    assert "no_levels" in blob                      # blocked rows lack levels
    assert "argmax" in blob and "require_exhaustion" in blob
    assert "trend-follower" in blob or "trend-following" in blob
    assert "funding" in blob and "candle-confirmation" in blob


def test_report_renders_every_section():
    rows = [_forecast(id=1), _blocked(id=2, blocked_gate="htf_filter")]
    outcomes = [{"forecast_id": 1, "realized_r": 1.5, "resolved": True}]
    text = bor.format_report(bor.summarize(rows, outcomes))
    for heading in ("by_analysis_status", "by_blocked_gate",
                    "by_candidate_direction", "argmax census",
                    "realized R", "no_levels"):
        assert heading in text


# --- 8. CLI -----------------------------------------------------------------

def test_cli_json_output_over_a_file(tmp_path, capsys):
    export = tmp_path / "export.json"
    export.write_text(json.dumps({
        "forecasts": [_forecast(id=1), _forecast(id=2, analysis_type="SWING")],
        "outcomes": [{"forecast_id": 1, "realized_r": 1.0, "resolved": True}],
    }), encoding="utf-8")

    assert bor.main(["--input", str(export), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["bounce_forecasts"] == 1
    assert summary["realized_r"]["computed"] == 1


def test_cli_with_no_source_reports_an_empty_ledger(capsys):
    assert bor.main([]) == 0
    assert "bounce_forecasts: 0" in capsys.readouterr().out


def test_load_json_parses_stringified_take_profit_levels(tmp_path):
    export = tmp_path / "export.json"
    export.write_text(json.dumps({
        "forecasts": [_forecast(id=1, take_profit_levels="[110.0, 120.0]")],
    }), encoding="utf-8")
    data = bor.load_json(str(export))
    assert data["forecasts"][0]["take_profit_levels"] == [110.0, 120.0]
    assert data["outcomes"] == []
