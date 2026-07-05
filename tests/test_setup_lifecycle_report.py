"""Stage 15A Step 2: offline lifecycle report CLI.

Доказывает: CLI грузит оба входных формата, детерминированно сортирует и
классифицирует потоки, честно агрегирует статусы/причины/потоки, уважает
пороги и --limit, корректно падает на битом входе, не мутирует ряды и остаётся
offline (никаких импортов БД/scheduler/pipeline/bot/signal_engine/ai/risk).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import tools.setup_lifecycle_report as rep

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "tools" / "setup_lifecycle_report.py"

T = "2026-07-05T12:00:00+00:00"


def _fc(id, status="WAIT", direction="long", conf=0.60, close="2026-07-05T12:00:00+00:00",
        symbol="BTCUSDT", atype="SWING", **kw) -> dict:
    base = {
        "id": id, "symbol": symbol, "analysis_type": atype,
        "strategy_version": "v1", "context_version": "v1",
        "analysis_status": status, "candidate_direction": direction,
        "final_bias": "LONG", "raw_confidence": conf,
        "long_score": 6.0, "short_score": 2.0,
        "signal_candle_close_time": close,
    }
    base.update(kw)
    return base


def _write(tmp_path, payload) -> str:
    p = tmp_path / "forecasts.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


# --- загрузка форматов ------------------------------------------------------

def test_loads_raw_list(tmp_path):
    path = _write(tmp_path, [_fc(1), _fc(2)])
    rows = rep.load_forecasts(path)
    assert len(rows) == 2


def test_loads_forecasts_object(tmp_path):
    path = _write(tmp_path, {"forecasts": [_fc(1), _fc(2), _fc(3)]})
    rows = rep.load_forecasts(path)
    assert len(rows) == 3


def test_empty_list_zero_transitions(tmp_path):
    path = _write(tmp_path, [])
    s = rep.summarize(rep.load_forecasts(path), rep.Thresholds())
    assert s["total_forecasts"] == 0
    assert s["total_transitions"] == 0
    assert all(v == 0 for v in s["by_status"].values())
    assert s["recent"] == []


# --- вывод ------------------------------------------------------------------

def test_text_output_includes_totals_and_counts(tmp_path, capsys):
    path = _write(tmp_path, [_fc(1, status="WAIT"),
                             _fc(2, status="ENTER",
                                 close="2026-07-05T16:00:00+00:00")])
    rc = rep.main(["--input", path])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Всего forecasts:" in out
    assert "По статусу жизненного цикла:" in out
    assert "UPGRADED" in out  # WAIT->ENTER


def test_json_output_valid_and_has_counts(tmp_path, capsys):
    path = _write(tmp_path, [_fc(1), _fc(2, close="2026-07-05T16:00:00+00:00")])
    rc = rep.main(["--input", path, "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total_forecasts"] == 2
    assert "by_status" in data and set(rep.STATUSES) <= set(data["by_status"])
    assert "top_reasons" in data and "by_stream" in data


# --- группировка и сортировка ----------------------------------------------

def test_groups_by_symbol_and_analysis_type(tmp_path):
    rows = [
        _fc(1, symbol="BTCUSDT", atype="SWING"),
        _fc(2, symbol="BTCUSDT", atype="INTRADAY"),
        _fc(3, symbol="ETHUSDT", atype="SWING"),
    ]
    s = rep.summarize(rows, rep.Thresholds())
    assert set(s["by_stream"]) == {
        "BTCUSDT|SWING", "BTCUSDT|INTRADAY", "ETHUSDT|SWING"}
    # каждый одиночный ряд в своём потоке -> NEW_SETUP
    assert s["by_status"]["NEW_SETUP"] == 3


def test_sorts_deterministically_before_classifying():
    # Подаём вперемешку по времени; правильный порядок -> WAIT затем ENTER = UPGRADED.
    late = _fc(2, status="ENTER", close="2026-07-05T16:00:00+00:00")
    early = _fc(1, status="WAIT", close="2026-07-05T12:00:00+00:00")
    recs = rep.build_transitions([late, early], rep.Thresholds())
    # порядок классификации: early(NEW_SETUP) -> late(UPGRADED)
    assert recs[0]["current_id"] == 1 and recs[0]["status"] == "NEW_SETUP"
    assert recs[1]["current_id"] == 2 and recs[1]["status"] == "UPGRADED"
    assert recs[1]["previous_id"] == 1


def test_previous_is_same_stream_only():
    rows = [
        _fc(1, symbol="BTCUSDT", atype="SWING", status="WAIT"),
        _fc(2, symbol="ETHUSDT", atype="SWING", status="ENTER",
            close="2026-07-05T16:00:00+00:00"),
    ]
    recs = rep.build_transitions(rows, rep.Thresholds())
    # разные symbol -> оба NEW_SETUP, previous не протекает между потоками
    assert all(r["status"] == "NEW_SETUP" for r in recs)
    assert all(r["previous_id"] is None for r in recs)


# --- limit и пороги ---------------------------------------------------------

def test_respects_limit_for_recent(tmp_path):
    rows = [_fc(i, close=f"2026-07-05T{12 + i:02d}:00:00+00:00")
            for i in range(1, 6)]
    s = rep.summarize(rows, rep.Thresholds(), limit=2)
    assert len(s["recent"]) == 2
    # самый свежий первым
    assert s["recent"][0]["current_id"] == 5


def test_respects_threshold_cli_args(tmp_path, capsys):
    # confidence 0.60 -> 0.68 (+0.08). С порогом 0.05 = UPGRADED, с 0.10 = CONTINUATION.
    rows = [_fc(1, conf=0.60),
            _fc(2, conf=0.68, close="2026-07-05T16:00:00+00:00")]
    path = _write(tmp_path, rows)

    rep.main(["--input", path, "--json", "--confidence-delta", "0.05"])
    up = json.loads(capsys.readouterr().out)
    assert up["by_status"]["UPGRADED"] == 1

    rep.main(["--input", path, "--json", "--confidence-delta", "0.10"])
    cont = json.loads(capsys.readouterr().out)
    assert cont["by_status"]["UPGRADED"] == 0
    assert cont["by_status"]["CONTINUATION"] == 1


def test_max_gap_hours_marks_expired(tmp_path, capsys):
    rows = [_fc(1, status="WAIT", close="2026-07-05T12:00:00+00:00"),
            _fc(2, status="WAIT", close="2026-07-05T12:30:00+00:00")]  # +30m
    path = _write(tmp_path, rows)
    rep.main(["--input", path, "--json", "--max-gap-hours", "0.25"])  # 15m окно
    data = json.loads(capsys.readouterr().out)
    assert data["by_status"]["EXPIRED"] == 1


# --- ошибки входа -----------------------------------------------------------

def test_invalid_json_exits_nonzero(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    rc = rep.main(["--input", str(p)])
    assert rc != 0
    assert "JSON" in capsys.readouterr().err


def test_missing_file_exits_nonzero(tmp_path, capsys):
    rc = rep.main(["--input", str(tmp_path / "nope.json")])
    assert rc != 0
    assert "не найден" in capsys.readouterr().err


# --- изоляция и чистота ------------------------------------------------------

def test_no_forbidden_imports():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("database", "scheduler", "pipeline", "bot", "signal_engine",
                 "ai", "risk", "asyncpg", "config")
    for name in imported:
        top = name.split(".")[0]
        assert top not in forbidden, f"запрещённый импорт: {name}"


def test_does_not_mutate_rows():
    rows = [_fc(1), _fc(2, status="ENTER", close="2026-07-05T16:00:00+00:00")]
    snapshot = json.dumps(rows, sort_keys=True)
    rep.summarize(rows, rep.Thresholds())
    rep.build_transitions(rows, rep.Thresholds())
    rep.sort_forecasts(rows)
    assert json.dumps(rows, sort_keys=True) == snapshot


def test_unknown_top_level_type_errors(tmp_path):
    path = _write(tmp_path, 42)
    with pytest.raises(rep.InputError):
        rep.load_forecasts(path)
