"""Stage 8: offline data-quality auditor (Option B).

Доказывает: coverage считается корректно чистыми функциями; NULL честно
учитываются; статус планируемого поля повышается до PRESENT только когда поле
реально заполнено; JSON-путь и пустой датасет работают; источник read-only
(нет пишущих SQL-глаголов); аудитор не импортируется production-кодом; golden
не трогаются; CLI выходит с кодом 0.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import tools.data_quality_audit as dqa

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "forecast_metrics_sample.json"
GOLDEN_DIR = REPO / "tests" / "golden"


@pytest.fixture(scope="module")
def data() -> dict:
    return dqa.load_json(str(FIXTURE))


@pytest.fixture(scope="module")
def report(data) -> dict:
    return dqa.audit(data)


# --- coverage math ----------------------------------------------------------

def test_field_coverage_counts_and_percent():
    rows = [
        {"a": 1, "b": None},
        {"a": 2, "b": 5},
        {"a": None, "b": 7},
    ]
    cov = dqa.field_coverage(rows)
    assert cov["a"] == {"non_null": 2, "total": 3, "nulls": 1, "coverage_pct": 66.67}
    assert cov["b"] == {"non_null": 2, "total": 3, "nulls": 1, "coverage_pct": 66.67}


def test_field_coverage_counts_absent_key_as_null():
    # ключ отсутствует в части строк -> считается NULL от общего числа строк.
    rows = [{"a": 1}, {"a": 2, "b": 9}]
    cov = dqa.field_coverage(rows)
    assert cov["b"]["non_null"] == 1
    assert cov["b"]["total"] == 2
    assert cov["b"]["nulls"] == 1


def test_field_coverage_empty_rows():
    cov = dqa.field_coverage([])
    assert cov == {}


def test_table_coverage_matches_fixture(report):
    tc = report["table_coverage"]
    assert tc["forecasts"]["rows"] == 8
    assert tc["forecast_outcomes"]["rows"] == 6
    assert tc["trades_journal"]["rows"] == 4
    # blocked_gate заполнен только у 2 из 8 прогнозов (NO_TRADE строки).
    assert tc["forecasts"]["fields"]["blocked_gate"]["coverage_pct"] == 25.0
    assert tc["forecasts"]["fields"]["analysis_status"]["coverage_pct"] == 100.0


def test_forecast_context_sidecar_not_persisted_yet(report):
    ctx = report["table_coverage"]["forecast_context"]
    assert ctx["rows"] == 0
    assert ctx["source"] == "not_persisted_yet"
    assert ctx["fields"] == {}


# --- planned field detection ------------------------------------------------

def _by_field(report: dict) -> dict:
    return {p["field"]: p for p in report["planned_fields"]}


def test_missing_planned_field_detection(report):
    by = _by_field(report)
    # market_regime считается в runtime, но не персистится в фикстуре.
    assert by["market_regime"]["status"] == dqa.RUNTIME_NOT_PERSISTED
    assert by["strategy_version"]["status"] == dqa.MISSING
    assert by["realized_r"]["status"] == dqa.NOT_AVAILABLE_YET


def test_present_planned_field_detection():
    # если планируемое поле реально заполнено в экспорте -> статус PRESENT.
    data = {
        "forecasts": [{"id": 1, "market_regime": "trend_up"}],
        "outcomes": [],
        "trades": [],
    }
    by = _by_field(dqa.audit(data))
    assert by["market_regime"]["status"] == dqa.PRESENT


def test_present_requires_non_null_value():
    # колонка есть, но значение NULL -> НЕ present, остаётся default_status.
    data = {"forecasts": [{"id": 1, "market_regime": None}], "outcomes": [], "trades": []}
    by = _by_field(dqa.audit(data))
    assert by["market_regime"]["status"] == dqa.RUNTIME_NOT_PERSISTED


def test_critical_missing_fields_are_the_expected_five(report):
    names = {p["field"] for p in report["critical_missing_fields"]}
    assert names == {
        "market_regime", "volatility_regime",
        "strategy_version", "context_version", "realized_r",
    }


def test_safe_candidates_exclude_not_recommended(report):
    names = {p["field"] for p in report["safe_stage9_candidates"]}
    # рекомендованные попадают, dangerous/redundant — нет.
    assert "market_regime" in names
    assert "confidence_bucket" not in names
    assert "tp3" not in names
    assert "expected_r" not in names


def test_not_recommended_fields(report):
    names = {p["field"] for p in report["fields_not_recommended"]}
    assert names == {
        "confidence_bucket", "tp3", "expected_r", "model_prompt_version_table",
    }
    for p in report["fields_not_recommended"]:
        assert p["recommended"] is False


# --- JSON input path / empty dataset ----------------------------------------

def test_load_json_defaults_missing_keys(tmp_path):
    f = tmp_path / "partial.json"
    f.write_text('{"forecasts": [{"id": 1}]}', encoding="utf-8")
    data = dqa.load_json(str(f))
    assert data["forecasts"] == [{"id": 1}]
    assert data["outcomes"] == []
    assert data["trades"] == []


def test_empty_dataset_runs_and_marks_all_missing():
    report = dqa.audit({"forecasts": [], "outcomes": [], "trades": []})
    # ни одно планируемое поле не может быть present без данных.
    assert all(p["status"] != dqa.PRESENT for p in report["planned_fields"])
    assert report["table_coverage"]["forecasts"]["rows"] == 0


def test_format_report_is_string(report):
    text = dqa.format_report(report)
    assert "data-quality audit" in text
    assert "CRITICAL MISSING" in text


# --- read-only guarantees ---------------------------------------------------

def test_source_has_no_write_sql_verbs():
    src = (REPO / "tools" / "data_quality_audit.py").read_text(encoding="utf-8")
    forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|TRUNCATE|"
                           r"UPSERT|MERGE|CREATE)\b", src, re.IGNORECASE)
    assert not forbidden, f"найдены пишущие SQL-глаголы: {set(forbidden)}"


def test_module_does_not_import_database():
    src = (REPO / "tools" / "data_quality_audit.py").read_text(encoding="utf-8")
    assert "import database" not in src
    assert "from database" not in src


def test_auditor_not_imported_by_production():
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        if "data_quality_audit" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"production ссылается на аудитор: {offenders}"


# --- no golden changes ------------------------------------------------------

def _golden_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_running_audit_does_not_touch_golden():
    before = _golden_digest()
    dqa.main(["--input", str(FIXTURE)])
    dqa.audit(dqa.load_json(str(FIXTURE)))
    assert _golden_digest() == before


# --- CLI --------------------------------------------------------------------

def test_cli_exit_zero_on_fixture():
    assert dqa.main(["--input", str(FIXTURE)]) == 0


def test_cli_exit_zero_json_mode():
    assert dqa.main(["--input", str(FIXTURE), "--json"]) == 0
