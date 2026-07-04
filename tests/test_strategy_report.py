"""Stage 7: offline strategy evaluation report (Option B).

Доказывает: evaluate() строит warnings/insights/recommendations_for_review
поверх forecast_metrics; каждое warning-правило срабатывает на synthetic dict
с достаточной выборкой и МОЛЧИТ при недостаточной; рекомендации помечены
«for review» и не содержат auto-action формулировок; CLI работает на
существующей фикстуре (text/markdown/json) и печатает sample size, censored
share и costs/drawdown disclaimer; источник read-only (нет пишущих SQL),
runner не импортируется production-кодом, golden не трогаются.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import tools.strategy_report as sr

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "forecast_metrics_sample.json"
GOLDEN_DIR = REPO / "tests" / "golden"


# ---------------------------------------------------------------------------
# Synthetic metrics builders (форма как у forecast_metrics.compute_all)
# ---------------------------------------------------------------------------

def _metrics(*, core=None, journal=None, buckets=None, calibration=None,
             execution=None) -> dict:
    return {
        "core": core or {},
        "journal": journal or {},
        "buckets": buckets or {},
        "calibration": calibration or {},
        "execution": execution or {},
        "missing_data": [{"metric": "TP3 metrics", "reason": "нет данных"}],
    }


def _codes(report: dict) -> set[str]:
    return {w["code"] for w in report["warnings"]}


# ---------------------------------------------------------------------------
# Структура evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_returns_expected_keys():
    report = sr.evaluate(_metrics())
    for key in ("sample", "warnings", "insights", "recommendations_for_review",
                "best_buckets", "worst_buckets"):
        assert key in report


# ---------------------------------------------------------------------------
# Каждое правило срабатывает при достаточной выборке
# ---------------------------------------------------------------------------

def test_low_sample_size_fires_on_small_sample():
    report = sr.evaluate(_metrics(core={"resolved": 3},
                                  journal={"total_trades": 2}))
    assert "low_sample_size" in _codes(report)
    assert report["sample"]["strong_conclusions_allowed"] is False


def test_low_coverage_fires():
    report = sr.evaluate(_metrics(
        core={"total_forecasts": 100, "coverage_enter_pct": 1.0, "resolved": 30},
        journal={"total_trades": 30}))
    assert "low_coverage" in _codes(report)


def test_weak_direction_accuracy_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30, "direction_accuracy_n": 40,
              "direction_accuracy_pct": 42.0},
        journal={"total_trades": 30}))
    assert "weak_direction_accuracy" in _codes(report)


def test_low_tp_hit_rate_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30, "tp1_hit_rate_pct": 25.0},
        journal={"total_trades": 30}))
    assert "low_tp_hit_rate" in _codes(report)


def test_high_stop_hit_rate_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30, "stop_hit_rate_pct": 55.0},
        journal={"total_trades": 30}))
    assert "high_stop_hit_rate" in _codes(report)


def test_negative_expectancy_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30},
        journal={"total_trades": 25, "expectancy_r": -0.4,
                 "max_drawdown_r": -8.0}))
    assert "negative_expectancy" in _codes(report)


def test_poor_mfe_mae_ratio_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30, "measured": 40, "mfe_mae_ratio": 0.6},
        journal={"total_trades": 30}))
    assert "poor_mfe_mae_ratio" in _codes(report)


def test_overconfidence_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30}, journal={"total_trades": 30},
        calibration={"n_resolved_with_confidence": 30, "overconfidence": 0.25}))
    assert "overconfidence" in _codes(report)


def test_stale_freshness_impact_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30}, journal={"total_trades": 30},
        execution={"stale_freshness_impact": {
            "[0,300)": {"n": 40, "win_rate_pct": 70.0},
            "[900,3600)": {"n": 15, "win_rate_pct": 30.0}}}))
    assert "stale_freshness_impact" in _codes(report)


def test_bad_bucket_fires():
    report = sr.evaluate(_metrics(
        core={"resolved": 30}, journal={"total_trades": 30},
        buckets={"by_timeframe": {
            "4h": {"resolved": 12, "win_rate_pct": 20.0, "avg_mfe": 100,
                   "avg_mae": 300}}}))
    assert "bad_bucket" in _codes(report)


# ---------------------------------------------------------------------------
# Правила МОЛЧАТ при недостаточной выборке (gates)
# ---------------------------------------------------------------------------

def test_rules_silent_below_sample_gates():
    # Значения «плохие», но выборки ниже порогов -> только low_sample_size.
    report = sr.evaluate(_metrics(
        core={"total_forecasts": 5, "coverage_enter_pct": 0.5, "resolved": 4,
              "measured": 4, "direction_accuracy_n": 4,
              "direction_accuracy_pct": 10.0, "tp1_hit_rate_pct": 5.0,
              "stop_hit_rate_pct": 90.0, "mfe_mae_ratio": 0.1},
        journal={"total_trades": 3, "expectancy_r": -2.0},
        calibration={"n_resolved_with_confidence": 4, "overconfidence": 0.9},
        buckets={"by_timeframe": {
            "4h": {"resolved": 2, "win_rate_pct": 0.0}}}))
    codes = _codes(report)
    assert codes == {"low_sample_size"}
    assert report["best_buckets"] == []
    assert report["worst_buckets"] == []


def test_bad_bucket_silent_when_bucket_sample_small():
    report = sr.evaluate(_metrics(
        core={"resolved": 30}, journal={"total_trades": 30},
        buckets={"by_timeframe": {
            "4h": {"resolved": 3, "win_rate_pct": 10.0}}}))
    assert "bad_bucket" not in _codes(report)


# ---------------------------------------------------------------------------
# Best/worst buckets только при strong sample
# ---------------------------------------------------------------------------

def test_best_worst_buckets_ranked_with_strong_sample():
    report = sr.evaluate(_metrics(
        core={"resolved": 40}, journal={"total_trades": 30},
        buckets={"by_analysis_type": {
            "SWING": {"resolved": 20, "win_rate_pct": 75.0, "avg_mfe": 1,
                      "avg_mae": 1},
            "INTRADAY": {"resolved": 15, "win_rate_pct": 30.0, "avg_mfe": 1,
                         "avg_mae": 1}}}))
    assert report["sample"]["strong_conclusions_allowed"] is True
    assert report["best_buckets"][0]["bucket"] == "by_analysis_type=SWING"
    assert report["worst_buckets"][0]["bucket"] == "by_analysis_type=INTRADAY"


# ---------------------------------------------------------------------------
# Рекомендации: for review + без auto-action формулировок
# ---------------------------------------------------------------------------

def test_recommendations_are_for_review_only():
    report = sr.evaluate(_metrics(
        core={"resolved": 30, "stop_hit_rate_pct": 60.0},
        journal={"total_trades": 25, "expectancy_r": -0.5,
                 "max_drawdown_r": -9.0}))
    recs = report["recommendations_for_review"]
    assert recs, "ожидались рекомендации для сработавших правил"
    forbidden = ("disable", "change threshold", "apply automatically",
                 "turn off", "increase weight", "auto-apply")
    for r in recs:
        assert "for review" in r.lower()
        low = r.lower()
        for bad in forbidden:
            assert bad not in low, f"auto-action формулировка: {bad!r} в {r!r}"


def test_positive_expectancy_insight_is_not_a_profitability_claim():
    report = sr.evaluate(_metrics(
        core={"resolved": 30},
        journal={"total_trades": 25, "expectancy_r": 0.8,
                 "max_drawdown_r": -3.0}))
    ins = [i for i in report["insights"]
           if i["code"] == "positive_expectancy_observed"]
    assert ins, "ожидался осторожный insight по expectancy"
    msg = ins[0]["message"].lower()
    assert "not a profitability claim" in msg
    assert "drawdown" in msg


# ---------------------------------------------------------------------------
# CLI + отчёт
# ---------------------------------------------------------------------------

def test_cli_exit_zero_on_fixture(capsys):
    code = sr.main(["--input", str(FIXTURE)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Strategy Evaluation Report" in out


def test_cli_markdown_and_json(capsys):
    assert sr.main(["--input", str(FIXTURE), "--format", "markdown"]) == 0
    md = capsys.readouterr().out
    assert md.startswith("# Strategy Evaluation Report")

    assert sr.main(["--input", str(FIXTURE), "--format", "json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert "metrics" in parsed and "evaluation" in parsed
    assert parsed["evaluation"]["sample"]["resolved"] == 5


def test_cli_empty_input_exit_zero(capsys):
    assert sr.main([]) == 0
    out = capsys.readouterr().out
    assert "Strategy Evaluation Report" in out


def test_report_contains_sample_censored_and_cost_disclaimer(capsys):
    sr.main(["--input", str(FIXTURE)])
    out = capsys.readouterr().out.lower()
    assert "sample size" in out
    assert "censored" in out
    assert "drawdown" in out
    assert "cost" in out
    assert "do not guarantee future performance" in out


# ---------------------------------------------------------------------------
# Read-only guarantees
# ---------------------------------------------------------------------------

def test_source_has_no_write_sql_verbs():
    src = (REPO / "tools" / "strategy_report.py").read_text(encoding="utf-8")
    forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|TRUNCATE|"
                           r"UPSERT|MERGE|CREATE)\b", src, re.IGNORECASE)
    assert not forbidden, f"найдены пишущие SQL-глаголы: {set(forbidden)}"


def test_module_does_not_import_database():
    src = (REPO / "tools" / "strategy_report.py").read_text(encoding="utf-8")
    assert "import database" not in src
    assert "from database" not in src


def test_runner_not_imported_by_production():
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        if "strategy_report" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"production ссылается на runner: {offenders}"


# ---------------------------------------------------------------------------
# Golden snapshots не меняются
# ---------------------------------------------------------------------------

def _golden_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_running_report_does_not_touch_golden():
    before = _golden_digest()
    sr.main(["--input", str(FIXTURE)])
    assert _golden_digest() == before
