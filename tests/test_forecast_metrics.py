"""Stage 6: offline forecast outcome analytics (Option B).

Доказывает: метрики считаются корректно чистыми функциями над синтетикой,
censored-строки не портят статистику, equity/drawdown берётся ТОЛЬКО из
непересекающихся journal-сделок, CLI работает на пустых/непустых данных,
источник read-only (нет пишущих SQL-глаголов), runner не импортируется
production-кодом, golden не трогаются.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import tools.forecast_metrics as fm

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "forecast_metrics_sample.json"
GOLDEN_DIR = REPO / "tests" / "golden"


@pytest.fixture(scope="module")
def data() -> dict:
    return fm.load_json(str(FIXTURE))


@pytest.fixture(scope="module")
def metrics(data) -> dict:
    return fm.compute_all(data)


# --- core -------------------------------------------------------------------

def test_core_counts_and_coverage(metrics):
    core = metrics["core"]
    assert core["total_forecasts"] == 8
    assert core["counts"] == {"ENTER": 5, "WAIT": 1, "NO_TRADE": 2}
    assert core["coverage_enter_pct"] == 62.5
    assert core["measured"] == 6
    assert core["resolved"] == 5
    assert core["unresolved_censored"] == 1


def test_core_hit_rates_over_resolved(metrics):
    core = metrics["core"]
    assert core["tp1_hit_rate_pct"] == 80.0
    assert core["tp2_hit_rate_pct"] == 60.0
    assert core["stop_hit_rate_pct"] == 20.0


def test_core_direction_accuracy_and_precision(metrics):
    core = metrics["core"]
    assert core["direction_accuracy_pct"] == 80.0
    assert core["direction_accuracy_n"] == 5
    assert core["enter_precision_pct"] == 75.0
    assert core["enter_precision_n"] == 4


def test_core_mfe_mae(metrics):
    core = metrics["core"]
    assert core["avg_mfe_points"] == 1350.0
    assert core["avg_mae_points"] == pytest.approx(716.6667, abs=1e-3)
    assert core["mfe_mae_ratio"] == pytest.approx(1.8837, abs=1e-3)


# --- trade journal (clean R + equity) ---------------------------------------

def test_journal_metrics(metrics):
    j = metrics["journal"]
    assert j["total_trades"] == 4
    assert (j["wins"], j["losses"], j["breakeven"]) == (3, 1, 0)
    assert j["avg_r"] == 2.0
    assert j["profit_factor"] == 9.0
    assert j["max_drawdown_r"] == -1.0
    assert j["final_equity_r"] == 8.0


def test_equity_drawdown_only_from_journal_not_forecasts(data):
    """Equity/drawdown не строится из перекрывающихся forecast-горизонтов —
    только из непересекающихся закрытых journal-сделок."""
    # У прогнозов НЕТ pnl_r -> journal_metrics(без trades) даёт пустую equity.
    empty = fm.journal_metrics([])
    assert empty["total_trades"] == 0
    assert empty["max_drawdown_r"] == 0.0
    assert empty["final_equity_r"] == 0.0
    # forecasts не участвуют в journal-метриках вообще.
    only_forecast_shaped = fm.journal_metrics(
        [{"analysis_status": "ENTER", "mfe_points": 999}])
    assert only_forecast_shaped["total_trades"] == 0


# --- censored / unresolved ---------------------------------------------------

def test_unresolved_is_censored_not_counted_in_hit_rates():
    """Unresolved outcome входит в measured, но не в resolved-hit-rate."""
    forecasts = [{"id": 1, "analysis_status": "ENTER", "candidate_direction": "long",
                  "raw_confidence": 0.6}]
    outcomes = [{"forecast_id": 1, "mfe_points": 100.0, "mae_points": 50.0,
                 "tp1_hit": False, "tp2_hit": False, "stop_hit": False,
                 "return_72h": None, "resolved": False}]
    core = fm.core_metrics(forecasts, outcomes)
    assert core["measured"] == 1
    assert core["resolved"] == 0
    assert core["unresolved_censored"] == 1
    assert core["tp1_hit_rate_pct"] is None  # нет резолвнутых -> честный None
    assert core["direction_accuracy_n"] == 0


# --- buckets ----------------------------------------------------------------

def test_bucket_by_analysis_type(metrics):
    b = metrics["buckets"]["by_analysis_type"]
    assert set(b) == {"SWING", "INTRADAY"}
    assert b["SWING"]["n"] == 7
    assert b["SWING"]["win_rate_pct"] == 80.0
    assert b["INTRADAY"]["resolved"] == 0


def test_bucket_blocked_gate_counts(metrics):
    assert metrics["buckets"]["by_blocked_gate"] == {"diversity": 1, "htf_filter": 1}


def test_bucket_confidence_and_score(metrics):
    conf = metrics["buckets"]["by_confidence_bucket"]
    assert conf["unknown"]["n"] == 2  # NO_TRADE без confidence
    score = metrics["buckets"]["by_score_bucket"]
    assert score["[9,inf)"]["n"] == 1  # intraday score 9.0


# --- calibration ------------------------------------------------------------

def test_calibration(metrics):
    c = metrics["calibration"]
    assert c["n_resolved_with_confidence"] == 5
    assert c["raw_brier_score"] == pytest.approx(0.1932, abs=1e-3)
    assert c["mean_confidence"] == 0.6
    assert c["realized_win_rate"] == 0.8
    assert c["overconfidence"] == -0.2  # underconfident на этой выборке
    assert "откалиброванная" in c["note"]


# --- execution realism ------------------------------------------------------

def test_execution_slippage_and_costs(metrics):
    e = metrics["execution"]
    assert e["close_vs_executable"]["n"] == 8
    assert e["close_vs_executable"]["avg_signed_points"] == pytest.approx(9.375, abs=1e-3)
    assert e["modeled_round_trip_cost_pct"] == 0.13
    assert e["net_after_costs"]["n"] == 5


def test_execution_freshness_impact_buckets(metrics):
    impact = metrics["execution"]["stale_freshness_impact"]
    assert impact["[0,300)"]["n"] == 6
    assert impact["[300,900)"]["win_rate_pct"] == 0.0


# --- missing data section ---------------------------------------------------

def test_missing_data_declared(metrics):
    names = {m["metric"] for m in metrics["missing_data"]}
    assert {"market_regime bucket", "volatility_regime bucket",
            "calibrated_confidence metrics", "TP3 metrics",
            "honest per-forecast R", "full live-execution PnL"} <= names


# --- CLI --------------------------------------------------------------------

def test_cli_exit_zero_on_fixture(capsys):
    code = fm.main(["--input", str(FIXTURE)])
    out = capsys.readouterr().out
    assert code == 0
    assert "forecast outcome analytics" in out
    assert "MISSING / UNAVAILABLE" in out


def test_cli_exit_zero_on_empty(capsys):
    code = fm.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "total_forecasts: 0" in out


def test_cli_json_output(capsys):
    code = fm.main(["--input", str(FIXTURE), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    parsed = json.loads(out)
    assert parsed["core"]["total_forecasts"] == 8


# --- read-only guarantees ---------------------------------------------------

def test_source_has_no_write_sql_verbs():
    src = (REPO / "tools" / "forecast_metrics.py").read_text(encoding="utf-8")
    forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|TRUNCATE|"
                           r"UPSERT|MERGE|CREATE)\b", src, re.IGNORECASE)
    assert not forbidden, f"найдены пишущие SQL-глаголы: {set(forbidden)}"


def test_module_does_not_import_database():
    src = (REPO / "tools" / "forecast_metrics.py").read_text(encoding="utf-8")
    assert "import database" not in src
    assert "from database" not in src


def test_runner_not_imported_by_production():
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        if "forecast_metrics" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"production ссылается на runner: {offenders}"


# --- no golden changes ------------------------------------------------------

def _golden_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_running_metrics_does_not_touch_golden():
    before = _golden_digest()
    fm.main(["--input", str(FIXTURE)])
    fm.compute_all(fm.load_json(str(FIXTURE)))
    assert _golden_digest() == before
