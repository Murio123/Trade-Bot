"""Stage 13: read-only CLI/агрегация Trading Journal v2.

Доказывает: сводка считается корректно чистыми функциями над детерминированными
in-memory строками; источник read-only (нет пишущих SQL-глаголов, нет
database.py, нет сети/Binance); CLI работает на пустых и непустых данных; runner
не импортируется production-кодом; golden не трогаются.
"""
from __future__ import annotations

import hashlib
import io
import re
import tokenize
from pathlib import Path

import tools.trading_journal as tj

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"
SOURCE = REPO / "tools" / "trading_journal.py"


def _code_only(path: Path) -> str:
    """Исходник без строковых литералов и комментариев — чтобы проза docstring
    (где перечислены запрещённые глаголы/имена) не давала ложных срабатываний.
    Проверяем реально исполняемый код, а не документацию."""
    skip = {tokenize.STRING, tokenize.COMMENT}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            skip.add(getattr(tokenize, name))
    out = []
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in skip:
                continue
            out.append(tok.string)
    return " ".join(out)


def _forecast(fid, status, direction="long", **kw):
    base = {
        "id": fid, "analysis_type": "SWING", "analysis_status": status,
        "candidate_direction": direction, "final_bias": "LONG",
        "stop_loss": 100.0, "take_profit_levels": [110.0, 120.0],
        "blocked_gate": None, "no_trade_reasons": None,
        "market_regime": "trend", "volatility_regime": "normal",
    }
    base.update(kw)
    return base


def _outcome(fid, **kw):
    base = {"forecast_id": fid, "tp1_hit": False, "tp2_hit": False,
            "stop_hit": False, "resolved": True, "realized_r": None}
    base.update(kw)
    return base


def _dataset():
    forecasts = [
        _forecast(1, "ENTER"),                       # good
        _forecast(2, "ENTER"),                       # bad
        _forecast(3, "ENTER"),                       # unresolved
        _forecast(4, "WAIT"),                        # avoided_loss
        _forecast(5, "NO_TRADE"),                    # missed_opportunity
        _forecast(6, "NO_TRADE", stop_loss=None,     # no_levels
                  take_profit_levels=None),
        _forecast(7, "NO_TRADE", candidate_direction=None,
                  market_regime=None, volatility_regime=None),  # none / missing dir+regime
    ]
    outcomes = [
        _outcome(1, resolved=True, realized_r=2.0, tp2_hit=True),
        _outcome(2, resolved=True, realized_r=-1.0, stop_hit=True),
        _outcome(3, resolved=False),
        _outcome(4, resolved=True, stop_hit=True),
        _outcome(5, resolved=True, tp2_hit=True),
        # forecast 6, 7 без outcome
    ]
    return forecasts, outcomes


# --- агрегация --------------------------------------------------------------

def test_summary_counts_and_classes():
    forecasts, outcomes = _dataset()
    s = tj.summarize(forecasts, outcomes)

    assert s["total_forecasts"] == 7
    assert s["by_analysis_status"] == {"ENTER": 3, "NO_TRADE": 3, "WAIT": 1}
    assert s["by_analysis_type"] == {"SWING": 7}
    assert s["by_candidate_direction"] == {"long": 6, "unknown": 1}

    assert s["by_classification"]["good_trade"] == 1
    assert s["by_classification"]["bad_trade"] == 1
    assert s["by_classification"]["unresolved"] == 1
    assert s["by_classification"]["avoided_loss"] == 1
    assert s["by_classification"]["missed_opportunity"] == 1
    assert s["by_classification"]["no_levels"] == 1
    assert s["by_classification"]["none"] == 1


def test_enter_and_non_enter_breakdown():
    s = tj.summarize(*_dataset())
    assert s["enter"] == {"good_trade": 1, "bad_trade": 1,
                          "unresolved": 1, "none": 0}
    assert s["non_enter"] == {"avoided_loss": 1, "missed_opportunity": 1,
                              "no_levels": 1, "none": 1}


def test_avg_enter_realized_r():
    s = tj.summarize(*_dataset())
    # ENTER с посчитанным R: 2.0 и -1.0 (unresolved -> None не участвует).
    assert s["avg_enter_realized_r"] == 0.5


def test_missing_data_inventory():
    s = tj.summarize(*_dataset())
    md = s["missing_data"]
    assert md["missing_direction"] == 1          # forecast 7
    assert md["missing_levels"] == 1             # forecast 6
    assert md["missing_market_regime"] == 1      # forecast 7
    assert md["missing_volatility_regime"] == 1  # forecast 7
    # forecast 3 (ENTER, unresolved) + forecast 6 (NO_TRADE, no outcome)
    assert md["unresolved_outcomes"] == 2
    assert md["missing_realized_r"] == 0


def test_deterministic_aggregation():
    forecasts, outcomes = _dataset()
    a = tj.summarize(forecasts, outcomes)
    b = tj.summarize(forecasts, outcomes)
    assert a == b


# --- CLI --------------------------------------------------------------------

def test_cli_on_empty_data(capsys):
    assert tj.main([]) == 0
    out = capsys.readouterr().out
    assert "Всего forecasts: 0" in out


def test_cli_json_on_file(tmp_path, capsys):
    import json
    forecasts, outcomes = _dataset()
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"forecasts": forecasts, "outcomes": outcomes}),
                    encoding="utf-8")
    assert tj.main(["--input", str(path), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["total_forecasts"] == 7
    assert parsed["by_classification"]["good_trade"] == 1


def test_load_json_parses_stringified_levels(tmp_path):
    import json
    path = tmp_path / "e.json"
    path.write_text(json.dumps({
        "forecasts": [{"id": 1, "analysis_status": "ENTER",
                       "take_profit_levels": "[110.0, 120.0]"}],
        "outcomes": [],
    }), encoding="utf-8")
    data = tj.load_json(str(path))
    assert data["forecasts"][0]["take_profit_levels"] == [110.0, 120.0]


# --- read-only гарантии -----------------------------------------------------

def test_sql_constants_are_select_only():
    # Реальные SQL-константы модуля: только SELECT, никаких пишущих глаголов.
    for sql in (tj._SQL_FORECASTS, tj._SQL_OUTCOMES):
        assert sql.strip().upper().startswith("SELECT")
        forbidden = re.findall(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|TRUNCATE|"
                               r"UPSERT|MERGE|CREATE)\b", sql, re.IGNORECASE)
        assert not forbidden, f"пишущие SQL-глаголы в запросе: {set(forbidden)}"


def test_module_does_not_import_forbidden_or_network():
    import ast
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    tops = {n.split(".")[0] for n in imported}
    # Не тянет database.py / decision-path / producer runtime.
    assert not (tops & {"database", "scheduler", "pipeline", "signal_engine",
                        "risk", "bot", "contracts"})
    # Никакого сетевого/биржевого клиента.
    code = _code_only(SOURCE).lower()
    assert "binance" not in code
    # Единственная зависимость от проекта — чистый leaf-классификатор.
    assert any(n == "analyzer.journal_classify" for n in imported)


def test_runner_not_imported_by_production():
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        if "trading_journal" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"production ссылается на runner: {offenders}"


def _golden_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_running_does_not_touch_golden(tmp_path):
    import json
    before = _golden_digest()
    forecasts, outcomes = _dataset()
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"forecasts": forecasts, "outcomes": outcomes}),
                    encoding="utf-8")
    tj.main(["--input", str(path)])
    assert _golden_digest() == before
