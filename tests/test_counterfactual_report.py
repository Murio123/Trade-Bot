"""Stage 13B Step 2: read-only offline counterfactual CLI.

Доказывает: агрегация avoided_loss / missed_opportunity / no_levels /
no_direction; ENTER пропускается; разбивки по type/gate; JSON и JSONL вход;
klines только из локального файла; детерминизм; пропуски не становятся
win/loss; источник read-only (нет сети/Binance/БД/write-SQL); golden не трогаются.
"""
from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tools.counterfactual_report as cr

UTC = timezone.utc
REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"
SOURCE = REPO / "tools" / "counterfactual_report.py"

START = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _rising_klines(hours: int = 80, price: float = 100_000.0,
                   step: float = 500.0, spread: float = 50.0) -> list[dict]:
    rows = []
    p = price
    for i in range(hours):
        ot = START + timedelta(hours=i)
        rows.append({"open_time": ot.isoformat(), "open": p,
                     "high": p + spread, "low": p - spread, "close": p + step})
        p += step
    return rows


def _forecasts() -> list[dict]:
    return [
        # 1) LONG на растущем рынке -> TP2 достигается -> missed_opportunity
        {"analysis_status": "NO_TRADE", "candidate_direction": "long",
         "analysis_type": "SWING", "blocked_gate": None,
         "market_regime": "trend", "volatility_regime": "normal",
         "decision_time": START.isoformat(), "reference_price": 100_000.0,
         "stop_loss": 95_000.0, "take_profit_levels": [101_000.0, 106_000.0]},
        # 2) SHORT на растущем рынке -> стоп сверху выбит -> avoided_loss
        {"analysis_status": "NO_TRADE", "candidate_direction": "short",
         "analysis_type": "SWING", "blocked_gate": "no_trade",
         "market_regime": "trend", "volatility_regime": "normal",
         "decision_time": START.isoformat(), "reference_price": 100_000.0,
         "stop_loss": 105_000.0, "take_profit_levels": [95_000.0, 90_000.0]},
        # 3) LONG, недавний якорь, недостижимые уровни -> unresolved
        {"analysis_status": "WAIT", "candidate_direction": "long",
         "analysis_type": "POSITIONAL", "blocked_gate": None,
         "market_regime": "range", "volatility_regime": "compression",
         "decision_time": (START + timedelta(hours=40)).isoformat(),
         "reference_price": 120_000.0, "stop_loss": 1.0,
         "take_profit_levels": [1_000_000_000.0]},
        # 4) нет уровней -> no_levels
        {"analysis_status": "NO_TRADE", "candidate_direction": "long",
         "analysis_type": "SWING", "blocked_gate": "htf_filter",
         "decision_time": START.isoformat(), "reference_price": 100_000.0,
         "stop_loss": None, "take_profit_levels": None},
        # 5) нет направления -> no_direction
        {"analysis_status": "NO_TRADE", "candidate_direction": None,
         "analysis_type": "SWING", "blocked_gate": "direction_conflict",
         "decision_time": START.isoformat(), "reference_price": 100_000.0,
         "stop_loss": 95_000.0, "take_profit_levels": [101_000.0]},
        # 6) ENTER -> skipped
        {"analysis_status": "ENTER", "candidate_direction": "long",
         "analysis_type": "SWING", "blocked_gate": None,
         "decision_time": START.isoformat(), "reference_price": 100_000.0,
         "stop_loss": 95_000.0, "take_profit_levels": [101_000.0, 106_000.0]},
    ]


def _summ():
    df = cr._build_klines_df(_rising_klines())
    now = cr._resolve_now(df, None)
    return cr.summarize(_forecasts(), df, now)


# --- 1-5,14 core categories -------------------------------------------------

def test_aggregates_avoided_loss():
    assert _summ()["avoided_loss"] == 1


def test_aggregates_missed_opportunity():
    assert _summ()["missed_opportunity"] == 1


def test_counts_no_levels():
    assert _summ()["no_levels"] == 1


def test_counts_no_direction():
    assert _summ()["no_direction"] == 1


def test_skips_enter():
    s = _summ()
    assert s["skipped_enter"] == 1
    # ENTER не попал в evaluated-категории (evaluated = avoided+missed+none).
    assert s["evaluated"] == 2  # avoided_loss(1) + missed_opportunity(1)


def test_counts_unresolved_not_as_win_or_loss():
    s = _summ()
    assert s["unresolved"] == 1
    # пропуски/censored не раздули win/loss
    assert s["avoided_loss"] == 1 and s["missed_opportunity"] == 1
    assert s["total_rows"] == 6


# --- 6-7 breakdowns ---------------------------------------------------------

def test_breakdown_by_analysis_type():
    bd = _summ()["by_analysis_type"]
    assert bd["POSITIONAL"] == {"unresolved": 1}
    assert bd["SWING"]["avoided_loss"] == 1
    assert bd["SWING"]["missed_opportunity"] == 1


def test_breakdown_by_blocked_gate():
    bd = _summ()["by_blocked_gate"]
    assert bd["no_trade"] == {"avoided_loss": 1}
    assert bd["htf_filter"] == {"no_levels": 1}
    assert bd["direction_conflict"] == {"no_direction": 1}


# --- 8-10 input formats -----------------------------------------------------

def test_json_input(tmp_path, capsys):
    fp = tmp_path / "f.json"
    fp.write_text(json.dumps({"forecasts": _forecasts()}), encoding="utf-8")
    kp = tmp_path / "k.json"
    kp.write_text(json.dumps(_rising_klines()), encoding="utf-8")
    assert cr.main(["--input", str(fp), "--klines", str(kp), "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["missed_opportunity"] == 1 and s["avoided_loss"] == 1


def test_jsonl_input(tmp_path, capsys):
    fp = tmp_path / "f.jsonl"
    fp.write_text("\n".join(json.dumps(r) for r in _forecasts()), encoding="utf-8")
    kp = tmp_path / "k.json"
    kp.write_text(json.dumps(_rising_klines()), encoding="utf-8")
    assert cr.main(["--input", str(fp), "--klines", str(kp), "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["total_rows"] == 6
    assert s["skipped_enter"] == 1


def test_csv_klines_only(tmp_path, capsys):
    fp = tmp_path / "f.json"
    fp.write_text(json.dumps(_forecasts()), encoding="utf-8")
    kp = tmp_path / "k.csv"
    lines = ["open_time,open,high,low,close"]
    for r in _rising_klines():
        lines.append(f"{r['open_time']},{r['open']},{r['high']},{r['low']},{r['close']}")
    kp.write_text("\n".join(lines), encoding="utf-8")
    assert cr.main(["--input", str(fp), "--klines", str(kp)]) == 0
    out = capsys.readouterr().out
    assert "avoided_loss" in out and "missed_opportunity" in out


# --- 13 determinism ---------------------------------------------------------

def test_deterministic():
    assert _summ() == _summ()


# --- 11-12 read-only / offline guarantees -----------------------------------

def test_no_network_or_db_imports():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    tops = {n.split(".")[0] for n in imported}
    assert not (tops & {"database", "scheduler", "pipeline", "signal_engine",
                        "risk", "bot", "contracts"})
    assert not (tops & {"asyncpg", "aiohttp", "requests", "httpx", "websockets"})


def test_no_binance_or_write_sql_in_code():
    # Исполняемый код (без строк/комментариев) не содержит сети/write-SQL.
    import io
    import tokenize
    skip = {tokenize.STRING, tokenize.COMMENT}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            skip.add(getattr(tokenize, name))
    code_tokens = []
    with SOURCE.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type not in skip:
                code_tokens.append(tok.string)
    code = " ".join(code_tokens).lower()
    assert "binance" not in code
    # Никаких БД-вызовов/курсоров/коннектов в исполняемом коде -> нет write-SQL.
    assert "execute" not in code
    assert "asyncpg" not in code
    assert ".connect" not in code


def test_runner_not_imported_by_production():
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        if "counterfactual_report" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, offenders


def _golden_digest() -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_does_not_touch_golden(tmp_path):
    before = _golden_digest()
    fp = tmp_path / "f.json"
    fp.write_text(json.dumps(_forecasts()), encoding="utf-8")
    kp = tmp_path / "k.json"
    kp.write_text(json.dumps(_rising_klines()), encoding="utf-8")
    cr.main(["--input", str(fp), "--klines", str(kp)])
    assert _golden_digest() == before
