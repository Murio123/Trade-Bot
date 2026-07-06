"""Stage 16B: tests for the offline trade pattern report CLI."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import tools.trade_patterns_report as cli

MODULE_PATH = Path(cli.__file__)


# ---------------------------------------------------------------------------
# Помощники: строки с достоверным исходом (совпадают с ядром journal_classify)
# ---------------------------------------------------------------------------

def _bad(confirming) -> dict:
    return {"analysis_status": "ENTER", "candidate_direction": "long",
            "stop_loss": 100.0, "take_profit_levels": [110.0, 120.0],
            "confirming_factors": list(confirming),
            "outcome": {"resolved": True, "realized_r": -1.0}}


def _good(confirming) -> dict:
    return {"analysis_status": "ENTER", "candidate_direction": "long",
            "stop_loss": 100.0, "take_profit_levels": [110.0, 120.0],
            "confirming_factors": list(confirming),
            "outcome": {"resolved": True, "realized_r": 2.0}}


def _missed(contradicting) -> dict:
    return {"analysis_status": "WAIT", "candidate_direction": "long",
            "stop_loss": 100.0, "take_profit_levels": [110.0, 120.0],
            "contradicting_factors": list(contradicting),
            "outcome": {"resolved": True, "tp2_hit": True, "tp1_hit": True,
                        "stop_hit": False}}


def _dataset() -> list[dict]:
    rows = [_bad(["toxic"]) for _ in range(3)]
    rows += [_good(["clean"]) for _ in range(3)]
    rows += [_missed(["fomo"]) for _ in range(3)]
    return rows


def _write(path: Path, rows, *, jsonl: bool) -> str:
    if jsonl:
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    else:
        path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Загрузка входа
# ---------------------------------------------------------------------------

def test_loads_json_array(tmp_path):
    path = _write(tmp_path / "rows.json", _dataset(), jsonl=False)
    rows = cli.load_rows(path)
    assert len(rows) == 9


def test_loads_jsonl(tmp_path):
    path = _write(tmp_path / "rows.jsonl", _dataset(), jsonl=True)
    rows = cli.load_rows(path)
    assert len(rows) == 9


def test_skips_empty_jsonl_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    lines = [json.dumps(_bad(["toxic"])), "", "   ", json.dumps(_good(["clean"]))]
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    rows = cli.load_rows(str(path))
    assert len(rows) == 2


def test_single_json_object_wrapped(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps(_bad(["toxic"])), encoding="utf-8")
    assert len(cli.load_rows(str(path))) == 1


def test_empty_file_returns_no_rows(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("   \n", encoding="utf-8")
    assert cli.load_rows(str(path)) == []


def test_malformed_json_raises_input_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{bad json here\n', encoding="utf-8")
    with pytest.raises(cli.InputError):
        cli.load_rows(str(path))


def test_missing_file_raises_input_error(tmp_path):
    with pytest.raises(cli.InputError):
        cli.load_rows(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# main() коды возврата и вывод
# ---------------------------------------------------------------------------

def test_malformed_json_exits_non_zero(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    code = cli.main(["--input", str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "Ошибка входа" in err


def test_json_flag_prints_valid_json(tmp_path, capsys):
    path = _write(tmp_path / "rows.json", _dataset(), jsonl=False)
    code = cli.main(["--input", path, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "summary" in payload and "patterns" in payload
    assert payload["summary"]["bad_trade_count"] == 3
    assert payload["summary"]["missed_opportunity_count"] == 3


def test_text_output_includes_summary_and_sections(tmp_path, capsys):
    path = _write(tmp_path / "rows.json", _dataset(), jsonl=False)
    code = cli.main(["--input", path])
    assert code == 0
    out = capsys.readouterr().out
    assert "Сводка:" in out
    assert "Топ факторов плохих сделок" in out
    assert "Топ факторов пропущенных возможностей" in out
    assert "toxic" in out
    assert "fomo" in out


def test_limit_limits_patterns_per_section(tmp_path, capsys):
    # Много разных токсичных факторов -> секция bad должна урезаться до --limit.
    rows = []
    for i in range(6):
        rows += [_bad([f"toxic{i}"]) for _ in range(3)]
    rows += [_good(["clean"]) for _ in range(3)]
    path = _write(tmp_path / "rows.json", rows, jsonl=False)
    code = cli.main(["--input", path, "--limit", "2"])
    assert code == 0
    out = capsys.readouterr().out
    bad_section = out.split("Топ факторов плохих сделок")[1].split(
        "Топ факторов пропущенных")[0]
    factor_lines = [ln for ln in bad_section.splitlines()
                    if ln.strip().startswith("confirming_factors=")]
    assert len(factor_lines) == 2


def test_min_count_marks_low_sample(tmp_path, capsys):
    # Один редкий токсичный фактор (total=1) при min-count=3 -> low_sample.
    rows = _dataset() + [_bad(["rare"])]
    path = _write(tmp_path / "rows.json", rows, jsonl=False)
    code = cli.main(["--input", path, "--min-count", "3", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    rare = next(p for p in payload["patterns"]
                if p["key"] == "confirming_factors:rare")
    assert rare["low_sample"] is True


def test_low_sample_marked_in_text(tmp_path, capsys):
    rows = _dataset() + [_bad(["rare"])]
    path = _write(tmp_path / "rows.json", rows, jsonl=False)
    cli.main(["--input", path, "--min-count", "3"])
    out = capsys.readouterr().out
    rare_line = next(ln for ln in out.splitlines()
                     if "confirming_factors=rare" in ln)
    assert "[low_sample]" in rare_line


# ---------------------------------------------------------------------------
# Изоляция: read-only CLI без runtime/exchange-зависимостей
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_ROOTS = {
    "database", "scheduler", "signal_engine", "risk", "bot", "ai",
}

_FORBIDDEN_TOKENS = (
    "create_order", "market_order", "limit_order", "api_key", "api_secret",
    "exchange.create",
)


def _imported_roots() -> set[str]:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_cli_does_not_import_runtime_layers():
    assert _imported_roots().isdisjoint(_FORBIDDEN_IMPORT_ROOTS)


def test_cli_has_no_forbidden_tokens():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_TOKENS:
        assert token not in source, f"forbidden token present: {token}"
