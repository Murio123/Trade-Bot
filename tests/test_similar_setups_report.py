"""Stage 17B: tests for the offline similar-setups report CLI."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import tools.similar_setups_report as cli

MODULE_PATH = Path(cli.__file__)


# ---------------------------------------------------------------------------
# Помощники: строки с достоверным исходом (совместимы с journal_classify)
# ---------------------------------------------------------------------------

def _enter(*, id, realized_r, setup="reversal", analysis_type="swing",
           direction="long", regime="trend", confirming=None) -> dict:
    return {
        "id": id,
        "analysis_status": "ENTER",
        "candidate_direction": direction,
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
        "setup_type": setup,
        "analysis_type": analysis_type,
        "market_regime": regime,
        "volatility_regime": "high",
        "setup_lifecycle_status": "NEW_SETUP",
        "raw_confidence": 0.72,
        "confirming_factors": list(confirming or ["fvg", "ob"]),
        "outcome": {"resolved": True, "realized_r": realized_r},
    }


def _history() -> list[dict]:
    return [
        _enter(id=1, realized_r=2.0),
        _enter(id=2, realized_r=-1.0),
        _enter(id=3, realized_r=1.5),
        _enter(id=5, realized_r=3.0, setup="breakout", analysis_type="scalp",
               direction="short", regime="range", confirming=["news"]),
    ]


def _write(path: Path, data, *, jsonl: bool) -> str:
    if jsonl and isinstance(data, list):
        path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Загрузка входа
# ---------------------------------------------------------------------------

def test_loads_json_array(tmp_path):
    path = _write(tmp_path / "h.json", _history(), jsonl=False)
    assert len(cli.load_rows(path)) == 4


def test_loads_jsonl(tmp_path):
    path = _write(tmp_path / "h.jsonl", _history(), jsonl=True)
    assert len(cli.load_rows(path)) == 4


def test_skips_empty_jsonl_lines(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(_enter(id=1, realized_r=1.0)) + "\n\n   \n"
                    + json.dumps(_enter(id=2, realized_r=1.0)) + "\n",
                    encoding="utf-8")
    assert len(cli.load_rows(str(path))) == 2


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": 1}\n{broken\n', encoding="utf-8")
    with pytest.raises(cli.InputError):
        cli.load_rows(str(path))


def test_load_target_single_object(tmp_path):
    path = _write(tmp_path / "t.json", _enter(id=99, realized_r=9.9),
                  jsonl=False)
    assert cli.load_target(path)["id"] == 99


def test_load_target_json_array_one_object(tmp_path):
    path = _write(tmp_path / "t.json", [_enter(id=99, realized_r=9.9)],
                  jsonl=False)
    assert cli.load_target(str(path))["id"] == 99


def test_load_target_empty_raises(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(cli.InputError):
        cli.load_target(str(path))


def test_resolve_target_by_id():
    rows = _history()
    assert cli.resolve_target(rows, "2")["id"] == 2


def test_resolve_target_missing_id_raises():
    with pytest.raises(cli.InputError):
        cli.resolve_target(_history(), "404")


# ---------------------------------------------------------------------------
# main() коды возврата и вывод
# ---------------------------------------------------------------------------

def test_json_flag_with_target_file(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    tgt = _write(tmp_path / "t.json", _enter(id=99, realized_r=9.9),
                 jsonl=False)
    code = cli.main(["--input", hist, "--target", tgt, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"target", "summary", "matches"}
    assert payload["summary"]["candidates"] == 4


def test_target_id_resolves_and_excludes_self(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    code = cli.main(["--input", hist, "--target-id", "1", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"]["id"] == 1
    assert all(m["id"] != 1 for m in payload["matches"])


def test_text_output_includes_sections(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    tgt = _write(tmp_path / "t.json", _enter(id=99, realized_r=9.9),
                 jsonl=False)
    code = cli.main(["--input", hist, "--target", tgt])
    assert code == 0
    out = capsys.readouterr().out
    assert "Similar Setups Report" in out      # title
    assert "Target:" in out                     # target summary
    assert "Сводка по похожим:" in out          # summary
    assert "Похожие сетапы" in out              # similar setups section


def test_limit_caps_matches_shown(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    code = cli.main(["--input", hist, "--target-id", "1", "--limit", "1",
                     "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["matches"]) == 1


def test_min_score_filters(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    tgt = _write(tmp_path / "t.json", _enter(id=99, realized_r=9.9),
                 jsonl=False)
    code = cli.main(["--input", hist, "--target", tgt, "--min-score", "0.9",
                     "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {m["id"] for m in payload["matches"]}
    assert ids == {1, 2, 3}          # breakout(id=5) отфильтрован


def test_malformed_target_exits_non_zero(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    tgt = tmp_path / "bad_target.json"
    tgt.write_text("{not json", encoding="utf-8")
    code = cli.main(["--input", hist, "--target", str(tgt)])
    assert code == 2
    assert "Ошибка входа" in capsys.readouterr().err


def test_malformed_input_exits_non_zero(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    code = cli.main(["--input", str(path), "--target-id", "1"])
    assert code == 2
    assert "Ошибка входа" in capsys.readouterr().err


def test_missing_target_id_exits_non_zero(tmp_path, capsys):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    code = cli.main(["--input", hist, "--target-id", "404"])
    assert code == 2
    assert "Ошибка входа" in capsys.readouterr().err


def test_requires_target_or_target_id(tmp_path):
    hist = _write(tmp_path / "h.json", _history(), jsonl=False)
    with pytest.raises(SystemExit):        # argparse: required mutually excl.
        cli.main(["--input", hist])


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
