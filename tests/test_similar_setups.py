"""Stage 17A: tests for the pure offline similar-setups history core."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from analyzer import similar_setups as ss

MODULE_PATH = Path(ss.__file__)


# ---------------------------------------------------------------------------
# Помощники: строки с достоверным исходом (совместимы с journal_classify)
# ---------------------------------------------------------------------------

_UNSET = object()


def _enter(*, realized_r, setup="reversal", analysis_type="swing",
           direction="long", regime="trend", vol="high",
           lifecycle="NEW_SETUP", conf=0.72, confirming=None,
           contradicting=None, reasons=None, outcome=_UNSET, **extra) -> dict:
    row = {
        "analysis_status": "ENTER",
        "candidate_direction": direction,
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
        "setup_type": setup,
        "analysis_type": analysis_type,
        "market_regime": regime,
        "volatility_regime": vol,
        "setup_lifecycle_status": lifecycle,
        "setup_lifecycle_reasons": list(reasons or []),
        "raw_confidence": conf,
        "confirming_factors": list(confirming or []),
        "contradicting_factors": list(contradicting or []),
        "outcome": {"resolved": True, "realized_r": realized_r}
        if outcome is _UNSET else outcome,
    }
    row.update(extra)
    return row


def _target() -> dict:
    return _enter(id=99, realized_r=9.9, confirming=["fvg", "ob"])


# ---------------------------------------------------------------------------
# Публичный API / контракт
# ---------------------------------------------------------------------------

def test_public_api_uses_limit_and_min_score_defaults():
    sig = inspect.signature(ss.find_similar_setups)
    assert "limit" in sig.parameters
    assert "top_k" not in sig.parameters
    assert sig.parameters["limit"].default == 10
    assert sig.parameters["min_score"].default == 0.0


def test_output_shape_target_summary_matches():
    result = ss.find_similar_setups(_target(), [_enter(id=1, realized_r=1.0)])
    assert set(result) == {"target", "summary", "matches"}
    required = {"candidates", "matches", "with_realized_r", "wins", "losses",
                "avg_realized_r", "median_realized_r", "best_realized_r",
                "worst_realized_r"}
    assert required <= set(result["summary"])


def test_match_object_fields():
    result = ss.find_similar_setups(_target(), [_enter(id=1, realized_r=1.0,
                                                       confirming=["fvg"])])
    m = result["matches"][0]
    required = {"id", "timestamp", "symbol", "analysis_type", "direction",
                "similarity_score", "shared_factors", "different_factors",
                "realized_r", "outcome_label", "setup_type", "market_regime",
                "volatility_regime", "confidence_bucket",
                "setup_lifecycle_status"}
    assert required <= set(m)
    assert "fvg" in m["shared_factors"]           # общий фактор
    assert "ob" in m["different_factors"]         # есть у target, нет у cand


# ---------------------------------------------------------------------------
# similarity()
# ---------------------------------------------------------------------------

def test_identical_setup_scores_one():
    a = _enter(realized_r=1.0, confirming=["fvg", "ob"])
    b = _enter(realized_r=-1.0, confirming=["fvg", "ob"])
    assert ss.similarity(a, b) == pytest.approx(1.0)


def test_completely_different_setup_scores_zero():
    a = _enter(realized_r=1.0)
    b = _enter(realized_r=1.0, setup="breakout", analysis_type="scalp",
               direction="short", regime="range", vol="low",
               lifecycle="INVALIDATED", conf=0.10, confirming=["news"])
    assert ss.similarity(a, b) == pytest.approx(0.0)


def test_setup_type_weight_dominates_direction():
    # Кандидат A: совпал только setup_type (вес 2.0).
    # Кандидат B: совпал только direction (вес 1.5). A должен быть похожее.
    target = {"setup_type": "reversal", "direction": "long"}
    a = {"setup_type": "reversal", "direction": "short"}
    b = {"setup_type": "breakout", "direction": "long"}
    assert ss.similarity(target, a) > ss.similarity(target, b)


def test_weighted_dimensions_meaningful():
    # Совпадение setup_type+direction+lifecycle+factors даёт высокий score,
    # рассогласование только по confidence bucket — незначительно снижает.
    target = _enter(realized_r=1.0, confirming=["fvg", "ob"])
    same_but_conf = _enter(realized_r=1.0, conf=0.10, confirming=["fvg", "ob"])
    score = ss.similarity(target, same_but_conf)
    # Присутствующий у target вес = 11.5; рассогласован только
    # confidence_bucket (1.0), поэтому score = 10.5 / 11.5.
    assert score == pytest.approx(10.5 / 11.5)


def test_similarity_zero_when_target_has_no_comparable_attrs():
    assert ss.similarity({}, _enter(realized_r=1.0)) == 0.0


def test_resolve_weights_merges_over_defaults():
    w = ss.resolve_weights({"setup_type": 5.0, "bogus": 9})
    assert w["setup_type"] == 5.0
    assert w["direction"] == 1.5
    assert "bogus" not in w


# ---------------------------------------------------------------------------
# find_similar_setups()
# ---------------------------------------------------------------------------

def _history() -> list[dict]:
    return [
        _enter(id=1, realized_r=2.0, confirming=["fvg", "ob"]),   # win
        _enter(id=2, realized_r=-1.0, confirming=["fvg", "ob"]),  # loss
        _enter(id=3, realized_r=1.5, confirming=["fvg", "ob"]),   # win
        _enter(id=4, realized_r=0.0, confirming=["fvg", "ob"]),   # none (R=0)
        _enter(id=5, realized_r=3.0, setup="breakout", analysis_type="scalp",
               direction="short", regime="range", vol="low",
               lifecycle="INVALIDATED", conf=0.10, confirming=["news"]),
    ]


def test_returns_json_serializable():
    result = ss.find_similar_setups(_target(), _history())
    assert json.loads(json.dumps(result)) == result


def test_default_min_score_shows_all_candidates():
    result = ss.find_similar_setups(_target(), _history())
    # min_score=0.0 -> все кандидаты (кроме target) попадают.
    assert result["summary"]["candidates"] == 5
    assert result["summary"]["matches"] == 5


def test_min_score_filters_dissimilar():
    result = ss.find_similar_setups(_target(), _history(), min_score=0.9)
    ids = {m["id"] for m in result["matches"]}
    assert ids == {1, 2, 3, 4}          # breakout(id=5) отфильтрован
    assert result["summary"]["matches"] == 4
    assert result["summary"]["candidates"] == 5


def test_win_loss_and_realized_r_stats():
    s = ss.find_similar_setups(_target(), _history(), min_score=0.9)["summary"]
    assert s["wins"] == 2
    assert s["losses"] == 1
    assert s["win_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert s["with_realized_r"] == 4
    # (2.0, -1.0, 1.5, 0.0) -> avg 0.625, median 0.75, best 2.0, worst -1.0
    assert s["avg_realized_r"] == pytest.approx(0.625)
    assert s["median_realized_r"] == pytest.approx(0.75)
    assert s["best_realized_r"] == pytest.approx(2.0)
    assert s["worst_realized_r"] == pytest.approx(-1.0)


def test_target_excluded_by_id_and_identity():
    target = _target()
    result = ss.find_similar_setups(target, _history() + [target])
    assert all(m["id"] != 99 for m in result["matches"])


def test_matches_sorted_by_similarity_desc():
    sims = [m["similarity_score"]
            for m in ss.find_similar_setups(_target(), _history())["matches"]]
    assert sims == sorted(sims, reverse=True)


def test_limit_caps_examples_not_summary():
    result = ss.find_similar_setups(_target(), _history(), min_score=0.9,
                                    limit=2)
    assert len(result["matches"]) == 2
    assert result["summary"]["matches"] == 4   # summary по всей выборке


def test_missed_and_avoided_counted():
    target = {"analysis_status": "WAIT", "candidate_direction": "long",
              "setup_type": "reversal", "market_regime": "trend",
              "confirming_factors": ["fvg"]}
    missed = {"analysis_status": "WAIT", "candidate_direction": "long",
              "setup_type": "reversal", "market_regime": "trend",
              "stop_loss": 100.0, "take_profit_levels": [110.0, 120.0],
              "confirming_factors": ["fvg"],
              "outcome": {"resolved": True, "tp2_hit": True, "tp1_hit": True,
                          "stop_hit": False}}
    avoided = {"analysis_status": "WAIT", "candidate_direction": "long",
               "setup_type": "reversal", "market_regime": "trend",
               "stop_loss": 100.0, "take_profit_levels": [110.0, 120.0],
               "confirming_factors": ["fvg"],
               "outcome": {"resolved": True, "tp1_hit": False,
                           "tp2_hit": False, "stop_hit": True}}
    s = ss.find_similar_setups(target, [missed, avoided])["summary"]
    assert s["missed"] == 1
    assert s["avoided"] == 1
    assert s["avg_realized_r"] is None   # non-ENTER: R нет


def test_unresolved_not_counted_as_win_loss():
    ghost = _enter(id=7, realized_r=1.0, confirming=["fvg", "ob"], outcome=None)
    s = ss.find_similar_setups(_target(), [ghost])["summary"]
    assert s["unresolved"] == 1
    assert s["wins"] == 0 and s["losses"] == 0
    assert s["avg_realized_r"] is None
    assert s["win_rate"] is None


def test_empty_history_safe():
    result = ss.find_similar_setups(_target(), [])
    assert result["summary"]["candidates"] == 0
    assert result["matches"] == []
    assert result["summary"]["win_rate"] is None


def test_malformed_rows_skipped():
    result = ss.find_similar_setups(_target(), ["nope", None, 123])
    assert result["summary"]["candidates"] == 0


def test_inputs_not_mutated():
    target = _target()
    row = _enter(id=1, realized_r=1.0, confirming=["fvg"])
    import copy
    t_before, r_before = copy.deepcopy(target), copy.deepcopy(row)
    ss.find_similar_setups(target, [row])
    assert target == t_before
    assert row == r_before


# ---------------------------------------------------------------------------
# Изоляция: leaf-модуль без runtime/exchange-зависимостей
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_ROOTS = {
    "database", "scheduler", "pipeline", "signal_engine", "risk", "bot", "ai",
    "tools",
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


def test_module_does_not_import_runtime_layers():
    assert _imported_roots().isdisjoint(_FORBIDDEN_IMPORT_ROOTS)


def test_module_has_no_forbidden_tokens():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_TOKENS:
        assert token not in source, f"forbidden token present: {token}"
