"""Stage 16A: tests for the pure offline trade pattern miner core."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from analyzer import trade_patterns as tp

MODULE_PATH = Path(tp.__file__)


# ---------------------------------------------------------------------------
# Помощники: строки с достоверным исходом через journal_classify.classify
# ---------------------------------------------------------------------------

def _enter(*, realized_r: float, confirming=None, **extra) -> dict:
    """ENTER-строка с разрешённым исходом -> good_trade / bad_trade."""
    row = {
        "analysis_status": "ENTER",
        "candidate_direction": "long",
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
        "confirming_factors": list(confirming or []),
        "outcome": {"resolved": True, "realized_r": realized_r},
    }
    row.update(extra)
    return row


def _wait_missed(*, contradicting=None, **extra) -> dict:
    """WAIT-строка, где сетап гипотетически отработал бы -> missed_opportunity."""
    row = {
        "analysis_status": "WAIT",
        "candidate_direction": "long",
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
        "contradicting_factors": list(contradicting or []),
        "outcome": {"resolved": True, "tp2_hit": True, "tp1_hit": True,
                    "stop_hit": False},
    }
    row.update(extra)
    return row


def _wait_avoided(*, contradicting=None, **extra) -> dict:
    """WAIT-строка с гипотетическим чистым стопом -> avoided_loss."""
    row = {
        "analysis_status": "WAIT",
        "candidate_direction": "long",
        "stop_loss": 100.0,
        "take_profit_levels": [110.0, 120.0],
        "contradicting_factors": list(contradicting or []),
        "outcome": {"resolved": True, "tp1_hit": False, "tp2_hit": False,
                    "stop_hit": True},
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# extract_pattern_factors
# ---------------------------------------------------------------------------

def test_extract_pulls_all_dimensions():
    row = {
        "confirming_factors": ["fvg", "ob"],
        "contradicting_factors": ["against_htf"],
        "setup_type": "reversal",
        "market_regime": "trend",
        "volatility_regime": "high",
        "raw_confidence": 0.72,
        "setup_lifecycle_status": "NEW_SETUP",
        "setup_lifecycle_reasons": ["confidence_up"],
        "analysis_type": "swing",
        "candidate_direction": "short",
    }
    dims = {f["dimension"] for f in tp.extract_pattern_factors(row)}
    assert dims == {
        "confirming_factors", "contradicting_factors", "setup_type",
        "market_regime", "volatility_regime", "confidence_bucket",
        "setup_lifecycle_status", "setup_lifecycle_reasons",
        "analysis_type", "direction",
    }
    # Каждая запись несёт согласованный key.
    for f in tp.extract_pattern_factors(row):
        assert f["key"] == f"{f['dimension']}:{f['value']}"


def test_list_dimensions_emit_one_factor_per_item():
    row = {"confirming_factors": ["a", "b", "c"]}
    keys = [f["key"] for f in tp.extract_pattern_factors(row)
            if f["dimension"] == "confirming_factors"]
    assert keys == ["confirming_factors:a", "confirming_factors:b",
                    "confirming_factors:c"]


def test_direction_falls_back_to_candidate_direction():
    assert any(f["key"] == "direction:short"
               for f in tp.extract_pattern_factors({"candidate_direction": "short"}))
    assert any(f["key"] == "direction:long"
               for f in tp.extract_pattern_factors({"direction": "long"}))


def test_lifecycle_status_and_reasons_included():
    row = {"setup_lifecycle_status": "UPGRADED",
           "setup_lifecycle_reasons": ["confidence_up", "score_up"]}
    keys = {f["key"] for f in tp.extract_pattern_factors(row)}
    assert "setup_lifecycle_status:UPGRADED" in keys
    assert "setup_lifecycle_reasons:confidence_up" in keys
    assert "setup_lifecycle_reasons:score_up" in keys


def test_duplicate_keys_deduplicated_within_row():
    row = {"confirming_factors": ["fvg", "fvg", "fvg"]}
    keys = [f["key"] for f in tp.extract_pattern_factors(row)
            if f["dimension"] == "confirming_factors"]
    assert keys == ["confirming_factors:fvg"]


@pytest.mark.parametrize("row", [
    {},
    {"confirming_factors": "not-a-list"},
    {"confirming_factors": None, "setup_type": None},
    {"contradicting_factors": [None, "", "  "]},
    {"setup_lifecycle_reasons": 123},
    {"market_regime": ""},
    "not-a-dict",
    None,
])
def test_missing_or_malformed_fields_do_not_raise(row):
    factors = tp.extract_pattern_factors(row)  # не должно бросать
    assert isinstance(factors, list)
    # confidence_bucket есть всегда для валидного dict-строки.
    if isinstance(row, dict):
        assert any(f["dimension"] == "confidence_bucket" for f in factors)


# ---------------------------------------------------------------------------
# confidence buckets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (40, "confidence_<50"),
    (0.4, "confidence_<50"),
    (55, "confidence_50_60"),
    (0.55, "confidence_50_60"),
    (65, "confidence_60_70"),
    (75, "confidence_70_80"),
    (0.72, "confidence_70_80"),
    (85, "confidence_80_plus"),
    (1.0, "confidence_80_plus"),
])
def test_confidence_buckets(value, expected):
    assert tp.confidence_bucket({"raw_confidence": value}) == expected


def test_confidence_missing_when_absent_or_non_number():
    assert tp.confidence_bucket({}) == tp.CONFIDENCE_MISSING
    assert tp.confidence_bucket({"raw_confidence": "high"}) == tp.CONFIDENCE_MISSING
    assert tp.confidence_bucket({"confidence": None}) == tp.CONFIDENCE_MISSING
    # bool не считается числом.
    assert tp.confidence_bucket({"confidence": True}) == tp.CONFIDENCE_MISSING


def test_calibrated_confidence_takes_priority():
    row = {"calibrated_confidence": 0.85, "raw_confidence": 0.10,
           "confidence": 5}
    assert tp.confidence_bucket(row) == "confidence_80_plus"
    # Если calibrated отсутствует — берётся raw.
    assert tp.confidence_bucket({"raw_confidence": 0.10}) == "confidence_<50"


# ---------------------------------------------------------------------------
# mine_trade_patterns
# ---------------------------------------------------------------------------

def _dataset() -> list[dict]:
    rows: list[dict] = []
    # 3 плохих ENTER с токсичным фактором, 3 хороших ENTER с чистым.
    rows += [_enter(realized_r=-1.0, confirming=["toxic"]) for _ in range(3)]
    rows += [_enter(realized_r=2.0, confirming=["clean"]) for _ in range(3)]
    # 3 пропущенных WAIT с фактором fomo, 3 корректных пропуска с cautious.
    rows += [_wait_missed(contradicting=["fomo"]) for _ in range(3)]
    rows += [_wait_avoided(contradicting=["cautious"]) for _ in range(3)]
    return rows


def test_summary_counts():
    result = tp.mine_trade_patterns(_dataset())
    s = result["summary"]
    assert s["rows"] == 12
    assert s["classified_rows"] == 12
    assert s["bad_trade_count"] == 3
    assert s["good_trade_count"] == 3
    assert s["missed_opportunity_count"] == 3
    assert s["correct_skip_count"] == 3
    assert s["insufficient_data_count"] == 0


def test_result_is_json_serializable():
    result = tp.mine_trade_patterns(_dataset())
    dumped = json.dumps(result)  # не должно бросать
    assert json.loads(dumped) == result


def _pattern(result: dict, key: str) -> dict:
    return next(p for p in result["patterns"] if p["key"] == key)


def test_bad_factor_has_count_and_lift_above_one():
    result = tp.mine_trade_patterns(_dataset())
    toxic = _pattern(result, "confirming_factors:toxic")
    assert toxic["count_bad"] == 3
    assert toxic["total"] == 3
    assert toxic["bad_lift"] is not None and toxic["bad_lift"] > 1.0
    assert toxic["low_sample"] is False


def test_missed_factor_has_count_and_lift_above_one():
    result = tp.mine_trade_patterns(_dataset())
    fomo = _pattern(result, "contradicting_factors:fomo")
    assert fomo["count_missed"] == 3
    assert fomo["missed_lift"] is not None and fomo["missed_lift"] > 1.0


def test_good_and_avoided_counts_tracked():
    result = tp.mine_trade_patterns(_dataset())
    clean = _pattern(result, "confirming_factors:clean")
    cautious = _pattern(result, "contradicting_factors:cautious")
    assert clean["count_good"] == 3
    assert clean["good_rate"] == 1.0
    assert cautious["count_avoided"] == 3
    assert cautious["avoided_rate"] == 1.0


def test_low_sample_flag_when_total_below_min_count():
    rows = _dataset() + [_enter(realized_r=-1.0, confirming=["rare"])]
    result = tp.mine_trade_patterns(rows, min_count=3)
    rare = _pattern(result, "confirming_factors:rare")
    assert rare["total"] == 1
    assert rare["low_sample"] is True


def test_insufficient_rows_not_counted_as_outcome_truth():
    # ENTER без outcome -> unresolved -> insufficient; его фактор не должен
    # попасть в статистику.
    ghost = {
        "analysis_status": "ENTER",
        "candidate_direction": "long",
        "stop_loss": 100.0,
        "take_profit_levels": [110.0],
        "confirming_factors": ["ghost_factor"],
        "outcome": None,
    }
    result = tp.mine_trade_patterns(_dataset() + [ghost])
    assert result["summary"]["insufficient_data_count"] == 1
    assert result["summary"]["classified_rows"] == 12
    assert all(p["key"] != "confirming_factors:ghost_factor"
               for p in result["patterns"])


def test_no_levels_row_is_insufficient():
    # WAIT без стопа/TP -> no_levels -> insufficient.
    row = {"analysis_status": "WAIT", "candidate_direction": "long",
           "confirming_factors": ["x"], "outcome": {"resolved": True}}
    result = tp.mine_trade_patterns([row])
    assert result["summary"]["insufficient_data_count"] == 1
    assert result["summary"]["classified_rows"] == 0
    assert result["patterns"] == []


def test_baseline_zero_gives_none_lift():
    # Только хорошие сделки: baseline_bad == 0 -> bad_lift == None.
    rows = [_enter(realized_r=2.0, confirming=["clean"]) for _ in range(3)]
    result = tp.mine_trade_patterns(rows)
    clean = _pattern(result, "confirming_factors:clean")
    assert clean["bad_lift"] is None
    assert clean["missed_lift"] is None


def test_patterns_sorted_normal_before_low_sample():
    rows = _dataset() + [_enter(realized_r=-1.0, confirming=["rare"])]
    patterns = tp.mine_trade_patterns(rows, min_count=3)["patterns"]
    low_indexes = [i for i, p in enumerate(patterns) if p["low_sample"]]
    normal_indexes = [i for i, p in enumerate(patterns) if not p["low_sample"]]
    assert max(normal_indexes) < min(low_indexes)


def test_empty_input_safe():
    result = tp.mine_trade_patterns([])
    assert result["summary"]["rows"] == 0
    assert result["patterns"] == []


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
