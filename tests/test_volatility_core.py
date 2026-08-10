"""C4.4: the report-only volatility ranking core.

The dangerous properties here are not "does it compute a number" but "can it
ever see the future", "can history be rewritten", and "can a category
silently change meaning". Those get the coverage.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from volatility import ledger, percentile, ranker


# --- ranker: observability -------------------------------------------------

def _labels(n=300):
    idx = np.arange(n)
    return idx, idx.astype(float)


def test_trailing_mean_stops_at_t_minus_horizon():
    idx, lab = _labels()
    got = ranker.trailing_mean(idx, lab, at_idx=100, window=10, horizon_bars=12)
    assert got == pytest.approx(83.5)          # last 10 of idx <= 88


def test_a_label_one_bar_too_recent_cannot_move_the_result():
    """The live-data analogue of the C4.3d anti-lookahead check. Here a
    mistake would not be caught by any walk-forward."""
    idx, lab = _labels()
    clean = ranker.trailing_mean(idx, lab, 100, window=10, horizon_bars=12)

    poisoned = lab.copy()
    poisoned[89] = 1e6                          # idx 89 == t-11
    assert ranker.trailing_mean(idx, poisoned, 100, window=10,
                                horizon_bars=12) == pytest.approx(clean)

    boundary = lab.copy()
    boundary[88] = 1e6                          # idx 88 == t-12, admissible
    assert ranker.trailing_mean(idx, boundary, 100, window=10,
                                horizon_bars=12) > clean


def test_no_observable_history_raises_instead_of_guessing():
    idx, lab = _labels()
    with pytest.raises(ranker.NotEnoughHistory):
        ranker.trailing_mean(idx, lab, at_idx=5, horizon_bars=12)


def test_rank_score_is_the_negated_trailing_mean():
    """The label mean-reverts at the horizon, so a high trailing ratio
    precedes a low one; without the negation the ranker points backwards."""
    idx, lab = _labels()
    at = 200
    assert ranker.rank_score(idx, lab, at) == pytest.approx(
        -ranker.trailing_mean(idx, lab, at))


def test_ranker_is_deterministic():
    idx, lab = _labels()
    first = ranker.rank_score(idx, lab, 250)
    assert all(ranker.rank_score(idx, lab, 250) == first for _ in range(5))


def test_ranker_rejects_unordered_or_duplicated_index():
    for bad in ([2, 1, 0], [0, 0, 1]):
        with pytest.raises(ValueError):
            ranker.trailing_mean(bad, [1.0, 2.0, 3.0], at_idx=50)


# --- percentile: frozen reference ------------------------------------------

def _ref(n=1000):
    return percentile.build_reference(np.linspace(0, 1, n), {"src": "test"})


def test_percentile_and_categories_are_monotone():
    ref = _ref()
    pcts = [ref.percentile_of(s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert pcts == sorted(pcts)
    assert [percentile.categorize(p) for p in (5, 40, 70, 95)] == \
        ["LOW", "NORMAL", "ELEVATED", "HIGH"]


def test_category_boundaries_are_exact():
    assert percentile.categorize(24.999) == "LOW"
    assert percentile.categorize(25.0) == "NORMAL"
    assert percentile.categorize(59.999) == "NORMAL"
    assert percentile.categorize(60.0) == "ELEVATED"
    assert percentile.categorize(84.999) == "ELEVATED"
    assert percentile.categorize(85.0) == "HIGH"
    assert percentile.categorize(100.0) == "HIGH"


def test_percentile_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        percentile.categorize(-0.1)
    with pytest.raises(ValueError):
        percentile.categorize(100.1)
    with pytest.raises(ValueError):
        _ref().percentile_of(float("nan"))
    with pytest.raises(ValueError):
        percentile.build_reference([], {})


def test_reference_survives_a_round_trip(tmp_path):
    ref = _ref()
    path = str(tmp_path / "ref.json")
    percentile.save_reference(ref, path)
    back = percentile.load_reference(path)
    assert np.array_equal(ref.scores, back.scores)
    assert back.version == percentile.DISTRIBUTION_VERSION


def test_a_reference_from_another_version_is_refused(tmp_path):
    """A silently re-fitted reference would change what every past category
    meant, making the whole ledger unreadable."""
    path = str(tmp_path / "ref.json")
    with open(path, "w") as fh:
        json.dump({"version": "something_else", "scores": [1, 2, 3],
                   "source": {}}, fh)
    with pytest.raises(ValueError, match="version mismatch"):
        percentile.load_reference(path)


def test_percentile_does_not_move_when_live_data_would():
    """The reference is frozen: feeding extreme new scores changes the
    percentile OF those scores, never the reference itself."""
    ref = _ref()
    before = ref.percentile_of(0.5)
    for extreme in (-100.0, 100.0, 1e9):
        ref.percentile_of(extreme)
    assert ref.percentile_of(0.5) == before


# --- ledger: immutability and maturation ------------------------------------

def _write(path, bar_idx=100, symbol="BTCUSDT", **kw):
    kw.setdefault("bar_close_utc", "2026-08-10T00:00:00Z")
    kw.setdefault("horizon_bars", 12)
    kw.setdefault("ranker_version", ranker.RANKER_VERSION)
    kw.setdefault("distribution_version", percentile.DISTRIBUTION_VERSION)
    kw.setdefault("score", 0.5)
    kw.setdefault("percentile", 71.0)
    kw.setdefault("category", "ELEVATED")
    return ledger.append_forecast(path, symbol=symbol, bar_idx=bar_idx, **kw)


def test_a_forecast_cannot_be_written_twice(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path)
    with pytest.raises(ledger.ImmutableRecordError):
        _write(path, percentile=5.0, category="LOW")


def test_maturation_appends_and_leaves_the_forecast_untouched(tmp_path):
    path = str(tmp_path / "l.jsonl")
    original = _write(path)
    ledger.append_maturation(path, forecast_id_="BTCUSDT:100", at_bar_idx=112,
                             realized_score=0.9, realized_percentile=80.0,
                             realized_category="ELEVATED")
    stored = [r for r in ledger.read_all(path) if r["kind"] == "forecast"][0]
    assert stored == original, "the forecast row was modified after the fact"


def test_maturation_before_the_horizon_is_refused(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path)
    for early in (100, 105, 111):
        with pytest.raises(ledger.LedgerError, match="matures at bar"):
            ledger.append_maturation(path, forecast_id_="BTCUSDT:100",
                                     at_bar_idx=early, realized_score=1.0,
                                     realized_percentile=50.0,
                                     realized_category="NORMAL")
    # exactly at the horizon is allowed
    ledger.append_maturation(path, forecast_id_="BTCUSDT:100", at_bar_idx=112,
                             realized_score=1.0, realized_percentile=50.0,
                             realized_category="NORMAL")


def test_an_outcome_is_recorded_exactly_once(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path)
    ledger.append_maturation(path, forecast_id_="BTCUSDT:100", at_bar_idx=112,
                             realized_score=1.0, realized_percentile=50.0,
                             realized_category="NORMAL")
    with pytest.raises(ledger.ImmutableRecordError):
        ledger.append_maturation(path, forecast_id_="BTCUSDT:100",
                                 at_bar_idx=113, realized_score=9.0,
                                 realized_percentile=99.0,
                                 realized_category="HIGH")


def test_maturing_an_unknown_forecast_is_refused(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path)
    with pytest.raises(ledger.LedgerError, match="no forecast"):
        ledger.append_maturation(path, forecast_id_="BTCUSDT:999",
                                 at_bar_idx=999, realized_score=1.0,
                                 realized_percentile=50.0,
                                 realized_category="NORMAL")


def test_pending_and_matured_views(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path, bar_idx=100)
    _write(path, bar_idx=101)
    assert len(ledger.pending(path)) == 2
    ledger.append_maturation(path, forecast_id_="BTCUSDT:100", at_bar_idx=112,
                             realized_score=1.0, realized_percentile=50.0,
                             realized_category="NORMAL")
    assert [r["bar_idx"] for r in ledger.pending(path)] == [101]
    assert len(ledger.matured_pairs(path)) == 1


def test_shadow_is_recorded_from_the_first_forecast(tmp_path):
    """Ridge rides along as a shadow so the comparison needs no backfill."""
    path = str(tmp_path / "l.jsonl")
    rec = _write(path, shadow={"ridge_percentile": 64.0, "ridge_category": "ELEVATED"},
                 baselines={"train_mean_percentile": 50.0})
    assert rec["shadow"]["ridge_percentile"] == 64.0
    assert rec["baselines"]["train_mean_percentile"] == 50.0


def test_ledger_carries_no_trading_decision_fields(tmp_path):
    """C4.4 may report relative expected volatility and nothing else."""
    path = str(tmp_path / "l.jsonl")
    _write(path)
    ledger.append_maturation(path, forecast_id_="BTCUSDT:100", at_bar_idx=112,
                             realized_score=1.0, realized_percentile=50.0,
                             realized_category="NORMAL")
    banned = {"direction", "side", "entry", "entry_price", "stop", "stop_loss",
              "target", "target_1", "take_profit", "position_size", "size",
              "leverage", "order", "order_id", "quantity", "signal", "action",
              "buy", "sell", "risk_percent"}
    for rec in ledger.read_all(path):
        leaked = banned & set(rec)
        assert not leaked, f"trading decision field(s) in ledger: {leaked}"


def test_schema_is_symbol_keyed_so_a_second_asset_needs_no_migration(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path, symbol="BTCUSDT", bar_idx=100)
    _write(path, symbol="ETHUSDT", bar_idx=100)   # same bar, different asset
    assert len(ledger.read_all(path)) == 2
    assert [r["symbol"] for r in ledger.pending(path, "ETHUSDT")] == ["ETHUSDT"]
    assert ledger.latest_forecast(path, "BTCUSDT")["symbol"] == "BTCUSDT"

    ledger.append_maturation(path, forecast_id_="ETHUSDT:100", at_bar_idx=112,
                             realized_score=1.0, realized_percentile=50.0,
                             realized_category="NORMAL")
    assert len(ledger.matured_pairs(path, "BTCUSDT")) == 0
    assert len(ledger.matured_pairs(path, "ETHUSDT")) == 1


def test_a_corrupt_line_is_reported_with_its_location(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path)
    with open(path, "a") as fh:
        fh.write("{not json\n")
    with pytest.raises(ledger.LedgerError, match=r":2 is not valid JSON"):
        ledger.read_all(path)


# --- architectural guards ---------------------------------------------------

def _imported_modules(path) -> set[str]:
    """Real imports only — a docstring naming a module is documentation, not
    a dependency, and a substring scan would conflate the two."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _volatility_sources():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "volatility"
    return sorted(root.glob("*.py"))


def test_volatility_package_does_not_import_the_offline_platform():
    """feature_store's own contract says it is never imported by runtime.
    The bot reads this package; the offline producer owns the features."""
    offline = ("tools.forecast_platform", "tools.deep_backtest",
               "tools.kline_dataset", "tools.deep_discovery")
    for path in _volatility_sources():
        for mod in _imported_modules(path):
            assert not mod.startswith(offline), \
                f"{path.name} imports offline-only {mod}"


def test_volatility_package_imports_nothing_from_the_trading_path():
    """A report-only ranker must not be able to reach the decision path."""
    forbidden = ("pipeline", "scheduler", "signal_engine", "contracts",
                 "risk", "bot", "database", "main")
    for path in _volatility_sources():
        for mod in _imported_modules(path):
            assert mod.split(".")[0] not in forbidden, \
                f"{path.name} imports trading-path module {mod}"


def test_volatility_package_places_no_orders():
    for path in _volatility_sources():
        src = path.read_text(encoding="utf-8")
        for banned in ("create_order", "place_order", "submit_order",
                       "ccxt", "api_key", "secret"):
            assert banned not in src, f"{path.name} references {banned}"
