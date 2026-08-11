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
        ["LOW", "NORMAL", "NORMAL", "HIGH"]


def test_category_boundaries_are_exact():
    """Three categories (C4.5). ELEVATED was merged into NORMAL because the
    freeze showed their median outcomes were indistinguishable."""
    assert percentile.CATEGORIES == ("LOW", "NORMAL", "HIGH")
    assert percentile.categorize(0.0) == "LOW"
    assert percentile.categorize(24.999) == "LOW"
    assert percentile.categorize(25.0) == "NORMAL"
    assert percentile.categorize(84.999) == "NORMAL"
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
    back = percentile.load_reference(
        path, expected_sha256=percentile.content_sha256(ref.scores))
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
        percentile.load_reference(path, allow_unpinned=True)


def test_a_refit_with_the_same_version_is_caught_by_the_hash(tmp_path):
    """Audit finding: the version is a code constant, so a re-fitted
    reference carries the SAME version string and used to load silently.
    Only the content can tell them apart."""
    original = _ref()
    path = str(tmp_path / "ref.json")
    percentile.save_reference(original, path)
    pinned = percentile.content_sha256(original.scores)

    # pinning the original hash: loads fine
    percentile.load_reference(path, expected_sha256=pinned)

    # someone re-fits on newer data; same version, different content
    refit = percentile.build_reference(np.linspace(0, 2, 1000), {"src": "refit"})
    percentile.save_reference(refit, path)
    assert refit.version == original.version, "the version cannot distinguish them"
    with pytest.raises(ValueError, match="re-fitted"):
        percentile.load_reference(path, expected_sha256=pinned)


def test_an_edited_reference_file_is_caught_even_without_pinning(tmp_path):
    path = str(tmp_path / "ref.json")
    percentile.save_reference(_ref(), path)
    with open(path) as fh:
        payload = json.load(fh)
    payload["scores"][0] = -999.0          # tamper, leave the stored hash
    with open(path, "w") as fh:
        json.dump(payload, fh)
    with pytest.raises(ValueError, match="corrupt"):
        percentile.load_reference(path, allow_unpinned=True)


def test_content_hash_ignores_ordering_but_not_values():
    a = percentile.content_sha256([3.0, 1.0, 2.0])
    assert a == percentile.content_sha256([1.0, 2.0, 3.0])
    assert a != percentile.content_sha256([1.0, 2.0, 3.5])


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
    kw.setdefault("category", "HIGH")
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
                             realized_category="NORMAL")
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
    rec = _write(path, shadow={"ridge_percentile": 64.0, "ridge_category": "NORMAL"},
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


def test_concurrent_producers_cannot_both_write_the_same_forecast(tmp_path):
    """Audit finding: the duplicate guard was read-then-append with no lock,
    so two producers firing on the same bar would both find nothing and both
    append. Exercised with real processes, not a mocked race."""
    import subprocess
    import sys
    import textwrap
    import time

    path = str(tmp_path / "l.jsonl")
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    # A shared wall-clock start beats time.sleep: process startup jitter is
    # larger than any fixed sleep, and without a tight barrier an unlocked
    # implementation slips through most runs.
    start_at = time.time() + 1.5
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {repo!r})
        from volatility import ledger
        while time.time() < {start_at!r}:
            pass
        try:
            ledger.append_forecast({path!r}, symbol="BTCUSDT", bar_idx=100,
                bar_close_utc="2026-08-10T00:00:00Z", horizon_bars=12,
                ranker_version="v", distribution_version="v",
                score=0.5, percentile=71.0, category="NORMAL")
            print("WROTE")
        except ledger.ImmutableRecordError:
            print("REFUSED")
    """)
    # Started together, not one after the other: subprocess.run in a loop
    # would serialise them and the second would refuse for the trivial
    # reason that the first had already finished.
    # Started together, not one after the other: subprocess.run in a loop
    # would serialise them and the later ones would refuse for the trivial
    # reason that the first had already finished. Several workers rather than
    # two, because with only two the collision window is narrow enough that
    # an unlocked implementation still passes most runs.
    procs = [subprocess.Popen([sys.executable, "-c", script],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for _ in range(6)]
    results = [p.communicate() for p in procs]
    outs = [out.strip() for out, _ in results]
    assert outs.count("WROTE") == 1, (outs, [err for _, err in results])
    assert outs.count("REFUSED") == 5, (outs, [err for _, err in results])

    rows = [r for r in ledger.read_all(path) if r["kind"] == "forecast"]
    assert len(rows) == 1, "the same forecast was written twice"


def test_a_duplicated_row_is_reported_rather_than_silently_overriding(tmp_path):
    """Dict views used to let the last duplicate win, which is how an
    append-only ledger quietly stops being append-only."""
    path = str(tmp_path / "l.jsonl")
    first = _write(path)
    with open(path, "a") as fh:                      # forge a second row
        forged = dict(first, percentile=5.0, category="LOW")
        fh.write(json.dumps(forged, sort_keys=True) + "\n")

    for call in (lambda: ledger.pending(path),
                 lambda: ledger.matured_pairs(path),
                 lambda: ledger.latest_forecast(path, "BTCUSDT")
                 and ledger.pending(path)):
        with pytest.raises(ledger.LedgerError, match="duplicate forecast"):
            call()


def test_a_duplicated_maturation_is_reported(tmp_path):
    path = str(tmp_path / "l.jsonl")
    _write(path)
    rec = ledger.append_maturation(path, forecast_id_="BTCUSDT:100",
                                   at_bar_idx=112, realized_score=1.0,
                                   realized_percentile=50.0,
                                   realized_category="NORMAL")
    with open(path, "a") as fh:
        fh.write(json.dumps(dict(rec, realized_percentile=99.0),
                            sort_keys=True) + "\n")
    with pytest.raises(ledger.LedgerError, match="duplicate maturation"):
        ledger.pending(path)


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


def test_unpinned_reference_load_is_refused_by_default(tmp_path):
    """Audit finding: an optional check nobody is obliged to use is not a
    defence. Pinning is the default posture; skipping it must be explicit."""
    path = str(tmp_path / "ref.json")
    percentile.save_reference(_ref(), path)
    with pytest.raises(ValueError, match="requires expected_sha256"):
        percentile.load_reference(path)
    percentile.load_reference(path, allow_unpinned=True)   # explicit opt-out


def test_latest_forecast_reports_a_duplicate_rather_than_treating_it_as_newer(tmp_path):
    """Audit finding: this read path scanned raw records and kept the last
    match, so a duplicate row looked like a legitimate update."""
    path = str(tmp_path / "l.jsonl")
    first = _write(path)
    with open(path, "a") as fh:
        fh.write(json.dumps(dict(first, percentile=5.0, category="LOW"),
                            sort_keys=True) + "\n")
    with pytest.raises(ledger.LedgerError, match="duplicate forecast"):
        ledger.latest_forecast(path, "BTCUSDT")


def test_matured_pairs_also_rejects_a_duplicated_outcome(tmp_path):
    """Audit finding: this path checked forecasts but not maturations."""
    path = str(tmp_path / "l.jsonl")
    _write(path)
    rec = ledger.append_maturation(path, forecast_id_="BTCUSDT:100",
                                   at_bar_idx=112, realized_score=1.0,
                                   realized_percentile=50.0,
                                   realized_category="NORMAL")
    with open(path, "a") as fh:
        fh.write(json.dumps(dict(rec, realized_percentile=99.0),
                            sort_keys=True) + "\n")
    with pytest.raises(ledger.LedgerError, match="duplicate maturation"):
        ledger.matured_pairs(path)


def test_a_ledger_path_colliding_with_a_sidecar_lock_is_refused(tmp_path):
    """`foo.lock` as a ledger would share a sidecar with the ledger `foo`."""
    with pytest.raises(ledger.LedgerError, match="must not end in"):
        _write(str(tmp_path / "ledger.lock"))


def test_read_all_refuses_a_corrupt_ledger_but_read_raw_does_not(tmp_path):
    """Audit finding: read_all was the one public reader that stayed quiet
    about duplicates. A caller that reads rows and acts on them must not be
    the path that misses corruption; read_raw is the deliberate escape for
    inspecting a ledger you already suspect is damaged."""
    path = str(tmp_path / "l.jsonl")
    first = _write(path)
    with open(path, "a") as fh:
        fh.write(json.dumps(dict(first, percentile=5.0), sort_keys=True) + "\n")

    with pytest.raises(ledger.LedgerError, match="duplicate forecast"):
        ledger.read_all(path)
    assert len(ledger.read_raw(path)) == 2, "read_raw must still show both rows"


def test_appending_without_file_locking_is_refused(monkeypatch):
    """Audit finding: fcntl was imported under try/except and the lock
    degraded to a no-op without it. Silently unprotected is worse than
    refusing, because the duplicate guards become read-then-write races."""
    monkeypatch.setattr(ledger, "fcntl", None)
    with pytest.raises(ledger.LedgerError, match="file locking is unavailable"):
        _write("/tmp/never-created-by-this-test.jsonl")


# --- the class of defect, not one instance of it ----------------------------

def _corrupt_with_duplicate_maturation(path):
    _write(path)
    rec = ledger.append_maturation(path, forecast_id_="BTCUSDT:100",
                                   at_bar_idx=112, realized_score=1.0,
                                   realized_percentile=50.0,
                                   realized_category="NORMAL")
    with open(path, "a") as fh:
        fh.write(json.dumps(dict(rec, realized_percentile=99.0),
                            sort_keys=True) + "\n")


def _corrupt_with_duplicate_forecast(path):
    first = _write(path)
    with open(path, "a") as fh:
        fh.write(json.dumps(dict(first, percentile=5.0), sort_keys=True) + "\n")


# Every public entry point, and whether it is allowed to ignore corruption.
# read_raw exists precisely to inspect a damaged ledger; forecast_id never
# reads one. Everything else must refuse.
_RAW_BY_DESIGN = {"read_raw", "forecast_id"}


def _public_ledger_functions():
    import inspect
    return {name: fn for name, fn in vars(ledger).items()
            if inspect.isfunction(fn) and not name.startswith("_")
            and fn.__module__ == ledger.__name__}


def test_every_public_ledger_function_is_classified():
    """If someone adds a public function, this fails until they decide
    whether it validates — which is how the previous four rounds of
    'you covered some paths but not others' happened."""
    known = _RAW_BY_DESIGN | {"read_all", "append_forecast", "append_maturation",
                              "pending", "matured_pairs", "latest_forecast"}
    assert set(_public_ledger_functions()) == known


@pytest.mark.parametrize("corrupt", [_corrupt_with_duplicate_forecast,
                                     _corrupt_with_duplicate_maturation])
def test_no_public_function_operates_on_a_corrupt_ledger(tmp_path, corrupt):
    """The instance-by-instance fixes kept missing a path. This enumerates
    them instead: every public function except the two raw ones must refuse
    BOTH kinds of corruption."""
    calls = {
        "read_all": lambda p: ledger.read_all(p),
        "pending": lambda p: ledger.pending(p),
        "matured_pairs": lambda p: ledger.matured_pairs(p),
        "latest_forecast": lambda p: ledger.latest_forecast(p, "BTCUSDT"),
        "append_forecast": lambda p: _write(p, bar_idx=200),
        "append_maturation": lambda p: ledger.append_maturation(
            p, forecast_id_="BTCUSDT:100", at_bar_idx=999, realized_score=1.0,
            realized_percentile=50.0, realized_category="NORMAL"),
    }
    assert set(calls) | _RAW_BY_DESIGN == set(_public_ledger_functions())

    for name, call in calls.items():
        path = str(tmp_path / f"{name}_{corrupt.__name__}.jsonl")
        corrupt(path)
        with pytest.raises(ledger.LedgerError, match="duplicate"):
            call(path)


@pytest.mark.parametrize("corrupt", [_corrupt_with_duplicate_forecast,
                                     _corrupt_with_duplicate_maturation])
def test_read_raw_still_works_on_a_corrupt_ledger(tmp_path, corrupt):
    path = str(tmp_path / "raw.jsonl")
    corrupt(path)
    assert len(ledger.read_raw(path)) >= 2
