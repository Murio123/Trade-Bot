"""Stage B3a.1: the backtest resolves the profile's declared stop_tf.

The walk used to lift the stop ATR only when ``stop_tf == htf``. That is true
for exactly one shipped profile (position: 1D/1D). Swing stops on 12H, intraday
on 1H and bounce on 4H — none of which is that profile's htf — so all three
silently fell back to the ENTRY ATR and sized stops tighter than the live
cascade does. Tighter stops inflate R, so the backtest was optimistic about
precisely the profiles it was used to tune.

The fix is to call the LIVE function, ``pipeline._stop_atr``, with every frame
the bar has already computed. These tests pin three things:

  1. the walk and the parity harness both delegate to the live function, so the
     lookup cannot drift into a second implementation again;
  2. the fix changes exactly the profiles whose stop_tf is not their htf, and
     leaves position byte-identical;
  3. an unavailable stop_tf still falls back to the entry ATR (fail-safe).
"""
from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

import backtest as bt
import pipeline
from signal_engine.profiles import PROFILES
from tests.test_golden_contexts import SCENARIOS
from tools import backtest_parity as bp

# ATRs keyed by timeframe: distinct values so the assertion identifies WHICH
# frame was read, not merely that some number came back.
_INDS: dict[str, dict[str, Any]] = {
    "15m": {"atr": 1.0},
    "1h": {"atr": 2.0},
    "4h": {"atr": 3.0},
    "12h": {"atr": 4.0},
    "1d": {"atr": 5.0},
}

# Scenarios that actually reach position sizing for each profile. Bounce is
# absent on purpose: require_exhaustion fails closed without funding history,
# so no synthetic bar reaches the sizing call (see test_bounce_profile.py).
_SIZING_SCENARIOS = {
    "swing": "downtrend_signal",
    "position": "downtrend_signal",
    "intraday": "range",
}


def _old_stop_atr(profile: dict[str, Any], inds: dict[str, Any],
                  entry_atr: float) -> float:
    """The pre-fix lookup: only ever consulted the HTF frame."""
    htf = profile["htf"]
    if profile.get("stop_tf") == htf and (inds.get(htf) or {}).get("atr"):
        return inds[htf]["atr"]
    return entry_atr


# --- 1. One implementation, not two -----------------------------------------

def test_backtest_delegates_to_the_live_stop_atr():
    """Identity, not a copy: the walk imports pipeline._stop_atr itself."""
    assert bt._stop_atr is pipeline._stop_atr


@pytest.mark.parametrize("module", [bt, bp])
def test_no_module_reimplements_the_htf_only_lookup(module):
    src = inspect.getsource(module)
    assert not re.search(r"stop_tf.{0,20}==\s*htf", src), \
        "the HTF-only stop_tf lookup is back"


@pytest.mark.parametrize("func", [bt._walk, bp.backtest_equiv_decision])
def test_sizing_paths_call_stop_atr(func):
    assert re.search(r"\b_stop_atr\(", inspect.getsource(func))


@pytest.mark.parametrize("func", [bt._walk, bp.backtest_equiv_decision])
def test_stop_atr_is_passed_entry_htf_and_zone_frames(func):
    """The merged dict is what makes a non-HTF stop_tf resolvable at all."""
    src = inspect.getsource(func)
    assert re.search(r"_stop_atr\(\s*profile,\s*\{\*\*zinds,\s*entry_tf:\s*ind,"
                     r"\s*htf:\s*ind_htf\s*\}", src)


# --- 2. Every profile's stop_tf is actually reachable ------------------------

@pytest.mark.parametrize("name", sorted(PROFILES))
def test_declared_stop_tf_is_computed_by_the_walk(name):
    """stop_tf must live in entry ∪ htf ∪ zone_tfs, or the walk cannot see it."""
    p = PROFILES[name]
    available = {p["entry"], p["htf"], *p["zone_tfs"]}
    assert p["stop_tf"] in available, (name, p["stop_tf"], sorted(available))


@pytest.mark.parametrize("name,stop_tf", [
    ("swing", "12h"), ("position", "1d"), ("intraday", "1h"), ("bounce", "4h"),
])
def test_stop_atr_reads_the_declared_timeframe(name, stop_tf):
    p = PROFILES[name]
    assert p["stop_tf"] == stop_tf
    got = pipeline._stop_atr(p, _INDS, entry_atr=99.0)
    assert got == _INDS[stop_tf]["atr"]
    assert got != 99.0                     # never the entry ATR when resolvable


@pytest.mark.parametrize("name", ["swing", "intraday", "bounce"])
def test_the_broken_profiles_are_exactly_those_whose_stop_tf_is_not_htf(name):
    """Documents the bug's blast radius: three profiles, not just bounce."""
    p = PROFILES[name]
    assert p["stop_tf"] != p["htf"]
    assert _old_stop_atr(p, _INDS, entry_atr=99.0) == 99.0      # fell back
    assert pipeline._stop_atr(p, _INDS, entry_atr=99.0) != 99.0  # now lifts


def test_position_was_already_correct():
    p = PROFILES["position"]
    assert p["stop_tf"] == p["htf"] == "1d"
    assert _old_stop_atr(p, _INDS, 99.0) == pipeline._stop_atr(p, _INDS, 99.0)


# --- 3. Fail-safe fallback ---------------------------------------------------

def test_unavailable_stop_tf_falls_back_to_the_entry_atr():
    p = {**PROFILES["bounce"], "stop_tf": "6h"}   # never computed by the walk
    assert pipeline._stop_atr(p, _INDS, entry_atr=99.0) == 99.0


def test_stop_tf_present_but_atr_missing_falls_back():
    p = PROFILES["bounce"]
    inds = {**_INDS, "4h": {}}                    # frame there, ATR not
    assert pipeline._stop_atr(p, inds, entry_atr=99.0) == 99.0


def test_profile_without_stop_tf_falls_back():
    p = {k: v for k, v in PROFILES["bounce"].items() if k != "stop_tf"}
    assert pipeline._stop_atr(p, _INDS, entry_atr=99.0) == 99.0


# --- 4. Behavioural regression on a real decision ----------------------------

@pytest.mark.parametrize("name,changes", [
    ("swing", True),        # 4h entry ATR -> 12h ATR
    ("intraday", True),     # 15m entry ATR -> 1h ATR
    ("position", False),    # 1d == htf, already lifted
])
def test_fix_moves_the_stop_for_exactly_the_affected_profiles(monkeypatch, name,
                                                              changes):
    params = SCENARIOS[_SIZING_SCENARIOS[name]]

    fixed = bp.backtest_equiv_decision(name, **params)["stop"]
    monkeypatch.setattr(bt, "_stop_atr", _old_stop_atr)
    old = bp.backtest_equiv_decision(name, **params)["stop"]

    assert fixed is not None and old is not None, "scenario reached no sizing"
    if changes:
        assert fixed != old
    else:
        assert fixed == old


@pytest.mark.parametrize("name,entry_tf,stop_tf", [
    ("swing", "4h", "12h"), ("intraday", "15m", "1h"), ("bounce", "1h", "4h"),
])
def test_the_stop_timeframe_atr_is_larger_than_the_entry_atr(name, entry_tf,
                                                             stop_tf):
    """Why the bug mattered: the ATR of a higher timeframe is larger.

    Reading the entry ATR therefore produced a TIGHTER stop than live, and a
    tighter stop inflates R — the backtest was optimistic exactly where it was
    used to pick SCORE_ALERT_MIN.
    """
    from analyzer.indicators import compute_indicators
    from tests.synthetic_market import make_klines

    assert PROFILES[name]["entry"] == entry_tf
    assert PROFILES[name]["stop_tf"] == stop_tf
    entry_atr = compute_indicators(make_klines(entry_tf, seed=3))["atr"]
    stop_tf_atr = compute_indicators(make_klines(stop_tf, seed=3))["atr"]
    assert stop_tf_atr > entry_atr
