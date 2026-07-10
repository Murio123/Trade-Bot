"""Stage B1: the require_exhaustion HTF policy and its exhaustion predicates.

Proves: counter-trend directions are admitted ONLY on confirmed exhaustion,
every mandatory condition is load-bearing, the bullish and bearish paths are
exact mirrors (no directional bias re-enters through the module meant to
remove it), a malformed context fails closed, and the shipped trend profiles
keep their historical block_counter_trend semantics untouched.
"""
from __future__ import annotations

import copy

import pytest

import config
from signal_engine.exhaustion import (REQUIRED_CTX_KEYS, bearish_exhaustion,
                                      bullish_exhaustion)
from signal_engine.htf_filter import (DEFAULT_HTF_POLICY, HTF_POLICIES,
                                      apply_htf_policy)
from signal_engine.profiles import PROFILES

EXHAUSTION_POLICY = {"htf_policy": "require_exhaustion"}


def bullish_ctx() -> dict:
    """Counter-trend LONG context: all 6 mandatory + exactly 1 optional.

    funding is mildly negative — enough to clear the crowded-funding veto,
    NOT enough (|z| < FUNDING_Z_SIGNAL) to count as a stretched-funding
    confirmation. The single optional is the bullish divergence.
    """
    return {
        "reversal": {"bull_strong": True, "bear_strong": False},
        "reversal_mtf": {"bull_tf_count": 2, "bear_tf_count": 0,
                         "bull_candle_confirm": True, "bear_candle_confirm": False},
        "equilibrium": {"zone": "discount", "pos": 0.2},
        "liquidity": {"liquidity_swept_below": True, "liquidity_swept_above": False,
                      "reversal_candle": True},
        "funding": {"current": -0.0001, "zscore": -0.5},
        "divergence": {"bullish_divergence": True, "bearish_divergence": False},
        "order_blocks": {"price_in_bullish_ob": False, "price_in_bearish_ob": False},
        "fvg": {"price_in_bullish_fvg": False, "price_in_bearish_fvg": False},
        "cvd": {"cvd_bullish": False, "cvd_bearish": False},
    }


def bearish_ctx() -> dict:
    """Exact mirror of bullish_ctx for a counter-trend SHORT."""
    return {
        "reversal": {"bull_strong": False, "bear_strong": True},
        "reversal_mtf": {"bull_tf_count": 0, "bear_tf_count": 2,
                         "bull_candle_confirm": False, "bear_candle_confirm": True},
        "equilibrium": {"zone": "premium", "pos": 0.8},
        "liquidity": {"liquidity_swept_below": False, "liquidity_swept_above": True,
                      "reversal_candle": True},
        "funding": {"current": 0.0001, "zscore": 0.5},
        "divergence": {"bullish_divergence": False, "bearish_divergence": True},
        "order_blocks": {"price_in_bullish_ob": False, "price_in_bearish_ob": False},
        "fvg": {"price_in_bullish_fvg": False, "price_in_bearish_fvg": False},
        "cvd": {"cvd_bullish": False, "cvd_bearish": False},
    }


def _without(ctx: dict, path: tuple[str, str], value) -> dict:
    ctx = copy.deepcopy(ctx)
    ctx[path[0]][path[1]] = value
    return ctx


# --- the happy path ---------------------------------------------------------

def test_full_bullish_context_admits_counter_trend_long():
    ok, reasons = bullish_exhaustion(bullish_ctx())
    assert ok
    assert len(reasons) == 7  # 6 mandatory + 1 optional


def test_full_bearish_context_admits_counter_trend_short():
    ok, reasons = bearish_exhaustion(bearish_ctx())
    assert ok
    assert len(reasons) == 7


def test_rejected_context_reports_no_reasons():
    """A blocked setup must not leak half-satisfied evidence."""
    ctx = _without(bullish_ctx(), ("reversal", "bull_strong"), False)
    assert bullish_exhaustion(ctx) == (False, [])


# --- every mandatory condition is load-bearing ------------------------------

BULL_MANDATORY_NEGATIONS = [
    pytest.param(("reversal_mtf", "bull_tf_count"), 1, id="too_few_reversal_tfs"),
    pytest.param(("reversal", "bull_strong"), False, id="reversal_not_strong"),
    pytest.param(("reversal_mtf", "bull_candle_confirm"), False, id="no_candle_confirm"),
    pytest.param(("equilibrium", "zone"), "premium", id="not_in_discount"),
    pytest.param(("liquidity", "liquidity_swept_below"), False, id="no_sweep"),
    pytest.param(("liquidity", "reversal_candle"), False, id="no_reclaim"),
]


@pytest.mark.parametrize("path,value", BULL_MANDATORY_NEGATIONS)
def test_each_bullish_mandatory_condition_blocks(path, value):
    assert bullish_exhaustion(_without(bullish_ctx(), path, value))[0] is False


def test_bullish_blocked_when_funding_crowded_long():
    """Extreme POSITIVE funding = crowded longs: squeeze fuel against us."""
    ctx = bullish_ctx()
    ctx["funding"] = {"current": 0.01, "zscore": 2.5}
    assert bullish_exhaustion(ctx)[0] is False


BEAR_MANDATORY_NEGATIONS = [
    pytest.param(("reversal_mtf", "bear_tf_count"), 1, id="too_few_reversal_tfs"),
    pytest.param(("reversal", "bear_strong"), False, id="reversal_not_strong"),
    pytest.param(("reversal_mtf", "bear_candle_confirm"), False, id="no_candle_confirm"),
    pytest.param(("equilibrium", "zone"), "discount", id="not_in_premium"),
    pytest.param(("liquidity", "liquidity_swept_above"), False, id="no_sweep"),
    pytest.param(("liquidity", "reversal_candle"), False, id="no_reclaim"),
]


@pytest.mark.parametrize("path,value", BEAR_MANDATORY_NEGATIONS)
def test_each_bearish_mandatory_condition_blocks(path, value):
    assert bearish_exhaustion(_without(bearish_ctx(), path, value))[0] is False


def test_bearish_blocked_when_funding_crowded_short():
    ctx = bearish_ctx()
    ctx["funding"] = {"current": -0.01, "zscore": -2.5}
    assert bearish_exhaustion(ctx)[0] is False


# --- optional confirmations -------------------------------------------------

def test_zero_optional_confirmations_blocks():
    ctx = _without(bullish_ctx(), ("divergence", "bullish_divergence"), False)
    assert bullish_exhaustion(ctx)[0] is False
    assert bearish_exhaustion(
        _without(bearish_ctx(), ("divergence", "bearish_divergence"), False))[0] is False


@pytest.mark.parametrize("path", [("order_blocks", "price_in_bullish_ob"),
                                  ("fvg", "price_in_bullish_fvg"),
                                  ("cvd", "cvd_bullish")])
def test_any_single_optional_confirmation_suffices(path):
    ctx = _without(bullish_ctx(), ("divergence", "bullish_divergence"), False)
    assert bullish_exhaustion(_without(ctx, path, True))[0] is True


def test_stretched_funding_counts_as_optional_confirmation():
    """Funding against the crowd we fade: below the veto extreme, above the
    signal threshold."""
    ctx = _without(bullish_ctx(), ("divergence", "bullish_divergence"), False)
    ctx["funding"] = {"current": -0.005, "zscore": -config.FUNDING_Z_SIGNAL}
    assert bullish_exhaustion(ctx)[0] is True


# --- fail closed ------------------------------------------------------------

@pytest.mark.parametrize("ctx", [None, {}, [], "ctx", 0])
def test_missing_or_malformed_ctx_fails_closed(ctx):
    assert bullish_exhaustion(ctx) == (False, [])
    assert bearish_exhaustion(ctx) == (False, [])


@pytest.mark.parametrize("key", REQUIRED_CTX_KEYS)
def test_each_missing_required_key_fails_closed(key):
    for build, predicate in ((bullish_ctx, bullish_exhaustion),
                             (bearish_ctx, bearish_exhaustion)):
        ctx = build()
        del ctx[key]
        assert predicate(ctx) == (False, [])


@pytest.mark.parametrize("key", REQUIRED_CTX_KEYS)
def test_each_non_dict_required_key_fails_closed(key):
    ctx = bullish_ctx()
    ctx[key] = None
    assert bullish_exhaustion(ctx) == (False, [])


def test_optional_sources_may_be_absent_entirely():
    """Optional evidence degrades to 'absent', never to an exception."""
    ctx = bullish_ctx()
    for key in ("order_blocks", "fvg", "cvd"):
        del ctx[key]
    assert bullish_exhaustion(ctx)[0] is True  # divergence still carries it


# --- fail closed on malformed NESTED values ---------------------------------
# A corrupted data source must block the gate, not crash it and not be coerced
# into a plausible number. "2" is not 2: it is evidence the feed is broken.

MALFORMED_NUMBERS = [
    pytest.param("2", id="numeric_string"),
    pytest.param(object(), id="object"),
    pytest.param([2], id="list"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(True, id="bool"),
]


@pytest.mark.parametrize("value", MALFORMED_NUMBERS)
def test_malformed_bull_tf_count_fails_closed(value):
    ctx = _without(bullish_ctx(), ("reversal_mtf", "bull_tf_count"), value)
    assert bullish_exhaustion(ctx) == (False, [])


@pytest.mark.parametrize("value", MALFORMED_NUMBERS)
def test_malformed_bear_tf_count_fails_closed(value):
    ctx = _without(bearish_ctx(), ("reversal_mtf", "bear_tf_count"), value)
    assert bearish_exhaustion(ctx) == (False, [])


@pytest.mark.parametrize("value", MALFORMED_NUMBERS)
@pytest.mark.parametrize("key", ["zscore", "current"])
def test_malformed_funding_fails_closed(key, value):
    """crowded_funding() would abs() this; we must never let it try."""
    assert bullish_exhaustion(_without(bullish_ctx(), ("funding", key), value)) == (False, [])
    assert bearish_exhaustion(_without(bearish_ctx(), ("funding", key), value)) == (False, [])


def test_malformed_funding_never_counts_as_optional_confirmation():
    """Even with every mandatory condition met, broken funding blocks — it can
    neither pass the crowding check nor supply a stretched-funding optional."""
    ctx = _without(bullish_ctx(), ("divergence", "bullish_divergence"), False)
    ctx["funding"] = {"current": "bad", "zscore": "bad"}
    assert bullish_exhaustion(ctx) == (False, [])


def test_absent_funding_keys_are_not_malformed():
    """A degraded funding source ({} from a failed exchange call) is ordinary:
    no crowding, no optional — not a hard block."""
    ctx = bullish_ctx()
    ctx["funding"] = {}
    assert bullish_exhaustion(ctx)[0] is True  # divergence still carries it
    ctx["funding"] = {"current": None, "zscore": None}
    assert bullish_exhaustion(ctx)[0] is True


@pytest.mark.parametrize("key", ["reversal", "reversal_mtf", "equilibrium",
                                 "liquidity", "funding", "divergence",
                                 "order_blocks", "fvg", "cvd"])
@pytest.mark.parametrize("value", MALFORMED_NUMBERS)
def test_garbage_in_any_source_never_raises(key, value):
    """The contract, stated directly: no input shape escapes as an exception.

    The VERDICT is not asserted here. Flag-shaped fields are read through
    bool(), where a truthy string is honestly truthy — type-validating every
    boolean is a different (and unwanted) contract. What must hold is that the
    numeric reads are strict and nothing ever propagates a TypeError.
    """
    for build, predicate in ((bullish_ctx, bullish_exhaustion),
                             (bearish_ctx, bearish_exhaustion)):
        ctx = build()
        ctx[key] = {k: value for k in ctx[key]}
        ok, reasons = predicate(ctx)
        assert isinstance(ok, bool) and isinstance(reasons, list)


@pytest.mark.parametrize("value", MALFORMED_NUMBERS)
def test_malformed_ctx_blocks_counter_trend_through_the_policy(value):
    """End-to-end: the gate stays shut, and apply_htf_policy does not raise."""
    bull = _without(bullish_ctx(), ("reversal_mtf", "bull_tf_count"), value)
    bear = _without(bearish_ctx(), ("reversal_mtf", "bear_tf_count"), value)
    assert apply_htf_policy("long", "bearish", EXHAUSTION_POLICY, bull) is None
    assert apply_htf_policy("short", "bullish", EXHAUSTION_POLICY, bear) is None


# --- the policy itself ------------------------------------------------------

def test_policy_is_registered():
    assert HTF_POLICIES["require_exhaustion"] is not HTF_POLICIES[DEFAULT_HTF_POLICY]


@pytest.mark.parametrize("direction,bias", [("long", "bullish"), ("short", "bearish"),
                                            ("long", "neutral"), ("short", "neutral")])
def test_aligned_and_neutral_directions_pass_through(direction, bias):
    """With-trend and no-trend directions are not counter-trend: untouched,
    and untouched even without a context."""
    assert apply_htf_policy(direction, bias, EXHAUSTION_POLICY, bullish_ctx()) == direction
    assert apply_htf_policy(direction, bias, EXHAUSTION_POLICY, None) == direction


def test_counter_trend_long_admitted_on_bullish_exhaustion():
    assert apply_htf_policy("long", "bearish", EXHAUSTION_POLICY, bullish_ctx()) == "long"


def test_counter_trend_short_admitted_on_bearish_exhaustion():
    assert apply_htf_policy("short", "bullish", EXHAUSTION_POLICY, bearish_ctx()) == "short"


def test_counter_trend_blocked_without_exhaustion():
    """The mirrored context is exhaustion for the OTHER side — not this one."""
    assert apply_htf_policy("long", "bearish", EXHAUSTION_POLICY, bearish_ctx()) is None
    assert apply_htf_policy("short", "bullish", EXHAUSTION_POLICY, bullish_ctx()) is None


@pytest.mark.parametrize("ctx", [None, {}, "nonsense"])
def test_counter_trend_without_ctx_falls_back_to_blocking(ctx):
    """A caller that cannot supply context (e.g. backtest.py) gets
    block_counter_trend semantics, never an open gate."""
    assert apply_htf_policy("long", "bearish", EXHAUSTION_POLICY, ctx) is None
    assert apply_htf_policy("short", "bullish", EXHAUSTION_POLICY, ctx) is None


def test_none_direction_stays_none():
    assert apply_htf_policy(None, "bearish", EXHAUSTION_POLICY, bullish_ctx()) is None


# --- regression: existing profiles are untouched ----------------------------
# Stage B2 gave the policy exactly ONE consumer: the disabled, observation-only
# "bounce" profile. Every other profile is trend-following and its gate must
# stay absolute — that is what these pin.

TREND_PROFILES = tuple(sorted(set(PROFILES) - {"bounce"}))


def test_only_bounce_uses_require_exhaustion():
    for name, profile in PROFILES.items():
        expected = "require_exhaustion" if name == "bounce" else DEFAULT_HTF_POLICY
        assert profile["htf_policy"] == expected, name


@pytest.mark.parametrize("name", TREND_PROFILES)
def test_trend_profiles_keep_blocking_counter_trend(name):
    profile = PROFILES[name]
    ctx = bullish_ctx()  # perfect exhaustion — must still not open the gate
    assert apply_htf_policy("long", "bearish", profile, ctx) is None
    assert apply_htf_policy("short", "bullish", profile, bearish_ctx()) is None
    assert apply_htf_policy("long", "bullish", profile, ctx) == "long"
    assert apply_htf_policy("short", "bearish", profile, ctx) == "short"


def test_bounce_profile_admits_counter_trend_only_on_exhaustion():
    """The one profile wired to the policy: exhaustion opens the gate, and
    nothing else does."""
    profile = PROFILES["bounce"]
    assert apply_htf_policy("long", "bearish", profile, bullish_ctx()) == "long"
    assert apply_htf_policy("short", "bullish", profile, bearish_ctx()) == "short"
    # Counter-trend without exhaustion, and with no context at all: blocked.
    assert apply_htf_policy("long", "bearish", profile, bearish_ctx()) is None
    assert apply_htf_policy("short", "bullish", profile, bullish_ctx()) is None
    assert apply_htf_policy("long", "bearish", profile, None) is None


@pytest.mark.parametrize("profile", [None, {}, {"htf_policy": None},
                                     {"htf_policy": "no_such_policy"}])
def test_unknown_policy_still_falls_back_to_blocking(profile):
    assert apply_htf_policy("long", "bearish", profile, bullish_ctx()) is None
    assert apply_htf_policy("short", "bullish", profile, bearish_ctx()) is None
