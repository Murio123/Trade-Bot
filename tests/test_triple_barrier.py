"""M02 tests: triple-barrier labeling.

The cases required by the G1 task: TP first, SL first, timeout, same-bar
double touch, the exact vertical boundary, gaps, a truncated tail, and
deterministic replay. Plus the anti-look-ahead properties from
ARCHITECTURE.md §5.1, which are the reason this module exists at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeling.triple_barrier import (LONG, NEUTRAL, SHORT, TRIPLE_BARRIER_VERSION,
                                     BarrierConfig, BarrierDataError,
                                     BarrierError, apply_barriers, knowable_at,
                                     label_spans, outcome_counts)

BAR_MS = 4 * 3_600_000
CFG = BarrierConfig(upper_mult=1.0, lower_mult=1.0, vertical_bars=3)


def frame(rows: list[tuple[float, float, float]], *, start_ms: int = 0,
          bar_ms: int = BAR_MS) -> pd.DataFrame:
    """`rows` are `(high, low, close)`; timestamps are a clean grid."""
    high, low, close = zip(*rows)
    return pd.DataFrame({
        "high": high, "low": low, "close": close,
        "close_time": [start_ms + i * bar_ms for i in range(len(rows))],
    })


def flat_sigma(n: int, value: float = 0.10) -> np.ndarray:
    return np.full(n, value)


# --- first-touch resolution --------------------------------------------------

def test_tp_first_for_a_long():
    # Entry close 100, sigma 0.10 -> upper 110, lower 90.
    f = frame([(100, 100, 100), (111, 99, 105), (100, 89, 95), (100, 100, 100)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(4), CFG)
    assert (e.barrier, e.outcome, e.label) == ("UPPER", "TP", 1)
    assert e.end_idx == 1
    assert e.exit_price == pytest.approx(110.0)
    assert e.r_multiple == pytest.approx(1.0)
    assert not e.timeout and not e.tie and e.resolved


def test_sl_first_for_a_long():
    f = frame([(100, 100, 100), (105, 89, 95), (111, 100, 105), (100, 100, 100)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(4), CFG)
    assert (e.barrier, e.outcome, e.label) == ("LOWER", "SL", -1)
    assert e.end_idx == 1
    assert e.r_multiple == pytest.approx(-1.0)


def test_a_short_mirrors_a_long_exactly():
    """The same price path, opposite side, opposite label and the same magnitude.
    Any asymmetry here is the defect NC4 caught in the tie rule."""
    f = frame([(100, 100, 100), (105, 89, 95), (100, 100, 100), (100, 100, 100)])
    (long_e,) = apply_barriers(f, [(0, LONG)], flat_sigma(4), CFG)
    (short_e,) = apply_barriers(f, [(0, SHORT)], flat_sigma(4), CFG)
    assert long_e.barrier == short_e.barrier == "LOWER"
    assert (long_e.outcome, long_e.label) == ("SL", -1)
    assert (short_e.outcome, short_e.label) == ("TP", 1)
    assert short_e.r_multiple == pytest.approx(-long_e.r_multiple)


def test_timeout_when_neither_barrier_is_touched():
    f = frame([(100, 100, 100), (105, 95, 102), (106, 96, 101), (104, 97, 103),
               (200, 10, 150)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(5), CFG)
    assert (e.barrier, e.outcome, e.label) == ("TIME", "TIME", 0)
    assert e.timeout and e.resolved
    assert e.end_idx == 3                       # start + vertical_bars
    assert e.exit_price == pytest.approx(103.0)  # the vertical bar's close


# --- the tie convention ------------------------------------------------------

def test_same_bar_double_touch_resolves_to_the_stop_for_a_long():
    f = frame([(100, 100, 100), (115, 85, 100), (100, 100, 100), (100, 100, 100)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(4), CFG)
    assert (e.barrier, e.outcome, e.label) == ("LOWER", "SL", -1)
    assert e.tie


def test_same_bar_double_touch_resolves_to_the_stop_for_a_short():
    """Regression for the defect NC4 caught: resolving every tie to LOWER made
    the "conservative" rule hand each short a free win, because LOWER is a
    short's target."""
    f = frame([(100, 100, 100), (115, 85, 100), (100, 100, 100), (100, 100, 100)])
    (e,) = apply_barriers(f, [(0, SHORT)], flat_sigma(4), CFG)
    assert (e.barrier, e.outcome, e.label) == ("UPPER", "SL", -1)
    assert e.tie
    assert e.r_multiple == pytest.approx(-1.0)


def test_a_tie_is_a_loss_for_both_sides():
    f = frame([(100, 100, 100), (115, 85, 100), (100, 100, 100), (100, 100, 100)])
    for side in (LONG, SHORT):
        (e,) = apply_barriers(f, [(0, side)], flat_sigma(4), CFG)
        assert e.r_multiple < 0, side


def test_a_neutral_tie_resolves_to_lower_by_frozen_convention():
    f = frame([(100, 100, 100), (115, 85, 100), (100, 100, 100), (100, 100, 100)])
    (e,) = apply_barriers(f, [(0, NEUTRAL)], flat_sigma(4), CFG)
    assert e.barrier == "LOWER" and e.tie


def test_ties_are_counted_separately_from_stops():
    f = frame([(100, 100, 100), (115, 85, 100), (100, 100, 100), (100, 100, 100),
               (100, 100, 100), (105, 89, 95), (100, 100, 100), (100, 100, 100)])
    events = apply_barriers(f, [(0, LONG), (4, LONG)], flat_sigma(8), CFG)
    counts = outcome_counts(events)
    assert counts["SL"] == 2 and counts["TIE"] == 1


# --- boundaries --------------------------------------------------------------

def test_the_decision_bar_is_never_scanned():
    """Bar 0's own range spans both barriers. Its label must come from later
    bars: the barriers are set from bar 0's close, so reading bar 0's high and
    low would be look-ahead and would resolve nearly every event instantly."""
    f = frame([(999, 1, 100), (105, 95, 100), (106, 96, 100), (104, 97, 100)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(4), CFG)
    assert e.outcome == "TIME"
    assert e.end_idx == 3


def barrier_levels(entry: float = 100.0, sigma: float = 0.10) -> tuple[float, float]:
    """The barriers exactly as the module computes them.

    Not the decimal values a reader would write down: `100 * (1 + 0.10)` is
    110.00000000000001, so a literal 110.0 is genuinely below the upper barrier.
    The non-strict convention is about the computed level, and testing it against
    a hand-written constant would be testing floating-point representation.
    """
    return entry * (1.0 + sigma), entry * (1.0 - sigma)


def test_a_touch_exactly_on_the_vertical_bar_is_a_touch_not_a_timeout():
    upper, _ = barrier_levels()
    f = frame([(100, 100, 100), (105, 95, 100), (105, 95, 100), (upper, 95, 105)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(4), CFG)
    assert (e.barrier, e.end_idx) == ("UPPER", 3)
    assert not e.timeout


def test_a_touch_one_bar_past_the_vertical_barrier_is_a_timeout():
    f = frame([(100, 100, 100), (105, 95, 100), (105, 95, 100), (105, 95, 100),
               (999, 95, 500)])
    (e,) = apply_barriers(f, [(0, LONG)], flat_sigma(5), CFG)
    assert e.outcome == "TIME" and e.end_idx == 3


def test_touches_are_non_strict():
    """`high >= upper` — reaching the barrier exactly is reaching it, and the
    smallest possible float below it is not."""
    upper, lower = barrier_levels()

    exact = frame([(100, 100, 100), (upper, 95, 105), (100, 100, 100),
                   (100, 100, 100)])
    assert apply_barriers(exact, [(0, LONG)], flat_sigma(4), CFG)[0].barrier == "UPPER"

    under = frame([(100, 100, 100), (np.nextafter(upper, 0.0), 95, 105),
                   (100, 100, 100), (100, 100, 100)])
    assert apply_barriers(under, [(0, LONG)], flat_sigma(4), CFG)[0].barrier == "TIME"

    # The same rule on the lower side.
    exact_lo = frame([(100, 100, 100), (105, lower, 95), (100, 100, 100),
                      (100, 100, 100)])
    assert apply_barriers(exact_lo, [(0, LONG)], flat_sigma(4), CFG)[0].barrier == "LOWER"

    over_lo = frame([(100, 100, 100), (105, np.nextafter(lower, 1e9), 95),
                     (100, 100, 100), (100, 100, 100)])
    assert apply_barriers(over_lo, [(0, LONG)], flat_sigma(4), CFG)[0].barrier == "TIME"


# --- spans -------------------------------------------------------------------

def test_span_is_never_zero_length():
    f = frame([(100, 100, 100)] * 8)
    events = apply_barriers(f, [0, 1, 2], flat_sigma(8), CFG)
    for e in events:
        assert e.end_idx > e.start_idx
    spans = label_spans(events)
    assert np.all(spans[:, 1] > spans[:, 0])


def test_label_spans_drops_unresolved_events_by_default():
    f = frame([(100, 100, 100)] * 5)
    events = apply_barriers(f, [0, 3], flat_sigma(5), CFG)
    assert [e.resolved for e in events] == [True, False]
    assert label_spans(events).shape == (1, 2)
    assert label_spans(events, resolved_only=False).shape == (2, 2)


def test_label_spans_of_nothing_is_an_empty_two_column_array():
    assert label_spans([]).shape == (0, 2)


def test_knowable_at_admits_only_finished_labels():
    f = frame([(100, 100, 100)] * 12)
    events = apply_barriers(f, [0, 4, 8], flat_sigma(12), CFG)
    resolved = [e for e in events if e.resolved]
    assert [e.start_idx for e in knowable_at(resolved, 3)] == [0]
    assert [e.start_idx for e in knowable_at(resolved, 7)] == [0, 4]
    assert [e.start_idx for e in knowable_at(resolved, 2)] == []


# --- truncation and gaps -----------------------------------------------------

def test_a_truncated_horizon_is_not_reported_as_a_timeout():
    """TIME asserts that a full horizon was observed and was empty. A frame that
    ends early cannot support that claim."""
    f = frame([(100, 100, 100)] * 3)
    (e,) = apply_barriers(f, [1], flat_sigma(3), CFG)
    assert (e.barrier, e.outcome) == ("TRUNCATED", "TRUNCATED")
    assert e.label is None and e.r_multiple is None and not e.resolved
    assert not e.timeout


def test_a_touch_before_the_frame_ends_resolves_even_if_the_horizon_is_short():
    f = frame([(100, 100, 100), (111, 99, 105), (100, 100, 100)])
    (e,) = apply_barriers(f, [0], flat_sigma(3), CFG)
    assert e.barrier == "UPPER" and e.resolved


def test_an_event_on_the_last_bar_has_no_label_at_all():
    f = frame([(100, 100, 100)] * 3)
    with pytest.raises(BarrierDataError):
        apply_barriers(f, [2], flat_sigma(3), CFG)


def test_a_gap_inside_the_horizon_fails_closed():
    f = frame([(100, 100, 100)] * 6)
    f.loc[3:, "close_time"] += 10 * BAR_MS          # three bars missing
    with pytest.raises(BarrierDataError, match="gap"):
        apply_barriers(f, [1], flat_sigma(6), CFG)


def test_a_gap_outside_the_horizon_does_not_block_an_event():
    f = frame([(100, 100, 100)] * 10)
    f.loc[8:, "close_time"] += 10 * BAR_MS
    events = apply_barriers(f, [1], flat_sigma(10), CFG)
    assert events[0].resolved


def test_millisecond_jitter_is_not_a_gap():
    f = frame([(100, 100, 100)] * 6)
    f.loc[2, "close_time"] += 2                      # …400002 for a …400000
    assert apply_barriers(f, [0], flat_sigma(6), CFG)[0].resolved


def test_unsorted_timestamps_fail_closed():
    f = frame([(100, 100, 100)] * 6)
    f.loc[3, "close_time"] = f.loc[1, "close_time"]
    with pytest.raises(BarrierDataError, match="increasing"):
        apply_barriers(f, [0], flat_sigma(6), CFG)


def test_skipping_the_continuity_check_must_be_named_explicitly():
    """A frame with no timestamp column is a programming decision, not a
    default: `time_col=None` has to appear at the call site."""
    f = frame([(100, 100, 100)] * 6).drop(columns=["close_time"])
    with pytest.raises(BarrierDataError, match="time_col"):
        apply_barriers(f, [0], flat_sigma(6), CFG)
    assert apply_barriers(f, [0], flat_sigma(6), CFG, time_col=None)


# --- input validation --------------------------------------------------------

def test_misaligned_sigma_is_rejected():
    f = frame([(100, 100, 100)] * 6)
    with pytest.raises(BarrierError, match="aligned"):
        apply_barriers(f, [0], flat_sigma(5), CFG)


def test_non_positive_sigma_fails_closed():
    f = frame([(100, 100, 100)] * 6)
    sig = flat_sigma(6)
    sig[0] = 0.0
    with pytest.raises(BarrierDataError, match="sigma"):
        apply_barriers(f, [0], sig, CFG)


def test_event_index_outside_the_frame_is_rejected():
    f = frame([(100, 100, 100)] * 6)
    for idx in (-1, 6, 99):
        with pytest.raises(BarrierError, match="outside"):
            apply_barriers(f, [idx], flat_sigma(6), CFG)


def test_bad_side_is_rejected():
    f = frame([(100, 100, 100)] * 6)
    with pytest.raises(BarrierError, match="side"):
        apply_barriers(f, [(0, 2)], flat_sigma(6), CFG)


def test_missing_price_columns_are_rejected():
    f = frame([(100, 100, 100)] * 6).drop(columns=["high"])
    with pytest.raises(BarrierError, match="high"):
        apply_barriers(f, [0], flat_sigma(6), CFG)


def test_config_validation():
    with pytest.raises(BarrierError):
        BarrierConfig(upper_mult=0.0, lower_mult=1.0, vertical_bars=3)
    with pytest.raises(BarrierError):
        BarrierConfig(upper_mult=1.0, lower_mult=-1.0, vertical_bars=3)
    with pytest.raises(BarrierError, match="vertical_bars"):
        BarrierConfig(upper_mult=1.0, lower_mult=1.0, vertical_bars=0)


def test_config_hash_is_stable_and_sensitive():
    a = BarrierConfig(1.0, 1.0, 12)
    assert a.content_sha256() == BarrierConfig(1.0, 1.0, 12).content_sha256()
    assert a.content_sha256() != BarrierConfig(2.0, 1.0, 12).content_sha256()
    assert a.content_sha256() != BarrierConfig(1.0, 1.0, 13).content_sha256()
    assert len(a.content_sha256()) == 12


# --- determinism -------------------------------------------------------------

def test_deterministic_replay():
    rng = np.random.default_rng(4)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    f = pd.DataFrame({"high": close * 1.01, "low": close * 0.99, "close": close,
                      "close_time": np.arange(n) * BAR_MS})
    sig = flat_sigma(n, 0.02)
    events = [(i, (-1) ** i) for i in range(0, n - 20, 5)]
    first = apply_barriers(f, events, sig, CFG)
    second = apply_barriers(f, events, sig, CFG)
    assert [e.as_dict() for e in first] == [e.as_dict() for e in second]


def test_version_is_pinned():
    assert TRIPLE_BARRIER_VERSION == "m02_triple_barrier_v1"
