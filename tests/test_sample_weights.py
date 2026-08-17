"""M03 tests: concurrency, uniqueness and effective sample size.

The cases the G1 task requires: no overlap, two identical events, partial
overlap, nesting, a high-concurrency burst, edge boundaries, and invalid spans
failing closed. Plus the invariant ARCHITECTURE.md M03 states as a specification
violation: the raw and effective counts are always reported together.
"""
from __future__ import annotations

import numpy as np
import pytest

from labeling.sample_weights import (SAMPLE_WEIGHTS_VERSION, SampleWeightError,
                                     average_uniqueness, concurrency,
                                     sequential_bootstrap, uniqueness_weights)


def spans(*pairs: tuple[int, int]) -> np.ndarray:
    return np.array(pairs, dtype=np.int64)


# --- concurrency -------------------------------------------------------------

def test_concurrency_counts_closed_intervals():
    """Spans are closed, so a label is live on the bar it starts and the bar it
    resolves. The half-open reading would call two labels non-overlapping when
    the second one's first bar is the first one's last."""
    c = concurrency(spans((0, 2), (2, 4)), n_bars=5)
    assert list(c) == [1, 1, 2, 1, 1]


def test_concurrency_of_a_single_span():
    assert list(concurrency(spans((1, 3)), n_bars=5)) == [0, 1, 1, 1, 0]


def test_concurrency_of_a_burst():
    c = concurrency(spans(*[(0, 9)] * 7), n_bars=10)
    assert list(c) == [7] * 10


def test_concurrency_axis_must_cover_every_span():
    with pytest.raises(SampleWeightError, match="n_bars"):
        concurrency(spans((0, 5)), n_bars=3)


def test_n_bars_is_inferred_when_omitted():
    assert concurrency(spans((0, 3))).size == 4


# --- uniqueness --------------------------------------------------------------

def test_no_overlap_gives_uniqueness_one():
    u = average_uniqueness(spans((0, 1), (2, 3), (4, 5)))
    assert np.allclose(u, 1.0)


def test_two_identical_events_each_get_one_half():
    u = average_uniqueness(spans((0, 5), (0, 5)))
    assert np.allclose(u, 0.5)


def test_n_identical_events_each_get_one_over_n():
    for n in (2, 3, 7, 20):
        u = average_uniqueness(spans(*[(0, 4)] * n))
        assert np.allclose(u, 1.0 / n), n


def test_partial_overlap_is_between_the_extremes():
    # [0,3] and [2,5] share bars 2 and 3 of four bars each.
    u = average_uniqueness(spans((0, 3), (2, 5)))
    assert np.allclose(u, (1 + 1 + 0.5 + 0.5) / 4)
    assert np.all((u > 0.5) & (u < 1.0))


def test_a_nested_event_is_less_unique_than_its_container():
    # [0,9] fully contains [4,5].
    u = average_uniqueness(spans((0, 9), (4, 5)))
    outer, inner = u[0], u[1]
    assert inner == pytest.approx(0.5)
    assert outer == pytest.approx((8 * 1.0 + 2 * 0.5) / 10)
    assert outer > inner


def test_uniqueness_is_always_in_zero_to_one():
    rng = np.random.default_rng(3)
    starts = rng.integers(0, 200, size=80)
    ends = starts + rng.integers(1, 30, size=80)
    u = average_uniqueness(np.column_stack([starts, ends]))
    assert np.all(u > 0.0) and np.all(u <= 1.0)


def test_adjacent_but_not_touching_events_are_fully_unique():
    u = average_uniqueness(spans((0, 2), (3, 5)))
    assert np.allclose(u, 1.0)


# --- weights and both sample sizes -------------------------------------------

def test_weights_sum_to_the_effective_sample_size():
    w = uniqueness_weights(spans((0, 5), (0, 5), (10, 12)))
    assert w.weights.sum() == pytest.approx(w.effective_count)
    assert w.effective_count == pytest.approx(0.5 + 0.5 + 1.0)
    assert w.raw_count == 3


def test_non_overlapping_spans_have_effective_equal_to_raw():
    w = uniqueness_weights(spans((0, 1), (2, 3), (4, 5), (6, 7)))
    assert w.effective_count == pytest.approx(w.raw_count)
    assert w.overlap_factor == pytest.approx(1.0)


def test_overlapping_spans_have_effective_strictly_below_raw():
    w = uniqueness_weights(spans(*[(i, i + 11) for i in range(40)]))
    assert w.effective_count < w.raw_count
    assert w.overlap_factor > 1.0


def test_the_project_scale_overlap_is_recovered():
    """A 12-bar label started on every bar is the project's own geometry, and the
    known answer is roughly a 12x overlap. This is the number ARCHITECTURE.md M03
    cites as the reason the module exists."""
    w = uniqueness_weights(spans(*[(i, i + 11) for i in range(400)]))
    assert 10.0 < w.overlap_factor < 12.5
    assert w.max_concurrency == 12


def test_weights_are_strictly_positive():
    w = uniqueness_weights(spans(*[(0, 20)] * 50))
    assert np.all(w.weights > 0.0)


def test_summary_reports_both_counts_and_never_only_one():
    s = uniqueness_weights(spans((0, 5), (0, 5))).summary()
    assert {"raw_count", "effective_count", "overlap_factor"} <= set(s)
    assert s["raw_count"] == 2
    assert s["effective_count"] == pytest.approx(1.0)


def test_as_dict_is_json_friendly():
    import json
    d = uniqueness_weights(spans((0, 2), (1, 3))).as_dict()
    assert json.loads(json.dumps(d))["raw_count"] == 2
    assert d["version"] == SAMPLE_WEIGHTS_VERSION


# --- fail closed -------------------------------------------------------------

def test_empty_span_set_fails_closed():
    with pytest.raises(SampleWeightError, match="no spans"):
        uniqueness_weights(np.empty((0, 2), dtype=np.int64))


def test_zero_length_span_fails_closed():
    with pytest.raises(SampleWeightError, match="end <= start"):
        uniqueness_weights(spans((0, 5), (7, 7)))


def test_reversed_span_fails_closed():
    with pytest.raises(SampleWeightError, match="end <= start"):
        uniqueness_weights(spans((9, 4)))


def test_negative_span_start_fails_closed():
    with pytest.raises(SampleWeightError, match="non-negative"):
        uniqueness_weights(spans((-1, 4)))


def test_wrong_shape_fails_closed():
    with pytest.raises(SampleWeightError, match="shape"):
        uniqueness_weights(np.arange(6, dtype=np.int64))


# --- sequential bootstrap ----------------------------------------------------

def test_sequential_bootstrap_is_deterministic_given_the_generator():
    s = spans(*[(i, i + 5) for i in range(20)])
    a = sequential_bootstrap(s, 15, np.random.default_rng(11))
    b = sequential_bootstrap(s, 15, np.random.default_rng(11))
    assert list(a) == list(b)


def test_sequential_bootstrap_returns_the_requested_size_in_range():
    s = spans(*[(i, i + 5) for i in range(20)])
    drawn = sequential_bootstrap(s, 30, np.random.default_rng(12))
    assert drawn.size == 30
    assert drawn.min() >= 0 and drawn.max() < 20


def test_sequential_bootstrap_prefers_less_overlapping_labels():
    """Ten labels stacked on the same bars plus one standing alone: the lone
    label must be drawn far more often than its 1-in-11 share, because ordinary
    bootstrap over-samples the redundant block."""
    s = spans(*([(0, 9)] * 10 + [(50, 59)]))
    drawn = sequential_bootstrap(s, 400, np.random.default_rng(13))
    lone_share = float(np.mean(drawn == 10))
    assert lone_share > 2 * (1 / 11)


def test_sequential_bootstrap_validates_size():
    with pytest.raises(SampleWeightError, match="size"):
        sequential_bootstrap(spans((0, 3)), 0, np.random.default_rng(1))
