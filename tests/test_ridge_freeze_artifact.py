"""C4.4a: the one-time Ridge artifact freeze.

Only the pure parts are exercised here — building the real artifact needs
the full dataset and takes minutes. What matters and is cheap to pin: the
training region cannot reach the sealed holdout, and the "scenario" table is
an empirical frequency rather than anything invented.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.ridge_freeze_artifact import (ARTIFACT_ID, realized_range_by_category,
                                         training_region)
from volatility import percentile as pct


def test_training_region_is_pulled_back_by_the_whole_horizon():
    """A bar within `horizon` of the holdout has a label built from bars
    inside it — training on it would leak the sealed region into the weights."""
    lo, hi = training_region(span_lo=1400, holdout_lo=8199, horizon_bars=12)
    assert (lo, hi) == (1400, 8187)
    assert hi + 12 == 8199, "the gap must be exactly the label horizon"


def test_training_region_scales_with_the_horizon():
    for horizon in (1, 12, 48):
        _, hi = training_region(0, 1000, horizon)
        assert hi == 1000 - horizon


def test_scenarios_are_empirical_frequencies_not_predictions():
    """Each category reports what actually happened on the training rows that
    fell into it — quantiles of realized range, in ATR multiples."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    ref = pct.build_reference(scores, {"src": "test"})
    # labels are log(ratio + eps); make them correlate with score so the
    # buckets are not degenerate
    labels = np.log(np.abs(scores) * 2 + 1.0)

    out = realized_range_by_category(ref, scores, labels)
    assert set(out) == set(pct.CATEGORIES)
    for name, stats in out.items():
        if not stats["n"]:
            continue
        assert stats["realized_atr_ratio_p10"] <= stats["realized_atr_ratio_median"]
        assert stats["realized_atr_ratio_median"] <= stats["realized_atr_ratio_p90"]
        assert stats["realized_atr_ratio_p10"] > 0, "ratios are positive by construction"


def test_scenario_rows_partition_the_input():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=1500)
    ref = pct.build_reference(scores, {"src": "test"})
    labels = np.log(np.abs(scores) + 1.0)
    out = realized_range_by_category(ref, scores, labels)
    assert sum(s["n"] for s in out.values()) == len(scores)


def test_scenarios_undo_the_frozen_log_transform():
    """The stored label is log(ratio + 1e-6); the table must report ratios,
    which is what a reader can picture."""
    ref = pct.build_reference(np.array([0.0, 1.0]), {"src": "test"})
    ratio = 3.5
    label = float(np.log(ratio + 1e-6))
    out = realized_range_by_category(ref, np.array([1.0]), np.array([label]))
    reported = [s for s in out.values() if s["n"]][0]
    assert reported["realized_atr_ratio_median"] == pytest.approx(ratio, abs=1e-6)


def test_artifact_id_is_stable():
    """The id is referenced by the ledger's shadow records; renaming it would
    orphan every forecast already written."""
    assert ARTIFACT_ID == "c44_ridge_frozen_v1"
