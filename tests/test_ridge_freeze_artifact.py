"""C4.4a: the one-time Ridge artifact freeze.

Only the pure parts are exercised here — building the real artifact needs
the full dataset and takes minutes. What matters and is cheap to pin: the
training region cannot reach the sealed holdout, and the "scenario" table is
an empirical frequency rather than anything invented.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from tools.forecast_platform.model_registry import ModelRegistry
from tools.ridge_freeze_artifact import (ARTIFACT_ID, FrozenArtifactError,
                                         _sha256_file, _write_frozen,
                                         realized_range_by_category,
                                         registered_sha256, training_region)
from volatility import percentile as pct


def _writer(text: str):
    def render(path: str) -> None:
        with open(path, "w") as fh:
            fh.write(text)
    return render


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


class TestFrozenWriteGuard:
    """The defect this class exists for: the freeze tool called model.save()
    and only afterwards asked the registry whether the id was taken. A re-run
    on a refreshed dataset therefore replaced an immutable artifact in place,
    and the registry's correct refusal came too late to matter."""

    def test_creates_when_absent(self, tmp_path):
        path = str(tmp_path / "a.json")
        sha, status = _write_frozen(path, _writer("one"))
        assert status == "created"
        assert sha == _sha256_file(path)
        assert open(path).read() == "one"

    def test_identical_rerun_is_a_no_op(self, tmp_path):
        path = str(tmp_path / "a.json")
        first, _ = _write_frozen(path, _writer("one"))
        second, status = _write_frozen(path, _writer("one"))
        assert (second, status) == (first, "unchanged")

    def test_refuses_to_replace_different_content(self, tmp_path):
        path = str(tmp_path / "a.json")
        _write_frozen(path, _writer("one"))
        with pytest.raises(FrozenArtifactError, match="never replaced"):
            _write_frozen(path, _writer("two"))
        assert open(path).read() == "one", "the original must survive intact"

    def test_refused_write_leaves_no_sidecar(self, tmp_path):
        """A stray .new next to a frozen artifact is its own hazard — the next
        reader cannot tell which of the two is the real one."""
        path = str(tmp_path / "a.json")
        _write_frozen(path, _writer("one"))
        with pytest.raises(FrozenArtifactError):
            _write_frozen(path, _writer("two"))
        assert os.listdir(tmp_path) == ["a.json"]

    def test_registry_pin_is_enforced_even_when_the_file_is_gone(self, tmp_path):
        """Deleting the artifact must not launder a re-freeze: the registry
        still pins the hash, so a different model cannot take the id."""
        path = str(tmp_path / "a.json")
        sha, _ = _write_frozen(path, _writer("one"))
        os.remove(path)
        with pytest.raises(FrozenArtifactError, match="registered with sha256"):
            _write_frozen(path, _writer("two"), expected_sha256=sha)
        assert not os.path.exists(path)

    def test_matching_pin_recreates_a_deleted_artifact(self, tmp_path):
        path = str(tmp_path / "a.json")
        sha, _ = _write_frozen(path, _writer("one"))
        os.remove(path)
        again, status = _write_frozen(path, _writer("one"), expected_sha256=sha)
        assert (again, status) == (sha, "created")

    def test_render_failure_does_not_touch_the_existing_file(self, tmp_path):
        path = str(tmp_path / "a.json")
        _write_frozen(path, _writer("one"))

        def explode(p):
            with open(p, "w") as fh:
                fh.write("partial")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _write_frozen(path, explode)
        assert open(path).read() == "one"

    def test_unregistered_id_reads_as_none(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "reg"))
        assert registered_sha256(registry, "nope") is None

    def test_corrupt_registry_entry_is_not_silently_unpinned(self, tmp_path):
        """Returning None on a damaged entry would drop the pin exactly when
        it is most needed."""
        registry = ModelRegistry(str(tmp_path / "reg"))
        with open(os.path.join(registry.registry_dir, "x.json"), "w") as fh:
            fh.write("{not json")
        with pytest.raises(Exception) as exc:
            registered_sha256(registry, "x")
        assert not isinstance(exc.value, FileNotFoundError)


def test_artifact_id_is_stable():
    """The id is referenced by the ledger's shadow records; renaming it would
    orphan every forecast already written."""
    assert ARTIFACT_ID == "c44_ridge_frozen_v1"
