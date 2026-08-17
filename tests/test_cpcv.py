"""M04 tests: combinatorial purged cross-validation.

The cases the G1 task requires: the combinatorial split count, no train/test
overlap, purge removing boundary contamination, embargo applied correctly,
deterministic manifests, tiny datasets failing closed, and — the one that
matters most — leakage governed by event spans rather than row indices.

The adversarial test ARCHITECTURE.md M04 demands is
`test_a_long_span_that_reaches_across_a_test_group_is_purged`: row indices alone
would keep that event, and it is the case a row-based implementation gets wrong.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pytest

from validation.cpcv import (CPCV_VERSION, SEALED_HOLDOUT_IDX_LO, CPCVConfig,
                             CPCVError, assemble_paths, backtest_paths,
                             combinatorial_splits, embargo_violations,
                             leakage_pairs, path_distribution, split_manifest)


def spans(*pairs: tuple[int, int]) -> np.ndarray:
    return np.array(pairs, dtype=np.int64)


def strided(n: int, width: int = 4, stride: int = 5, start: int = 0) -> np.ndarray:
    """`n` non-overlapping events, `width` bars each."""
    return np.array([[start + i * stride, start + i * stride + width]
                     for i in range(n)], dtype=np.int64)


CFG = CPCVConfig(n_groups=6, n_test_groups=2, embargo_bars=0, purge_bars=0,
                 holdout_idx_lo=None, min_train_events=1)


# --- geometry counts ---------------------------------------------------------

def test_split_and_path_counts_match_the_combinatorics():
    for m, k in ((6, 2), (5, 1), (8, 3), (4, 2)):
        cfg = CPCVConfig(n_groups=m, n_test_groups=k, embargo_bars=0,
                         holdout_idx_lo=None)
        assert cfg.n_splits == comb(m, k)
        assert cfg.n_paths == comb(m - 1, k - 1)
        # The identity the docstring claims: C(M,k)*k/M == C(M-1,k-1).
        assert cfg.n_paths == comb(m, k) * k // m


def test_the_built_geometry_has_the_promised_number_of_splits():
    g = combinatorial_splits(strided(60), CFG)
    assert g.n_splits == comb(6, 2) == 15
    assert g.n_paths == comb(5, 1) == 5
    assert g.version == CPCV_VERSION


def test_every_group_is_tested_in_exactly_n_paths_splits():
    g = combinatorial_splits(strided(60), CFG)
    counts = {i: 0 for i in range(6)}
    for s in g.splits:
        for grp in s.test_groups:
            counts[grp] += 1
    assert set(counts.values()) == {g.n_paths}


def test_each_split_tests_exactly_k_groups():
    g = combinatorial_splits(strided(60), CFG)
    assert all(len(s.test_groups) == 2 for s in g.splits)
    assert len({s.test_groups for s in g.splits}) == 15


def test_groups_partition_the_admitted_events():
    g = combinatorial_splits(strided(60), CFG)
    for s in g.splits:
        assert set(s.train_idx) | set(s.test_idx) | set(s.purged_idx) == set(
            int(i) for i in g.admitted_rows)


# --- the correctness property -------------------------------------------------

def test_train_and_test_never_share_a_row():
    g = combinatorial_splits(strided(60), CFG)
    for s in g.splits:
        assert not (set(s.train_idx) & set(s.test_idx))


def test_no_leakage_on_non_overlapping_events():
    s = strided(60)
    assert leakage_pairs(combinatorial_splits(s, CFG), s) == 0


def test_no_leakage_on_heavily_overlapping_events():
    """Every event 12 bars long, started every bar — the project's own geometry,
    where naive splitting leaks on every boundary."""
    s = np.array([[i, i + 12] for i in range(300)], dtype=np.int64)
    cfg = CPCVConfig(n_groups=6, n_test_groups=2, embargo_bars=2, purge_bars=0,
                     holdout_idx_lo=None, min_train_events=5)
    assert leakage_pairs(combinatorial_splits(s, cfg), s) == 0


def test_a_long_span_that_reaches_across_a_test_group_is_purged():
    """The adversarial case ARCHITECTURE.md M04 asks for.

    Event 0 starts at bar 0 and resolves at bar 500, so its label was determined
    by bars belonging to every later group. By row index it is the first row and
    looks safely in the past; by span it contaminates everything. A row-based
    purge keeps it and leaks.
    """
    # 60 short events tile bars 0..303 in six groups of ten, so each group spans
    # roughly 50 bars. The long event starts inside group 1 and resolves inside
    # group 3, crossing group 2 entirely.
    long_event = spans((55, 160))
    s = np.vstack([strided(60), long_event])
    long_row = 60
    g = combinatorial_splits(s, CFG)

    assert leakage_pairs(g, s) == 0

    crossed_groups = [grp for grp, (lo, hi) in g.group_windows.items()
                      if lo <= 160 and hi >= 55]
    assert len(crossed_groups) >= 3, crossed_groups

    purged_somewhere = False
    for sp in g.splits:
        touches_a_crossed_group = bool(set(sp.test_groups) & set(crossed_groups))
        if touches_a_crossed_group:
            # By row index this event sits early in the frame and looks safely in
            # the past. By span it was labelled by bars the test group owns.
            assert long_row not in set(sp.train_idx), sp.test_groups
            if long_row in set(sp.purged_idx):
                purged_somewhere = True
        elif long_row in set(sp.train_idx):
            # Kept only when no test group overlaps it — which is the point of
            # purging by span rather than dropping the event globally.
            for t in sp.test_idx:
                assert not (s[t][0] <= 160 and 55 <= s[t][1])
    assert purged_somewhere, "the long span was never actually purged"


def test_purge_bars_widens_the_exclusion_beyond_the_spans():
    s = strided(60)
    tight = combinatorial_splits(s, CFG)
    wide = combinatorial_splits(s, CPCVConfig(
        n_groups=6, n_test_groups=2, embargo_bars=0, purge_bars=25,
        holdout_idx_lo=None, min_train_events=1))
    assert sum(sp.purged_idx.size for sp in wide.splits) > \
        sum(sp.purged_idx.size for sp in tight.splits)
    assert leakage_pairs(wide, s) == 0


def test_embargo_removes_training_events_just_after_a_test_group():
    s = strided(60)
    with_embargo = combinatorial_splits(s, CPCVConfig(
        n_groups=6, n_test_groups=2, embargo_bars=20, purge_bars=0,
        holdout_idx_lo=None, min_train_events=1))
    assert embargo_violations(with_embargo, s) == 0
    assert sum(sp.purged_idx.size for sp in with_embargo.splits) > 0


def test_without_an_embargo_there_is_nothing_to_violate():
    s = strided(60)
    assert embargo_violations(combinatorial_splits(s, CFG), s) == 0


def test_embargo_is_right_edge_only():
    """A training event ending just BEFORE a test group starts is kept: span
    purge already covers backward contamination exactly, so a left embargo would
    discard data for no stated reason."""
    s = spans((0, 4), (10, 14), (20, 24), (30, 34), (40, 44), (50, 54),
              (60, 64), (70, 74), (80, 84), (90, 94), (100, 104), (110, 114))
    cfg = CPCVConfig(n_groups=6, n_test_groups=1, embargo_bars=3, purge_bars=0,
                     holdout_idx_lo=None, min_train_events=1)
    g = combinatorial_splits(s, cfg)
    for sp in g.splits:
        for grp in sp.test_groups:
            g_lo, _ = g.group_windows[grp]
            before = [i for i in sp.train_idx if s[i][1] < g_lo]
            # Events entirely before the group are retained.
            assert before or g_lo == 0


# --- the sealed holdout -------------------------------------------------------

def test_events_reaching_the_sealed_holdout_are_dropped():
    s = np.vstack([strided(50, start=0),
                   spans((SEALED_HOLDOUT_IDX_LO - 2, SEALED_HOLDOUT_IDX_LO + 5),
                         (SEALED_HOLDOUT_IDX_LO + 10, SEALED_HOLDOUT_IDX_LO + 20))])
    g = combinatorial_splits(s, CPCVConfig(n_groups=5, n_test_groups=2,
                                           embargo_bars=0, min_train_events=1))
    assert g.n_events_held_out == 2
    assert g.n_events_admitted == 50
    held = {50, 51}
    for sp in g.splits:
        assert not (held & set(sp.train_idx))
        assert not (held & set(sp.test_idx))


def test_the_holdout_boundary_is_exclusive_on_the_span_end():
    """A span ending exactly at the boundary index is inside the holdout and is
    dropped; one ending the bar before is admitted."""
    lo = SEALED_HOLDOUT_IDX_LO
    # Two events at the boundary itself, plus a normal population well before it.
    # Both boundary events are short: a span reaching back hundreds of bars would
    # be purged for its width rather than for the holdout, and the test would no
    # longer isolate the boundary rule.
    s = spans((lo - 4, lo), (lo - 5, lo - 1),
              *[(100 + i * 5, 100 + i * 5 + 4) for i in range(20)])
    g = combinatorial_splits(s, CPCVConfig(n_groups=4, n_test_groups=1,
                                           embargo_bars=0, min_train_events=1))
    assert g.n_events_held_out == 1
    assert 0 not in set(g.admitted_rows)      # ends exactly at lo -> dropped
    assert 1 in set(g.admitted_rows)          # ends at lo - 1 -> admitted


def test_everything_inside_the_holdout_fails_closed():
    s = spans(*[(SEALED_HOLDOUT_IDX_LO + i, SEALED_HOLDOUT_IDX_LO + i + 3)
                for i in range(20)])
    with pytest.raises(CPCVError, match="sealed holdout"):
        combinatorial_splits(s, CPCVConfig(n_groups=4, n_test_groups=1,
                                           embargo_bars=0))


def test_the_holdout_can_be_disabled_only_explicitly():
    assert CPCVConfig(n_groups=4, n_test_groups=1,
                      embargo_bars=0).holdout_idx_lo == SEALED_HOLDOUT_IDX_LO


# --- fail closed --------------------------------------------------------------

def test_a_tiny_dataset_fails_closed_rather_than_degrading():
    with pytest.raises(CPCVError, match="cannot be partitioned"):
        combinatorial_splits(strided(4), CFG)


def test_config_validation_rejects_impossible_geometry():
    with pytest.raises(CPCVError, match="n_groups"):
        CPCVConfig(n_groups=1, n_test_groups=1, embargo_bars=0)
    with pytest.raises(CPCVError, match="n_test_groups"):
        CPCVConfig(n_groups=4, n_test_groups=0, embargo_bars=0)
    with pytest.raises(CPCVError, match="must be <"):
        CPCVConfig(n_groups=4, n_test_groups=4, embargo_bars=0)
    with pytest.raises(CPCVError, match="non-negative"):
        CPCVConfig(n_groups=4, n_test_groups=1, embargo_bars=-1)


def test_an_over_wide_purge_refuses_rather_than_returning_an_unpurged_split():
    """The explicit "never falls back to ordinary K-fold" requirement: if purge
    and embargo consume the training set, the answer is a refusal."""
    with pytest.raises(CPCVError, match="after purge"):
        combinatorial_splits(strided(60), CPCVConfig(
            n_groups=6, n_test_groups=2, embargo_bars=0, purge_bars=10_000,
            holdout_idx_lo=None, min_train_events=1))


def test_min_train_events_is_enforced():
    with pytest.raises(CPCVError, match="below the required"):
        combinatorial_splits(strided(12), CPCVConfig(
            n_groups=6, n_test_groups=2, embargo_bars=0, holdout_idx_lo=None,
            min_train_events=50))


def test_malformed_spans_fail_closed():
    with pytest.raises(CPCVError, match="shape"):
        combinatorial_splits(np.arange(10, dtype=np.int64), CFG)
    with pytest.raises(CPCVError, match="no events"):
        combinatorial_splits(np.empty((0, 2), dtype=np.int64), CFG)
    with pytest.raises(CPCVError, match="end > start"):
        combinatorial_splits(spans(*[(i, i) for i in range(20)]), CFG)


# --- manifests ----------------------------------------------------------------

def test_the_manifest_is_deterministic():
    s = strided(60)
    a = split_manifest(combinatorial_splits(s, CFG))
    b = split_manifest(combinatorial_splits(s, CFG))
    assert a == b
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert len(a["manifest_sha256"]) == 16


def test_the_manifest_hash_changes_with_the_geometry():
    s = strided(60)
    base = split_manifest(combinatorial_splits(s, CFG))["manifest_sha256"]
    other = split_manifest(combinatorial_splits(s, CPCVConfig(
        n_groups=6, n_test_groups=2, embargo_bars=5, holdout_idx_lo=None,
        min_train_events=1)))["manifest_sha256"]
    assert base != other


def test_the_manifest_records_the_shared_training_caveat_and_counts():
    m = split_manifest(combinatorial_splits(strided(60), CFG))
    assert m["paths_share_training_data"] is True
    assert m["n_splits"] == 15 and m["n_paths"] == 5
    assert m["n_events_held_out"] == 0
    assert len(m["splits"]) == 15


def test_the_config_hash_is_stable_and_sensitive():
    a = CPCVConfig(n_groups=6, n_test_groups=2, embargo_bars=2)
    assert a.content_sha256() == CPCVConfig(
        n_groups=6, n_test_groups=2, embargo_bars=2).content_sha256()
    assert a.content_sha256() != CPCVConfig(
        n_groups=6, n_test_groups=2, embargo_bars=3).content_sha256()


def test_the_manifest_is_json_serialisable():
    import json
    m = split_manifest(combinatorial_splits(strided(60), CFG))
    assert json.loads(json.dumps(m))["n_splits"] == 15


# --- paths --------------------------------------------------------------------

def test_every_path_covers_every_group_exactly_once():
    g = combinatorial_splits(strided(60), CFG)
    paths = assemble_paths(g)
    assert len(paths) == g.n_paths
    for path in paths:
        assert sorted(grp for grp, _ in path) == list(range(6))


def test_paths_use_different_splits_for_the_same_group():
    g = combinatorial_splits(strided(60), CFG)
    paths = assemble_paths(g)
    for grp in range(6):
        used = [dict(p)[grp] for p in paths]
        assert len(set(used)) == len(used)


def test_backtest_paths_scores_each_group_once_per_split():
    g = combinatorial_splits(strided(60), CFG)
    calls: list[tuple[int, int]] = []

    def score(train_idx, test_idx):
        calls.append((train_idx.size, test_idx.size))
        return float(test_idx.size)

    res = backtest_paths(g, score)
    assert len(calls) == g.n_splits * CFG.n_test_groups
    assert res.path_scores.size == g.n_paths
    assert res.paths_share_training_data is True


def test_backtest_paths_is_deterministic():
    g = combinatorial_splits(strided(60), CFG)

    def score(train_idx, test_idx):
        return float(train_idx.sum() % 97) / 97.0

    a = backtest_paths(g, score).path_scores
    b = backtest_paths(g, score).path_scores
    assert list(a) == list(b)


def test_path_distribution_reports_the_variance_caveat():
    g = combinatorial_splits(strided(60), CFG)
    rng = np.random.default_rng(5)
    dist = path_distribution(backtest_paths(g, lambda tr, te: float(rng.normal())))
    assert dist["n_paths"] == g.n_paths
    assert "variance_caveat" in dist
    assert dist["paths_share_training_data"] is True
    for key in ("mean", "median", "std", "min", "max", "q05", "q95",
                "share_positive"):
        assert key in dist


def test_the_scorer_receives_one_group_at_a_time_not_the_whole_test_set():
    """A path stitches together one group at a time, so a scorer given the
    concatenated test set could not be reassembled into paths."""
    g = combinatorial_splits(strided(60), CFG)
    sizes: list[int] = []
    backtest_paths(g, lambda tr, te: sizes.append(te.size) or 0.0)
    group_sizes = {len(rows) for rows in
                   [[i for i in g.admitted_rows] for _ in range(1)]}
    assert max(sizes) < 60 // CFG.n_groups + 2
