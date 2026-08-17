"""M04 — a distribution of out-of-sample results instead of one number.

Three walk-forward folds give three numbers, and no distribution can be formed
from three numbers. C4.3e hit this and had to recover an interval post hoc with
a block bootstrap. Combinatorial purged cross-validation takes the same data and
extracts many out-of-sample paths from it, so the uncertainty is visible by
construction rather than reconstructed afterwards.

The mechanism: partition the events into `M` contiguous groups, test on every
combination of `k` of them, train on the rest. That is `C(M, k)` splits, and
because each group is tested in `C(M-1, k-1)` of them, the splits reassemble
into exactly `C(M-1, k-1)` complete traversals of the history — the paths.

**The one correctness property.** No test event's label span may intersect any
training event's span. Everything else in this module is bookkeeping; this is
the property that decides whether the results mean anything, and it gets an
exact adversarial count (`leakage_pairs`) rather than a statistical check.

**Purge is two-sided here, and that is the difference from the existing
geometry.** `tools/deep_backtest.fold_windows` purges forward only, which is
correct for a strictly forward walk: training always precedes validation, so
only the training set's forward horizon can reach into it. In a combinatorial
split a test group has training data on *both* sides, and a training event that
starts before the test window can still resolve inside it while a training event
that starts after it was labelled by bars the test window contains. Reusing the
one-sided rule here would leak on every interior group. The existing geometry is
extended, not replaced — the conventions are identical (half-open bar axis,
purge derived from label span rather than a fixed constant, sealed tail removed
first, fail-closed on insufficient geometry).

**Embargo is right-edge only.** Span-based purge already removes backward
contamination exactly, so a left embargo would discard training data for no
stated reason. The right edge is different: serial correlation in the features
outlives the label span, and the embargo is the stated allowance for it.

**Paths are not independent.** They share training data, so the spread of the
path distribution understates the true variance. This is reported with every
output (`paths_share_training_data`) rather than left for the reader to
remember.

Fails closed, always. A geometry that cannot support the requested `M` and `k`
raises; it never silently degrades to ordinary K-fold, which would produce
plausible-looking leaky numbers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb
from typing import Any, Callable

import numpy as np

CPCV_VERSION = "m04_cpcv_v1"

# The C4.1 dataset's sealed holdout boundary on the entry-idx axis
# (reports/c45/BASELINE.md §holdout, ARCHITECTURE.md §5.2). Any event whose label
# span reaches this index is removed before splitting. It is a dataset geometry
# constant, not a strategy parameter, and it is named rather than inlined so
# that a run which deliberately passes something else is visible in the manifest.
SEALED_HOLDOUT_IDX_LO = 8199


class CPCVError(Exception):
    """The requested geometry cannot be produced. Never downgraded to a warning."""


@dataclass(frozen=True)
class CPCVConfig:
    """Frozen split geometry, hashed so a run can be pinned to it.

    `M`, `k`, purge and embargo are all tunable and are therefore all frozen
    before the first run (ARCHITECTURE.md §5.3). `content_sha256` is what makes
    that checkable.
    """
    n_groups: int                                   # M
    n_test_groups: int                              # k
    embargo_bars: int
    purge_bars: int = 0
    holdout_idx_lo: int | None = SEALED_HOLDOUT_IDX_LO
    min_train_events: int = 1

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise CPCVError(f"n_groups must be >= 2, got {self.n_groups}")
        if self.n_test_groups < 1:
            raise CPCVError(
                f"n_test_groups must be >= 1, got {self.n_test_groups}")
        if self.n_test_groups >= self.n_groups:
            raise CPCVError(
                f"n_test_groups={self.n_test_groups} must be < "
                f"n_groups={self.n_groups}; testing on every group leaves no "
                "training data and is not a cross-validation")
        for name in ("embargo_bars", "purge_bars", "min_train_events"):
            if getattr(self, name) < 0:
                raise CPCVError(f"{name} must be non-negative")

    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """`C(M-1, k-1)` — equivalently `C(M,k) * k / M`.

        Each group is tested in exactly this many splits, so the splits
        reassemble into this many complete traversals of the history.
        """
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def content_sha256(self) -> str:
        payload = json.dumps({"version": CPCV_VERSION, **asdict(self)},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class CPCVSplit:
    """One `C(M, k)` combination, already purged.

    `test_idx_by_group` keeps the test groups separate rather than concatenated:
    a path stitches together one group at a time from different splits, so
    collapsing them here would make path assembly impossible.
    """
    split_index: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_idx_by_group: dict[int, np.ndarray]
    purged_idx: np.ndarray
    group_windows: dict[int, tuple[int, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_index": self.split_index,
            "test_groups": list(self.test_groups),
            "n_train": int(self.train_idx.size),
            "n_test": int(self.test_idx.size),
            "n_purged": int(self.purged_idx.size),
            "test_idx_by_group": {str(g): [int(i) for i in idx]
                                  for g, idx in sorted(self.test_idx_by_group.items())},
            "train_idx": [int(i) for i in self.train_idx],
            "test_idx": [int(i) for i in self.test_idx],
            "purged_idx": [int(i) for i in self.purged_idx],
        }


@dataclass(frozen=True)
class CPCVGeometry:
    """The splits plus everything needed to audit them."""
    version: str
    config: CPCVConfig
    n_events: int
    n_events_admitted: int
    n_events_held_out: int
    group_bounds: tuple[tuple[int, int], ...]
    group_windows: dict[int, tuple[int, int]]
    splits: tuple[CPCVSplit, ...]
    admitted_rows: np.ndarray
    paths_share_training_data: bool = True

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    @property
    def n_paths(self) -> int:
        return self.config.n_paths


def _validate_spans(spans: np.ndarray) -> np.ndarray:
    s = np.asarray(spans, dtype=np.int64)
    if s.ndim != 2 or s.shape[1] != 2:
        raise CPCVError(
            f"spans must be an (n, 2) array of [start, end], got shape {s.shape}")
    if s.shape[0] == 0:
        raise CPCVError("no events to split")
    if np.any(s[:, 1] <= s[:, 0]):
        raise CPCVError(
            "every span needs end > start; a zero-length span cannot be purged "
            "against because it has no interval to intersect")
    return s


def combinatorial_splits(spans: np.ndarray, config: CPCVConfig) -> CPCVGeometry:
    """Build every purged `C(M, k)` split, or refuse.

    `spans` are the M02 label spans as closed `[start_idx, end_idx]` bar
    intervals, one row per event, in the caller's own row order. Row indices in
    the returned splits refer to that order, so weights from M03 line up without
    a reindexing step.
    """
    s = _validate_spans(spans)
    n_events = int(s.shape[0])

    if config.holdout_idx_lo is None:
        admitted = np.arange(n_events, dtype=np.int64)
    else:
        admitted = np.flatnonzero(s[:, 1] < config.holdout_idx_lo).astype(np.int64)
    n_held = n_events - int(admitted.size)
    if admitted.size == 0:
        raise CPCVError(
            f"every one of {n_events} events reaches into the sealed holdout at "
            f"idx {config.holdout_idx_lo}; there is nothing to cross-validate")

    # Groups are contiguous in time, so a group's bar window is an interval.
    # Ordering by span start (end as tiebreak) is what makes that true; grouping
    # in arbitrary row order would give interleaved windows and purge would then
    # have to exclude nearly everything.
    order = admitted[np.lexsort((s[admitted, 1], s[admitted, 0]))]
    M, k = config.n_groups, config.n_test_groups
    if order.size < M:
        raise CPCVError(
            f"{order.size} admitted events cannot be partitioned into {M} groups; "
            "reduce n_groups or supply more events")

    bounds = np.linspace(0, order.size, M + 1).astype(int)
    groups: list[np.ndarray] = []
    for g in range(M):
        rows = order[bounds[g]:bounds[g + 1]]
        if rows.size == 0:
            raise CPCVError(
                f"group {g} is empty for M={M} over {order.size} events")
        groups.append(rows)

    group_windows = {g: (int(s[rows, 0].min()), int(s[rows, 1].max()))
                     for g, rows in enumerate(groups)}
    group_bounds = tuple((int(bounds[g]), int(bounds[g + 1])) for g in range(M))

    splits: list[CPCVSplit] = []
    for i, test_groups in enumerate(combinations(range(M), k)):
        test_rows = np.concatenate([groups[g] for g in test_groups])
        test_set = set(int(r) for r in test_rows)
        candidate = np.array([int(r) for r in order if int(r) not in test_set],
                            dtype=np.int64)

        # Purge: drop any candidate whose span touches a forbidden window.
        # Forbidden = each test group's bar window, widened by purge_bars on both
        # sides and by embargo_bars on the right only.
        keep_mask = np.ones(candidate.size, dtype=bool)
        for g in test_groups:
            g_lo, g_hi = group_windows[g]
            lo = g_lo - config.purge_bars
            hi = g_hi + config.purge_bars + config.embargo_bars
            starts, ends = s[candidate, 0], s[candidate, 1]
            # Closed-interval intersection: [a, b] meets [lo, hi].
            keep_mask &= ~((ends >= lo) & (starts <= hi))

        train_rows = candidate[keep_mask]
        purged_rows = candidate[~keep_mask]

        if train_rows.size < max(1, config.min_train_events):
            raise CPCVError(
                f"split {i} (test groups {test_groups}) has {train_rows.size} "
                f"training events after purge, below the required "
                f"{max(1, config.min_train_events)}. Purge={config.purge_bars}, "
                f"embargo={config.embargo_bars}, M={M}, k={k} cannot be "
                "satisfied by this event geometry. Refusing rather than "
                "falling back to an unpurged split.")

        splits.append(CPCVSplit(
            split_index=i, test_groups=tuple(test_groups),
            train_idx=np.sort(train_rows), test_idx=np.sort(test_rows),
            test_idx_by_group={g: np.sort(groups[g]) for g in test_groups},
            purged_idx=np.sort(purged_rows),
            group_windows={g: group_windows[g] for g in test_groups}))

    if len(splits) != config.n_splits:  # pragma: no cover - arithmetic guard
        raise CPCVError(
            f"built {len(splits)} splits but C({M},{k}) = {config.n_splits}")

    return CPCVGeometry(
        version=CPCV_VERSION, config=config, n_events=n_events,
        n_events_admitted=int(admitted.size), n_events_held_out=n_held,
        group_bounds=group_bounds, group_windows=group_windows,
        splits=tuple(splits), admitted_rows=np.sort(admitted))


def leakage_pairs(geometry: CPCVGeometry, spans: np.ndarray) -> int:
    """Count (train, test) pairs whose label spans intersect. Must be 0.

    An exact count, not a sample and not a statistic. G1_SPEC.md T6 treats any
    non-zero result as an immediate failure, so this function's job is to be
    brutally literal about the definition of leakage rather than clever about
    performance.
    """
    s = _validate_spans(spans)
    total = 0
    for split in geometry.splits:
        tr, te = split.train_idx, split.test_idx
        if tr.size == 0 or te.size == 0:
            continue
        a_lo = s[tr, 0][:, None]
        a_hi = s[tr, 1][:, None]
        b_lo = s[te, 0][None, :]
        b_hi = s[te, 1][None, :]
        total += int(np.count_nonzero((a_lo <= b_hi) & (b_lo <= a_hi)))
    return total


def embargo_violations(geometry: CPCVGeometry, spans: np.ndarray) -> int:
    """Training events starting inside a test group's right-edge embargo.

    Separate from `leakage_pairs` because the embargo is an allowance for
    correlation that outlives the span, so a violation here is not a span
    intersection and the span check cannot see it.
    """
    s = _validate_spans(spans)
    cfg = geometry.config
    if cfg.embargo_bars == 0:
        return 0
    total = 0
    for split in geometry.splits:
        for g in split.test_groups:
            _, g_hi = geometry.group_windows[g]
            starts = s[split.train_idx, 0]
            total += int(np.count_nonzero(
                (starts > g_hi) & (starts <= g_hi + cfg.embargo_bars)))
    return total


def split_manifest(geometry: CPCVGeometry) -> dict[str, Any]:
    """A deterministic, hashable record of exactly which rows went where.

    Two runs of the same configuration on the same spans produce byte-identical
    manifests, including the hash. That is the only way a claim about a past
    CPCV result can be checked later.
    """
    body = {
        "version": geometry.version,
        "config": asdict(geometry.config),
        "config_sha256": geometry.config.content_sha256(),
        "n_events": geometry.n_events,
        "n_events_admitted": geometry.n_events_admitted,
        "n_events_held_out": geometry.n_events_held_out,
        "n_splits": geometry.n_splits,
        "n_paths": geometry.n_paths,
        "paths_share_training_data": geometry.paths_share_training_data,
        "group_bounds": [list(b) for b in geometry.group_bounds],
        "group_windows": {str(g): list(w)
                          for g, w in sorted(geometry.group_windows.items())},
        "splits": [sp.as_dict() for sp in geometry.splits],
    }
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return body


def assemble_paths(geometry: CPCVGeometry) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group the splits into complete traversals of the history.

    Returns `n_paths` tuples, each holding one `(group, split_index)` pair per
    group — so every path covers every group exactly once, using a different
    split each time. The `j`-th path takes, for each group, the `j`-th split in
    which that group was tested; splits are visited in index order, which makes
    the assembly deterministic.
    """
    occurrences: dict[int, list[int]] = {g: [] for g in range(geometry.config.n_groups)}
    for split in geometry.splits:
        for g in split.test_groups:
            occurrences[g].append(split.split_index)

    expected = geometry.n_paths
    for g, splits in occurrences.items():
        if len(splits) != expected:
            raise CPCVError(  # pragma: no cover - arithmetic guard
                f"group {g} is tested in {len(splits)} splits but each group must "
                f"appear in exactly C(M-1,k-1) = {expected}")
    return tuple(
        tuple((g, occurrences[g][j]) for g in range(geometry.config.n_groups))
        for j in range(expected))


@dataclass(frozen=True)
class PathResults:
    """Per-group scores, the paths built from them, and the caveat."""
    version: str
    n_splits: int
    n_paths: int
    group_scores: dict[tuple[int, int], float]
    path_scores: np.ndarray
    paths_share_training_data: bool = True

    def distribution(self) -> dict[str, Any]:
        p = np.asarray(self.path_scores, dtype=float)
        finite = p[np.isfinite(p)]
        if finite.size == 0:
            return {"n_paths": 0, "note": "no finite path scores"}
        return {
            "n_paths": int(finite.size),
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "q05": float(np.quantile(finite, 0.05)),
            "q95": float(np.quantile(finite, 0.95)),
            "share_positive": float(np.mean(finite > 0.0)),
            "paths_share_training_data": self.paths_share_training_data,
            "variance_caveat": ("paths share training data, so this spread "
                                "understates the true out-of-sample variance"),
        }


def backtest_paths(geometry: CPCVGeometry,
                   score_fn: Callable[[np.ndarray, np.ndarray], float],
                   ) -> PathResults:
    """Score every (split, test group) and stitch the results into paths.

    `score_fn(train_idx, test_idx)` is called once per test group — not once per
    split — because a path takes one group at a time. It receives row indices
    into the caller's original span order and returns one float.

    Deterministic: groups and splits are visited in index order.
    """
    group_scores: dict[tuple[int, int], float] = {}
    for split in geometry.splits:
        for g in sorted(split.test_idx_by_group):
            group_scores[(g, split.split_index)] = float(
                score_fn(split.train_idx, split.test_idx_by_group[g]))

    paths = assemble_paths(geometry)
    path_scores = np.array(
        [np.mean([group_scores[(g, si)] for g, si in path]) for path in paths],
        dtype=float)
    return PathResults(version=CPCV_VERSION, n_splits=geometry.n_splits,
                       n_paths=len(paths), group_scores=group_scores,
                       path_scores=path_scores)


def path_distribution(results: PathResults) -> dict[str, Any]:
    """The induced distribution, with the shared-training caveat attached."""
    return results.distribution()
