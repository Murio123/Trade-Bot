"""M03 — how many independent observations there actually are.

The project already knows the answer is not "one per row". C4.3e discovered the
~12x overlap on the *evaluation* side and works around it with a block
bootstrap. Training still treats every row as independent, so a model fitted on
4062 overlapping labels believes it has twelve times more information than it
does, and every standard error it produces is too small by roughly `sqrt(12)`.

The correction, from Lopez de Prado's *Advances in Financial Machine Learning*
ch. 4: a label spanning bars `[a, b]` shares each of those bars with every other
label that also spans it. Weight each bar of a label by the reciprocal of how
many labels are concurrent there, average over the label's own bars, and the
result — average uniqueness — is the fraction of that observation that is its
own. Summed over observations it is the effective sample size.

Two numbers are reported together, everywhere, without exception: the raw count
and the effective count. ARCHITECTURE.md M03 states the rule as a specification
violation rather than a style preference — "a metric quoted against 4062 rows
when the effective count is 338" — and the only structural defence is that this
module never returns one without the other.

Time decay is deliberately absent. It is a modelling choice (recent data matters
more) dressed as a correction, and it must be argued for separately rather than
arriving switched on inside a bug-fix module.

`tools/ridge_ranking_significance.effective_sample_size` computes a cruder
proxy — `n / horizon` plus a lag-1 autocorrelation — for the fixed-horizon C4.3e
labels. It is not deleted here: it is part of a frozen C4.3e result path, and
changing what that tool prints would change a published number. This module
supersedes it for all new work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SAMPLE_WEIGHTS_VERSION = "m03_sample_weights_v1"


class SampleWeightError(Exception):
    """A span set that cannot be weighted — a programming or data defect."""


def _validate(spans: np.ndarray, n_bars: int | None) -> tuple[np.ndarray, int]:
    s = np.asarray(spans, dtype=np.int64)
    if s.size == 0:
        raise SampleWeightError(
            "no spans to weight; an empty span set has no effective sample size "
            "and returning 0 would be indistinguishable from a real zero")
    if s.ndim != 2 or s.shape[1] != 2:
        raise SampleWeightError(
            f"spans must be an (n, 2) array of [start, end], got shape {s.shape}")
    starts, ends = s[:, 0], s[:, 1]
    if np.any(starts < 0):
        raise SampleWeightError("span starts must be non-negative")
    bad = np.flatnonzero(ends <= starts)
    if bad.size:
        raise SampleWeightError(
            f"{bad.size} span(s) have end <= start (first at row {int(bad[0])}: "
            f"[{int(starts[bad[0]])}, {int(ends[bad[0]])}]); a zero-length or "
            "reversed span is a defect in the labeling, not an observation with "
            "no width")
    required = int(ends.max()) + 1
    if n_bars is None:
        n_bars = required
    elif n_bars < required:
        raise SampleWeightError(
            f"n_bars={n_bars} but a span reaches bar {required - 1}; the "
            "concurrency axis must cover every span or the labels past its end "
            "would be silently unweighted")
    return s, int(n_bars)


def concurrency(spans: np.ndarray, n_bars: int | None = None) -> np.ndarray:
    """How many labels are live at each bar.

    Spans are closed, `[start, end]`, so both endpoints count: a label is live
    on the bar it starts and on the bar it resolves. The half-open reading would
    make two labels that meet end-to-start look non-overlapping when the second
    one's first bar is the first one's last.
    """
    s, n = _validate(spans, n_bars)
    counts = np.zeros(n + 1, dtype=np.int64)
    np.add.at(counts, s[:, 0], 1)
    np.add.at(counts, s[:, 1] + 1, -1)
    return np.cumsum(counts)[:n]


def average_uniqueness(spans: np.ndarray, n_bars: int | None = None) -> np.ndarray:
    """Per-label mean of `1 / concurrency` over the label's own bars.

    In `(0, 1]`: exactly 1.0 when a label overlaps nothing, smaller the more it
    shares. Never zero — a label is always concurrent with itself, so the
    reciprocal is at most 1 and at least `1/n_labels`.
    """
    s, n = _validate(spans, n_bars)
    conc = concurrency(s, n).astype(float)
    # Every bar covered by a span has concurrency >= 1 by construction; the
    # clip guards only the uncovered bars, which no span averages over.
    inv = 1.0 / np.clip(conc, 1.0, None)
    cumulative = np.concatenate(([0.0], np.cumsum(inv)))
    starts, ends = s[:, 0], s[:, 1]
    totals = cumulative[ends + 1] - cumulative[starts]
    widths = (ends - starts + 1).astype(float)
    return totals / widths


@dataclass(frozen=True)
class SampleWeights:
    """Weights and both sample sizes, inseparably.

    `weights` are the average uniquenesses themselves, so they are strictly
    positive and sum to `effective_count` by construction rather than by a
    normalisation step that could be skipped.
    """
    version: str
    raw_count: int
    effective_count: float
    mean_uniqueness: float
    max_concurrency: int
    weights: np.ndarray

    @property
    def overlap_factor(self) -> float:
        """Raw / effective — the factor by which naive standard errors are too
        small. 1.0 means no overlap; the project's fixed-horizon labels sit near
        12."""
        return self.raw_count / self.effective_count

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weights"] = [float(w) for w in self.weights]
        d["overlap_factor"] = self.overlap_factor
        return d

    def summary(self) -> dict[str, Any]:
        """The report-facing form: both counts, no per-row payload."""
        return {"version": self.version, "raw_count": self.raw_count,
                "effective_count": round(self.effective_count, 4),
                "overlap_factor": round(self.overlap_factor, 4),
                "mean_uniqueness": round(self.mean_uniqueness, 6),
                "max_concurrency": self.max_concurrency}


def uniqueness_weights(spans: np.ndarray,
                       n_bars: int | None = None) -> SampleWeights:
    """The weights, the raw count and the effective count, in one object.

    Returning a bare array here was the obvious design and is the wrong one:
    the caller would then be free to report the row count next to the weighted
    metric, which is the specification violation this module exists to prevent.
    """
    s, n = _validate(spans, n_bars)
    u = average_uniqueness(s, n)
    if np.any(u <= 0.0):
        raise SampleWeightError(
            "computed a non-positive uniqueness weight; this is arithmetically "
            "impossible for valid spans and indicates a defect in concurrency")
    return SampleWeights(
        version=SAMPLE_WEIGHTS_VERSION, raw_count=int(s.shape[0]),
        effective_count=float(u.sum()), mean_uniqueness=float(u.mean()),
        max_concurrency=int(concurrency(s, n).max()), weights=u)


def sequential_bootstrap(spans: np.ndarray, size: int,
                         rng: np.random.Generator,
                         n_bars: int | None = None) -> np.ndarray:
    """Draw `size` labels, preferring ones that overlap what is already drawn
    the least.

    Ordinary bootstrap on overlapping labels resamples the same information
    repeatedly and produces confidence intervals that are too narrow — the exact
    failure this module addresses on the fitting side. Here each draw's
    probability is proportional to its average uniqueness *given the sample so
    far*, so a label already represented becomes progressively less likely.

    Deterministic given `rng`. No G1 threshold depends on it; it is provided
    because M03's specification lists it and because the alternative is that a
    future caller writes a worse version inline.
    """
    s, n = _validate(spans, n_bars)
    if size < 1:
        raise SampleWeightError(f"bootstrap size must be >= 1, got {size}")
    n_labels = int(s.shape[0])
    starts, ends = s[:, 0], s[:, 1]

    drawn: list[int] = []
    # Concurrency contributed by the labels drawn so far. A candidate's
    # uniqueness is evaluated against this plus itself.
    live = np.zeros(n, dtype=np.int64)
    for _ in range(size):
        avg_u = np.empty(n_labels, dtype=float)
        for i in range(n_labels):
            a, b = int(starts[i]), int(ends[i])
            window = live[a:b + 1] + 1
            avg_u[i] = float(np.mean(1.0 / window))
        total = avg_u.sum()
        if total <= 0.0:  # pragma: no cover - impossible for valid spans
            raise SampleWeightError("all bootstrap probabilities collapsed to zero")
        pick = int(rng.choice(n_labels, p=avg_u / total))
        drawn.append(pick)
        live[int(starts[pick]):int(ends[pick]) + 1] += 1
    return np.array(drawn, dtype=np.int64)
