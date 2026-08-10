"""C4.4: percentile and category against a FROZEN reference distribution.

The reference is captured once from the training region and stored on disk.
It is never recomputed from live data, because a category whose meaning
drifts month to month is worse than no category: "HIGH" has to mean the same
thing in January and in June for a ledger of past forecasts to be readable
at all.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

DISTRIBUTION_VERSION = "c44_ref_v1"


def content_sha256(scores) -> str:
    """Fingerprint of the reference itself.

    The version string is a code constant, so it cannot tell a re-fitted
    reference from the original — both come out as c44_ref_v1. Only the
    content can. Callers that must not silently absorb a re-fit pin this.
    """
    arr = np.sort(np.asarray(scores, dtype=float))
    return hashlib.sha256(arr.tobytes()).hexdigest()

# Frozen cut points. Deliberately asymmetric: the interesting statements are
# "unusually quiet" and "unusually busy", so the middle is wide.
CATEGORY_EDGES = ((25.0, "LOW"), (60.0, "NORMAL"), (85.0, "ELEVATED"),
                  (100.1, "HIGH"))
CATEGORIES = tuple(name for _, name in CATEGORY_EDGES)


@dataclass(frozen=True)
class ReferenceDistribution:
    """Sorted reference scores plus the provenance needed to trust them."""
    version: str
    scores: np.ndarray
    source: dict[str, Any]

    def percentile_of(self, score: float) -> float:
        """Share of the reference at or below `score`, in percent.

        Uses the midpoint convention for ties so a score sitting exactly on a
        cluster does not get the whole cluster's weight pushed to one side.
        """
        if not np.isfinite(score):
            raise ValueError(f"score must be finite, got {score!r}")
        below = float(np.searchsorted(self.scores, score, side="left"))
        equal = float(np.searchsorted(self.scores, score, side="right")) - below
        return float((below + 0.5 * equal) / len(self.scores) * 100.0)

    def category_of(self, score: float) -> tuple[str, float]:
        pct = self.percentile_of(score)
        return categorize(pct), pct


def categorize(percentile: float) -> str:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile out of range: {percentile!r}")
    for edge, name in CATEGORY_EDGES:
        if percentile < edge:
            return name
    return CATEGORY_EDGES[-1][1]


def build_reference(scores, source: dict[str, Any]) -> ReferenceDistribution:
    arr = np.sort(np.asarray(scores, dtype=float))
    if arr.size == 0:
        raise ValueError("reference distribution cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("reference distribution contains non-finite values")
    return ReferenceDistribution(DISTRIBUTION_VERSION, arr, dict(source))


def save_reference(ref: ReferenceDistribution, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"version": ref.version, "source": ref.source,
                   "content_sha256": content_sha256(ref.scores),
                   "scores": ref.scores.tolist()}, fh, indent=2, default=str)


def load_reference(path: str, expected_sha256: str | None = None
                   ) -> ReferenceDistribution:
    """Load a frozen reference.

    Three separate checks, because they catch different failures:
      - version: the file was written by incompatible code;
      - stored hash vs recomputed: the file was edited or truncated;
      - `expected_sha256`: the file is a DIFFERENT reference than the caller
        was built against. Only this one catches a re-fit, since a re-fit
        keeps the same version string and its own hash is self-consistent.
    """
    with open(path) as fh:
        payload = json.load(fh)
    if payload.get("version") != DISTRIBUTION_VERSION:
        raise ValueError(
            f"reference distribution version mismatch: file has "
            f"{payload.get('version')!r}, code expects "
            f"{DISTRIBUTION_VERSION!r}")

    scores = payload["scores"]
    actual = content_sha256(scores)
    stored = payload.get("content_sha256")
    if stored is not None and stored != actual:
        raise ValueError(
            f"reference distribution is corrupt: stored hash {stored} does "
            f"not match its own contents ({actual})")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"reference distribution was re-fitted: expected {expected_sha256}, "
            f"file contains {actual} — loading it would silently change what "
            f"every category recorded in the ledger meant")
    return build_reference(scores, payload.get("source", {}))
