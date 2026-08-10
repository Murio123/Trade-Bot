"""C4.4: percentile and category against a FROZEN reference distribution.

The reference is captured once from the training region and stored on disk.
It is never recomputed from live data, because a category whose meaning
drifts month to month is worse than no category: "HIGH" has to mean the same
thing in January and in June for a ledger of past forecasts to be readable
at all.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

DISTRIBUTION_VERSION = "c44_ref_v1"

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
                   "scores": ref.scores.tolist()}, fh, indent=2, default=str)


def load_reference(path: str) -> ReferenceDistribution:
    with open(path) as fh:
        payload = json.load(fh)
    if payload.get("version") != DISTRIBUTION_VERSION:
        raise ValueError(
            f"reference distribution version mismatch: file has "
            f"{payload.get('version')!r}, code expects "
            f"{DISTRIBUTION_VERSION!r} — a silently re-fitted reference would "
            f"change what every past category meant")
    return build_reference(payload["scores"], payload.get("source", {}))
