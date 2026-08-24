"""S3: cross-sectional evaluation of one ranking rule at a time.

Read-only, offline. Consumes S2's frozen labels and S2's write-once universe
snapshots; changes neither.

Three properties of the procedure are the whole reason it is written this way,
and each answers a specific way this measurement could lie:

  * **The unit of observation is a date, not a row.** Metrics are computed
    inside each monthly cross-section and then averaged across dates with equal
    weight. Pooling 7,324 rows would weight a 238-name date twenty times more
    than a 25-name date, and would treat events with 180-day overlapping
    windows as independent observations. The pooled number is still reported,
    labelled as secondary.
  * **Uncertainty is estimated over blocks, not over events.** Consecutive runs
    of six monthly dates span one horizon, so a block is the smallest unit that
    does not overlap the next one. There are about eleven of them, and that —
    not 7,324 — is what any confidence statement rests on.
  * **The random baseline is matched and seeded.** It draws the same K names
    from the same eligible universe on the same dates, so it answers "is this
    ranking better than no ranking" rather than "is this better than nothing".

There is deliberately no function here that evaluates two features together.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

BLOCK_DATES = 6          # six monthly dates = one 180-day horizon
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260824
CI_LOW, CI_HIGH = 5.0, 95.0    # a 90% percentile interval
RANDOM_DRAWS = 1000
MIN_UNIVERSE = 25
TOP_FRACTION = 0.20
SECONDARY_TOP_N = 10


def _median(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else None


def _mean(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation, computed without scipy (this repo has none).

    Average ranks for ties, then Pearson on the ranks — the textbook
    definition, and the tie handling matters because features like F1 take few
    distinct values.
    """
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(np.asarray(x, dtype=float)), _ranks(np.asarray(y, float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


@dataclass(frozen=True)
class DateResult:
    """One monthly cross-section, for one ranking rule."""
    asof: str
    regime: str
    n_eligible: int
    n_resolved: int
    k: int
    selection_rate: float | None
    universe_rate: float | None
    selection_median_return: float | None
    universe_median_return: float | None
    selection_median_excess: float | None
    universe_median_excess: float | None
    selection_median_mae: float | None
    universe_median_mae: float | None
    spearman: float | None
    random_mean: float | None
    random_p95: float | None
    secondary_top_n_rate: float | None
    coverage: float

    @property
    def lift(self) -> float | None:
        if not self.selection_rate or not self.universe_rate:
            return None
        return self.selection_rate / self.universe_rate


def evaluate_date(*, asof: str, regime: str, values: dict[str, float],
                  labels: dict[str, dict[str, Any]], descending: bool,
                  n_eligible: int, rng: np.random.Generator) -> DateResult:
    """Rank one cross-section and measure it. Resolved labels only.

    `values` holds the feature for every eligible symbol that had enough
    history; `labels` holds S2's outcome for every eligible symbol that
    resolved. A symbol missing from either is excluded here and counted in
    `coverage`, never silently treated as a zero.
    """
    from spot.features import rank_symbols, top_quantile_k

    ranked = [s for s in rank_symbols(values, descending) if s in labels]
    resolved = [s for s in labels if s in values]
    coverage = len(values) / n_eligible if n_eligible else 0.0

    def rate(symbols: Iterable[str]) -> float | None:
        outcomes = [labels[s]["clean_2x"] for s in symbols
                    if labels[s].get("clean_2x") is not None]
        return float(np.mean(outcomes)) if outcomes else None

    def med(symbols: Iterable[str], field: str) -> float | None:
        return _median([labels[s].get(field) for s in symbols])

    k = top_quantile_k(len(ranked), TOP_FRACTION) if ranked else 0
    top = ranked[:k]

    random_rates: list[float] = []
    if ranked and k:
        pool = np.array(ranked)
        for _ in range(RANDOM_DRAWS):
            draw = rng.choice(pool, size=k, replace=False)
            r = rate(draw.tolist())
            if r is not None:
                random_rates.append(r)

    xs = [values[s] for s in resolved]
    ys = [labels[s]["terminal_return"] for s in resolved]
    pairs = [(a, b) for a, b in zip(xs, ys) if b is not None]

    return DateResult(
        asof=asof, regime=regime, n_eligible=n_eligible,
        n_resolved=len(ranked), k=k,
        selection_rate=rate(top), universe_rate=rate(ranked),
        selection_median_return=med(top, "terminal_return"),
        universe_median_return=med(ranked, "terminal_return"),
        selection_median_excess=med(top, "excess_vs_btc"),
        universe_median_excess=med(ranked, "excess_vs_btc"),
        selection_median_mae=med(top, "mae"),
        universe_median_mae=med(ranked, "mae"),
        spearman=spearman([p[0] for p in pairs], [p[1] for p in pairs]),
        random_mean=_mean(random_rates),
        random_p95=(float(np.percentile(random_rates, 95))
                    if random_rates else None),
        secondary_top_n_rate=rate(ranked[:SECONDARY_TOP_N]),
        coverage=coverage,
    )


def blocks(dates: Sequence[Any], size: int = BLOCK_DATES) -> list[list[int]]:
    """Consecutive non-overlapping runs of `size` dates, by position.

    A trailing partial run is kept: dropping it would silently discard the most
    recent months, which are the ones a reader most wants included.
    """
    return [list(range(i, min(i + size, len(dates))))
            for i in range(0, len(dates), size)]


def block_bootstrap(per_date: Sequence[float | None], *,
                    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
                    size: int = BLOCK_DATES) -> dict[str, Any]:
    """Percentile interval for the mean, resampling whole blocks.

    Resampling dates individually would assume months are exchangeable, which
    they are not: two dates a month apart share five-sixths of their outcome
    window.
    """
    idx_blocks = blocks(list(range(len(per_date))), size)
    usable = [[i for i in b if per_date[i] is not None] for b in idx_blocks]
    usable = [b for b in usable if b]
    if len(usable) < 2:
        return {"mean": _mean(per_date), "low": None, "high": None,
                "blocks": len(usable), "draws": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        chosen = rng.integers(0, len(usable), size=len(usable))
        vals = [per_date[i] for c in chosen for i in usable[c]]
        means.append(float(np.mean(vals)))
    return {
        "mean": _mean(per_date),
        "low": float(np.percentile(means, CI_LOW)),
        "high": float(np.percentile(means, CI_HIGH)),
        "blocks": len(usable),
        "draws": draws,
    }


def aggregate(results: Sequence[DateResult]) -> dict[str, Any]:
    """Date-weighted aggregation, with the block-bootstrap interval."""
    if not results:
        return {"dates": 0}
    diff = [(r.selection_rate - r.universe_rate)
            if (r.selection_rate is not None and r.universe_rate is not None)
            else None for r in results]
    lifts = [r.lift for r in results]
    beat_random = [1.0 if (r.selection_rate is not None
                           and r.random_p95 is not None
                           and r.selection_rate > r.random_p95) else 0.0
                   for r in results]
    sel = _mean([r.selection_rate for r in results])
    uni = _mean([r.universe_rate for r in results])
    return {
        "dates": len(results),
        "events": int(sum(r.n_resolved for r in results)),
        "blocks": len(blocks(results)),
        "selection_rate": sel,
        "universe_rate": uni,
        # S3_SPEC §6 M-B: the selection rate divided by the universe rate,
        # both aggregated across dates first. Dividing per date and averaging
        # the ratios instead is not the same number and is badly behaved: a
        # date where the universe rate is 2% and the selection rate is 10%
        # contributes a ratio of 5, so a handful of quiet months can push a
        # mean-of-ratios above 1 while the selection rate sits *below* the
        # universe rate all along. The first implementation did exactly that
        # and reported lift > 1 for all six rules whose rates were lower than
        # the universe's. Both are reported; only this one is binding.
        "lift": (sel / uni if (sel is not None and uni) else None),
        "mean_of_per_date_lift_ratios": _mean(lifts),
        "rate_difference": block_bootstrap(diff),
        "selection_median_return": _mean(
            [r.selection_median_return for r in results]),
        "universe_median_return": _mean(
            [r.universe_median_return for r in results]),
        "selection_median_excess": _mean(
            [r.selection_median_excess for r in results]),
        "universe_median_excess": _mean(
            [r.universe_median_excess for r in results]),
        "selection_median_mae": _mean(
            [r.selection_median_mae for r in results]),
        "universe_median_mae": _mean([r.universe_median_mae for r in results]),
        "spearman": block_bootstrap([r.spearman for r in results]),
        "random_mean": _mean([r.random_mean for r in results]),
        "dates_beating_random_p95": _mean(beat_random),
        "secondary_top_n_rate": _mean(
            [r.secondary_top_n_rate for r in results]),
        "coverage": _mean([r.coverage for r in results]),
    }


def block_stability(results: Sequence[DateResult]) -> dict[str, Any]:
    """The share of blocks whose mean lift exceeds 1 (§10 condition 4)."""
    idx_blocks = blocks(results)
    per_block = []
    for b in idx_blocks:
        lifts = [results[i].lift for i in b if results[i].lift is not None]
        per_block.append(float(np.mean(lifts)) if lifts else None)
        continue
    scored = [v for v in per_block if v is not None]
    return {
        "blocks": len(scored),
        "blocks_with_lift_above_1": sum(1 for v in scored if v > 1.0),
        "share": (sum(1 for v in scored if v > 1.0) / len(scored)
                  if scored else None),
        "per_block": [None if v is None else round(v, 4) for v in per_block],
    }


def classify(agg: dict[str, Any], stability: dict[str, Any],
             by_regime: dict[str, dict[str, Any]]) -> tuple[str, dict[str, bool]]:
    """The frozen go/no-go rule of S3_SPEC.md §10. No judgement, no exceptions."""
    # Condition 1 of §10 is worded as "the top-quintile CLEAN_2X rate exceeds
    # the eligible universe rate", so it is a comparison of rates, not of an
    # average of ratios.
    sel, uni = agg.get("selection_rate"), agg.get("universe_rate")
    lift = agg.get("lift")
    diff = agg.get("rate_difference", {})
    sel_ex = agg.get("selection_median_excess")
    uni_ex = agg.get("universe_median_excess")
    regime_lifts = [v.get("lift") for v in by_regime.values()
                    if v.get("lift") is not None]

    checks = {
        "lift_above_1": bool(sel is not None and uni is not None and sel > uni),
        "beats_random_p95": bool(
            (agg.get("dates_beating_random_p95") or 0) > 0.5),
        "better_excess_vs_btc": bool(sel_ex is not None and uni_ex is not None
                                     and sel_ex > uni_ex),
        "stable_in_60pct_blocks": bool((stability.get("share") or 0.0) >= 0.60),
        "positive_in_2_of_3_regimes": sum(1 for v in regime_lifts
                                          if v > 1.0) >= 2,
        "bootstrap90_excludes_zero": bool(
            diff.get("low") is not None and diff.get("high") is not None
            and (diff["low"] > 0 or diff["high"] < 0)),
    }
    if all(checks.values()):
        return "PRELIMINARY_KEEP", checks
    negative = (lift is not None and lift < 1.0
                and diff.get("high") is not None and diff["high"] < 0)
    if negative:
        return "PRELIMINARY_REMOVE", checks
    return "PRELIMINARY_UNKNOWN", checks
