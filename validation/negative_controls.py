"""G1 — the suite that tries to make the apparatus lie.

Every module in C5.0 is built to detect edge. None of them can tell you whether
they detect edge *that is not there*, and that is the only question G1 asks. So
the apparatus is run against data constructed to contain nothing: zero-drift
prices, permuted signals, coin-flip directions, labels drawn from their own
marginal. If a significant result comes out of that, the apparatus is broken and
every verdict it has ever given on real features is void.

Thresholds live in `reports/c50/G1_SPEC.md` §4.1 and were committed at `197a765`,
before this file existed. That ordering is the entire methodological content of
this suite — a false-positive rate is only evidence if the bar was set first.

Two structural choices worth stating:

**Everything is synthetic.** The real-data matched-coverage and
matched-timestamp baselines already exist in
`tools/swing_hypothesis_walkforward.py`. Running them is Phase A′ evaluation,
which G1 forbids, and it would also answer a different question: whether *this
strategy* beats a random one, rather than whether *the apparatus* can tell. The
synthetic versions here test the apparatus.

**Costs are transaction-only, and that is not a shortcut.** Synthetic bars have
invented timestamps, so there is no real funding bill; `FUNDING_NOT_MODELLED`
is passed explicitly and propagated into every result, exactly as P1 requires.
It also means the null expectancy is expected to be slightly *negative* — a
random entry pays real costs — which G1_SPEC.md T2 anticipates and permits,
while a null expectancy strictly *above* zero is an immediate failure.

Determinism: `rng_for(i)` seeds replication `i` from a fixed master seed, and no
control reads a global random state. Two runs produce identical numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from labeling.sample_weights import uniqueness_weights
from labeling.triple_barrier import (BarrierConfig, apply_barriers, label_spans,
                                     outcome_counts)
from validation.cpcv import (CPCVConfig, assemble_paths, backtest_paths,
                             combinatorial_splits, leakage_pairs)
from validation.deflated_sharpe import (InsufficientData, deflated_sharpe,
                                        sharpe_stats)
from validation.trade_costs import transaction_cost_r

NEGATIVE_CONTROLS_VERSION = "g1_negative_controls_v1"

# Frozen in G1_SPEC.md §4. Changing it invalidates every recorded control run,
# which is the point of writing it down rather than passing it in.
MASTER_SEED = 20260817

BAR_MS = 4 * 3_600_000          # the swing profile's 4h bar
SIGMA_PER_BAR = 0.012           # ~1.2% per 4h, the order of magnitude BTCUSDT runs
START_PRICE = 30_000.0


def rng_for(replication: int) -> np.random.Generator:
    """The generator for replication `i`. Never a global seed, never time-based."""
    return np.random.default_rng([MASTER_SEED, int(replication)])


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def synthetic_ohlcv(n_bars: int, rng: np.random.Generator, *,
                    mu_per_bar: float = 0.0,
                    sigma_per_bar: float = SIGMA_PER_BAR,
                    start_price: float = START_PRICE,
                    bar_ms: int = BAR_MS,
                    start_ms: int = 1_600_000_000_000) -> pd.DataFrame:
    """A zero-drift GBM with plausible intrabar ranges.

    `mu_per_bar = 0.0` is the whole point: any expectancy the apparatus reports
    on this frame is manufactured. The intrabar high and low are drawn as
    half-normal excursions beyond the bar's own open/close range, so barrier
    touches are possible rather than degenerate — a frame whose high always
    equals its close would make every M02 event resolve TIME and the control
    would prove nothing.
    """
    if n_bars < 2:
        raise ValueError("n_bars must be at least 2")
    log_ret = rng.normal(mu_per_bar, sigma_per_bar, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate(([start_price], close[:-1]))

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    # Excursions scale with the same sigma, so the range/body ratio is stable
    # across the path instead of drifting with price level.
    up = np.abs(rng.normal(0.0, sigma_per_bar * 0.6, size=n_bars))
    dn = np.abs(rng.normal(0.0, sigma_per_bar * 0.6, size=n_bars))
    high = body_hi * (1.0 + up)
    low = body_lo * (1.0 - dn)

    close_time = start_ms + np.arange(n_bars, dtype=np.int64) * bar_ms
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "close_time": close_time})


def realized_sigma(ohlcv: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Trailing volatility of log closes, as a fraction, strictly backward-looking.

    `shift(1)` is not decoration: without it bar `i`'s barriers would be scaled
    by a standard deviation that includes bar `i`'s own return, which is
    look-ahead — small, invisible in aggregate, and exactly the kind of leak this
    suite exists to catch. The head is backfilled with the first valid value
    rather than dropped, so row indices stay aligned with the frame.
    """
    logc = np.log(ohlcv["close"].to_numpy(dtype=float))
    ret = pd.Series(np.diff(logc, prepend=logc[0]))
    sig = ret.rolling(window).std(ddof=1).shift(1)
    sig = sig.bfill().fillna(SIGMA_PER_BAR)
    out = sig.to_numpy(dtype=float)
    # A flat opening window can still yield 0.0; barriers need a positive scale.
    return np.where(out > 0.0, out, SIGMA_PER_BAR)


def ar1_signal(n: int, rng: np.random.Generator, phi: float = 0.85) -> np.ndarray:
    """A signal with the autocorrelation of a real indicator and no information.

    §6.3 requires the controls to have "the autocorrelation structure of the real
    ones but no predictive relationship". White noise would be an easier test
    that the apparatus could pass while still failing on real features, because
    persistence is what lets a lucky run look like a trend.
    """
    x = np.empty(n, dtype=float)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, np.sqrt(1.0 - phi ** 2))
    return x


# ---------------------------------------------------------------------------
# Statistics used by the thresholds
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(x: Sequence[float] | np.ndarray, rng: np.random.Generator,
                      n_resamples: int = 2000, alpha: float = 0.05
                      ) -> dict[str, float]:
    """Percentile bootstrap CI of the mean. Used for G1_SPEC.md T2 and T4."""
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 2:
        raise ValueError("bootstrap_mean_ci needs at least 2 finite values")
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    means = a[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {"mean": float(a.mean()), "ci_lo": lo, "ci_hi": hi,
            "n": int(a.size), "n_resamples": int(n_resamples)}


def ci_verdict(ci: dict[str, float]) -> str:
    """`CONTAINS_ZERO` / `ABOVE_ZERO` / `BELOW_ZERO`.

    T2's operative sentences are the explicit ones: strictly above zero is an
    immediate failure, strictly below zero is reported and is not a failure
    (a random entry paying real costs should lose money).
    """
    if ci["ci_lo"] > 0.0:
        return "ABOVE_ZERO"
    if ci["ci_hi"] < 0.0:
        return "BELOW_ZERO"
    return "CONTAINS_ZERO"


def share_band(p: float, n: int, sigmas: float = 3.0) -> tuple[float, float]:
    """`p ± sigmas * sqrt(p(1-p)/n)` — the bands T3 and T5 are stated with."""
    se = np.sqrt(p * (1.0 - p) / n)
    return (p - sigmas * se, p + sigmas * se)


# ---------------------------------------------------------------------------
# Shared event construction
# ---------------------------------------------------------------------------

# Symmetric on purpose, and the reason is the whole validity of the suite. With
# `upper == lower` a zero-drift process has an analytically zero label
# expectancy, so any expectancy the apparatus reports is manufactured. With the
# project's real 2:1 target the null expectancy is not zero — it depends on M02's
# same-bar tie rule, which is a deliberate pessimistic bias — and a control built
# on it would confound "the apparatus invents edge" with "M02 is conservative by
# design". That bias is measured separately by `barrier_bias_diagnostic`, where
# it is a reported number rather than a contaminant.
DEFAULT_BARRIERS = BarrierConfig(upper_mult=1.0, lower_mult=1.0, vertical_bars=12)
PROJECT_BARRIERS = BarrierConfig(upper_mult=2.0, lower_mult=1.0, vertical_bars=12)
DEFAULT_STRIDE = 4


@dataclass(frozen=True)
class SyntheticRun:
    """One replication's data and labels, shared by several controls."""
    ohlcv: pd.DataFrame
    sigma: np.ndarray
    events: list[Any]
    spans: np.ndarray
    r_multiples: np.ndarray
    net_r: np.ndarray
    entry_idx: np.ndarray
    outcomes: dict[str, int]
    raw_count: int
    effective_count: float

    def weights_summary(self) -> dict[str, Any]:
        return uniqueness_weights(self.spans).summary()


def build_synthetic_run(rng: np.random.Generator, *, n_bars: int = 1200,
                        stride: int = DEFAULT_STRIDE,
                        config: BarrierConfig = DEFAULT_BARRIERS,
                        mu_per_bar: float = 0.0,
                        sides: np.ndarray | None = None,
                        entry_idx: np.ndarray | None = None) -> SyntheticRun:
    """Zero-drift bars, M02 labels on them, M03 accounting for the overlap.

    `time_col=None` is passed to M02 deliberately: these timestamps are
    synthetic, so a continuity check over them would be theatre. On real data it
    must never be skipped, which is why it is an explicit argument rather than a
    default.
    """
    ohlcv = synthetic_ohlcv(n_bars, rng, mu_per_bar=mu_per_bar)
    sigma = realized_sigma(ohlcv)

    last_start = n_bars - config.vertical_bars - 1
    if last_start <= 0:
        raise ValueError("frame is too short for the requested vertical barrier")
    if entry_idx is None:
        entry_idx = np.arange(1, last_start, stride, dtype=np.int64)
    else:
        entry_idx = np.asarray(entry_idx, dtype=np.int64)
    if sides is None:
        sides = np.zeros(entry_idx.size, dtype=np.int64)      # direction-neutral
    else:
        sides = np.asarray(sides, dtype=np.int64)
    if sides.size != entry_idx.size:
        raise ValueError("sides and entry_idx must be the same length")

    events = apply_barriers(ohlcv, list(zip(entry_idx.tolist(), sides.tolist())),
                            sigma, config, time_col=None)
    resolved = [e for e in events if e.resolved]
    spans = label_spans(events)
    if spans.shape[0] == 0:
        raise ValueError("no event resolved; the control would measure nothing")

    r = np.array([e.r_multiple for e in resolved], dtype=float)
    # Cost in R at each event's own stop distance, through the P1 layer. Not a
    # local formula: a test forbids one.
    cost = np.array([transaction_cost_r(e.entry_price,
                                        abs(e.entry_price - (e.lower if e.side in (0, 1)
                                                             else e.upper)))
                     for e in resolved], dtype=float)
    weights = uniqueness_weights(spans)
    return SyntheticRun(
        ohlcv=ohlcv, sigma=sigma, events=resolved, spans=spans, r_multiples=r,
        net_r=r - cost, entry_idx=np.array([e.start_idx for e in resolved],
                                          dtype=np.int64),
        outcomes=outcome_counts(events), raw_count=weights.raw_count,
        effective_count=weights.effective_count)


def barrier_bias_diagnostic(replication: int = 0, *, n_bars: int = 4000
                            ) -> dict[str, Any]:
    """How much the same-bar tie rule costs, in R, on data with no drift.

    Not a threshold and not a pass/fail: a measurement. M02's tie convention
    resolves an ambiguous bar as a stop, which is defensible and documented, but
    if it were expensive enough it would dominate every downstream expectancy —
    and nobody would know, because the convention is invisible in the aggregate
    outcome counts. This reports the tie rate and the null expectancy under both
    the symmetric configuration the controls use and the 2:1 configuration the
    project's swing profile actually targets.
    """
    out: dict[str, Any] = {}
    for name, cfg in (("symmetric", DEFAULT_BARRIERS),
                      ("project_2to1", PROJECT_BARRIERS)):
        run = build_synthetic_run(rng_for(replication), n_bars=n_bars, config=cfg)
        ties = int(run.outcomes.get("TIE", 0))
        resolved = sum(run.outcomes.get(k, 0) for k in ("TP", "SL", "TIME"))
        out[name] = {
            "upper_mult": cfg.upper_mult, "lower_mult": cfg.lower_mult,
            "vertical_bars": cfg.vertical_bars,
            "config_sha256": cfg.content_sha256(),
            "n_events": resolved, "ties": ties,
            "tie_rate": ties / resolved if resolved else float("nan"),
            "mean_gross_r": float(run.r_multiples.mean()),
            "mean_net_r": float(run.net_r.mean()),
            "outcomes": run.outcomes,
        }
    return out


def _dsr_or_insufficient(series: np.ndarray, n_trials: int) -> dict[str, Any]:
    """DSR, or an explicit insufficient-data record. Never a fabricated 0.5."""
    try:
        res = deflated_sharpe(series, n_trials=n_trials)
    except InsufficientData as exc:
        return {"status": "INSUFFICIENT_DATA", "reason": str(exc),
                "significant": False}
    d = res.as_dict()
    d["status"] = "OK"
    d["significant"] = res.significant
    return d


# ---------------------------------------------------------------------------
# NC1 — randomized labels
# ---------------------------------------------------------------------------

def nc1_randomized_labels(replication: int) -> dict[str, Any]:
    """Labels drawn IID from their own empirical marginal.

    The marginal — and therefore the mean, the skew and the fat tails — is
    preserved exactly; only the pairing between an event and its outcome is
    destroyed. Any significance left is significance the apparatus invented from
    distribution shape alone, which is precisely what M01's skew and kurtosis
    terms are supposed to absorb.

    `n_trials = 1` is the strictest available setting: it makes the benchmark
    zero, so the test is whether the base statistic is calibrated rather than
    whether the deflation is generous. A higher trial count only raises the bar.

    T1 is applied to the **gross** series. Net returns carry a real ~0.1 R cost
    per event, which pushes the null so far below zero that no upward
    one-sided test could ever fire — an FPR of zero that measures the cost
    model, not the statistic. The net DSR is reported too, but the gross one is
    the number the threshold binds to, and that is strictly harder than what
    G1_SPEC.md §4.1 demands.
    """
    rng = rng_for(replication)
    run = build_synthetic_run(rng)
    idx = rng.integers(0, run.r_multiples.size, size=run.r_multiples.size)
    gross, net = run.r_multiples[idx], run.net_r[idx]
    g = _dsr_or_insufficient(gross, n_trials=1)
    n = _dsr_or_insufficient(net, n_trials=1)
    return {"control": "NC1", "replication": replication,
            "raw_count": run.raw_count, "effective_count": run.effective_count,
            "mean_gross_r": float(gross.mean()), "mean_net_r": float(net.mean()),
            "dsr": g.get("dsr"), "dsr_net": n.get("dsr"),
            "status": g["status"], "significant": g["significant"],
            "significant_net": n["significant"],
            "tie_rate": run.outcomes.get("TIE", 0) / max(1, run.raw_count),
            "funding_modelled": False}


# ---------------------------------------------------------------------------
# NC2 — time-shuffled signal
# ---------------------------------------------------------------------------

def nc2_time_shuffled_signal(replication: int) -> dict[str, Any]:
    """A persistent signal, permuted in time against the returns it "predicts".

    A permutation preserves the signal's marginal distribution exactly and
    destroys only its alignment with the future. The position is `sign(signal)`
    and the payoff is the next bar's return, so an apparatus that reports this as
    significant is reporting a relationship that a permutation cannot leave
    behind.
    """
    rng = rng_for(replication)
    ohlcv = synthetic_ohlcv(1200, rng)
    close = ohlcv["close"].to_numpy(dtype=float)
    fwd = np.diff(close) / close[:-1]                     # bar i -> i+1

    signal = ar1_signal(fwd.size, rng)
    shuffled = rng.permutation(signal)
    pnl = np.sign(shuffled) * fwd
    unshuffled_pnl = np.sign(signal) * fwd

    out = _dsr_or_insufficient(pnl, n_trials=1)
    marginal_preserved = bool(np.allclose(np.sort(signal), np.sort(shuffled)))
    return {"control": "NC2", "replication": replication,
            "raw_count": int(pnl.size), "effective_count": float(pnl.size),
            "mean_r": float(pnl.mean()), "dsr": out.get("dsr"),
            "status": out["status"], "significant": out["significant"],
            "marginal_preserved": marginal_preserved,
            "unshuffled_sharpe": float(sharpe_stats(unshuffled_pnl).sharpe),
            "shuffled_sharpe": float(sharpe_stats(pnl).sharpe)}


# ---------------------------------------------------------------------------
# NC3 — random entry, matched coverage
# ---------------------------------------------------------------------------

def nc3_random_entry_matched_coverage(replication: int) -> dict[str, Any]:
    """Random entry timestamps, with the reference set's count and per-group
    coverage.

    Matching coverage matters: an unmatched random baseline can differ from the
    reference simply by holding positions at different times, and then the
    comparison measures exposure rather than skill. The entries are drawn
    uniformly inside each CPCV group's bar window, one for one against the
    reference.
    """
    rng = rng_for(replication)
    reference = build_synthetic_run(rng)
    n_bars = int(len(reference.ohlcv))
    last_start = n_bars - DEFAULT_BARRIERS.vertical_bars - 1

    # Matched coverage: same number of entries inside each decile of the walked
    # region, so the random baseline is exposed to the same market segments.
    edges = np.linspace(1, last_start, 11).astype(int)
    drawn: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        count = int(np.count_nonzero((reference.entry_idx >= lo)
                                     & (reference.entry_idx < hi)))
        if count:
            drawn.extend(rng.integers(lo, max(lo + 1, hi), size=count).tolist())
    entries = np.sort(np.array(drawn, dtype=np.int64))

    run = build_synthetic_run(rng_for(replication), entry_idx=entries,
                              sides=np.zeros(entries.size, dtype=np.int64))
    return {"control": "NC3", "replication": replication,
            "raw_count": run.raw_count, "effective_count": run.effective_count,
            "reference_count": reference.raw_count,
            "coverage_matched": bool(run.raw_count == reference.raw_count),
            "mean_net_r": float(run.net_r.mean()),
            "mean_gross_r": float(run.r_multiples.mean()),
            "funding_modelled": False}


# ---------------------------------------------------------------------------
# NC4 — random direction, matched timestamps
# ---------------------------------------------------------------------------

def nc4_random_direction_matched_timestamps(replication: int) -> dict[str, Any]:
    """The reference entry bars, with the side decided by a coin flip.

    This is the control that catches an asymmetric sign convention anywhere in
    the label or cost path — the failure P1 was specifically at risk of
    introducing. If longs and shorts are not treated as mirror images, a
    coin-flip direction drifts away from zero, and T3's band is what notices.
    """
    rng = rng_for(replication)
    reference = build_synthetic_run(rng)
    sides = rng.choice([1, -1], size=reference.entry_idx.size)
    # A fresh generator from the same seed reproduces the same bars, so this is
    # the identical market with only the direction changed.
    run = build_synthetic_run(rng_for(replication),
                              entry_idx=reference.entry_idx, sides=sides)
    # Read the side back off the resolved events rather than slicing the input:
    # if an event were dropped, a positional slice would silently mislabel every
    # side after it.
    resolved_sides = np.array([e.side for e in run.events], dtype=np.int64)
    long_mask = resolved_sides == 1
    return {"control": "NC4", "replication": replication,
            "raw_count": run.raw_count, "effective_count": run.effective_count,
            "mean_net_r": float(run.net_r.mean()),
            "mean_gross_r": float(run.r_multiples.mean()),
            "mean_net_r_long": float(run.net_r[long_mask].mean())
            if long_mask.any() else float("nan"),
            "mean_net_r_short": float(run.net_r[~long_mask].mean())
            if (~long_mask).any() else float("nan"),
            # Gross, per side, is what T3 binds to: costs are identical for both
            # sides, so they cannot create an asymmetry but they do bury it.
            "mean_gross_r_long": float(run.r_multiples[long_mask].mean())
            if long_mask.any() else float("nan"),
            "mean_gross_r_short": float(run.r_multiples[~long_mask].mean())
            if (~long_mask).any() else float("nan"),
            "n_long": int(long_mask.sum()), "n_short": int((~long_mask).sum()),
            "funding_modelled": False}


# ---------------------------------------------------------------------------
# NC5 — feature permutation
# ---------------------------------------------------------------------------

def _ols_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                     w_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Weighted least squares with an intercept. Deliberately the simplest
    model that can overfit: a complex one would confound "the apparatus leaks"
    with "the model memorised"."""
    a = np.column_stack([np.ones(x_train.shape[0]), x_train])
    sw = np.sqrt(w_train)[:, None]
    beta, *_ = np.linalg.lstsq(a * sw, y_train * np.sqrt(w_train), rcond=None)
    return np.column_stack([np.ones(x_test.shape[0]), x_test]) @ beta


def _oos_event_pnl(geometry: Any, x: np.ndarray, r: np.ndarray,
                   w: np.ndarray) -> np.ndarray:
    """One out-of-sample P&L per event, pooled over the splits that held it out.

    Each event is a test observation in several splits; its P&L is averaged over
    them so every event contributes exactly once to the series M01 sees.
    Concatenating instead would repeat each event `C(M-1,k-1)` times and inflate
    the sample length that PSR's `sqrt(n-1)` multiplies by — the same
    overlap-counting error M03 exists to prevent, reintroduced through the back
    door.

    The payoff is measured against the **training** mean, never the pooled mean.
    See `_centred` for why that matters here and why using the pooled mean would
    be look-ahead.
    """
    total = np.zeros(r.size, dtype=float)
    count = np.zeros(r.size, dtype=np.int64)
    for split in geometry.splits:
        pred = _ols_fit_predict(x[split.train_idx], r[split.train_idx],
                                w[split.train_idx], x[split.test_idx])
        total[split.test_idx] += np.sign(pred) * _centred(
            r, split.train_idx, split.test_idx, w)
        count[split.test_idx] += 1
    seen = count > 0
    return total[seen] / count[seen]


def _centred(r: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
             w: np.ndarray) -> np.ndarray:
    """Test returns, centred on the *training* set's weighted mean.

    This exists because of a finding, not a preference. M02's same-bar tie rule
    resolves an ambiguous bar as a stop, and on synthetic data that convention
    alone injects a drift of roughly -0.11 R per event (measured:
    `barrier_bias_diagnostic`). So the control data is not actually a null — it
    has a real, artifact, negative mean. Any directional model fitted on it
    learns "always take the other side" and earns that drift back, and a naive
    control then reports a significant DSR: NC5 did exactly that at DSR 0.96
    before this correction.

    That is not the apparatus manufacturing edge; it is the apparatus correctly
    detecting a drift the *control construction* put there. But a control that
    fires on its own artifact cannot distinguish the two, so the statistic has to
    be immune to a common additive drift. Centring on the training mean removes
    it while remaining strictly point-in-time — the pooled mean would leak the
    test groups' own average into the score, which is precisely the leak M04
    exists to prevent.
    """
    wt = w[train_idx]
    train_mean = float(np.sum(wt * r[train_idx]) / np.sum(wt))
    return r[test_idx] - train_mean


def nc5_feature_permutation(replication: int) -> dict[str, Any]:
    """Features with no relationship to the label, one of them permuted.

    The label is the M02 outcome on zero-drift bars; the features are persistent
    AR(1) noise. A model fitted through M04's purged splits and scored on the
    held-out groups should show nothing, permuted or not — and the permutation
    must not *improve* anything, because there is nothing to damage.
    """
    rng = rng_for(replication)
    run = build_synthetic_run(rng)
    n = run.raw_count
    features = np.column_stack([ar1_signal(n, rng) for _ in range(3)])
    y = run.net_r
    weights = uniqueness_weights(run.spans)

    cfg = CPCVConfig(n_groups=6, n_test_groups=2, embargo_bars=2,
                     purge_bars=DEFAULT_BARRIERS.vertical_bars,
                     holdout_idx_lo=None, min_train_events=10)
    geometry = combinatorial_splits(run.spans, cfg)

    def scorer(x: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
        def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
            pred = _ols_fit_predict(x[train_idx], y[train_idx],
                                    weights.weights[train_idx], x[test_idx])
            return float(np.mean(np.sign(pred) * _centred(
                y, train_idx, test_idx, weights.weights)))
        return score

    permuted = features.copy()
    permuted[:, 0] = rng.permutation(permuted[:, 0])

    base = backtest_paths(geometry, scorer(features))
    perm = backtest_paths(geometry, scorer(permuted))

    # The DSR is taken on the per-event out-of-sample P&L, not on the ten path
    # scores: ten observations are below M01's floor, so a DSR over them would
    # be a permanent INSUFFICIENT_DATA and the threshold would bind to nothing.
    #
    # And it is taken on the PAIRED DIFFERENCE between the real and the permuted
    # feature set, not on either alone. Permuting a feature that carries no
    # information should change nothing, so the difference is centred at zero by
    # construction whatever the data's own drift — which makes this a test of the
    # apparatus rather than of the control's artifacts.
    pnl_base = _oos_event_pnl(geometry, features, run.r_multiples,
                              weights.weights)
    pnl_perm = _oos_event_pnl(geometry, permuted, run.r_multiples,
                              weights.weights)
    event_pnl = pnl_base - pnl_perm
    out = _dsr_or_insufficient(event_pnl, n_trials=1)

    return {"control": "NC5", "replication": replication,
            "n_oos_events": int(event_pnl.size),
            "mean_event_pnl": float(event_pnl.mean()),
            "raw_count": weights.raw_count,
            "effective_count": weights.effective_count,
            "n_paths": geometry.n_paths, "n_splits": geometry.n_splits,
            "leakage_pairs": leakage_pairs(geometry, run.spans),
            "base_mean_path": float(base.path_scores.mean()),
            "permuted_mean_path": float(perm.path_scores.mean()),
            "permutation_improved": bool(perm.path_scores.mean()
                                         > base.path_scores.mean()),
            "dsr": out.get("dsr"), "status": out["status"],
            "significant": out["significant"], "funding_modelled": False}


# ---------------------------------------------------------------------------
# NC6 — the full pipeline on IID zero-drift data
# ---------------------------------------------------------------------------

def nc6_pipeline_zero_drift(replication: int) -> dict[str, Any]:
    """M02 -> M03 -> M04 -> M01, end to end, on data with no drift.

    The two numbers that matter are the *best* path's DSR and the leak count.
    Taking the maximum over paths is exactly how a backtest lies, so the
    deflation has to absorb it: `n_trials` is the path count, which is the honest
    trial number for a selection made over paths. The leak count must be zero
    exactly, not small.
    """
    rng = rng_for(replication)
    run = build_synthetic_run(rng, n_bars=1600)
    weights = uniqueness_weights(run.spans)

    cfg = CPCVConfig(n_groups=6, n_test_groups=2, embargo_bars=2,
                     purge_bars=DEFAULT_BARRIERS.vertical_bars,
                     holdout_idx_lo=None, min_train_events=10)
    geometry = combinatorial_splits(run.spans, cfg)

    # A signal with real persistence and no information, so the direction the
    # training set suggests is a genuine coin flip per split.
    signal = ar1_signal(weights.raw_count, rng)

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
        """The smallest strategy that still *learns*: pick the sign of the
        signal's weighted association with the label on train, apply it to the
        held-out group, and measure against the training mean.

        The dependence on `train_idx` is required, not incidental. A scorer that
        reads only the test group returns the same number for a given group in
        every split, so all C(M-1,k-1) paths collapse to one identical value and
        the "path distribution" has zero spread by construction — a control that
        would sail through T4 while measuring nothing at all. The first version
        of this function had exactly that defect.
        """
        wt, we = weights.weights[train_idx], weights.weights[test_idx]
        centred_train = _centred(run.r_multiples, train_idx, train_idx,
                                 weights.weights)
        lean = np.sign(np.sum(wt * signal[train_idx] * centred_train))
        if lean == 0.0:  # pragma: no cover - exact zero needs a measure-zero draw
            lean = 1.0
        payoff = _centred(run.r_multiples, train_idx, test_idx, weights.weights)
        return float(lean * np.sum(we * np.sign(signal[test_idx]) * payoff)
                     / np.sum(we))

    results = backtest_paths(geometry, score)
    # Sharpe per path, over that path's own M group scores.
    per_path_sharpe: list[float] = []
    for path in assemble_paths(geometry):
        vals = np.array([results.group_scores[(g, si)] for g, si in path],
                        dtype=float)
        sd = vals.std(ddof=1)
        per_path_sharpe.append(float(vals.mean() / sd) if sd > 0 else 0.0)

    # Per-event out-of-sample P&L, per split, reused twice below.
    def split_pnl(split: Any) -> np.ndarray:
        wt = weights.weights[split.train_idx]
        centred_train = _centred(run.r_multiples, split.train_idx,
                                 split.train_idx, weights.weights)
        lean = np.sign(np.sum(wt * signal[split.train_idx] * centred_train)) or 1.0
        payoff = _centred(run.r_multiples, split.train_idx, split.test_idx,
                          weights.weights)
        return lean * np.sign(signal[split.test_idx]) * payoff

    oos = np.zeros(weights.raw_count, dtype=float)
    seen = np.zeros(weights.raw_count, dtype=np.int64)
    for split in geometry.splits:
        oos[split.test_idx] += split_pnl(split)
        seen[split.test_idx] += 1
    oos_pnl = oos[seen > 0] / seen[seen > 0]

    # T4's second half, evaluated literally: the DSR of the BEST path.
    #
    # A path visits every group exactly once, so its own P&L series has one entry
    # per event — comfortably above M01's 30-observation floor, which means a
    # genuine per-path DSR is computable and no reinterpretation is needed. An
    # earlier version graded the pooled series instead and merely argued that
    # `n_trials = n_paths` represented the selection; the independent audit was
    # right that this did not evaluate the stated condition. `n_trials` is the
    # path count because the maximum is taken over exactly that many paths, which
    # is the selection the deflation has to absorb.
    path_dsr: list[dict[str, Any]] = []
    for path in assemble_paths(geometry):
        series = np.concatenate([split_pnl(geometry.splits[si])[
            np.isin(geometry.splits[si].test_idx,
                    geometry.splits[si].test_idx_by_group[g])]
            for g, si in path])
        path_dsr.append(_dsr_or_insufficient(series, n_trials=geometry.n_paths))

    graded = [p for p in path_dsr if p["status"] == "OK"]
    best_path_dsr = max((p["dsr"] for p in graded), default=None)
    best_path_significant = any(p["significant"] for p in path_dsr)

    pooled = _dsr_or_insufficient(oos_pnl, n_trials=geometry.n_paths)
    pooled_net = _dsr_or_insufficient(run.net_r, n_trials=geometry.n_paths)
    best = float(np.max(per_path_sharpe)) if per_path_sharpe else float("nan")

    return {"control": "NC6", "replication": replication,
            "dsr_net": pooled_net.get("dsr"),
            "best_path_dsr": best_path_dsr,
            # T1 for NC6 binds to the best path, not the pooled series: taking
            # the maximum over paths is how a backtest lies, so that is the
            # quantity whose false-positive rate matters.
            "significant_pooled": pooled["significant"],
            "dsr_pooled": pooled.get("dsr"),
            "n_paths_insufficient": len(path_dsr) - len(graded),
            "tie_rate": run.outcomes.get("TIE", 0) / max(1, weights.raw_count),
            "raw_count": weights.raw_count,
            "effective_count": weights.effective_count,
            "overlap_factor": weights.overlap_factor,
            "n_paths": geometry.n_paths, "n_splits": geometry.n_splits,
            "leakage_pairs": leakage_pairs(geometry, run.spans),
            "path_median": float(np.median(results.path_scores)),
            "path_mean": float(results.path_scores.mean()),
            "best_path_sharpe": best,
            "mean_net_r": float(run.net_r.mean()),
            "mean_gross_r": float(run.r_multiples.mean()),
            "outcomes": run.outcomes,
            "dsr": best_path_dsr,
            "status": "OK" if graded else "INSUFFICIENT_DATA",
            "significant": best_path_significant, "funding_modelled": False}


# ---------------------------------------------------------------------------
# NC7 — false ranking superiority (G1_SPEC.md T5)
# ---------------------------------------------------------------------------

def nc7_false_ranking(replication: int) -> dict[str, Any]:
    """Two independent null strategies, ranked against each other.

    Neither has an edge and neither is better, so the winner must be a coin
    flip. A systematic winner means the ranking machinery favours something
    other than performance — ordering, group assignment, or the weighting.
    """
    rng = rng_for(replication)
    run = build_synthetic_run(rng)
    weights = uniqueness_weights(run.spans)
    cfg = CPCVConfig(n_groups=6, n_test_groups=2, embargo_bars=2,
                     purge_bars=DEFAULT_BARRIERS.vertical_bars,
                     holdout_idx_lo=None, min_train_events=10)
    geometry = combinatorial_splits(run.spans, cfg)

    n = run.raw_count
    sig_a, sig_b = ar1_signal(n, rng), ar1_signal(n, rng)

    def score_for(sig: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
        def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float:
            payoff = _centred(run.r_multiples, train_idx, test_idx,
                              weights.weights)
            return float(np.mean(np.sign(sig[test_idx]) * payoff))
        return score

    a = backtest_paths(geometry, score_for(sig_a)).path_scores.mean()
    b = backtest_paths(geometry, score_for(sig_b)).path_scores.mean()
    advantage = float(a - b)

    # T5's second half: is the advantage *reported as significant*? Judged on the
    # per-event paired difference, with n_trials=2 because a winner was picked out
    # of two. Paired, so any drift common to both strategies cancels and what is
    # left is the difference the ranking claims to have found.
    def event_pnl(sig: np.ndarray) -> np.ndarray:
        total = np.zeros(n, dtype=float)
        count = np.zeros(n, dtype=np.int64)
        for split in geometry.splits:
            payoff = _centred(run.r_multiples, split.train_idx, split.test_idx,
                              weights.weights)
            total[split.test_idx] += np.sign(sig[split.test_idx]) * payoff
            count[split.test_idx] += 1
        seen = count > 0
        return total[seen] / count[seen]

    diff = event_pnl(sig_a) - event_pnl(sig_b)
    if advantage < 0:
        diff = -diff          # test the winner's claimed advantage, not A's
    out = _dsr_or_insufficient(diff, n_trials=2)

    return {"control": "NC7", "replication": replication,
            "raw_count": weights.raw_count,
            "effective_count": weights.effective_count,
            "leakage_pairs": leakage_pairs(geometry, run.spans),
            "a_mean_path": float(a), "b_mean_path": float(b),
            "advantage": advantage, "a_wins": bool(a > b),
            "dsr": out.get("dsr"), "status": out["status"],
            "significant": out["significant"],
            "funding_modelled": False}


# ---------------------------------------------------------------------------
# NC8 — M01 calibration on nothing but noise
# ---------------------------------------------------------------------------

def nc8_iid_noise_calibration(replication: int) -> dict[str, Any]:
    """IID zero-mean noise straight into M01, with no labeling in between.

    Added after the first control run, and the reason is worth recording. Every
    label-based control turned out to sit on a small negative drift injected by
    M02's tie convention, which makes a one-sided upward test essentially
    unable to fire. Those controls therefore prove "no manufactured edge" — the
    thing G1 asks — but they cannot measure a false-positive *rate*, because
    their null is not centred.

    This one is centred by construction. Under a correct M01 the DSR of a
    zero-mean series is *asymptotically* uniform on (0, 1), so the share above
    0.95 estimates the false-positive rate directly and the mean DSR should sit
    near 0.5. That makes T1 a real measurement rather than a formality.

    "Asymptotically" is not hedging. PSR is a normal approximation with estimated
    skew and kurtosis, so uniformity is exact only in the limit; for a discrete
    return series it cannot be exact at all, since a finite set of attainable
    sample means gives a finite set of attainable DSRs. All three families drawn
    here are continuous, and n = 250 is where the approximation is being checked
    — which is the point of measuring the rate instead of asserting it.

    Fat tails and skew are drawn in deliberately: a Student-t with 4 degrees of
    freedom, sign-tilted, is the case where a naive Sharpe test over-rejects, and
    it is exactly what M01's skew and kurtosis terms exist to absorb.
    """
    rng = rng_for(replication)
    n = 250
    kind = replication % 3
    if kind == 0:
        x = rng.normal(0.0, 1.0, size=n)
    elif kind == 1:
        x = rng.standard_t(4, size=n)
    else:
        # Skewed and centred: an exponential minus its own mean.
        x = rng.exponential(1.0, size=n) - 1.0
    stats = sharpe_stats(x)
    out = _dsr_or_insufficient(x, n_trials=1)
    return {"control": "NC8", "replication": replication,
            "family": ("normal", "student_t4", "skewed_exponential")[kind],
            "raw_count": n, "effective_count": float(n),
            "mean_r": float(x.mean()), "observed_sharpe": stats.sharpe,
            "skew": stats.skew, "kurtosis": stats.kurtosis,
            "dsr": out.get("dsr"), "status": out["status"],
            "significant": out["significant"]}


CONTROLS: dict[str, tuple[Callable[[int], dict[str, Any]], int]] = {
    # name -> (function, replications frozen in G1_SPEC.md §4)
    "NC1": (nc1_randomized_labels, 500),
    "NC2": (nc2_time_shuffled_signal, 500),
    "NC3": (nc3_random_entry_matched_coverage, 200),
    "NC4": (nc4_random_direction_matched_timestamps, 200),
    "NC5": (nc5_feature_permutation, 500),
    "NC6": (nc6_pipeline_zero_drift, 500),
    "NC7": (nc7_false_ranking, 500),
    "NC8": (nc8_iid_noise_calibration, 500),
}
