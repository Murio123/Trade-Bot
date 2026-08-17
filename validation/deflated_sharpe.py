"""M01 — the Sharpe ratio, corrected for how many times we looked.

A Sharpe ratio computed on the winner of 40 experiments is not an estimate of
that strategy's skill; it is an estimate of the maximum of 40 draws from a
distribution whose mean may well be zero. This module makes that correction
explicit, following Bailey & Lopez de Prado (2014), *The Deflated Sharpe
Ratio*, and Bailey & Lopez de Prado (2012), *The Sharpe Ratio Efficient
Frontier* for the probabilistic Sharpe ratio it is built on.

Three quantities are kept strictly separate, because conflating them is the
usual way this correction is defeated:

- **observed Sharpe** — what the return series says, per observation;
- **expected maximum Sharpe** — what the best of `n_trials` null strategies
  would show anyway, given the dispersion of Sharpes across trials;
- **deflated Sharpe** — the probability that the observed Sharpe exceeds that
  benchmark, given the sample's length, skew and kurtosis.

Conventions, frozen in `reports/c50/G1_SPEC.md` §2:

**Per observation, never annualized.** There is no `sqrt(252)` in this file and
there must never be one. The project's trade series are irregular in time, so
any annualization factor is an invented parameter; and the deflation compares a
Sharpe against a distribution of Sharpes computed the same way, so a scale
factor cancels only as long as it is never applied. A caller who wants an
annual number multiplies outside and owns the assumption.

**Kurtosis is non-excess** — 3.0 for a Gaussian. The PSR variance term
`1 - g3*SR + (g4-1)/4 * SR^2` requires it. Passing excess kurtosis would make
the term too small and every p-value too confident, which is the failure
direction that matters.

**Insufficient data raises.** A 4-trade series has no Sharpe worth deflating,
and returning 0.5 for it would let a caller treat noise as a measurement.
`InsufficientData` is a distinct exception so it can be caught and reported as
an outcome (ARCHITECTURE.md §5.7: insufficient data is an outcome, not an
error) without being confused with a bug.

Not a selector. Nothing here chooses a strategy; G1 forbids that until the
trial ledger of ARCHITECTURE.md §6.1 exists, because a DSR quoted against a
hand-counted `n_trials` is only as honest as that count.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

DSR_VERSION = "m01_dsr_v1"

# Below this the moment estimates that the correction depends on -- skew and
# kurtosis above all -- are noise, and a deflated Sharpe built on them is a
# number with no content. 30 is the conventional floor and is stated here rather
# than passed in, so no caller can lower it for one inconvenient series.
MIN_OBSERVATIONS = 30

EULER_MASCHERONI = 0.5772156649015329


class DeflatedSharpeError(Exception):
    """Malformed input — a programming error."""


class InsufficientData(DeflatedSharpeError):
    """The series cannot support a deflated Sharpe.

    Deliberately a separate type: callers report this as an explicit
    insufficient-data outcome and never as a Sharpe of zero.
    """


# ---------------------------------------------------------------------------
# Normal distribution, without scipy
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf — exact to double precision."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation coefficients (|error| < 1.15e-9), refined
# below by one Halley step against the exact erfc, which takes the result to
# full double precision. scipy is not a dependency of this project and adding
# one for a single inverse CDF is not worth the deployment surface.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Standard normal quantile.

    Raises on p outside (0, 1): the infinite tails are never a legitimate
    answer here, and returning +-inf would propagate silently into an expected
    maximum Sharpe of infinity.
    """
    if not (0.0 < p < 1.0):
        raise DeflatedSharpeError(f"norm_ppf needs p in (0, 1), got {p!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q
             + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r
             + _A[5]) * q / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r
                              + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q
              + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)

    # Halley refinement.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# ---------------------------------------------------------------------------
# Sample moments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharpeStats:
    """The four numbers the deflation needs, and nothing else.

    `sharpe` is per observation. `kurtosis` is non-excess.
    """
    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sharpe_stats(returns: Sequence[float] | np.ndarray) -> SharpeStats:
    """Per-observation Sharpe plus the shape terms.

    `std` uses ddof=1 — the sample standard deviation, matching the T-1 in the
    PSR denominator. Skew and kurtosis are the plain standardised moments
    (Fisher-Pearson without the small-sample correction), which is the
    convention the source paper's formula assumes.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0:
        raise InsufficientData("empty return series")
    if not np.all(np.isfinite(r)):
        raise DeflatedSharpeError("return series contains non-finite values")
    n = int(r.size)
    if n < MIN_OBSERVATIONS:
        raise InsufficientData(
            f"{n} observations is below the {MIN_OBSERVATIONS}-observation floor; "
            "skew and kurtosis are not estimable and a deflated Sharpe built on "
            "them would be meaningless")

    mean = float(r.mean())
    std = float(r.std(ddof=1))
    # Not `std <= 0.0`. A genuinely constant series does not have a std of
    # exactly zero in floating point: `np.full(100, 0.4).std(ddof=1)` is
    # 1.1e-16, which sails past an equality guard and yields a Sharpe of 3.6e15.
    # Degeneracy has to be judged against the data's own scale.
    scale = max(float(np.max(np.abs(r))), 1.0)
    if std <= scale * 1e-12:
        raise InsufficientData(
            f"return series is constant to within floating-point precision "
            f"(std={std:.3g} against a scale of {scale:.3g}); a Sharpe ratio is "
            "undefined and reporting one as 0.0, as infinite, or as 3.6e15 "
            "would all be wrong")

    centred = (r - mean) / std
    # Population standardised moments about the sample mean, scaled by the
    # ddof=1 std that the PSR formula uses.
    skew = float(np.mean(centred ** 3))
    kurtosis = float(np.mean(centred ** 4))
    return SharpeStats(n=n, mean=mean, std=std, sharpe=mean / std,
                       skew=skew, kurtosis=kurtosis)


def _psr_variance_term(sharpe: float, skew: float, kurtosis: float) -> float:
    """`1 - g3*SR + (g4-1)/4 * SR^2`, the shape correction to the Sharpe's own
    sampling variance.

    Negative skew and fat tails both make a Sharpe less trustworthy, and this
    is the term that says so. It can go non-positive for extreme inputs, which
    is not a defect of the data but a statement that the asymptotic
    approximation has left its domain — so it fails closed rather than taking
    the square root of a negative number.
    """
    term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if term <= 0.0:
        raise InsufficientData(
            f"the PSR variance term is {term:.6g} (<= 0) for sharpe={sharpe:.4f}, "
            f"skew={skew:.4f}, kurtosis={kurtosis:.4f}; the asymptotic "
            "approximation does not hold for this distribution shape")
    return term


def probabilistic_sharpe(stats: SharpeStats, benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), given length and distribution shape.

    Bailey & Lopez de Prado (2012). The `n - 1` is theirs, not a fencepost
    slip: the estimator's variance carries the sample-size correction.
    """
    term = _psr_variance_term(stats.sharpe, stats.skew, stats.kurtosis)
    z = (stats.sharpe - benchmark) * math.sqrt(stats.n - 1) / math.sqrt(term)
    return norm_cdf(z)


# ---------------------------------------------------------------------------
# Selection pressure
# ---------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """What the best of `n_trials` null strategies shows by luck alone.

    The Gumbel approximation to the expected maximum of `n_trials` draws:

        E[max] = sharpe_std * [(1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e))]

    with `g` the Euler-Mascheroni constant and `Z` the normal quantile.

    `n_trials == 1` returns exactly 0.0. There is no selection in a single
    trial, so the benchmark is the null mean; the formula itself degenerates
    there (`Z(0)` is `-inf`) and special-casing it is the honest reading rather
    than a numerical patch.
    """
    if n_trials < 1:
        raise DeflatedSharpeError(f"n_trials must be >= 1, got {n_trials}")
    if sharpe_std < 0.0:
        raise DeflatedSharpeError("sharpe_std must be non-negative")
    if n_trials == 1 or sharpe_std == 0.0:
        return 0.0
    n = float(n_trials)
    return sharpe_std * ((1.0 - EULER_MASCHERONI) * norm_ppf(1.0 - 1.0 / n)
                         + EULER_MASCHERONI * norm_ppf(1.0 - 1.0 / (n * math.e)))


def _sharpe_std_from_trials(trial_sharpes: Sequence[float] | np.ndarray) -> float:
    s = np.asarray(trial_sharpes, dtype=float).ravel()
    if s.size < 2:
        raise DeflatedSharpeError(
            "trial_sharpes needs at least 2 trials to have a dispersion")
    if not np.all(np.isfinite(s)):
        raise DeflatedSharpeError("trial_sharpes contains non-finite values")
    return float(s.std(ddof=1))


def sampling_sharpe_std(stats: SharpeStats) -> float:
    """The dispersion of Sharpe estimates attributable to sampling noise alone.

    `sqrt(psr_variance_term / (n - 1))` — the asymptotic standard error of the
    Sharpe estimator under the observed distribution shape.

    This is the **fallback** when the trial Sharpes themselves were not
    recorded, and its bias direction must be stated: it assumes the trials
    differ only by noise and share one true Sharpe. Real trials also differ in
    construction, so their true dispersion is larger, the true expected maximum
    is larger, and a DSR computed from this fallback is therefore **optimistic**
    — it under-deflates. Pass `trial_sharpes` whenever they exist. This is why
    the trial ledger is the next blocker after G1 and not an optional nicety.
    """
    term = _psr_variance_term(stats.sharpe, stats.skew, stats.kurtosis)
    return math.sqrt(term / (stats.n - 1))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Every intermediate kept, so the number can be argued with."""
    version: str
    n_observations: int
    n_trials: int
    observed_sharpe: float
    skew: float
    kurtosis: float
    sharpe_std: float
    sharpe_std_source: str
    expected_max_sharpe: float
    probabilistic_sharpe: float
    dsr: float
    p_value: float

    @property
    def significant(self) -> bool:
        """DSR >= 0.95 — the one-sided alpha = 0.05 of G1_SPEC.md §4.1."""
        return self.dsr >= 0.95

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def deflated_sharpe(returns: Sequence[float] | np.ndarray, *, n_trials: int,
                    trial_sharpes: Sequence[float] | np.ndarray | None = None,
                    ) -> DeflatedSharpeResult:
    """The deflated Sharpe ratio of a return series.

    `dsr` is a probability: P(true Sharpe > expected max under `n_trials`).
    `p_value = 1 - dsr` is the one-sided p-value against that benchmark.

    Deterministic: no sampling, no random state, no iteration order.
    """
    stats = sharpe_stats(returns)
    if trial_sharpes is None:
        sharpe_std = sampling_sharpe_std(stats)
        source = "sampling_noise_fallback"
    else:
        sharpe_std = _sharpe_std_from_trials(trial_sharpes)
        source = "measured_trial_dispersion"

    benchmark = expected_max_sharpe(n_trials, sharpe_std)
    dsr = probabilistic_sharpe(stats, benchmark)
    return DeflatedSharpeResult(
        version=DSR_VERSION, n_observations=stats.n, n_trials=int(n_trials),
        observed_sharpe=stats.sharpe, skew=stats.skew, kurtosis=stats.kurtosis,
        sharpe_std=sharpe_std, sharpe_std_source=source,
        expected_max_sharpe=benchmark,
        probabilistic_sharpe=probabilistic_sharpe(stats, 0.0),
        dsr=dsr, p_value=1.0 - dsr)


def min_track_record_length(stats: SharpeStats, benchmark: float = 0.0,
                            alpha: float = 0.05) -> float:
    """Observations needed before the observed Sharpe clears `benchmark` at
    confidence `1 - alpha`.

    Returns `inf` when the observed Sharpe is at or below the benchmark: no
    amount of further data makes a non-existent excess significant, and
    returning a large finite number would suggest otherwise.
    """
    if not (0.0 < alpha < 1.0):
        raise DeflatedSharpeError(f"alpha must be in (0, 1), got {alpha!r}")
    if stats.sharpe <= benchmark:
        return math.inf
    term = _psr_variance_term(stats.sharpe, stats.skew, stats.kurtosis)
    z = norm_ppf(1.0 - alpha)
    return 1.0 + term * (z / (stats.sharpe - benchmark)) ** 2


# ---------------------------------------------------------------------------
# Probability of backtest overfitting
# ---------------------------------------------------------------------------

def pbo(in_sample: np.ndarray, out_of_sample: np.ndarray) -> float:
    """Probability of backtest overfitting, by the CSCV logit rule.

    Both inputs are `(n_splits, n_strategies)` performance matrices — the same
    strategies scored in-sample and out-of-sample on each split. For every
    split the in-sample winner is selected, its out-of-sample relative rank
    `w` is taken, and PBO is the share of splits whose logit `log(w/(1-w))` is
    below zero, i.e. the winner landed in the bottom half out of sample.

    Implemented and tested as a primitive; deliberately not part of any G1
    threshold, because it needs a real trial population to mean anything.
    """
    is_m = np.asarray(in_sample, dtype=float)
    oos_m = np.asarray(out_of_sample, dtype=float)
    if is_m.ndim != 2 or is_m.shape != oos_m.shape:
        raise DeflatedSharpeError(
            "pbo needs two (n_splits, n_strategies) matrices of equal shape, "
            f"got {is_m.shape} and {oos_m.shape}")
    n_splits, n_strategies = is_m.shape
    if n_strategies < 2:
        raise DeflatedSharpeError("pbo needs at least 2 strategies to rank")
    if not (np.all(np.isfinite(is_m)) and np.all(np.isfinite(oos_m))):
        raise DeflatedSharpeError("pbo inputs contain non-finite values")

    below = 0
    for s in range(n_splits):
        best = int(np.argmax(is_m[s]))
        # Rank of the winner among the out-of-sample scores, 1-based, ties
        # resolved to the average rank so a plateau is not scored as a win.
        order = np.argsort(np.argsort(oos_m[s]))
        rank = float(order[best]) + 1.0
        ties = np.flatnonzero(oos_m[s] == oos_m[s][best])
        if ties.size > 1:
            rank = float(np.mean([order[t] + 1.0 for t in ties]))
        omega = rank / (n_strategies + 1.0)
        if math.log(omega / (1.0 - omega)) < 0.0:
            below += 1
    return below / float(n_splits)
