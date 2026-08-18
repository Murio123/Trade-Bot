# C5.0 G1 — apparatus trustworthiness: result

Run 2026-08-17. Anchor `7c8f37e` (P1 closed). Criteria frozen at `197a765`,
before any code existed. Apparatus at `d2e86b3`.

## Verdict: **G1_INDETERMINATE**

**Phase A′ stays blocked.** Indeterminate is an outcome, not an error
(ARCHITECTURE.md §5.7), and it blocks exactly as a failure does.

Every apparatus property G1 asks about is demonstrated: leakage is exactly zero,
the false-positive rate is at or below nominal on every control that can measure
one, no control shows manufactured edge, and the two defects the suite found are
fixed. The suite returns `CONTROLS_INDETERMINATE` at the frozen replication
counts (R = 500 / 200, master seed 20260817) with a single outstanding item.

**What is unresolved: T3 as frozen fails, at 0.0350 against `[0.394, 0.606]`.**
An earlier draft of this document reported G1_PASS by evaluating T3 as the
long/short gap under amendment A3 instead. An independent audit called that a
post-hoc relaxation, and on the point that matters it was right: a failing frozen
criterion was replaced with a passing one after the results were seen, and the
code being measured cannot be the judge of whether the criterion it fails is
defective or merely inconvenient. A3 is therefore recorded as a **proposal that
does not gate**, T3 as frozen is the binding criterion, and the verdict follows
from it.

This is a defect in the criterion, not evidence that the apparatus manufactures
edge — which is why the outcome is indeterminate rather than `G1_FAIL`.
`G1_FAIL` is reserved by §5 for a T2 CI above zero, a T1 breach, or a T6 leak,
and none occurred. The three ways to resolve it are set out in `G1_SPEC.md` §7
A3; the decision is the project owner's, not the implementer's.

**Also unresolved, and independent of the above: the trial ledger does not
exist**, so M01 cannot verify its own `n_trials`. G1 proves the statistic is
correct; it cannot prove a future caller will report the trial count honestly.

---

## 1. M01 — Deflated Sharpe Ratio

`validation/deflated_sharpe.py`, 32 tests.

Observed Sharpe, expected maximum Sharpe under selection, and the deflated
probability are three separate fields, never collapsed. Per observation
throughout — no annualization, enforced by a test that scans the parsed AST for
numeric literals (a text scan would flag the docstring's own "no `sqrt(252)`").
Kurtosis is non-excess; the wrong convention would make every p-value too
confident. `InsufficientData` below 30 observations or on a degenerate series.

**Calibration, measured rather than assumed** (NC8, R = 500, three
distribution families):

| statistic | value | expected under a correct M01 |
|---|---|---|
| mean DSR | 0.4961 | 0.5 |
| P(DSR ≥ 0.95) | **0.0340** | 0.05 |

The DSR of a mean-zero series is uniform, so that share *is* the false-positive
rate. It holds on Gaussian, Student-t(4) and skewed-exponential inputs.

**A real defect found by its own test.** The degeneracy guard read
`std <= 0.0`. A constant array's sample std is `1.1e-16`, not zero, so
`np.full(100, 0.4)` produced a Sharpe of 3.6e15 and sailed into the PSR formula.
The guard is now scale-relative.

**Documented bias.** With no recorded trial Sharpes, the expected maximum is
computed from the sampling-noise-only dispersion, which assumes trials differ
only by luck. Real trials differ in construction too, so their true dispersion
is larger and this fallback **under-deflates**. It is named in the output
(`sharpe_std_source`) so a result can never silently claim more than it has.

---

## 2. M02 — Triple Barrier

`labeling/triple_barrier.py`, 34 tests.

First-touch resolution over bars `i+1 .. i+v` inclusive. The decision bar is
never scanned. A touch on the vertical bar is a touch, not a timeout. Touches
are non-strict against the computed barrier. A truncated horizon resolves
`TRUNCATED` with a null label, never `TIME` — `TIME` asserts that a full horizon
was observed and was empty. A gap inside the scan window raises; skipping the
continuity check requires `time_col=None` at the call site.

**The defect NC4 caught.** The tie rule resolved to LOWER unconditionally and
was described as conservative. LOWER is the stop only for a long; for a short it
is the *target*. The "pessimistic" rule was handing every short a free win:

| | gross R, longs | gross R, shorts | gap |
|---|---|---|---|
| as first written | −0.22 | −0.01 | **−0.21** |
| after the fix | −0.09497 | −0.10632 | +0.01135, CI [−0.0041, +0.0287] |

The pooled mean was −0.002 before the fix — a clean-looking null. No aggregate
statistic could have seen this; only the side-by-side comparison could.
Amendment A1.

**The tie rule's cost, now measured** (`barrier_bias_diagnostic`, 997 events):

| barriers | tie rate | gross R/event | net R/event |
|---|---|---|---|
| symmetric 1:1 | 11.23% | −0.05717 | −0.17406 |
| project 2:1 | 3.01% | +0.02852 | −0.08837 |

At the project's 2:1 geometry the convention costs about 0.03 R per event. It is
a deliberate pessimistic bias in the safe direction, but it is not negligible,
and Phase A′ must not read a small negative expectancy as evidence of no edge
without subtracting it. It was previously invisible: `outcome_counts` now reports
`TIE` separately from `SL`.

Barrier multipliers are **not** chosen here. They are two free parameters and
the module's main overfitting surface; freezing them needs the trial ledger and
belongs to A′.

---

## 3. M03 — Sample Weights / Uniqueness

`labeling/sample_weights.py`, 28 tests.

Concurrency over closed spans, average uniqueness per event, weights that are
the uniquenesses themselves — so they are strictly positive and sum to the
effective sample size by construction rather than by a normalisation step
someone could skip. `uniqueness_weights` returns an object carrying both counts;
returning a bare array would have let a caller quote a weighted metric against
the row count, which ARCHITECTURE.md M03 defines as a specification violation.

Recovers the project's known geometry: 12-bar labels started every bar give
`max_concurrency = 12` and an overlap factor between 10 and 12.5 — the ~12×
figure C4.3e found empirically on the evaluation side.

`tools/ridge_ranking_significance.effective_sample_size` (an `n/horizon` proxy)
is left untouched: it is part of a frozen C4.3e result path and changing what it
prints would change a published number. It is superseded for all new work.

### Raw vs effective, across the controls

| control | raw | effective | overlap |
|---|---|---|---|
| NC1 / NC5 / NC7 (stride 4, 12-bar) | 297 | 292.3 | ×1.016 |
| NC3 (random entry, clustered) | 297 | 220.1 | **×1.350** |
| NC6 (stride 4, longer frame) | 397 | 390.6 | ×1.016 |
| NC2 / NC8 (bar series, no spans) | 1199 / 250 | identical | ×1.000 |

NC3 is the informative row: random entries clump, so the same nominal count
carries 26% less information than the strided sets. Non-overlapping spans give
effective exactly equal to raw (T7), and no control ever reported effective
above raw.

---

## 4. M04 — CPCV

`validation/cpcv.py`, 33 tests.

`C(M,k)` splits reassembling into `C(M-1,k-1)` paths; at M=6, k=2 that is 15
splits and 5 paths, verified against the combinatorics including the
`C(M,k)·k/M` identity. Deterministic manifests with a content hash. Fails closed
on `k >= M`, on too few events, and when purge consumes the training set — it
never degrades to K-fold.

**Purge is two-sided, and this is the substantive difference from the existing
geometry.** `tools/deep_backtest.fold_windows` purges forward only, which is
correct for a strictly forward walk and insufficient here: a combinatorial test
group has training data on both sides. The adversarial test
ARCHITECTURE.md M04 asks for plants an event spanning bars 55–160, crossing three
groups. By row index it is early in the frame and looks safely in the past; by
span it contaminates everything. It is purged from every split that tests a
group it crosses, and retained where it does not — which is the point of purging
by span rather than dropping it globally.

**T6 — leakage is zero exactly**, not small: 0 intersecting (train, test) span
pairs across 1500 splits per control over NC5, NC6 and NC7. Embargo violations
0. Sealed holdout (idx 8199+) excluded before splitting, boundary exclusive on
the span end, and everything-inside-the-holdout fails closed.

**Reported with every output:** paths share training data, so the path spread
understates the true variance. `paths_share_training_data` and an explicit
`variance_caveat` string travel with the distribution.

---

## 5. Negative controls — definitions

All synthetic, fixed seeds, byte-reproducible. The real-data matched-coverage
baselines already exist in `tools/swing_hypothesis_walkforward.py`; running them
is Phase A′ evaluation, which G1 forbids, and they answer a different question.

| # | control | construction | R |
|---|---|---|---|
| NC1 | randomized labels | M02 outcomes redrawn IID from their own marginal | 500 |
| NC2 | time-shuffled signal | AR(1) signal permuted against next-bar returns, marginal preserved exactly | 500 |
| NC3 | random entry, matched coverage | entry count matched per decile of the walked region | 200 |
| NC4 | random direction, matched timestamps | reference entry bars, side Bernoulli(0.5) | 200 |
| NC5 | feature permutation | one AR(1) feature permuted, scored through purged CPCV | 500 |
| NC6 | IID zero-drift pipeline | M02 → M03 → M04 → M01 end to end | 500 |
| NC7 | false ranking superiority | two independent null strategies ranked against each other | 500 |
| NC8 | M01 calibration | IID zero-mean noise straight into M01, no labeling | 500 |

Barriers are **symmetric** in the controls. With `upper == lower` a zero-drift
process has an analytically zero label expectancy, so any expectancy reported is
manufactured. At the project's 2:1 geometry the null depends on the tie rule, and
a control built on it would confound "the apparatus invents edge" with "M02 is
conservative by design".

NC7 and NC8 were added during the work. NC8 exists because every label-based
control sits on the tie rule's small negative drift, which makes a one-sided
upward test nearly unable to fire: those controls prove "no manufactured edge",
but they cannot measure a false-positive *rate*. NC8's null is centred by
construction, so T1 becomes a real measurement. NC7 is T5's control, which the
frozen spec required but did not name.

---

## 6. Negative controls — results

| control | threshold | value | verdict |
|---|---|---|---|
| NC1 | T1 FPR ≤ 0.075 | 0.0020 | pass |
| NC2 | T1 FPR ≤ 0.075 | 0.0480 | pass |
| NC3 | T2 net / gross CI not above zero | −0.2101 / −0.0973, both BELOW_ZERO | pass |
| NC4 | T2 net / gross CI not above zero | −0.2133 / −0.1005, both BELOW_ZERO | pass |
| NC4 | **T3 as frozen** | 0.0350 vs [0.394, 0.606] | **FAIL — binding** |
| NC4 | T3 long/short gap CI contains zero | +0.01135, CI [−0.0041, +0.0287] | pass (A3, proposal only) |
| NC5 | T1 FPR ≤ 0.075 | 0.0680 | pass |
| NC6 | T1 FPR ≤ 0.075, **best path** | 0.0020 | pass |
| NC6 | T4 path median CI contains zero | −0.00248, CI [−0.0072, +0.0026] | pass |
| NC7 | T5 winner share in [0.433, 0.567] | 0.4800 | pass |
| NC7 | T1 FPR ≤ 0.075 | 0.0200 | pass |
| NC8 | T1 FPR ≤ 0.075 | 0.0340 | pass |
| NC2 | must not rank above its own unshuffled null | 0.4900 vs band top 0.567 | pass |
| NC5/6/7 | T6 leakage == 0 | 0 | pass |
| all | T7 both counts reported by every record | yes, effective ≤ raw everywhere | pass |

Notes a reader should have:

- **The T2 CIs are all below zero, and that is correct.** A random entry paying a
  real ~0.11 R round-trip cost is supposed to lose money. T2's operative
  sentences are the explicit ones: strictly above zero fails, strictly below zero
  is reported and does not.
- **NC5's 0.0680 is the closest call**, 1.8 binomial SE above the nominal 0.05
  and inside the frozen bound. It is a paired statistic on ~297 out-of-sample
  events, and it is the one number in this table worth re-measuring if the
  apparatus changes.
- **NC6's path spread is 0.056, not zero.** This is checked because a degenerate
  path distribution would satisfy T4's CI test while measuring nothing — which
  is exactly what the first version of that control did.

### Four gate defects found by the independent audit

Recorded separately from the ones I found, because these are the ones that got
past me and two of them made the gate weaker than it claimed to be.

1. **T4's second half was never evaluated.** The frozen text asks whether the
   *best path* clears `dsr >= 0.95`; the runner graded the pooled series and
   argued that `n_trials = n_paths` represented the selection. That was a
   reinterpretation, and an unnecessary one: a path visits every group exactly
   once, so its own series has one entry per event — well above M01's
   30-observation floor — and a genuine per-path DSR is computable. Now it is
   computed, and the best-path FPR is **0.0020**.
2. **NC2's stated condition was recorded and never graded.** The spec requires
   the shuffled signal not to rank above its own unshuffled null. Both Sharpes
   were computed and neither was compared, so every replication could have had
   the shuffled signal winning and NC2 would still have passed on T1 alone.
   Measured: **0.4900** against a band top of 0.567.
3. **T7 passed by omission.** `reported` was true if *any* record carried both
   counts, and the effective-vs-raw comparison was nested inside that check — so
   a control that stopped reporting effective sizes would have satisfied T7 by
   not reporting them. Both halves are now enforced independently.
4. **The M01 calibration claim was overstated.** "The DSR of a mean-zero series
   is uniform" is exact only asymptotically, and for a discrete return series it
   cannot be exact at all: finitely many attainable sample means give finitely
   many attainable DSRs. The claim is now qualified.

The audit also found no leakage path in M04, no look-ahead in M02, and no
arithmetic defect in M03 — and it was the source of the verdict change above.

### Three invalid controls, found and fixed

Recorded because a control that cannot fail is worse than no control, and two of
these would have passed while proving nothing.

1. **NC5 reported DSR 0.96 and was right to.** M02's tie rule injects a ~0.11 R
   drift into the synthetic data, so an OLS whose features carry no information
   still learns "always take the other side" from the intercept and earns that
   drift back. The apparatus was correctly detecting a drift the *control
   construction* had planted. Fixed by making the statistic a paired difference
   between the real and permuted feature sets, which is immune to any common
   additive drift.
2. **NC6's first scorer ignored `train_idx`.** A scorer reading only the test
   group returns the same value for a given group in every split, so all five
   paths collapsed to one identical number and the path distribution had zero
   spread by construction. It would have passed T4 comfortably.
3. **Payoffs are centred on the training mean, never the pooled mean.** The
   pooled mean would leak the test groups' own average into the score — the
   precise leak M04 exists to prevent.

---

## 7. Verification

- Targeted: 32 (M01) + 34 (M02) + 28 (M03) + 33 (M04) + 30 (controls) + 14
  (integration) = **171 new tests**.
- Full suite: **1789 passed**, 0 failed (1618 at the anchor).
- `git diff --check` clean.
- No frozen artifact opened: `validation/cost_sensitivity.py` still hashes to
  `1cd4db9fe8e8`; `ARCHITECTURE.md`, `BASELINE.md`, `P1_TASK.md` and
  `P1_RESULT.md` unmodified since `7c8f37e`, enforced by a test.
- No sealed holdout read. No strategy rule, threshold, model or dataset touched.
- Import direction intact: nothing under `validation/` or `labeling/` imports
  `tools.*`; no production module imports `labeling`.
- One duplicate primitive removed rather than added: modal-interval and gap
  detection now live once, in `validation/bar_grid.py`.

Generated artifacts are untracked, under `reports/g1/`.

---

## 8. Next allowed step

**Phase A′ is blocked**, by the indeterminate verdict. Two things stand between
here and it, in this order.

**First, a decision that is not mine to make: what to do about T3.** The three
options are in `G1_SPEC.md` §7 A3. The recommendation is option 3 — re-freeze T3
for a future gate, record G1 as indeterminate on that one line, and take the
apparatus properties as demonstrated — because it neither rewrites history nor
discards work. Options 1 and 2 are both defensible and both belong to the owner.

**Second, the §6.1 trial ledger.** M01 takes `n_trials` as an argument and cannot
check it. Understating that count is, per ARCHITECTURE.md M01, "the single
largest failure mode", and it is addressed structurally by the ledger, which does
not exist. Running A′ before it means every deflated number rests on a trial
count reconstructed after the fact from git history — exactly the reconstruction
§6.1 says cannot be trusted. The ledger also needs retroactive population from
the project's history (H1, H2, confluence variants, the `/backtest` threshold
grid, the four profiles, C4.3a–e), with unknowns rounded up.

Only then A′: re-run H1, H2, the confluence score and C4.3d/C4.3e through this
apparatus. Both outcomes are written into ARCHITECTURE.md §3.2 in advance — a
rejection may have been made on insufficient grounds, or the +0.13 may not
survive deflation and the C4.4/C4.5 ranker loses its stated justification. The
second must be accepted if it occurs.

Two smaller items A′ must not forget:

- M02's barrier multipliers are unchosen and must be frozen and ledgered before
  the first A′ run.
- M02's tie rule costs ~0.03 R per event at the project's 2:1 geometry. A′ must
  subtract that before reading a small negative expectancy as absence of edge.
