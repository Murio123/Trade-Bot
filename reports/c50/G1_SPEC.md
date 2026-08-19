# C5.0 G1 — apparatus trustworthiness: frozen specification

Declared 2026-08-17, **before any control was run and before any real project
result was re-interpreted**. Anchor commit: `7c8f37e` (P1 closed, full suite
1618 green). This document exists so the pass/fail thresholds cannot be chosen
after seeing the numbers. `G1_RESULT.md` may report against these criteria; it
may not amend them.

G1 asks one question: **does the research apparatus manufacture edge where
there is none?** It does not ask whether any edge exists. No strategy rule,
threshold, model or frozen downstream artifact is read, changed or
re-interpreted by anything specified here.

---

## 1. Scope

| Module | Path (fixed by ARCHITECTURE.md §1) | Purpose |
|---|---|---|
| M01 | `validation/deflated_sharpe.py` | deflate a Sharpe for selection pressure |
| M02 | `labeling/triple_barrier.py` | path-dependent first-touch labels |
| M03 | `labeling/sample_weights.py` | concurrency, uniqueness, effective sample size |
| M04 | `validation/cpcv.py` | combinatorial purged cross-validation |
| NC | `validation/negative_controls.py` + `tools/g1_negative_controls.py` | the null suite |

Supporting extraction: `validation/bar_grid.py` — modal bar interval and gap
intervals, moved out of `validation/funding.py` so M02 and P1 share one
implementation instead of two. Behaviour is unchanged and the 59 existing
funding tests are the proof.

---

## 2. Conventions frozen here

**Frequency (M01).** Every Sharpe in this apparatus is **per observation** —
per trade for trade series, per bar for bar series. There is no annualization
anywhere, no `sqrt(252)`, no `sqrt(365*6)`. A caller who wants an annual figure
multiplies outside the module and owns that assumption. The reason: the
project's trade series are irregular in time, so any annualization factor is a
fabricated parameter, and DSR compares a Sharpe against a distribution of
Sharpes computed the same way — the scale cancels only if it is never applied.

**Kurtosis (M01).** `kurtosis` means the **non-excess** fourth standardised
moment: 3.0 for a Gaussian. Bailey & López de Prado's PSR variance term is
`1 - g3*SR + (g4-1)/4 * SR^2`, which requires the non-excess convention; the
excess convention would silently inflate confidence.

**Barrier touch (M02).** Non-strict: `high >= upper` is a touch, `low <=
lower` is a touch. When both barriers fall inside one bar's range the outcome
is **SL for the position's side** — LOWER for a long, UPPER for a short, and
LOWER for a direction-neutral event as a frozen deterministic convention. The
intrabar path is unknown from OHLC and the pessimistic branch is taken
deliberately (ARCHITECTURE.md M02 assumptions). The vertical barrier
bar is **included** in the touch scan: an event with horizon `v` scans bars
`start+1 .. start+v` inclusive, and resolves `TIME` only if no barrier is
touched in that closed range. The decision bar itself is never scanned — its
own high/low are already known when the barriers are set, so scanning it would
be look-ahead of the trivial kind.

**Span (M02/M03/M04).** A label span is the closed bar interval
`[start_idx, end_idx]` with `end_idx > start_idx` always. Zero-length is a
defect, not a fast fill. The label of bar `u` is knowable only at `end_idx(u)`.

**Purge (M04).** Two-sided and span-based, not row-based. A training event is
purged if its span intersects any test group's span-expanded window. The
existing walk-forward geometry (`tools/deep_backtest.fold_windows`) purges only
forward, which is correct for a strictly forward walk and insufficient for
combinatorial folds where a test group has training data on both sides.

**Embargo (M04).** Applied on the **right edge of every test group**, in bars,
after purge. Left edge needs no embargo: span purge already covers backward
contamination exactly.

**Sealed holdout.** `holdout_lo` defaults to idx 8199. Any event whose span
reaches idx >= `holdout_lo` is dropped before splitting and counted in the
manifest. The holdout is never a CPCV group. No module specified here reads
it.

---

## 3. Fail-closed policy

Every module raises rather than degrading. Specifically:

- M01 raises `InsufficientData` for `n < 30` observations or zero variance. It
  never returns a Sharpe of 0.0 or a p-value of 0.5 for a degenerate input.
- M02 raises `BarrierDataError` when the bar grid has a gap inside an event's
  horizon, and when the frame is too short for the requested vertical barrier
  at the requested events. A truncated tail resolves as an explicit
  `TRUNCATED` outcome, not as `TIME`.
- M03 raises `SampleWeightError` on a zero-length, reversed, or out-of-range
  span.
- M04 raises `CPCVError` when `k >= M`, `M < 2`, any split has an empty train
  or test set, or purge/embargo consume a group entirely. **It never falls back
  to ordinary K-fold.**

---

## 4. Negative-control suite

All controls run on **synthetic** data with fixed seeds. The real-data
matched-coverage and matched-timestamp baselines already exist in
`tools/swing_hypothesis_walkforward.py` and running them is Phase A′
evaluation, which G1 forbids; the synthetic versions test the apparatus, which
is what G1 is for.

| # | Control | Construction | What it must not do |
|---|---|---|---|
| NC1 | Randomized labels | true bar returns, labels drawn IID from the empirical label marginal | produce expectancy or DSR significance |
| NC2 | Time-shuffled signal | signal permuted, marginal distribution preserved exactly | rank above its own unshuffled null |
| NC3 | Random entry, matched coverage | entry count and per-fold coverage matched to a reference event set, timestamps random | show positive expectancy |
| NC4 | Random direction, matched timestamps | reference entry timestamps kept, side drawn Bernoulli(0.5) | show positive expectancy |
| NC5 | Feature permutation | one feature column permuted within the CPCV training block | improve any CPCV path metric |
| NC6 | Synthetic IID zero-drift | GBM with `mu = 0`, realistic sigma, full M02→M03→M04→M01 pipeline | produce significant DSR |

**Replications and seeds.** `R = 500` replications for the DSR-based controls
(NC1, NC2, NC5, NC6), `R = 200` for the expectancy-based baselines (NC3, NC4).
Master seed `20260817`; replication `i` uses `numpy.random.default_rng([20260817, i])`.
Seeds are recorded in the manifest and the run is byte-reproducible.

**Nominal α = 0.05**, one-sided, throughout.

### 4.1 Pass/fail thresholds — frozen

**T1 — false-positive rate.** For each DSR-based control, the empirical share
of replications with `dsr >= 0.95` must satisfy

    FPR <= 0.075   (R = 500)

Derivation, recorded so the number is not arbitrary: under a correct apparatus
`FPR ~ Binomial(500, 0.05)/500`, whose standard error is
`sqrt(0.05*0.95/500) = 0.00975`. A one-sided 99% bound is
`0.05 + 2.326*0.00975 = 0.0727`, rounded up to 0.075. A correct apparatus
fails T1 with probability < 1%; an apparatus with a true FPR of 0.10 fails it
with probability > 97%.

**T2 — null expectancy is centred on zero.** For each expectancy-based control
(NC3, NC4), the bootstrap 95% CI of the mean null expectancy (in R, net of the
P1 cost model) must **contain zero**. A CI strictly above zero is the
apparatus manufacturing edge and is an immediate G1_FAIL. A CI strictly below
zero is *not* a failure — costs are real and a random entry paying costs
should lose — but it must be reported.

**T3 — no directional bias.** For NC4 the share of replications with positive
mean expectancy must lie inside `0.5 ± 3*sqrt(0.25/200) = 0.5 ± 0.106`, i.e.
`[0.394, 0.606]`. A random-direction control drifting outside this band means
the sign convention in the cost or label path is asymmetric.

**T4 — CPCV path distribution is centred on zero.** For NC6, the median across
CPCV paths of the per-path Sharpe, aggregated over replications, must have a
95% CI containing zero, and the share of replications whose *best* path clears
`dsr >= 0.95` must satisfy T1. The second half is the one that matters: taking
the maximum over paths is exactly how a backtest lies, and the deflation must
absorb it.

**T5 — no false ranking superiority.** Two independent null strategies are
ranked against each other by CPCV mean path Sharpe over `R = 500`
replications. The winner's share must lie in `0.5 ± 3*sqrt(0.25/500) =
[0.433, 0.567]`, and the share of replications where the winner's advantage is
reported as significant must satisfy T1.

**T6 — leakage is structurally zero, not statistically small.** Across every
split of every replication, the count of (train event, test event) pairs whose
spans intersect must be **exactly 0**. This is not a threshold; any non-zero
count is G1_FAIL.

**T7 — raw vs effective sample size.** For overlapping spans the reported
effective count must be strictly less than the raw count, and for
non-overlapping spans exactly equal. Both numbers appear in every control's
output. A control reporting only the raw count is a specification violation.

---

## 5. G1 verdict rule

Exactly one of `G1_PASS` / `G1_FAIL` / `G1_INDETERMINATE`.

`G1_PASS` requires **all** of:

1. M01, M02, M03, M04 targeted tests pass, including every anti-look-ahead and
   fail-closed case listed in §3.
2. T1–T7 all hold.
3. No sealed or frozen downstream artifact is opened, overwritten or
   re-interpreted. `validation/cost_sensitivity.py` still hashes to
   `1cd4db9fe8e8`; `reports/c45/BASELINE.md`, `reports/c50/P1_TASK.md` and
   `reports/c50/ARCHITECTURE.md` are unmodified.
4. The full test suite is green and no existing strategy output changes.

`G1_FAIL` if any control shows the apparatus manufacturing edge — a T2 CI
strictly above zero, a T1 breach, or any T6 leak. The response to a G1_FAIL is
to fix the apparatus, never to adjust a strategy parameter and never to move a
threshold in this document.

`G1_INDETERMINATE` if a control cannot be run at the specified replication
count or geometry, or if a stop condition in the G1 task fires. Indeterminate
is an outcome, not an error, and it blocks Phase A′ exactly as a failure does.

---

## 6. Out of scope, recorded so it is not forgotten

- **The trial ledger (ARCHITECTURE.md §6.1) is not built here.** M01 accepts
  `n_trials` as an argument and cannot verify it. Until the ledger exists, any
  DSR quoted against a hand-counted trial number is only as honest as that
  count. This is the next blocker after G1 and is why §5 forbids using M01 as
  a selector yet.
- PBO (`pbo(...)`) is implemented and tested as a primitive but is not part of
  any G1 threshold: it needs a real trial population to be meaningful.
- M02's barrier multipliers are **not** chosen here. They are two tunable
  parameters and the most obvious overfitting surface in the module; freezing
  them requires the trial ledger and belongs to Phase A′.
- Sequential bootstrap (M03) is implemented and tested; no G1 threshold
  depends on it.

---

## 7. Amendments

Amendments exist so that nothing above is ever edited silently. Each records the
original text, what was wrong with it, and what caught it. **No amendment may
weaken a threshold in §4.1** — those are frozen against the results, which is
the entire methodological content of this document. Amendments correct
*conventions* and *readings*, and only on demonstrated defect.

### A1 — the same-bar tie convention was asymmetric (2026-08-17)

§2 originally read: *"When both barriers fall inside one bar's range the outcome
is **SL** ... For a long that is the stop, so the pessimistic branch is taken
deliberately."* The implementation matched: resolve to LOWER, unconditionally.

That is wrong for shorts. LOWER is the stop only for a long; for a short it is
the target. The rule described as conservative was therefore handing every short
a free win on every ambiguous bar.

**Caught by NC4.** Random-direction entries with matched timestamps came out at
−0.22 R for longs against −0.01 R for shorts. The asymmetry cancels almost
exactly in the pooled mean — NC4's overall gross expectancy was −0.002, which
looks like a clean null — so no aggregate statistic could have seen it. Only the
side-by-side comparison T3 asks for exposed it. This is precisely the failure
mode T3 was written to catch, and it is the strongest single argument for having
frozen the controls before the code.

**Corrected to:** the tie resolves to the adverse barrier for the position's
side. Direction-neutral events keep LOWER, as a stated deterministic choice
rather than as conservatism.

The fix is to the apparatus, not to any threshold or strategy parameter, which is
what §5 requires of a failed control.

### A2 — T3 is evaluated on gross expectancy (2026-08-17)

T3 says "the share of replications with positive mean expectancy" without
qualifying the quantity. T2 explicitly qualifies its own as "net of the P1 cost
model"; T3 does not.

T3 is evaluated on **gross**. Under the net reading a ~0.11 R round-trip cost
drives the share of positive replications to zero for *every* apparatus, correct
or broken, so the band `[0.394, 0.606]` would be unpassable by construction and
the threshold would test nothing. Gross is the only reading under which T3
measures what its own closing sentence claims to measure — whether the sign
convention is asymmetric. Costs are identical for both sides, so they cannot
create an asymmetry, only bury it.

Both figures are reported. The band binds to gross. This is a reading of an
ambiguous sentence, not a relaxation: the gross reading is the one that can fail,
and it did — see A1.

### A3 — PROPOSED, NOT ADOPTED: replace T3's statistic with the long/short gap (2026-08-17)

**Status: proposal. It does not gate. T3 as frozen is the binding criterion and
it FAILS.** Adopting this would be the one thing the freeze exists to prevent —
turning a failing frozen criterion into a passing one after the results were
seen. An independent audit called an earlier draft of this section a post-hoc
relaxation, and on the procedural point it was right: the code that is failing a
criterion cannot be the judge of whether the criterion is defective or merely
inconvenient. Only the project owner can adopt this.

The consequence of leaving it unadopted is recorded in `G1_RESULT.md`: the
verdict is **G1_INDETERMINATE**, which blocks Phase A′ exactly as a failure
does.

The evidence for the proposal follows, so that the decision can be made on it.

**T3 as written** requires the share of replications with positive mean gross
expectancy to lie in `[0.394, 0.606]`. Measured at the frozen `R = 200`:

| apparatus | T3 as written (share) | long/short gap (R) |
|---|---|---|
| with the A1 defect | **0.0000** | −0.21 |
| after the A1 fix | **0.0350** | +0.0114, CI [−0.0041, +0.0287] |

Two facts follow, and both are measurements rather than arguments.

1. **The as-written statistic has no power for the property T3 names.** It read
   0.0000 with a real, severe sign-convention defect present and 0.0350 with it
   fixed — never inside the band in either state, and barely moved by the very
   asymmetry its closing sentence defines as the failure condition.
2. **It appears to be unpassable by any apparatus that uses M02's labels** —
   stated as analysis, not as a proof. M02's tie rule is a deliberate pessimistic
   bias (`barrier_bias_diagnostic`: −0.057 R gross per event on symmetric
   barriers, from an 11.2% tie rate). That bias is common to both sides, so the
   gross null is systematically negative and the share of positive replications
   sits near zero regardless of whether the code is correct.

   The counter-argument, which a reader should weigh: this reasoning is offered
   by the same work that is failing the criterion, and "the test is impossible"
   is what a failing implementation would also say. It could be checked
   independently — a labelling rule that resolved ties by coin flip would centre
   the null and make T3-as-frozen passable — and that check has not been run,
   because building an alternative labeller to rescue a threshold is itself the
   behaviour the freeze exists to discourage.

**The proposed statistic:** the bootstrap 95% CI of the per-replication
difference `mean gross R (long) − mean gross R (short)` must contain zero. That
is exactly the property T3's own text describes — longs and shorts must be mirror
images — and it has demonstrated power in the correct direction: it failed with
the A1 defect present and passes with it fixed.

**The argument against adopting it, which is the stronger one.** The ordering was:
run the controls, watch T3 fail, fix a genuine defect, watch T3 fail again,
then replace the statistic. The last step came after seeing that the fix did not
rescue the criterion. That is post-hoc, and "the statistic is defective" is also
precisely what someone rationalising would say. The freeze is worth nothing if the
party being measured can retire an inconvenient measurement.

**What is true either way.** The property T3 targets — directional symmetry of
the label and cost path — is demonstrated by the gap CI regardless of whether A3
is adopted, and the A1 defect it exposed is fixed. What is *not* settled is
whether the frozen criterion can be retired. That is the owner's call.

**Three ways this could be resolved**, none of which is for the implementer to
pick:

1. **Adopt A3.** The verdict becomes G1_PASS. Requires accepting a statistic
   changed after seeing results, on the evidence above.
2. **Keep T3 as frozen.** The verdict stays G1_INDETERMINATE and A′ stays
   blocked until the criteria are re-frozen and the controls re-run — the clean
   but slow path.
3. **Re-freeze T3 for a future gate** and record G1 as indeterminate on this one
   line, with the apparatus properties taken as demonstrated. This is the
   recommendation: it neither rewrites history nor discards work.

### A3 — owner decision (2026-08-19): option 3, and A3 stays NOT ADOPTED

The three-way choice above was put to the project owner and resolved:
**option 3**. Nothing in A3 above is edited; this records what was decided
about it.

- **A3 is not adopted for G1.** The G1 verdict remains **`G1_INDETERMINATE`**,
  and T3 as frozen remains recorded as failed. Option 1 was declined on the
  argument A3 itself states against it.
- **The apparatus properties are accepted as demonstrated** — leakage
  controls, purge/embargo/CPCV, sample weights and uniqueness, DSR
  calibration, negative-control FPR — which is what makes Phase A′ eligible
  without a re-run.
- **T3 must be re-frozen prospectively** before any future gate that depends
  on the directional-symmetry property, and the replacement criterion **must
  not be chosen using Phase A′ results**.
- The corrected semantics apply to gates declared **after 2026-08-18**
  only — the cutoff frozen in `TRIAL_REGISTRY_SPEC.md` §9 item 3 before
  this decision was taken. The decision is dated 2026-08-19 and does not
  move it.

Governed record: `reports/c50/G1_T3_DECISION.md`, machine-readable in
`reports/c50/g1_governance.json`.
