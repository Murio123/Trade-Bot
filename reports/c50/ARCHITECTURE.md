# C5.0 — Research Architecture Specification (FROZEN)

Status: **specification only**. No code, no experiments, no implementation.
Frozen at: 2026-08-11.
Scope: the research apparatus for deciding whether the swing engine has an edge.

## 0. What this document is and is not

It is a frozen record of design decisions taken while they were fresh, so that
implementation months from now does not silently re-derive them differently.

It is **not** a commitment to build all 15 modules. Module M00 alone may
invalidate half of them. Each module carries an explicit kill condition, and
the roadmap has re-evaluation gates between phases.

**This document must not influence current code.** Nothing in `bot/`,
`signal_engine/`, `volatility/`, or `tools/` may import from, reference, or be
restructured toward anything specified here until its module is separately
approved and implemented. The document is a map, not a dependency.

### 0.1 The binding constraint

Every priority in this document follows from one measured fact:

> Effective sample size is ~338 observations (4062 rows / 12-bar overlap).
> The only surviving measured edge in the project is +0.13 Spearman on
> volatility *ranking*, roughly 2.4 SE from zero.

Consequences that shape the whole architecture:

- Adding features cannot help before the measurement apparatus is trustworthy.
- Any slicing of the data (by regime, by year, by exchange) burns statistical
  power that the project does not have. Slices must be coarse and pre-declared.
- Most modules will honestly return `INSUFFICIENT_DATA`. That is a correct
  result, not a failure, and the architecture treats it as a first-class output
  rather than an error.

### 0.2 No fabricated numbers

No module in this specification is assigned an expected Sharpe delta. Any such
number would be invented. Priority is ordinal, with stated reasoning.
This mirrors the D1.2 decision to remove unearned confidence from the UI.

---

## 1. Module index

| ID | Module | Tier | Priority | Gate |
|----|--------|------|----------|------|
| M00 | `validation/cost_sensitivity.py` | viability | **Critical — Sprint 0** | — |
| M01 | `validation/deflated_sharpe.py` | apparatus | Critical | G1 |
| M02 | `labeling/triple_barrier.py` | apparatus | Critical | G1 |
| M03 | `labeling/sample_weights.py` | apparatus | Critical | G1 |
| M04 | `validation/cpcv.py` | apparatus | Critical | G1 |
| M05 | `models/meta_labeling.py` | alpha | Critical | G2 |
| M06 | `regime/hmm_regime.py` | conditioning | High | G3 |
| M07 | `validation/invariant_alpha.py` | apparatus | Critical | G3 |
| M08 | `validation/causal_stress_test.py` | apparatus | Conditional | G3 |
| M09 | `forecast/bayesian_edge.py` | decision | High | G3 |
| M10 | `forecast/probabilistic.py` | decision | High | G3 |
| M11 | `derivatives/term_structure.py` | alpha | High | G4 |
| M12 | `microstructure/liquidation_model.py` | alpha | High | G4 |
| M13 | `research/analog_search.py` | presentation | Medium | G4 |
| M14 | `microstructure/order_flow.py` | alpha | Medium | deferred |
| M15 | `risk/adaptive_sizing.py` | decision | Low | deferred |

Retired during design: `validation/feature_stability.py` — its scope
(permutation importance stability, PSI drift) is a strict subset of M07.
Two modules answering overlapping questions would be the parallel
implementation the project standards forbid.

---

## 2. Dependency graph

```
                            M00 cost_sensitivity
                                    │
                            [GATE G0: is there a bar to clear?]
                                    │
        ┌───────────────┬───────────┴───────────┬──────────────┐
        │               │                       │              │
   M02 triple_barrier   │                  M01 deflated_sharpe │
        │               │                       │              │
        └──────► M03 sample_weights ────────────┤              │
                        │                       │              │
                        └────────► M04 cpcv ◄───┘              │
                                    │                          │
                            [GATE G1: apparatus trustworthy?]  │
                                    │                          │
                        ┌───────────┴───────────┐              │
                        │                       │              │
                  M05 meta_labeling       M06 hmm_regime ◄─────┘
                        │                       │
                    [GATE G2]                   │
                        │                       │
                        └───────────┬───────────┘
                                    │
                            M07 invariant_alpha
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
          M08 causal_stress   M09 bayesian_edge  M10 probabilistic
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                            [GATE G3: any STABLE_EDGE?]
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
          M11 term_structure  M12 liquidation  M13 analog_search
                                    │
                            [GATE G4: new data justified?]
                                    │
                        ┌───────────┴───────────┐
                   M14 order_flow        M15 adaptive_sizing
                        (deferred)          (deferred)
```

### 2.1 Edge semantics

Two distinct kinds of dependency, never conflated:

- **hard** — the downstream module cannot run without the upstream artifact.
- **validation** — the downstream module *may* run, but its output is not
  admissible as evidence without the upstream module. Running it anyway
  produces a number, and that number is inadmissible.

| From | To | Kind | Reason |
|------|-----|------|--------|
| M02 | M03 | hard | weights are computed from label spans |
| M02 | M04 | hard | purge width is derived from barrier geometry |
| M03 | M04 | hard | folds must carry uniqueness weights |
| M01 | M04 | validation | a CPCV Sharpe distribution undeflated is inadmissible |
| M01 | M05,M07,M08 | validation | every reported metric is deflated or inadmissible |
| M04 | M05,M07 | hard | all out-of-sample estimation goes through CPCV |
| M02 | M05 | hard | meta-labels are triple-barrier outcomes |
| M06 | M07 | hard | regime slices are the HMM posterior |
| M07 | M08 | hard | M08 consumes only `STABLE_EDGE` candidates |
| M06 | M09 | hard | posteriors are hierarchical over regimes |
| M07 | M11,M12 | validation | new features enter the same classifier |
| M09,M10 | M15 | hard | sizing consumes a posterior, not a point estimate |

**Cycle check:** the graph is a DAG. The one apparent cycle — M07 classifies
features, and M11/M12 add features that M07 must then classify — is resolved by
gate G4: M11/M12 are a *second pass* through M07, not a back-edge. The
classifier is frozen before the second pass and its trial count incremented.

---

## 3. Execution order

Strictly sequential across gates; parallel within a phase is permitted.

| Phase | Modules | Exit gate |
|-------|---------|-----------|
| Sprint 0 | M00 | **G0** — does any candidate edge exceed round-trip cost? |
| A | M02 → M03 → M04 → M01 | **G1** — negative controls pass |
| A′ | re-evaluate H1, H2, confluence, C4.3e under the new apparatus | — |
| B | M05 | **G2** — meta-label lift admissible after deflation |
| C | M06 → M07 → (M09, M10 parallel) → M08 | **G3** — any `STABLE_EDGE`? |
| D | M11 → M12 → M13 | **G4** — new data justified |
| E | M14, M15 | deferred indefinitely |

### 3.1 Gate definitions

Gates are **stop conditions**, not checkpoints. A failed gate halts the phase
and forces re-evaluation of everything downstream.

**G0 — viability.** Round-trip cost for the target instrument is quantified
under realistic fees, slippage, and funding. If no measured or plausibly
attainable edge clears it, phases A–E are cancelled and the project pivots to
report-only forecasting (the C4.4/C4.5 line) permanently.

**G1 — apparatus trustworthy.** The negative-control suite (§6.3) runs clean:
synthetic no-edge features are not reported as significant above the nominal
false-positive rate. If G1 fails, no downstream result is admissible.

**G2 — meta-labeling.** Precision lift over the unfiltered signal is positive
with a deflated CI excluding zero on CPCV paths. Failing G2 does not
automatically stop phase C, but it removes the only near-term path to improved
live results and must be reported as such.

**G3 — any structural edge.** At least one feature or signal classified
`STABLE_EDGE` by M07 and `MECHANISM_CONFIRMED` by M08. If zero, phase D is not
started: adding data sources to a system with no established edge expands the
search space without justification.

**G4 — new data justified.** G3 passed, and the marginal cost of the new data
source is stated against a pre-declared hypothesis for it.

### 3.2 Phase A′ is mandatory

After the apparatus exists, the already-decided results are re-run through it:
H1 (rejected), H2 (rejected), the confluence score (never validated), and
C4.3d/C4.3e (+0.13, accepted). Two outcomes are possible and both are
informative:

- a rejection was made on insufficient grounds and the hypothesis is reopened;
- the +0.13 does not survive deflation and the C4.4/C4.5 ranker loses its
  stated justification.

The second outcome must be accepted if it occurs. It is written here, in
advance, so that it cannot later be argued away.

---

## 4. Module specifications

Each entry: purpose, inputs, outputs, interface shape, assumptions, invariants,
prerequisites, priority reasoning, kill condition.

Interfaces are given as **shapes, not signatures** — deliberately, so
implementation retains freedom. They fix what crosses a boundary, not how.

---

### M00 — `validation/cost_sensitivity.py`

**Purpose.** Establish the profitability threshold any strategy must clear,
before building apparatus to measure whether it does.

**Inputs.** Historical OHLCV; fee schedule (maker/taker, tier); funding history;
a slippage model (initially: fixed bps by order size, escalating to depth-based
if L2 becomes available); the existing signal history.

**Outputs.** A sensitivity surface: net expectancy as a function of
(cost_bps × execution_delay_bars × holding_period). Plus a single reported
scalar: **minimum gross edge required for break-even** at current costs.

**Interface shape.**
- `cost_model(order_notional, side, regime) -> bps`
- `net_expectancy(gross_returns, cost_model, delay_bars) -> distribution`
- `break_even_edge(horizon_bars, turnover) -> float`

**Assumptions.**
- Taker execution by default. Maker fills are not assumed without a fill model.
- Funding is paid at the exchange schedule while a position is held.
- Delay is measured in bars, not seconds — the engine is bar-close driven.

**Invariants.**
- Costs are never optional in any downstream backtest. Once M00 exists, a gross
  number reported without it is a specification violation.
- The cost model is versioned and hashed like a frozen artifact.

**Prerequisites.** None. This is the only module with no dependencies.

**Why this priority.** It is the cheapest module and the only one that can
terminate the entire program before investment. A 4h swing on BTC perps pays
roughly 0.1% per round trip in fees alone, before slippage and funding. A
ranking edge of +0.13 Spearman is not obviously worth more than that. Answering
this takes a day; the remaining 14 modules take a quarter. Fail fast.

**Kill condition.** None — this module runs unconditionally.

---

### M01 — `validation/deflated_sharpe.py`

**Purpose.** Correct every performance estimate for the number of hypotheses
tried, sample length, and return distribution shape.

**Inputs.** Return or IC series per trial; the **trial ledger** (§6.1); sample
length; observed skew and kurtosis.

**Outputs.** Deflated Sharpe Ratio; probability of backtest overfitting (PBO);
minimum track record length for the observed estimate.

**Interface shape.**
- `deflated_sharpe(returns, n_trials, skew, kurtosis) -> (dsr, p_value)`
- `pbo(in_sample_ranks, out_of_sample_ranks) -> float`
- `min_track_record_length(sharpe, target, skew, kurtosis) -> int`

**Assumptions.**
- The trial count is honestly recorded. This is the single largest failure mode
  and is addressed structurally by §6.1, not by discipline.
- Returns are IID after uniqueness weighting (M03). Without M03 this assumption
  is false by a factor of ~12.

**Invariants.**
- Every metric published anywhere in the research pipeline passes through this
  module or is marked `UNDEFLATED — INADMISSIBLE`.
- Trial count is monotonically non-decreasing. It is never reset, never
  recounted downward, and abandoned experiments still count.

**Prerequisites.** Trial ledger exists and is retroactively populated with the
project's history: H1, H2, confluence variants, the `/backtest` threshold grid,
the four profiles, C4.3a–e. Best-effort reconstruction from git history, with
unknowns rounded **up**.

**Why this priority.** It changes the interpretation of every past and future
result, at negligible cost. Without it the research process systematically
manufactures false discoveries and cannot detect that it is doing so.

**Kill condition.** None.

---

### M02 — `labeling/triple_barrier.py`

**Purpose.** Replace fixed-horizon labels with path-dependent outcomes.

**Inputs.** OHLCV; volatility estimate at the decision bar; barrier
configuration (upper/lower multipliers, vertical horizon).

**Outputs.** Per-bar: which barrier hit first (`TP`/`SL`/`TIME`), the realized
return at the touch, the touch bar index, and the label span `[t, t_touch]`.

**Interface shape.**
- `apply_barriers(ohlcv, events, sigma, multipliers, vertical) -> labels_frame`
- `label_span(labels_frame) -> array of (start_idx, end_idx)`

**Assumptions.**
- Intrabar path is unknown from OHLC alone. When both barriers fall inside one
  bar's range, the outcome is **conservative**: the stop is assumed hit first.
  This is a deliberate pessimistic bias and is documented, not hidden.
- Barriers scale with contemporaneous volatility, not fixed percentages.

**Invariants.**
- Barrier multipliers are frozen before any run, as C4.1 froze the volatility
  label. They are two new tunable parameters and are the most obvious
  overfitting surface in this module.
- `end_idx > start_idx` always; a zero-length span is a bug, not a fast fill.
- The label of bar `u` is knowable only at `u + span(u)`. Every point-in-time
  computation admits only `u` with `end_idx(u) <= t`.

**Prerequisites.** Frozen barrier configuration, recorded in the trial ledger.

**Why this priority.** The current fixed-horizon label answers a question
nobody trades: "what was the range 12 bars later, unconditionally." Real
outcomes are path-dependent. Every downstream module inherits this defect.

**Kill condition.** None.

---

### M03 — `labeling/sample_weights.py`

**Purpose.** Correct for overlapping labels inside training, not only inside
evaluation.

**Inputs.** Label spans from M02.

**Outputs.** Per-observation uniqueness weight; average uniqueness; a
sequential-bootstrap sampler.

**Interface shape.**
- `concurrency(spans, n_bars) -> array`
- `uniqueness_weights(spans) -> array`
- `sequential_bootstrap(spans, size, rng) -> indices`

**Assumptions.**
- Weight is inversely proportional to concurrent label count.
- Optional time decay is **off by default**: it is a modelling choice, not a
  correction, and must be justified separately if enabled.

**Invariants.**
- Weights are strictly positive and sum to the effective sample size, which is
  reported alongside the nominal row count **everywhere**. A metric quoted
  against 4062 rows when the effective count is 338 is a specification
  violation.
- No training run is permitted with uniform weights on overlapping labels.

**Prerequisites.** M02.

**Why this priority.** The project already discovered the ~12× overlap on the
evaluation side and works around it with block bootstrap. Training still treats
all rows as independent, so the model believes it has twelve times more
information than it does. This is a correction to fitting, not just reporting.

**Kill condition.** None.

---

### M04 — `validation/cpcv.py`

**Purpose.** Produce a *distribution* of out-of-sample performance rather than
a single number.

**Inputs.** Feature matrix; labels and spans (M02); weights (M03); group count
`M`; test group count `k`; purge width; embargo width.

**Outputs.** Per-path metrics across `C(M,k)` combinations; the induced
distribution; the count of paths for the trial ledger.

**Interface shape.**
- `combinatorial_splits(n, M, k, purge, embargo) -> iterator of (train, test)`
- `backtest_paths(splits, fit_fn, score_fn) -> paths_frame`
- `path_distribution(paths_frame) -> distribution`

**Assumptions.**
- Purge width equals the maximum label span, not a fixed 12 bars — with M02,
  spans vary.
- Embargo is applied on the right edge of every test group.
- **Paths are not independent.** They share training data, so the distribution
  understates variance. Reported explicitly with every output.

**Invariants.**
- No test observation's label span may intersect any training observation's
  span. This is the single correctness property of the module and gets a
  dedicated adversarial test.
- The sealed holdout (idx 8199+) is excluded from every split. It is not a
  CPCV group and never becomes one.
- `M`, `k`, purge, and embargo are frozen before the first run.

**Prerequisites.** M02, M03. Existing `build_folds` / `partition_rows` geometry
is correct and is extended, not replaced.

**Why this priority.** Three folds yield three numbers, from which no
distribution can be formed; C4.3e had to recover one post hoc via block
bootstrap. CPCV extracts many paths from the same data and makes the
uncertainty visible by construction.

**Kill condition.** None.

---

### M05 — `models/meta_labeling.py`

**Purpose.** Leave the primary signal's *direction* alone; learn whether a
given signal will be profitable.

**Inputs.** Primary signal events (existing confluence cascade output);
triple-barrier outcomes on those events only (M02); secondary features
(confluence components, regime posterior, C4.4 volatility rank, derivatives,
session/time-of-day).

**Outputs.** `P(profitable | signal fired)`; a precision/recall curve over the
threshold; the filtered signal stream.

**Interface shape.**
- `build_meta_dataset(primary_events, barrier_outcomes, features) -> frame`
- `fit_meta(frame, weights, cv) -> model`
- `p_success(model, features_now) -> float`

**Assumptions.**
- The primary model's direction call is taken as given and is never overridden.
  M05 filters and sizes; it does not flip.
- Training rows exist only where the primary fired, so the sample is small and
  selection-biased. Both are stated with every result.

**Invariants.**
- **M05 must never change a signal's side.** Structurally enforced: the
  interface returns a probability, never a direction.
- Its output is a probability, never a "confidence score" for display. The
  D1.2 removal of fake confidence stands.
- Precision improves at the cost of recall; both are always reported together.

**Prerequisites.** M02, M04; M01 as validation. A signal history long enough to
have any positive class at all — checked before fitting, not after.

**Why this priority.** Highest-value alpha work available, because it does not
require solving direction. Direction was attempted twice (H1, H2) and rejected
twice. "Will this specific setup work?" is a strictly easier question, uses the
machinery already built, and addresses the system's actual failure mode: false
positives, not missed trades.

**Kill condition.** Positive-class count below the pre-declared minimum → the
module is not fitted and reports `INSUFFICIENT_DATA`. Fitting anyway would
produce a model of noise wearing the clothes of a filter.

---

### M06 — `regime/hmm_regime.py`

**Purpose.** Probabilistic market-state classification to condition everything
downstream.

**Inputs.** Realized volatility; return autocorrelation; volume; funding term
structure (M11 when available, degraded feature set before that).

**Outputs.** Per-bar posterior over `K` states; the transition matrix; state
descriptions.

**Interface shape.**
- `fit_regime(features, K, seed) -> model`
- `posterior(model, features_upto_t) -> probability vector`
- `state_labels(model) -> descriptions`

**Assumptions.**
- `K` is selected by BIC **on the training region only** and then frozen. It is
  a hyperparameter chosen from data and therefore a known overfitting surface.
- Posteriors are forward-filtered at inference: **no smoothing**, which would
  use future data.

**Invariants.**
- Inference is causal. A smoothed posterior may appear in research plots and
  never in a point-in-time computation.
- Output is a distribution, never a hard label. Downstream consumers weight by
  probability; the boundary between regimes is where a hard label does the most
  damage.
- Existing heuristic `signal_engine/regime.py` is **not** modified by this
  module. Two regime notions coexist until one is retired deliberately.

**Prerequisites.** Frozen `K`; frozen feature set; frozen seed.

**Why this priority.** Sharpe is regime-dependent almost universally; pooling
across regimes hides both good and bad. It was raised from High to Critical
during design because M07 cannot run without regime slices.

**Kill condition.** If BIC does not clearly prefer `K > 1`, the module reports a
single-regime world and phase C proceeds without conditioning — a valid and
informative outcome.

---

### M07 — `validation/invariant_alpha.py`

**Purpose.** Separate a structural edge from a historical correlation.

**Inputs.** Per-feature/signal edge estimates; regime posteriors (M06);
volatility regime; calendar year; cross-asset data (ETH, SOL); cross-exchange
data **for derivatives features only**; the pre-registered `mechanism` string
per candidate.

**Outputs.** One of `STABLE_EDGE`, `REGIME_SPECIFIC_EDGE`, `TIME_LIMITED_EDGE`,
`INSUFFICIENT_DATA`, `LIKELY_SPURIOUS`; the supporting slice table; alpha
half-life slope with CI; the weight policy.

**Interface shape.**
- `slice_edges(feature, slices, cv) -> per-slice estimates`
- `leave_one_regime_out(feature, regimes, cv) -> generalization estimate`
- `alpha_half_life(ic_series) -> (slope, ci)`
- `classify(evidence, mechanism, trial_count) -> classification`
- `weight_policy(classification) -> weight`

**Assumptions.**
- Slicing multiplies the multiple-testing problem. Slices are **pre-declared**,
  **one dimension at a time**, never a 3×3×5 cross-product.
- Minimum cell size: **40 effective observations**. Below it, the cell is
  `INSUFFICIENT_DATA` — never "unstable".
- Cross-exchange testing is near-worthless for price/TA features: BTC perps on
  Binance/Bybit/OKX are ~99% correlated on 4h returns, so it repeats one test
  three times. It is retained **only** for funding, OI, basis, and liquidation
  features, where venues genuinely differ. Cross-**asset** (ETH, SOL) replaces
  it as the independent-data test: same mechanism, different data.
- Alpha half-life is not estimable as a decay curve at n_eff ≈ 338. Only the
  sign and CI of a single slope parameter are estimable. The module reports
  "decay indistinguishable from zero" when that is the truth, and does not fit
  an exponential to noise.

**Invariants.**
- **Default classification is `LIKELY_SPURIOUS`.** Promotion requires evidence;
  the residual is not assumed benign.
- `REGIME_SPECIFIC_EDGE` requires a `mechanism` registered **before** the test
  and hashed with the run config. Without it, a one-slice hit is
  observationally identical to luck and is classified `LIKELY_SPURIOUS`. This
  is an information limit, not a methodological gap — no test on the same data
  can separate the two.
- Slice estimates are deflated by the **total** number of slices examined.
- The last calendar year is reserved to validate the classification itself and
  does not participate in producing it.
- Weight changes are recomputed offline **quarterly**, frozen and hashed
  between recomputations, and appended to a policy ledger so the weights in
  force at any past forecast are recoverable.
- The reweighting rule is itself validated as a strategy (adaptive vs frozen
  weights, same CPCV + deflation). Unvalidated adaptation is not enabled.

**Prerequisites.** M01, M04, M06. A pre-registered mechanism per candidate.
Cross-asset data pipelines for ETH and SOL.

**Why this priority.** It is the module the entire apparatus exists to enable —
the difference between "this worked" and "this works." It absorbed the retired
`feature_stability` scope. It is sequenced after M01/M03/M04/M06 because
stability of an edge cannot be measured before the edge is measured properly.

**Kill condition.** If the negative-control suite classifies synthetic noise as
`STABLE_EDGE` above the nominal rate, the classifier is broken and its verdicts
on real features are void.

**Expected outcome, stated in advance.** Most likely, zero features reach
`STABLE_EDGE` and the majority land in `INSUFFICIENT_DATA`. That is the module
working correctly at n_eff ≈ 338, and it identifies the true constraint — data
volume — rather than feature quality.

---

### M08 — `validation/causal_stress_test.py`

**Purpose.** Verify a candidate edge depends on its hypothesized mechanism.

**Inputs.** `STABLE_EDGE` candidates from M07; the **mechanism DAG** per
candidate; perturbation specifications; execution parameters.

**Outputs.** `MECHANISM_CONFIRMED` / `MECHANISM_WEAKENED` /
`MECHANISM_REJECTED` / `MECHANISM_INCONCLUSIVE`; sensitivity curves with
confidence bands; the paired difference distribution.

**Interface shape.**
- `register_mechanism(candidate, dag) -> hash`
- `perturb(data, variable, magnitude, residualize_against) -> perturbed`
- `paired_delta(edge_fn, base, perturbed, shared_seeds) -> distribution`
- `sensitivity_curve(edge_fn, variable, magnitudes) -> curve with bands`
- `classify(curves, mde) -> verdict`

**Assumptions.**
- **This is sensitivity analysis, not causal inference.** `do(X)` is
  unavailable on observational market data. The docstring must state the
  boundary explicitly. Only execution parameters — delay, cost, size — are
  genuine interventions, because they are under our control.
- Market variables are mutually correlated; volatility, liquidity, funding, OI
  and spread co-move, most strongly in stress. An "unrelated" perturbation
  therefore does not exist naively — perturbation is applied in a
  **residualized** space, orthogonalized against the DAG's mechanism-carrying
  variables.
- Counterfactual replay against an impact model is permitted only with the
  impact model's assumptions published and the verdict's sensitivity to them
  reported.

**Invariants.**
- **Paired comparison, never two independent significance tests.** Same folds,
  same seeds, same block structure; only the perturbation differs. At n_eff ≈
  338 an edge 2.4 SE from zero collapses to non-significance under almost any
  perturbation, so "significant before, not after" would grant
  `MECHANISM_CONFIRMED` to pure noise. The test is on the distribution of Δ.
- Minimum detectable difference is declared before the run. If MDE ≥ the
  observed edge, the verdict is `INCONCLUSIVE` and no other verdict is
  reachable.
- The mechanism DAG is registered and hashed before testing, like M07's
  `mechanism`. Declared after the fact, it is not evidence.
- The cliff criterion ("edge halves at 1σ perturbation of a mechanism
  variable") is pre-declared. Curve shape always finds a post-hoc reading.
- Sensitivity curves always carry confidence bands.
- Any verdict other than `CONFIRMED` downgrades the candidate in M07; a
  non-confirmed mechanism does not retain full weight.

**Prerequisites.** M07 producing at least one `STABLE_EDGE`. A registered DAG.

**Why this priority — and why conditional.** The falsification requirement
(edge must *disappear* when its mechanism is destroyed) is the strongest
discriminator in the whole specification: ordinary validation only asks
"does it still work," never "does it break correctly." But M08 consumes M07's
output, and if M07 yields zero candidates the module has no input. Building it
early would mean writing a consumer for an empty stream.

**Kill condition.** M07 returns no `STABLE_EDGE` → M08 is not built.

---

### M09 — `forecast/bayesian_edge.py`

**Purpose.** Make slow evidence accumulation useful immediately, instead of
binary "not enough data yet."

**Inputs.** The C4.4/C4.5 forecast ledger; regime posteriors (M06); a
**skeptical prior** centered on no edge.

**Outputs.** Posterior over win rate / expectancy per regime; credible
intervals; posterior predictive.

**Interface shape.**
- `update(prior, observations) -> posterior`
- `hierarchical_pool(regime_posteriors) -> pooled`
- `credible_interval(posterior, mass) -> (lo, hi)`

**Assumptions.**
- Beta-Binomial for hit rate; Normal-Inverse-Gamma for expectancy.
- Hierarchical pooling across regimes shrinks thin cells toward the global
  estimate rather than reporting them as extremes.
- The prior is fixed before observing data and published with every result.

**Invariants.**
- Prior is skeptical by construction: at zero observations the posterior says
  "no edge," never "unknown, could be large."
- The posterior is never used to justify a decision the frequentist apparatus
  rejected. It quantifies uncertainty; it does not launder a failed test.
- Ledger reads are append-only and go through `volatility/ledger.py` validation.

**Prerequisites.** M06 for regime conditioning; a populated forecast ledger.
Published prior.

**Why this priority.** C4.3e established that forward confirmation of the +0.13
edge needs ~2830 matured forecasts, roughly 15 months. A binary "wait 15
months" wastes every observation collected in the interim. A posterior gives a
defensible answer at every point: near-prior at 30 observations, meaningfully
shifted at 300.

**Kill condition.** None; degrades gracefully to the prior.

---

### M10 — `forecast/probabilistic.py`

**Purpose.** Predictive distributions instead of point forecasts.

**Inputs.** Model outputs; a calibration set; the target.

**Outputs.** Predictive intervals with coverage guarantees; CRPS; reliability
diagrams.

**Interface shape.**
- `conformal_intervals(model, calibration, alpha) -> interval fn`
- `crps(predicted_distribution, realized) -> float`
- `reliability(predicted, realized, bins) -> diagram`

**Assumptions.**
- Split conformal prediction; the calibration set is held out and consumes data
  the project can barely spare — an explicit, accepted tradeoff.
- Conformal coverage assumes exchangeability, which time series violate. Block
  or adaptive conformal variants are required; naive split conformal is not
  valid here and its use is a specification violation.

**Invariants.**
- Evaluation is by CRPS and calibration, never by MAE on the point estimate.
  C4.3c already rejected the point predictor; re-reporting MAE would relitigate
  a settled question.
- Intervals shown to a user carry their coverage level.

**Prerequisites.** A model worth wrapping — i.e. G2 or G3 passed.

**Why this priority.** With R² near zero, a point forecast is not merely useless
but misleading. Distributions serve both correctness and the project's standing
commitment not to display unearned precision.

**Kill condition.** No admissible model to wrap.

---

### M11 — `derivatives/term_structure.py`

**Purpose.** Positioning signal from basis and funding structure.

**Inputs.** Perpetual funding across venues; quarterly futures prices; spot.

**Outputs.** Basis by expiry; curve slope; cross-venue funding spread;
annualized carry.

**Interface shape.**
- `basis(spot, futures, days_to_expiry) -> annualized`
- `curve_slope(basis_by_expiry) -> float`
- `funding_spread(venues) -> frame`

**Assumptions.**
- Quarterly liquidity is adequate on the venues used; thin books make basis
  noisy and the module reports a liquidity flag alongside every value.
- Basis is point-in-time and never interpolated across a roll.

**Invariants.**
- Roll dates are handled explicitly; a discontinuity is not smoothed away.
- All values are point-in-time; no future expiry information leaks backward.
- Enters M07 classification as a new candidate, incrementing the trial count.

**Prerequisites.** G4. A multi-venue derivatives data pipeline.

**Why this priority.** Best cost-to-value ratio among new data sources: cheap to
obtain, and positioning expressed through basis is harder to fake than volume.
It also feeds M06's feature set.

**Kill condition.** G3 fails → not built.

---

### M12 — `microstructure/liquidation_model.py`

**Purpose.** Model forced-selling cascades.

**Inputs.** Open interest by venue; funding; the liquidation feed where
published; price; existing `analyzer/liquidation_map.py` tiers as a baseline.

**Outputs.** Estimated distribution of liquidation prices; expected cascade
amplitude conditional on breaching the nearest cluster; a fragility score.

**Interface shape.**
- `liquidation_density(oi, funding, price_history) -> density over price`
- `cascade_amplitude(density, breach_level, impact_model) -> expected move`
- `fragility(density, current_price) -> float`

**Assumptions.**
- True leverage distribution is unobservable and is inferred indirectly from OI
  and funding dynamics. The inference is a model, and its error is reported.
- Exchange liquidation feeds are incomplete and delayed — Binance in particular
  publishes a throttled subset. Treated as a biased sample, never as ground
  truth.
- Cascade modelled as a branching process: a trigger at level L moves price by
  ΔP, which may reach the next cluster.

**Invariants.**
- Point-in-time only; a cluster estimate never uses OI recorded after the bar.
- Enters M07 classification, incrementing the trial count.
- Replaces the fixed-tier heuristic only after passing M07; both must not run
  in parallel indefinitely.

**Prerequisites.** G4. Multi-venue OI and liquidation data.

**Why this priority.** Forced liquidation is mechanical rather than
informational, making it one of the few crypto settings where predictability
arises from market structure instead of from anticipating others' beliefs — and
one of the few where the required data is actually obtainable at this scale.

**Kill condition.** G3 fails → not built.

---

### M13 — `research/analog_search.py`

**Purpose.** Empirical distribution of forward outcomes from historical
analogs, conditioned on current state.

**Inputs.** Normalized feature space; current state vector; regime filter (M06);
label spans for purging (M02).

**Outputs.** k nearest historical analogs; the empirical distribution of their
forward outcomes; a similarity-quality measure.

**Interface shape.**
- `find_analogs(state, history, k, regime_filter, purge) -> indices`
- `outcome_distribution(analog_indices, labels) -> distribution`
- `analog_quality(distances) -> float`

**Assumptions.**
- Dimensionality reduction is required before the neighbor search: nearest
  neighbors in 19 dimensions are dominated by the curse of dimensionality.
- Analogs from a different regime mislead; the regime filter is mandatory, not
  optional.

**Invariants.**
- Purging by label span is mandatory — the nearest neighbor of bar `t` is
  usually `t±1`, whose outcome overlaps `t`'s. Without purging the module
  reports the target as its own prediction.
- Output is a distribution, never a single "similar case."
- Existing `analyzer/historical.py` is superseded, not duplicated.

**Prerequisites.** M02, M06. A dimensionality-reduction choice frozen in advance.

**Why this priority.** Mostly a communication instrument: it makes uncertainty
legible without asserting predictive power. Same construction as the C4.4a
scenario table, conditioned on current state instead of on category.

**Kill condition.** Analog quality below a pre-declared threshold → reports
"no comparable historical state," which is a legitimate answer.

---

### M14 — `microstructure/order_flow.py` *(deferred)*

**Purpose.** Order flow imbalance and trade-size structure.

**Inputs.** Trade-level or L2 book data.

**Outputs.** OFI at multiple horizons; trade-size distribution; aggressor
ratio; Kyle's lambda.

**Assumptions.** Requires tick or L2 storage the project does not have. OFI
predictive power decays over minutes, while the engine's horizon is 4h.

**Why deferred.** The idea is sound and standard at funds — but their horizon is
not this project's. The binding constraint is data cost, and the horizon
mismatch means the payoff is unclear even after paying it. Listed for
completeness, not scheduled.

---

### M15 — `risk/adaptive_sizing.py` *(deferred)*

**Purpose.** Fractional Kelly sizing from the M09 posterior.

**Inputs.** Posterior over expectancy; cost model (M00); risk limits.

**Assumptions.** Kelly requires a reliable edge estimate. Sizing multiplies the
edge — with an edge indistinguishable from zero, it multiplies noise.

**Invariants.** Fractional Kelly only (≤ 0.25), hard caps independent of the
posterior, DRY_RUN unchanged.

**Why deferred, deliberately.** Technically the correct answer to position
sizing, and building it before M01–M07 would mean optimizing a multiplier
against an unknown multiplicand. Recorded here so its absence is visibly a
decision rather than an oversight.

---

## 5. Cross-cutting invariants

Binding on every module. A violation is a defect regardless of results.

1. **Point-in-time.** No computation admits data unavailable at the decision
   bar. The label of bar `u` is knowable only at `u + span(u)`.
2. **Sealed holdout.** idx 8199+ is not read by any module in this document.
   Reading it is a one-way action that cannot be undone by discarding results.
3. **Freeze before running.** Every parameter that could be tuned to an
   observed result is fixed and hashed beforehand.
4. **Trial ledger is append-only.** Abandoned experiments count. See §6.1.
5. **Effective, not nominal.** Every sample size reported is effective.
6. **Deflate or mark inadmissible.** No exceptions, including for results that
   look obviously good.
7. **Default to the null.** Unclassified means spurious; unconfirmed means not
   confirmed; insufficient data is an outcome, not an error.
8. **Symbol-parameterized.** No module hardcodes BTCUSDT. Cross-asset validation
   in M07 depends on this, and `reports/d13/multi_asset_blockers.md` already
   records what breaks otherwise — notably `db.open_trades()`
   (`database.py:391`), which is a correctness bug the moment a second asset
   exists.
9. **Research does not touch production.** Nothing under `bot/` or
   `signal_engine/` imports research modules. The existing direction
   constraint (`volatility/` must not import `tools.forecast_platform`)
   generalizes: production consumes ledgers and frozen artifacts, never
   research code.
10. **DRY_RUN.** No module in this document creates an order.

---

## 6. Shared infrastructure

### 6.1 Trial ledger

The single largest failure mode of M01 is an understated trial count, and
discipline is not a control. Structural design:

- Append-only, same guarantees as `volatility/ledger.py`: exclusive lock,
  fsync, duplicate detection, no rewriting.
- One entry per hypothesis **at declaration time**, before results exist.
- Entry: hypothesis, pre-registered mechanism, frozen parameters, config hash,
  declaration timestamp.
- Outcome appended separately, referencing the declaration.
- **A result whose hypothesis has no prior declaration is inadmissible.** This
  is what makes the count honest: it cannot be reconstructed favorably after
  the fact.
- Retroactive population from git history for pre-C5.0 work, with unknowns
  rounded up.

### 6.2 Frozen artifact discipline

Every reference distribution, model artifact, cost model, regime model, and
weight policy follows the C4.4 pattern: content hash, mandatory pinning by
consumers, version string, and **refusal to overwrite**.

Note, recorded as a design input rather than a hypothetical: the C4.4a freeze
tool wrote its artifact before checking the registry, so a re-run silently
replaced an immutable artifact with a model trained on different data. The
registry correctly refused re-registration — after the file was already gone.
Every artifact writer specified here must check-then-write, and treat a hash
mismatch as a hard failure rather than an update.

### 6.3 Negative-control suite

Mandatory for M01, M05, M07, M08. Generate synthetic features with the
autocorrelation structure of the real ones but no predictive relationship, run
the full pipeline, and measure the false-positive rate.

If it exceeds nominal α, the module is broken and its verdicts on real features
are void. This is roughly an hour of work per module and is the only way to
*measure* the false-positive rate rather than assume it.

Includes sign-flip and within-regime block-preserving shuffles.

### 6.4 Data requirements not yet met

| Requirement | Needed by | Status |
|---|---|---|
| ETH, SOL OHLCV, same pipeline | M07 | not built |
| Multi-venue funding | M07, M11 | partial |
| Quarterly futures | M11 | not built |
| Multi-venue OI | M12 | partial |
| Liquidation feed | M12 | not built |
| L2 / tick data | M14 | not built, expensive |
| Fee schedule + slippage model | M00 | **must exist for Sprint 0** |

---

## 7. Priority rationale, consolidated

The ordering is deliberately unusual: validation apparatus outranks alpha
generation. The reasoning, in one line each.

| Rank | Module | Because |
|---|---|---|
| 0 | M00 | can terminate the program in one day; costs are not optional |
| 1 | M01 | changes the meaning of every result, past and future |
| 2 | M05 | only near-term path to better live results; avoids the direction problem |
| 3 | M02+M03 | fixes a ~12× overstatement of information inside training |
| 4 | M04 | distribution instead of a point; directly addresses the power problem |
| 5 | M06 | pooling across regimes hides both good and bad; M07 needs it |
| 6 | M07 | the question the whole apparatus exists to answer |
| 7 | M08 | falsification — the strongest discriminator, but needs M07's output |
| 8 | M09 | makes 15 months of slow accumulation useful from month one |
| 9 | M11 | best cost-to-value among new data |
| 10 | M12 | mechanical predictability, crypto-specific |
| 11 | M10 | honest uncertainty; point forecasts are misleading at R²≈0 |
| 12 | M13 | communication of uncertainty |
| 13 | M14 | sound but data-bound and horizon-mismatched |
| 14 | M15 | multiplier against an unknown multiplicand |

**The top four add no new features.** That is the central claim of this
architecture. A system that cannot distinguish +0.13 from zero does not become
better by acquiring more things to measure; it becomes better at measuring.

---

## 8. What is deliberately excluded

- **More indicators.** 19 features did not beat a constant. 460 would not.
- **LLMs in the decision path.** Non-determinism plus training-data leakage.
- **Automated hypothesis search.** Without strict multiplicity control it is a
  false-discovery generator, and the trial ledger cannot keep up with it.
- **Live execution changes.** Out of scope for the entire document.

---

## 9. Re-evaluation protocol

This specification is frozen, not permanent. It is revised only at a gate, and
a revision records: which gate, what was learned, what changed, and what the
change costs in trial count.

Scheduled re-evaluation after M00 and after phase A′. Both may cancel large
parts of what is written above — that is the intended behavior, and the earlier
it happens, the more it saves.

---

## 10. Amendment 1 — gate G0, 2026-08-11

**Gate:** G0. **Result:** PASS. Full numbers in `M00_RESULT.md`.

**What was learned.** The 4h swing horizon costs 0.19% per round trip and needs
a 37.6% win rate at 2R against 33.3% at zero cost. Survivable. The prediction
that cost would likely kill the swing engine was wrong, and the programme is
not cancelled. The binding variable turned out to be stop distance, not holding
period: below a ~0.8% stop, execution takes more than a fifth of every R.

**What changed.** One new prerequisite, inserted before phase A′:

> **P1 — retrofit funding into the shared backtest cost.** Seven call sites
> compute `cost_pct = (2*TAKER_FEE_PCT + SLIPPAGE_PCT)/100` and hold positions
> for hours without paying funding. Every net result the project has produced
> is overstated. Phase A′ re-evaluates past hypotheses, and re-running them
> against the same overstated costs would reproduce the original error with new
> machinery.

P1 is not a research module. It is a correctness fix to existing code, it
changes historical results, and it needs its own task and audit.

**Direction of the bias, for the record.** H1 and H2 were *rejected* on
overstated numbers, so correcting the bias makes those rejections stronger, not
weaker — neither reopens. The confluence score was never validated, and its
unvalidated numbers are overstated by the same amount.

**Modules cancelled:** none. **Modules re-prioritised:** none. **Trial count
impact:** none — M00 tested no hypothesis about the market.
